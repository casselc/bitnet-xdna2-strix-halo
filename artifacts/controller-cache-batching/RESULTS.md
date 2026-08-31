# Controller prompt-cache, prefix reuse and batching

The previous pass measured the controller service with `cache_prompt=False` on
every request. This determines what changes when the reuse mechanisms the pinned
server already implements are switched on.

| | |
|---|---|
| branch base | `service-cotenancy` @ `9295df0a6167eaa43c983c13f702fce1033e4b1f` |
| branch | `controller-cache-batching` |
| controller | BitNet-b1.58-2B-4T I2_S, promoted XDNA runtime, `llama-server` t8, `-np 8` |
| prompt | structured system-state report, **1954 tokens**, identical across requests |
| semantics | `SEMANTICS.md` (read from the pinned source, not upstream docs) |

| tag | meaning |
|---|---|
| **[MEASURED]** | observed on this machine |
| **[DERIVED]** | arithmetic over measured quantities |
| **[DEFERRED]** | not done, with the reason |
| **[INVALID]** | measured, then found wrong; retained with the reason |

Token accounting is kept explicit throughout — **supplied**, **reused**,
**evaluated** — because the server reports `prompt_n` as tokens *evaluated* and
`cache_n` as tokens *reused*, so the supplied prompt is their sum. Reading
`prompt_n` as "prompt size" is wrong the moment caching engages.

---

## 1. The identical-prompt A/B [MEASURED]

Same warm service, same 1954-token prompt, only `cache_prompt` differs.
Raw: `concurrency_cache0.csv`, `concurrency_cache1.csv`.

| | `cache_prompt=false` | `cache_prompt=true` |
|---|---:|---:|
| tokens supplied | 1954 | 1954 |
| **tokens reused** | **0** | **1953** |
| **tokens evaluated** | **1954** | **1** |
| prompt_ms | 2340 | **17** |
| **TTFT** | **2357 ms** | **33 ms** |
| total, 32 output tokens | 2890 ms | **540 ms** |

**TTFT falls 71x and total latency 5.4x.** The previous pass's 2366 ms TTFT
characterises full prefill on every request. That is a real operating point — it
is what cold, all-distinct work costs — but it is not steady state, and the
earlier figure must be read with that scope.

## 2. Concurrency reverses [MEASURED]

t8, 24 requests per cell, 2 warmup requests discarded so a cold slot is not
averaged into a warm measurement.

| c | req/s off | **req/s on** | total p50 off | **total p50 on** | queue p95 off | **queue p95 on** | TTFT p50 on |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.345 | **1.834** | 2902 | **544** | 4 | 4 | 33 |
| 2 | 0.345 | **2.348** | 5798 | **853** | 2359 | **4** | 52 |
| 4 | 0.363 | **3.632** | 11040 | **1099** | 2529 | **5** | 67 |
| 8 | 0.382 | **4.299** | 20903 | **1863** | 12433 | **7** | 115 |

**The "saturates at concurrency 1" conclusion does not survive.** With reuse on,
throughput scales **2.34x from c=1 to c=8** (1.834 → 4.299 req/s) and queue wait
stays at **4–7 ms** instead of growing to 12.4 s. At c=8 the cached service does
**11.3x** the work of the uncached one.

Latency still grows with concurrency — 544 → 1863 ms total — but that is decode
sharing a fixed compute stream, not admission queueing: the queue component is
7 ms of 1863 ms.

## 3. Prompt caching and NPU offload are mutually exclusive here [MEASURED]

The finding with the widest architectural consequence, and it was not the
question being asked.

The runtime offloads a `mul_mat` only when the micro-batch has at least
`kMTile = 1024` tokens. With the prefix reused, a repeat request evaluates
**1 token**, so the threshold is never met:

| request | tokens evaluated | **NPU lease acquisitions per request** |
|---|---:|---:|
| cold, `cache_prompt=false` | 1954 | **~149** |
| warm, `cache_prompt=true` | 1 | **0** |

**A warm cached controller request never touches the NPU.** It runs entirely on
Zen 5: prefill is one token, and decode was already the CPU GEMV path.

This does not make the offload worthless — it accelerates exactly the case
caching cannot help, namely **cold prefill of state the slot has not seen**. But
it reframes the architecture:

> XDNA2 accelerates **cache misses**, not steady state.

It also explains the `leasewait=None` in every cached cell above: there were no
lease acquisitions to measure, not a broken counter.

## 4. Cache correctness [MEASURED]

Prompt reuse is an optimisation, not authority, so outputs were compared before
any speedup was believed. Identical prompt, temperature 0, seed 42:

| condition | output matches uncached reference |
|---|---|
| cache off (reference) | — |
| cache on, cold slot | **yes** |
| cache on, warm repeat | **yes** |
| cache on, **after an unrelated prompt through the slot pool** | **yes** |
| cache off again | **yes** |

No stale KV bleed across prompts. The post-unrelated request still reused 1953
tokens, so slot routing recovered the correct slot rather than silently
re-prefilling.

**Caveat on what this validates.** The base BitNet model continues the state
report rather than emitting an action verb; it is not instruction-tuned for this
prompt. That is adequate for a timing benchmark and for detecting KV
contamination — identical inputs must give identical outputs either way — but it
means the action-selection experiments measure decode mechanics, not decision
quality.

---

## 5. Production-shaped prefix reuse [MEASURED]

The identical-prompt test only proves the mechanism. A real controller sends a
stable context followed by **changed** state, so the question is how reuse
degrades as the shared fraction falls. Families are sized in **tokens** via the
server's own tokenizer — character ratios differ from token ratios by ~2x on
this structured text. Raw: `shared_prefix.csv`.

| family | supplied | reused | evaluated | TTFT p50 | total p50 |
|---|---:|---:|---:|---:|---:|
| ~95% shared | 2112 | 2034 | **79** | **257 ms** | 778 ms |
| ~75% shared | 2007 | 1629 | 379 | 750 ms | 1277 ms |
| ~50% shared | 1874 | 1116 | 759 | 1369 ms | 1908 ms |
| ~25% shared | 1741 | 603 | 1139 | 1363 ms | 1913 ms |
| ~5% shared | 1636 | 198 | 1439 | 1475 ms | 2014 ms |
| **control: stable text placed LAST** | 1850 | **1** | 1850 | **2208 ms** | 2762 ms |

**TTFT tracks tokens *evaluated*, not tokens supplied.** At the realistic
production shape (~95% stable context, changed state appended) warm TTFT is
**257 ms** against 2357 ms uncached — **9.2x**.

**The control matters as much as the families.** Identical amounts of shared
text, but placed *after* the changing part, reuses exactly **one token** and
costs 2208 ms. The cache matches a common *prefix*; a prompt layout that varies
its head throws the entire benefit away. Stable-context-first is not a style
preference, it is the whole mechanism.

### The cache and the NPU are complementary, and hand off at 1024 tokens [MEASURED]

~50% and ~25% have near-identical TTFT (1369 vs 1363 ms) despite 759 vs 1139
evaluated tokens. That straddles `kMTile`, so it was checked rather than
explained away:

| family | evaluated | lease acquisitions / request | NPU | TTFT |
|---|---:|---:|---|---:|
| ~95% | 79 | 0 | no | 268 ms |
| ~50% | 759 | 0 | no | 1360 ms |
| ~25% | **1139** | **139** | **YES** | 1376 ms |
| ~5% | **1439** | **149** | **YES** | 1459 ms |

The NPU engages exactly when evaluated tokens cross **1024**, and 50% more work
then costs 1% more time because the NPU absorbs it. The two mechanisms cover
opposite regimes:

> **Prompt cache owns the high-reuse regime. XDNA2 owns the low-reuse regime.
> They meet at `kMTile = 1024` evaluated tokens.**

*(A first attempt at this probe reported `eval=1` for the low-reuse families: the
prompts were still resident in slots from the preceding benchmark run, so the
probe measured a cache hit rather than the intended miss. Re-run with epochs no
earlier run had used.)*

## 6. Semantic singleflight [MEASURED]

N identical deterministic requests should produce **one** model execution.
Coalescing is keyed on model, prompt version, tenant, policy version, authority,
temperature, seed and output length — not on prompt bytes — because reuse across
a policy version or security domain would be an isolation bug, not an
optimisation. Raw: `singleflight.csv`.

| burst | server queue: executions / wall | singleflight: executions / wall | speedup | outputs identical |
|---:|---|---|---:|---|
| 2 | 2 / 1196 ms | **1 / 569 ms** | 2.1x | yes |
| 4 | 4 / 4132 ms | **1 / 570 ms** | 7.2x | yes |
| 8 | 8 / 9006 ms | **1 / 570 ms** | **15.8x** | yes |

Singleflight latency is **flat at ~570 ms regardless of burst size** — it is one
execution plus a wait. Executions avoided: 50%, 75%, **87.5%**.

The previous pass's concurrency benchmark submitted identical requests and let
them queue as independent work. For a deterministic controller that is the wrong
serving semantics, and it inflated the apparent cost of concurrency.

## 7. Controller output size [MEASURED]

Once the prefix is reused, **decode is the whole request**. Raw:
`short_output.csv`.

| variant | generated | decode | total | vs prose |
|---|---:|---:|---:|---:|
| A 32-token prose | 32 | 524 ms | 544 ms | — |
| C JSON grammar `{"action":"SCALE"}` | 11 | 214 ms | 240 ms | 2.3x |
| B action verb only | 4 | 50 ms | 70 ms | **7.7x** |
| D single token | 1 | 0 ms | **20 ms** | **27x** |

Grammar-constrained output works in the pinned stack with no new decoding
machinery. **Shrinking controller output is now worth more than any remaining
prefill work**, which was not true when prefill cost 2.4 s.

## 8. Final service concurrency [MEASURED]

Best honest combination: prefix reuse on, action-only output. t8, 24 requests
per cell, 2 warmup discarded. Raw: `concurrency_final_cache_short.csv`.

| c | req/s | TTFT p50 | TTFT p95 | total p50 | total p95 | queue p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13.469 | 31 | 34 | 74 | 81 | 4 |
| 2 | 18.151 | 45 | 46 | 108 | 119 | 19 |
| 4 | 27.944 | 60 | 62 | 144 | 149 | 4 |
| 8 | **32.916** | 105 | 109 | **244** | 246 | **6** |

Against the previous pass at c=8 (0.382 req/s, total p50 20903 ms):
**86x the throughput and 86x lower latency.** Throughput still scales at c=8
(2.44x from c=1), so the ceiling was not found. Queue wait stays ~6 ms.

---

## 9. Three kinds of concurrency, kept separate [DERIVED]

The previous pass measured **class 1 as though it were class 3**, which is the
root of its concurrency conclusion.

| class | what it is | right mechanism | measured cost at burst/c=8 |
|---|---|---|---|
| **1. duplicate** | same semantic request | **singleflight** | 1 execution, 570 ms flat |
| **2. shared-prefix distinct** | same stable context, changed state | **prefix reuse** (+ NPU past 1024 evaluated tokens) | 33 req/s, total p50 244 ms |
| **3. cold distinct** | unrelated state, nothing shared | genuine queueing | 0.38 req/s, total p50 20.9 s |

Class 3 is the only one that behaves like the previous pass described — and its
numbers there are correct *for that class*.

## 10. Security / cache identity [DERIVED — design note, not implemented]

The minimum coordinate under which two requests may share an execution or a
cached result:

```
model coordinate          + tokenizer coordinate
stable prompt-prefix hash + system-prompt version
tenant / security domain  + policy / authority version
tool schema version       + inference params affecting semantics
                            (temperature, seed, max output, grammar)
```

`tools/singleflight_bench.py` keys on this tuple rather than prompt bytes, and
all reuse in this pass stays inside one explicit benchmark domain. Two properties
must hold in any real implementation and are stated here so they are not
rediscovered later: **a policy or state version change must invalidate**, and
**reuse must never cross a tenant or authority boundary**. Prompt-byte equality
is not sufficient for either.

Slot-level KV reuse in the pinned server is already tenant-unsafe by
construction: any request may be routed to any slot, so a shared deployment would
need per-domain slot pools or per-request cache disabling. Not a problem for a
single-tenant controller; a blocker for a shared one.

## 11. Not done [DEFERRED]

- **Explicit 20–50 turn state-delta sequence with periodic rebase (Task 5).**
  The ~95%-shared family is the same shape — stable context, changed state
  appended — and gives the amortised number (257 ms TTFT, 79 evaluated tokens).
  What is *not* measured is KV growth over many turns and the cost of a rebase.
  The architectural invariant is untested, not violated: nothing here makes the
  KV cache authoritative, and every measured request reconstructs from a
  full prompt the client holds.
- **Completed-result memoization (Task 7).** Singleflight covers the in-flight
  case; a TTL/version cache is a small addition but needs the invalidation
  semantics of section 10 to be real rather than benchmarked.
- **Multi-sequence batching of distinct suffixes (Task 8).** The pinned server
  has `--cont-batching`, but proving no KV cross-contamination between sequence
  IDs needs a correctness harness this pass did not build. This is the most
  interesting remaining question, because class-2 requests sharing a prefix are
  exactly what a shared batch should exploit — and the XDNA path wants ≥1024
  tokens per dispatch, which several distinct suffixes could reach together
  where one cannot.
- **Logit/candidate scoring for action selection (Task 11).** The single-token
  variant (20 ms) bounds what it could buy; the remaining gain is small.
- **Heterogeneous NPU/CPU lanes (Task 14)** and **worker MTP (Task 15)** — no
  concrete need arose, and the brief gates both on one.

## 12. Verdict

### **CACHE/DELTA CONTROLLER ARCHITECTURE VALIDATED**

Every reuse mechanism the pinned server implements works, is correct, and was
switched off by the previous benchmark. With them on, the controller service is
a different machine.

### The eight questions

| # | question | answer |
|---|---|---|
| 1 | warm TTFT, production-shaped reused prefix | **257 ms** (~95% shared); **31 ms** for an identical prompt |
| 2 | amortised TTFT for state-delta turns | **257 ms**, 79 evaluated tokens of 2112 supplied |
| 3 | full executions avoided by singleflight | **50% / 75% / 87.5%** at burst 2 / 4 / 8 |
| 4 | can distinct shared-prefix requests batch? | **[DEFERRED]** — mechanism present, correctness harness not built |
| 5 | new maximum useful client concurrency | **≥ 8, ceiling not found** — 32.9 req/s at c=8 and still scaling |
| 6 | is admission control still necessary? | **Only for cold distinct requests (class 3).** Not for classes 1 or 2 |
| 7 | how much does shorter output help? | **7.7x** (544 → 70 ms); **27x** for a single token |
| 8 | is 2.4 s TTFT still the student target? | **No.** The target is **31–257 ms**, and the binding cost is now decode |

### The controller service budget, revised

| quantity | previous pass | **this pass** |
|---|---:|---:|
| warm TTFT | 2366 ms | **31–257 ms** |
| total latency, small output | 2914 ms | **70 ms** (action) / 244 ms (c=8) |
| useful concurrency | 1 | **≥ 8** |
| req/s at c=8 | 0.382 | **32.9** |

### What this means for the controller specialist

The previous pass concluded that TTFT was the number to design against and that
a specialist needing less context would help most. **That is now wrong.** With
prefix reuse, prefill of a stable context is nearly free, and the binding cost is
**decode of the controller's own output**. The design pressure moves from *less
context* to **fewer output tokens** — a specialist that emits one structured
action rather than a sentence of justification is worth 7.7x to 27x, where
further context reduction is worth little.

### And one consequence for the hardware

**A warm cached controller request never touches the NPU** (section 3). XDNA2
earns its place on **cold prefill and cache misses**, which is precisely where
the cache cannot help — the two are complementary, meeting at 1024 evaluated
tokens. A deployment serving mostly-warm controller traffic would see the NPU
idle; one serving diverse or first-touch state would not. This does not
invalidate the runtime promotion, but it does mean **NPU utilisation is a
function of cache hit rate**, which no previous pass measured or could have.
