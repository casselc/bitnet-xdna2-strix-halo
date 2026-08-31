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
