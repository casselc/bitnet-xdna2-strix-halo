# Distinct-suffix batching, XDNA aggregation and the state-spine question

| | |
|---|---|
| branch base | `controller-cache-batching` @ `2ca2e51916f1aa3fddfae94ef4db4ceb02d02296` |
| branch | `controller-state-scheduler` |
| controller | BitNet-b1.58-2B-4T I2_S, promoted XDNA runtime, `llama-server` t8, `-np 8` |

| tag | meaning |
|---|---|
| **[MEASURED]** | observed on this machine |
| **[DERIVED]** | arithmetic over measured quantities |
| **[DEFERRED]** | not done, with the reason |
| **[INVALID]** | measured, then found wrong; retained with the reason |

---

## 0. Scope qualification of the previous branch [not a retraction]

`controller-cache-batching` measured, correctly:

- prompt-prefix reuse (1953 of 1954 tokens reused, TTFT 2357 → 33 ms)
- singleflight on identical requests (87.5% of executions avoided at burst 8)
- constrained output (544 → 70 → 20 ms)
- the cache/XDNA handoff at 1024 evaluated tokens

**Its final `~33 req/s at c=8` cell sent the same controller prompt to every
client thread.** That is concurrency class 1 (duplicate). It is a valid
measurement of that class and the numbers stand, but it must **not** be read as
evidence that distinct shared-prefix suffixes batch correctly or efficiently.
That is class 2, and it is what this branch measures.

---

## 1. Instrumenting the actual batch shape [MEASURED]

Throughput scaling cannot distinguish "several sequences combined into one large
matrix" from "several small matrices processed quickly". So the runtime now
records **ne11** — the token dimension of the batch reaching each I2_S
`mul_mat` — in power-of-two buckets, behind `BITNET_XDNA_NE11_STATS`. Disabled,
it costs one relaxed atomic load.

`bitnet_xdna_worth_it(ne11)` is called on *every* I2_S mul_mat, including the
ones it declines, which makes it the one honest place to observe this.

**Validated against known inputs before use:** a 2125-token request produces a
`b2048` bucket, a 1754-token request a `b1024` bucket.

*(Two harness defects were found and fixed before any result was believed: the
snapshot parser stopped at the first torn append rather than skipping it, so one
early partial line froze the snapshot and every window reported a zero delta;
and `max` is a running maximum since process start, so its delta is meaningless
— the window's largest batch is now taken from the highest bucket that moved.)*

## 2. Distinct suffixes DO batch [MEASURED]

Shared stable prefix, per-request unique suffix, `cache_prompt=true`, t8.
Raw: `distinct_batching.csv`.

| family | c | new/req | aggregate | top bucket | XDNA | req/s | TTFT p50 | total p95 |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| ~79 new | 1 | 95 | 95 | **64** | no | 3.09 | 280 | 323 |
| ~79 new | 4 | 95 | 380 | **256** | no | 3.995 | 886 | 1001 |
| ~79 new | 8 | 95 | 760 | **512** | no | 4.29 | 1660 | 1864 |
| ~379 new | 1 | 395 | 395 | 256 | no | 1.213 | 778 | 824 |
| ~379 new | 4 | 402 | 1608 | **1024** | **YES** | 1.571 | 2459 | 2546 |
| ~379 new | 8 | 402 | 3216 | **1024** | **YES** | 1.290 | 4109 | 6199 |
| ~759 new | 2 | 782 | 1564 | **1024** | **YES** | 0.902 | 2154 | 2217 |
| ~1139 new | 1 | 1155 | 1155 | 1024 | YES | 0.694 | 1375 | 1441 |
| ~1139 new | 8 | 1155 | 9240 | 1024 | YES | 0.604 | 5158 | 13244 |

**The top bucket tracks the aggregate, not the per-request size.** Four 95-token
suffixes produce a 256-bucket batch; eight produce a 512-bucket batch. The
sequences are genuinely combined into one token matrix.

**And aggregation does re-engage XDNA.** Four ~402-token suffixes aggregate to
1608 tokens and cross `kMTile`, as do two ~782-token suffixes at 1564. Requests
that individually would run on the CPU reach the NPU *because they were
batched*. The hypothesis in the brief is confirmed.

## 3. But it does not help [MEASURED — the important negative]

Throughput does not improve when XDNA re-engages, and tails degrade badly:

| family | c=1 | c=2 | c=4 | c=8 |
|---|---:|---:|---:|---:|
| ~379 new, req/s | 1.213 | 1.196 | **1.571** | 1.290 |
| ~759 new, req/s | 0.685 | 0.902 | 0.805 | 0.806 |
| ~1139 new, req/s | 0.694 | 0.604 | 0.572 | 0.604 |
| ~1139 new, total p95 | 1441 | 3309 | 6994 | **13244** |

The best case is ~379 at c=4: **1.571 vs 1.213 req/s, a 1.30x gain** — for
4x the in-flight work and 3.1x the p95 (824 → 2546 ms). At ~1139, where XDNA is
engaged at every concurrency, throughput *falls* with concurrency while p95 rises
9.2x.

This is **outcome D** of the brief's Task 4: aggregation happens, XDNA engages,
and the combination does not pay. Adding sequences adds proportional work; the
compute stream is already saturated by one request's worth of tokens, so
batching redistributes latency rather than creating throughput.

**The 1024-token threshold was not lowered to manufacture an offload**, and
should not be: the offload engages correctly and simply does not help here.

## 4. Correctness: no cross-sequence contamination [MEASURED]

Concurrent output was compared against each suffix run alone. Sporadic
mismatches appeared — ~3% of requests — and needed to be resolved before any
performance number could be reported.

**First reading was wrong and is retained here as [INVALID].** Using
sequential-integer session tags, four requests appeared to emit a neighbour's
tag, which looked like an off-by-one contamination signature. It was not: the
tags were guessable. The model emits plausible nearby numbers drawn from the
state text — in other trials it produced `ZQ123` and `ZQ-1`, tags belonging to no
request at all — so "emitted 9102 while being 9101" is a guess, not a leak.

Repeated with **10-hex-character random tags**, where emitting another request's
tag by chance is ~2^-40:

| | of 32 concurrent requests |
|---|---:|
| echoed its **own** tag | 7 |
| echoed **another request's** tag | **0** |
| echoed neither (base model did not copy) | 25 |

**No cross-sequence KV contamination.** The residual ~3% output variation is
floating-point non-determinism from differing batch shapes — reduction order
changes with batch composition, and a 4-token greedy continuation can flip when
two logits are near-tied. It is not a correctness failure of the batching path.

*(The base model copies its own tag only 7 times in 32. That weakens it as a
positive control but not as a contamination detector: the question is whether a
foreign tag ever appears, and it never does.)*

## 5. Micro-batch coalescing window [DEFERRED — gated task, gate did not open]

Task 5 was conditional on Task 3 showing that aggregate XDNA batching wins. Section 3
measured the opposite: distinct suffixes *do* coalesce into a large enough `ne11` to
re-engage the NPU, and doing so is slower than letting them stay on the CPU. A
coalescing window can only increase the amount of work steered into that losing path,
so it was not built. Reopen only if the Section 3 result reverses.

## 6. The state spine [MEASURED]

`tools/state_spine.py`. An `Authoritative` object holds an append-only event log and
derives a monotonic `version()`. The *spine* is the long canonical prefix built from
that log; a *query* is a short suffix appended to it. Three regimes over 50 turns:

| regime | eval/turn | reused/turn | TTFT p50 | total p50 |
|---|---|---|---|---|
| A — rebuild the whole prompt each turn | 2328 | 0 | 3023 ms | — |
| B — spine reuse, append query only | **28** | 2084 | **147 ms** | — |
| C — rebase every 10 turns | 28 | 2084 | 147 ms | 122 ms |
| C — rebase every 25 turns | 28 | 2084 | 145 ms | 130 ms |

Spine reuse removes **98.8% of prefill token work** (2328 -> 28 evaluated tokens per
turn) and cuts TTFT **20.6x**. Rebase cadence between 10 and 25 turns is not
distinguishable at this scale — the rebase itself is amortized over enough turns that
its cost disappears into the noise.

Note the direction this pushes the NPU question: regime A evaluates 2328 tokens per
turn, comfortably above `kMTile = 1024`, so it offloads. Regime B evaluates 28, far
below, so it never does. **The winning configuration is the one that keeps the NPU
idle.** This is the same tension recorded in the previous branch, now measured on the
stateful path rather than the stateless one.

## 7. Ephemeral query forks [MEASURED — 4/4 invariants hold]

`tools/state_spine.py`, Task 8. A fork is a throwaway continuation of the spine that is
never written back to the authoritative log.

| invariant | result |
|---|---|
| forks do not change the authoritative version | PASS |
| an authoritative delta *does* change the version | PASS |
| the projection equals an independent replay of S+D | PASS |
| a post-fork query matches a fresh-replay query | PASS |

So forking is safe: speculative branches leave no residue in the authority, and the
spine after forking is byte-identical to a spine that never forked.

## 8. Pre-warming [MEASURED]

`tools/prewarm_memo.py`. Four arms, each measuring the *query* TTFT after the spine has
(or has not) been placed in a slot ahead of time.

| arm | query TTFT | prewarm cost | notes |
|---|---|---|---|
| A — cold, no prewarm | 2665 ms | — | baseline |
| B — prewarmed spine | **34 ms** | 2507 ms | 78x faster query |
| C — prewarmed, GPU tenant active | 89 ms | 4057 ms | 100.8 W package |
| D — prewarmed, verifier tenant active | 33 ms | — | verifier held 1230 ops/s, p95 1.196 ms, p99 1.554 ms |

Prewarming moves the cost, it does not remove it: arm B pays 2507 ms up front to save
2631 ms at query time. That is only a win when the prewarm can be issued during idle
time that would otherwise be wasted — which is exactly the controller's duty cycle, and
is the case the state spine creates.

Co-tenancy degrades prewarm but not the warm query. Under GPU load the *prewarm* slows
1.6x (2507 -> 4057 ms) because it is real prefill competing for memory bandwidth, while
the warm query only moves 34 -> 89 ms. Under the CPU verifier the warm query is
unaffected (33 ms) and the verifier keeps full throughput. This matches the tri-device
result from the previous branch: the contended resource is bandwidth during prefill, not
the NPU and not the decode path.

## 9. Memoization [MEASURED — 8/8 invalidation cases correct]

`tools/prewarm_memo.py`. The memo key is the full tuple
`(model, tokenizer, tenant, authority, policy_version, state_version, objective,
tool_schema, grammar, temperature, seed, max_tokens)`.

All eight cases pass: a repeat with an identical key hits, and a change to *any* of
objective, state version, policy version, authority, tenant, or grammar misses. The two
identity fields — `tenant` and `authority` — are part of the key precisely so that a
token-prefix match can never by itself produce a reuse across a security domain
(Task 14). Prefix similarity is necessary for reuse but never sufficient; the key is
checked first.

## 10. Residency: how many warm state domains actually fit [MEASURED]

This started as an anomaly. A warm domain survived **20 other domains passing through an
8-slot server** and still came back with `eval=1`, which no slot-LRU model explains. The
first sweep made it worse: 8, 16, 32 and then 48 distinct domains all came back 100%
warm, with RSS completely flat.

**Mechanism, read from the pinned source.** Residency is not bounded by slots. There is a
second tier — `server_prompt_cache` (`server-task.cpp`), enabled by default with
`--cache-ram 8192` (MiB), which this controller runs with since it never passes the flag:

- `slot::prompt_save` copies a displaced slot's sequence state into the cache
  (`llama_state_seq_get_data_ext`).
- `slot::prompt_load` restores it and then `states.erase(it_best)` — the state is **moved
  out** of the cache, and `data.clear(); data.shrink_to_fit()` returns the pages.
- `server_prompt_cache::update()` evicts with `states.pop_front()` — plain LRU — against
  `limit_size` (8192 MiB) and `limit_tokens` (= `n_ctx`), where
  `limit_tokens_cur = max(limit_tokens, limit_size/size_per_token)`, so at our state size
  both limits collapse to the same ~8 GiB budget.

So the cache holds roughly *(domains visited − slots)* states, and RSS stays flat because
every load frees exactly what the next save allocates. **Slot count sets concurrency;
`--cache-ram` sets residency.** These are independent, and only the second one answers
"how many warm domains fit".

**Capacity, measured.** Predicted knee: 8192 MiB / ~110 MiB per 1491-token state ≈ 74.
Warm 128 distinct domains, then probe newest-to-oldest (LRU keeps the newest) and stop at
the first run of misses:

```
  ..........................................................................XXX
  74 warm, then a cliff
  warm TTFT median   26 ms
  cold TTFT median 1397 ms
```

**74 domains.** The implied per-state size is 8192 MiB / 74 = **111 MiB = 76.0 KiB/token**,
against **75.0 KiB/token** derived independently from the model geometry
(5 KV heads x 128 head_dim x 30 layers x 2 for K and V x 2 bytes f16). The two agree to
1.3%, so the capacity is fully explained by KV size and the RAM budget — there is nothing
unaccounted for.

**The knee is a cliff, not a slope: 26 ms -> 1397 ms, 54x, with no middle ground.**

### The failure mode that matters more than the capacity

The first attempt to measure this probed the 128 domains **in the order they were warmed**
and reported `0 / 128` warm. That number is real but it is not an eviction measurement —
it is LRU thrash. Revisiting an evicted domain rebuilds it and pushes a state in, which
`pop_front`s the entry that was about to be visited next. A cyclic scan over a working set
larger than the cache therefore degrades to a **0% hit rate**, not to a partial one, and
per-request latency fell from 26 ms to ~13 s while the controller burned 7.6 cores.

That is the honest operational warning: **exceeding cache capacity does not cost you a
proportional share of your hits, it can cost you all of them.** The reverse-order probe
above exists specifically because the forward probe destroys the evidence it is trying to
collect.

### Consequences

- ~74 concurrent warm state domains at ~1500 tokens each, on the default 8192 MiB budget.
  It scales linearly in `--cache-ram` and inversely in spine length — a 3000-token spine
  halves it to ~37.
- Residency is **tunable**: `--cache-ram` is the knob, and this box has 122 GiB, so a much
  larger budget is available if the working set demands it. It was not raised here because
  the brief's question was what the current configuration does.
- Admission control matters more than cache size. Because the degradation is a cliff and a
  cyclic pattern zeroes the hit rate, a scheduler should refuse or defer work that would
  push the working set past capacity rather than let it thrash. That is a real finding
  about *this* server's fixed model, exactly the bottleneck the brief asked to be recorded
  if it appeared. **It appeared.**

## 11. Batching background cache fills [MEASURED]

`artifacts/controller-state-scheduler/batch_fills.csv`. Issue 1, 2, 4 and 8 simultaneous
cold spine fills and measure how long the batch takes.

| simultaneous fills | wall | per-fill p50 | per-fill max | fills/s | speedup vs serial |
|---|---|---|---|---|---|
| 1 | 1565 ms | 1565 ms | 1565 ms | 0.64 | 1.00x |
| 2 | 3388 ms | 3387 ms | 3387 ms | 0.59 | 1.66x |
| 4 | 6202 ms | 6201 ms | 6201 ms | 0.64 | 3.12x |
| 8 | 13657 ms | 9440 ms | 13656 ms | 0.59 | 5.07x |

**Fills per second is flat at ~0.6 regardless of batch width.** The apparent "speedup vs
serial" is arithmetic, not throughput: batching 8 fills makes each individual fill 8.7x
slower (1565 -> 13656 ms worst case) while completing the same ~0.6 fills/s.

This is the expected result and it is worth stating plainly, because it is easy to
misread the 5.07x column as a win. Cold fills are bandwidth-bound prefill, and the box
was already bandwidth-saturated at one fill. Batching background fills buys nothing on
throughput; it only decides who waits. The one thing it does buy is *predictability* —
issue them one at a time and each completes in a known ~1.6 s, which is what a background
refresh scheduler actually wants.

## 12. Cache affinity as a scheduling signal [MEASURED — it is not one]

`id_slot` is settable per request, so warm-correct / wrong-prefix / idle-cold routing can
be **forced** rather than inferred from `slot_prompt_similarity`.

| spine | routing | eval | reused | prefill p50 |
|---|---|---|---|---|
| 1490 tok | warm-correct slot (0) | 1 | 1490 | 17.4 ms |
| 1490 tok | wrong-prefix slot (1) | 1 | 1490 | 17.2 ms |
| 1490 tok | idle cold slot (5) | 1 | 1490 | 17.4 ms |
| 2306 tok | warm-correct slot (0) | 1 | 2306 | 19.5 ms |
| 2306 tok | wrong-prefix slot (1) | 1 | 2306 | 19.8 ms |
| 2306 tok | idle cold slot (5) | 1 | 2306 | 19.7 ms |

**Routing to the "wrong" slot costs nothing** — `eval_n` is 1 in every case and the spread
across all six cells is 0.6 ms, well inside noise. The reason is Section 10: the RAM
prompt cache sits underneath the slots and rescues any routing decision by restoring the
state into whichever slot was picked. Slot affinity is only a signal on a server *without*
that second tier.

So the answer to "should the scheduler use cache affinity?" is **no, use residency**. The
question that matters is not *which slot* holds a domain but *whether the domain is still
within the ~74-entry cache at all* — that is the difference between 17 ms and 1397 ms, and
it is the only cache signal on this server worth scheduling against.

## 13. Action scoring [MEASURED — the cheap method does not work, the affordable one does]

Five candidate actions (`SCALE_UP`, `SCALE_DOWN`, `ROLLBACK`, `HOLD`, `PAGE_ONCALL`)
against a warm spine.

**(a) One forward pass, read the top-20 first-token distribution — 20 ms, and it fails.**
The returned distribution is dominated by formatting tokens, not action tokens:

```
   0  ' '        p=0.1518      5  ' d'       p=0.0192
   1  ' ['       p=0.1401      6  ' p'       p=0.0175
   2  ' SELECT'  p=0.0405      7  ' Choose'  p=0.0175
   3  ' <'       p=0.0302      8  ' Select'  p=0.0161
   4  ' action'  p=0.0281      9  ' "'       p=0.0120
```

**Zero of the five actions are reachable from the top-20 first tokens.** A 2B model at
this prompt is still deciding on syntax at the first position, so first-token logprobs are
not a usable action score. This would have been an easy thing to assume works.

**(b) Score each action explicitly — 130 ms for five, and it does work.**

| action | eval | reused | latency |
|---|---|---|---|
| SCALE_UP | 3 | 1518 | 30.2 ms |
| SCALE_DOWN | 1 | 1520 | 20.1 ms |
| ROLLBACK | 3 | 1518 | 30.0 ms |
| HOLD | 1 | 1518 | 19.6 ms |
| PAGE_ONCALL | 4 | 1518 | 30.4 ms |

Every candidate reuses the full spine and evaluates only its own 1-4 token suffix, so
**K-way scoring costs K short suffix evals, not K prefills** — ~26 ms per candidate. That
is the whole reason the spine makes scoring affordable: without prefix reuse each of these
would be a 1.5 s rebuild and a five-way decision would cost 7.5 s instead of 130 ms.

A grammar-constrained single pass would likely also work and is cheaper still, but that
was not measured here and is not claimed.

---

# Final answers

**1. Can distinct cached suffixes batch so their aggregate new-token work forms a large
matrix and re-engages XDNA?** **Yes — measured, and it does not pay.** Distinct suffixes
do coalesce past `kMTile = 1024` and the NPU does engage (Section 2). Throughput does not
improve: the best cell is 1.571 vs 1.213 req/s (1.30x) for 4x the in-flight work and 3.1x
the p95, and at ~1139 new tokens throughput *falls* with concurrency while p95 rises 9.2x
(Section 3). Outcome D of Task 4.

**2. Can we maintain a disposable model-KV state spine, fork ephemeral branches, and
pre-warm/rebase in the background?** **Yes, on all three counts, and this is where the
wins are.** Spine reuse removes 98.8% of per-turn prefill (2328 -> 28 evaluated tokens)
and cuts TTFT 20.6x (3023 -> 147 ms). Forks pass all four invariants. Prewarming makes a
query 78x faster (2665 -> 34 ms).

**3. What is the real per-turn cost of a stateful controller?** 28 evaluated tokens,
~147 ms TTFT, ~122-130 ms total at a 10-25 turn rebase cadence. Against 2328 tokens and
3023 ms for the stateless equivalent.

**4. Does the NPU help the stateful path?** **No, and that is the honest headline.** The
regime that offloads (rebuild, 2328 tokens) is the slow one; the regime that wins (spine
reuse, 28 tokens) is permanently below `kMTile` and never dispatches. Every optimisation
that helped in this brief moved work *away* from the NPU.

**5. What is the cost of a scored decision?** ~26 ms per candidate, 130 ms for five
(Section 13). The one-pass first-token method is 20 ms and does not work — none of the
five actions appears in the top-20.

**6. Does batching background cache fills help?** No. Fills/s is flat at ~0.6 across
1/2/4/8 (Section 11). It changes who waits, not how much gets done.

**7. Is cache affinity a useful scheduling signal?** No (Section 12). Forced routing to a
wrong-prefix or idle-cold slot is indistinguishable from the warm-correct slot — 0.6 ms
spread across six cells — because the RAM prompt cache restores state into whatever slot
is chosen.

**8. What signal *should* the scheduler use?** Residency. Whether a domain is inside the
~74-entry prompt cache is a 17 ms vs 1397 ms difference — 82x, and the only cache
distinction on this server large enough to schedule against.

**9. How many warm state domains fit comfortably?** **74**, at ~1500 tokens each, on the
default `--cache-ram 8192`. Implied per-state size 111 MiB = 76.0 KiB/token, matching the
75.0 KiB/token derived from model geometry to 1.3% (Section 10). Scales linearly in
`--cache-ram` and inversely in spine length. Separately, each *slot* is capped at
`n_ctx / n_parallel` = 2560 tokens here, so spine length and concurrency trade off
directly — a 2560-token spine allows 8 concurrent sequences, a 5120-token spine allows 4.

**10. What happens past capacity?** A cliff, not a slope: 26 ms -> 1397 ms. And a cyclic
access pattern over an oversized working set collapses to a **0% hit rate** rather than a
partial one, with per-request latency reaching ~13 s (Section 10). This is the strongest
argument for admission control in the whole pass.

**11. Is state reuse safe?** Yes, within the boundary that was built. No cross-sequence
contamination in 32/32 high-entropy trials (Section 4); forks leave the authoritative
version untouched (Section 7); memoization invalidates on all 8 cases including tenant,
authority and policy version (Section 9). `tenant` and `authority` are part of the memo
key precisely so a token-prefix match can never on its own authorise reuse (Task 14).

**12. Does co-tenancy break any of this?** No. Under an active GPU worker the *prewarm*
slows 1.6x (bandwidth-bound prefill) but the warm query only moves 34 -> 89 ms; under the
CPU verifier the warm query is unaffected at 33 ms and the verifier holds 1230 ops/s
(Section 8). Consistent with the tri-device result on the previous branch.

---

# Recommendation

**Build the controller around a pre-warmed state spine with residency-aware admission
control, and stop trying to route work to the NPU.**

Concretely: keep one canonical spine per state domain, append only the query suffix, rebase
every 10-25 turns, pre-warm during idle time, and score candidate actions by explicit
per-action suffix evaluation. Admit work only while the active domain set stays inside the
cache — raise `--cache-ram` to widen it, but treat the limit as hard, because exceeding it
costs every hit rather than a proportional share.

This gives ~147 ms TTFT and ~26 ms per scored action against 3023 ms and 1.5 s for the
stateless equivalent, and it does so with the NPU idle.

**On the NPU, stated plainly.** Across this brief, every measured improvement reduced the
evaluated-token count and therefore moved work below `kMTile` and off the NPU. The one
configuration that reliably engages XDNA — full prefill every turn — is the one worth
20.6x less. The offload is not broken and was not disabled or retuned to reach this
conclusion; it engages correctly when the shape calls for it (Section 2) and simply does
not pay at controller scale (Section 3). **This does not generalise beyond the 2B
controller measured here**: a larger model, a longer irreducible suffix, or a workload
without a reusable prefix would all put more work above the threshold, and none of those
were measured.

**Reopening conditions.** Revisit the NPU for this path if any of: the irreducible
per-turn suffix exceeds ~1024 tokens for structural reasons; the model grows enough that a
28-token step is no longer bandwidth-trivial; or `kMTile` is re-derived against a
materially different GEMM kernel. Absent one of those, the spine is the optimisation and
the NPU is not.
