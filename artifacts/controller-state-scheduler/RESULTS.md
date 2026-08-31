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
