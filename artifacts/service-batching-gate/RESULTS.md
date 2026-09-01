# Service batching gate: is controller concurrency ~1 fundamental or configured?

| | |
|---|---|
| branch base | `service-cotenancy` @ `9295df0a6167eaa43c983c13f702fce1033e4b1f` |
| branch | `service-batching-gate` |
| controller | BitNet-b1.58-2B-4T I2_S, promoted XDNA runtime, `llama-server` |

| tag | meaning |
|---|---|
| **[MEASURED]** | observed on this machine |
| **[DERIVED]** | arithmetic over measured quantities |
| **[CORRECTION]** | a prior published claim found wrong here |
| **[DEFERRED]** | not done, with the reason |

`service-cotenancy` is frozen. Nothing on this branch rewrites it; corrections are
recorded here and point at what they correct.

---

## 1. Two qualifications to the prior result

### Qualification A — what the concurrency result does and does not prove [DERIVED]

`service-cotenancy` §3 proves, and this branch does not dispute:

> throughput saturates in the tested server configuration, and every additional
> in-flight request adds roughly its full service time to everyone's latency.

It does **not** prove "the hardware supports only concurrency 1". The tested
configuration was a single point:

```
-np 8   -b 2048   -ub 2048   -t {4,6,8}   (-tb never set, so -tb == -t)
controller prompt = 1954 tokens, n_predict = 32
```

Two 1954-token prompts are 3908 prompt tokens, which cannot occupy one 2048-token
physical batch. Whether a larger `-b`/`-ub` admits useful multi-slot batch formation
was never measured. The prior conclusion is therefore correctly scoped as *"in this
configuration"*, and this branch measures whether the configuration is the cause.

### Qualification B — the lease-hold sentence is wrong [CORRECTION]

`service-cotenancy` §3 states:

> The lease is held for essentially the whole prefill — mean hold **14.87 ms** over
> **~210** invocations per 1954-token request

Neither number appears in any committed cell. Re-derived from the raw CSVs
(`lease_acquisitions / requests`, and `lease_hold_ns / lease_acquisitions`):

| cell | requests | acquisitions | acq/request | mean hold |
|---|---:|---:|---:|---:|
| conc-t6-c1 | 16 | 2336 | **146.0** | **7.08 ms** |
| conc-t6-c2 | 16 | 2368 | 148.0 | 6.63 ms |
| conc-t4-c1 | 12 | 1760 | 146.7 | 7.31 ms |
| conc-t8-c1 | 12 | 1760 | 146.7 | 7.30 ms |
| baseline-C-t8 | 5 | 704 | 140.8 | 7.96 ms |
| mixed-t6-c2 | 8 | 1152 | 144.0 | 11.18 ms |
| soak-t8 | 150 | 14688 | 97.9 | 7.65 ms |

**Across every committed cell the range is 97.9–148.0 acquisitions/request and
5.54–11.18 ms mean hold. No cell is near 210 or 14.87 ms.**

`210` is very likely the *architectural* figure from the project plan — 7 I2_S
matmuls x 30 layers = 210 offloadable linear nodes per prefill — carried into the
prose as if it had been measured. The measured 146 is ~70% of it, which is a real
and separate question (which nodes decline the offload) that this branch's `ne11`
histogram is designed to answer.

The qualitative claim is also overstated. For `conc-t6-c1`, summing `prompt_ms`
over the 16 committed per-request records in `requests.jsonl` gives 43.2 s of
prefill against 16.5 s of total lease hold:

**lease hold is 38% of prefill, not "essentially the whole" of it.**

What survives unchanged: lease wait is exactly zero and the contended fraction is
zero at every concurrency, validated against a deliberately contended run. The
lease is still not the bottleneck. Only the size of the held region was misstated.

---

## 2. Measurement hygiene [MEASURED]

Three defects in the prior harness were fixed before any new numbers were taken.
All three were hypotheses in the brief; two are confirmed as real contamination.

### H2 CONFIRMED — the first request after a restart is not warm

`/health` returning ok does not mean the XDNA path is ready: the runtime expands
ternary weights to int8 and uploads them to device buffers lazily, on the first
prefill that clears the offload threshold. `tools/service_warmup.py` now issues real
1954-token requests after every controller start and refuses to proceed until two
consecutive prefills agree within 5%.

Two independent controller restarts, `t8 -b 2048 -ub 2048 -np 8`:

| restart | cold prefill | steady prefill | one-time cost |
|---|---:|---:|---:|
| 1 | 3021.3 ms | 2323.0 ms | **+698.3 ms (+30.1%)** |
| 2 | 2967.0 ms | 2383.1 ms | **+583.9 ms (+24.5%)** |

**The first request after every restart carries a ~0.6 s one-time residency cost.**
`service-cotenancy` cells were as small as 5 requests, so a single contaminated
first request moved a 5-sample mean by ~120 ms and could dominate a max or p95
outright. This is now excluded from statistics and recorded separately, because the
cold cost is real operational information about restart behaviour.

### H3 CONFIRMED — TTFT was reconstructed, not observed

The prior harness computed `ttft = prompt_ms + predicted_ms/predicted_n` from the
server's own timings. That cannot see admission queueing, scheduler delay or
transport. A streaming (SSE) path now timestamps the first chunk that actually
carries content. Both are kept:

- `client_ttft_ms` — measured first-token arrival (now reported as `ttft_ms`)
- `ttft_derived_ms` — the old reconstruction, retained under an explicit name

At `c=1` warmed they agree closely (2538.23 measured vs 2546.77 derived), which is
the expected result when nothing is queueing — the divergence to look for is under
concurrency, where the reconstruction is blind by construction.

### `queue_ms` renamed to `non_compute_wall_ms` [CORRECTION]

The prior harness reported `client wall - server compute` as `queue_ms`. That
quantity bundles admission queueing, scheduler delay, HTTP transport and client-side
scheduling. It is now named for what it is, and a `SlotSampler` polls `/slots` at
20 Hz so statements about admission rest on server-reported slot state rather than a
subtraction.

## 3. What token batch sizes actually reach XDNA [MEASURED]

New `ne11` histogram at the ggml/XDNA boundary (`bitnet_xdna_observe_node`). The
offload gate is evaluated by all `nth` threads, so the observer is called from
thread 0 only — instrumenting inside `worth_it()` would overcount every node by a
factor of `nth`. Declined nodes are recorded too, which resolves an open question.

One warmed 1954-token request, `n_predict=1`:

| quantity | value |
|---|---:|
| nodes seen (one prefill graph) | **210** |
| declined, ne11 below threshold | 3 |
| declined, **shape unsupported** | **60** |
| offloaded to XDNA | **147** |
| ne11 histogram of offloaded work | `1024:147` (all in [1024, 2048)) |

### Why a prefill produces ~147 invocations and not 210 [MEASURED]

This closes the gap that Qualification B exposed. `plan_for()` requires
`N ∈ {2560, 6912}`, but BitNet-2B's `attn_k` and `attn_v` are `[2560, 640]` —
**2 of the 7 I2_S matmuls per layer have no NPU plan**, so 2 x 30 = **60 nodes per
prefill decline for shape** and run on the CPU. 210 - 60 = 150 offloadable, of which
147 are observed.

So the earlier prose figure of "210 invocations" was the architectural node count,
and the measured 146-148 is that count minus the k/v projections. Both numbers are
now explained rather than one being asserted over the other.

**All offloaded work sits in a single bucket, ne11 ∈ [1024, 2048).** That is the
baseline against which Task 5 asks its question: if two concurrent 1954-token
requests ever formed one batch, offloaded work would appear in the [2048, 4096)
bucket. It does not, at `-b 2048 -ub 2048`.

## 4. Lease instrumentation overhead [MEASURED — negligible]

Every service timing on this branch depends on the lease CSV writer being cheap, so
it was measured. The env var is read once at startup, so ON and OFF cannot be
interleaved per request; they are interleaved per **cell**, alternating which arm
leads each round so a monotone drift (thermal, page cache, another tenant) cannot
line up with one arm the way a block-ordered A-then-B run would.

`t8 c1`, warmed, 1954-token prompt, `n_predict=1`, 3 rounds x 12 requests per arm:

| round | order | OFF prefill mean | ON prefill mean |
|---|---|---:|---:|
| 0 | off, on | 2341.5 ms | 2377.6 ms |
| 1 | on, off | 2406.0 ms | 2378.4 ms |
| 2 | off, on | 2374.8 ms | 2382.3 ms |

Pooled, n=36 per arm:

```
OFF  mean 2374.09 ms   p50 2381.61
ON   mean 2379.43 ms   p50 2384.05
relative effect  +0.23%   (bootstrap 95% CI  -0.41% .. +0.87%)
```

**LEASE INSTRUMENTATION OVERHEAD NEGLIGIBLE.** The confidence interval contains
zero and both bounds are inside ±1%, so the instrumentation stays on for every
experiment on this branch. Note the round-1 ordering: OFF was *slower* than ON in
that round, which is what run-to-run dispersion at this scale looks like and is the
reason a single A/B pair would not have supported the claim.

Raw: `lease_overhead.csv`, `lease_overhead.json`.

## 5. The primary discriminator: does `-b/-ub 2048` cause "concurrency ~1"? [MEASURED]

`tools/batch_gate.py`. Warmed `t8`, 1954-token prompts, `n_predict=32`, 8 requests per
cell, every cell inside the ne11 histogram, lease window, `/slots` sampler and power.

| cell | req/s | TTFT p50 | TTFT p95 | total p95 | ne11 max | lease acq | slots busy | W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b2048 np1 c1 | 0.345 | 2340 | 2378 | 2930 | 1024 | 1184 | 0.83 | 95 |
| b2048 np2 c1 | 0.343 | 2368 | 2395 | 2950 | 1024 | 1184 | 0.83 | 93 |
| b2048 np2 c2 | 0.359 | 4699 | 4761 | 5616 | 1024 | 1184 | 1.52 | 96 |
| b2048 np8 c2 | 0.356 | 4725 | 4799 | 5651 | 1024 | 1184 | 1.62 | 96 |
| b2048 np8 c4 | 0.375 | 4829 | 9615 | 15469 | 1024 | 1184 | 2.62 | 96 |
| b4096 np1 c1 | 0.341 | 2364 | 2418 | 2974 | 1024 | 1184 | 0.87 | 93 |
| b4096 np2 c1 | 0.343 | 2347 | 2398 | 2971 | 1024 | 1184 | 0.85 | 92 |
| **b4096 np2 c2** | **0.394** | **4232** | **4238** | **5101** | **2048** | **576** | 1.61 | **91** |
| **b4096 np8 c2** | **0.390** | **4252** | **4266** | **5166** | **2048** | **576** | 1.68 | **91** |
| b4096 np8 c4 | 0.390 | 6851 | 9106 | 14826 | 2048 | 896 | 2.77 | 94 |

### H1 is confirmed as a mechanism [MEASURED]

**At `-ub 2048` two concurrent 1954-token requests never share a graph. At `-ub 4096`
they do.** The ne11 histogram shows it directly: offloaded work moves from bucket
[1024, 2048) to bucket [2048, 4096) exactly when the ceiling is raised and `c >= 2`.

The lease counter corroborates it independently: **1184 -> 576 acquisitions for the
same 8 requests**. Two prompts sharing one graph means one set of NPU invocations
instead of two, so the count halves. Two instruments, different mechanisms, same
conclusion.

`np` alone changes nothing (np1/np2/np8 at c1 are 0.341–0.345). Slot count is not the
gate; the **ubatch ceiling** is.

### H1 is refuted as the cause of the throughput result [MEASURED]

Batch formation was genuinely suppressed, and fixing it is worth having:

| going from | to | throughput | TTFT p50 | TTFT p95 | power |
|---|---|---|---|---|---|
| b2048 c2 | b4096 c2 | +9.7% | −9.9% | −11.0% | −5 W |
| b2048 c4 | b4096 c4 | +4.0% | +41.9% | −5.3% | −2 W |

But it does not change the regime. At the best configuration (`b4096`), going from
one in-flight request to two buys **+14.9% throughput (0.343 -> 0.394 req/s) and costs
+80% TTFT (2347 -> 4232 ms)**. Going to four buys nothing further (0.390) and costs a
2.9x total p95. Prefill is bandwidth-bound; combining two prompts into one larger
matrix does not make the memory system faster, it only removes per-graph overhead —
which is where the ~15% and the 5 W come from.

## 6. Explicit multi-prompt oracle [MEASURED — server batch formation is NOT the limit]

The pinned `/completion` accepts a list of prompts, so an explicit batch can be
compared against independent simultaneous requests.

| config | independent | explicit multi-prompt | ne11 max (both) |
|---|---:|---:|---:|
| b2048 np2, k=2 | 0.356–0.359 | 0.357 | 1024 |
| b2048 np8, k=4 | 0.375 | 0.387 | 1024 |
| b4096 np2, k=2 | 0.394 | 0.394 | 2048 |
| b4096 np8, k=2 | 0.390 | 0.395 | 2048 |
| b4096 np8, k=4 | 0.390 | 0.394 | 2048 |

**Explicit batching and independent requests are indistinguishable.** More precisely:

- At `ub=2048` **neither** forms a combined graph. An explicit two-prompt request is
  still capped at 1024-bucket work — the ceiling binds regardless of how the work is
  submitted, so this is not a scheduling decision the server is getting wrong.
- At `ub=4096` **both** form combined graphs and reach identical throughput
  (0.394 vs 0.394 at np2).

This is the brief's CASE 2/3 boundary, and it resolves cleanly: once the ubatch
ceiling permits combination, `llama-server`'s continuous batching already forms the
batch that an explicit multi-prompt request would. **There is no scheduling work to
recover here.** The residual +1–3% for the explicit form is one HTTP round trip and
one result assembly, not better batching.

## 7. Controller output length [MEASURED]

`tools/output_thread_gate.py --mode output`, at the winning batch config
(`b4096 ub4096 np8 t8`), deterministic sampling, 8 requests per cell.

| n_predict | c | req/s | TTFT p50 | total p50 | total p95 | gen mean | W |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | **0.421** | 2370 | **2370** | 2450 | — | 95.8 |
| 4 | 1 | 0.408 | 2389 | 2443 | 2539 | 54.8 | 95.0 |
| 8 | 1 | 0.399 | 2378 | 2514 | 2538 | 128.4 | 94.8 |
| 32 | 1 | 0.341 | 2375 | 2934 | 2958 | 561.1 | 93.5 |
| 1 | 2 | **0.473** | 4223 | 4223 | 4264 | — | 90.2 |
| 32 | 2 | 0.382 | 4233 | 5121 | 5622 | 1164.1 | 92.8 |
| 1 | 4 | **0.477** | 8398 | 8399 | 8401 | — | 90.0 |
| 32 | 4 | 0.417 | 8498 | 9604 | 9633 | 3222.7 | 91.5 |

**TTFT is invariant to output length.** At c=1 it is 2370–2389 ms across every
`n_predict` from 1 to 32; at c=2, 4218–4268 ms. This is the expected result and it is
worth stating because it settles what the controller budget is made of: **the entire
first-token latency is prefill**, and nothing about the generation length touches it.

Shortening the output from 32 to 1 token is nonetheless a real gain on the other two
axes: **+23.5% throughput and −19.2% total latency at c=1** (0.341 -> 0.421 req/s,
2934 -> 2370 ms), holding at +23.8% at c=2 and +14.4% at c=4. Generation costs
~17.5 ms/token (561 ms for 32).

**It does not change useful concurrency.** At `n_predict=1`, c1 -> c2 is +12.4%
throughput for +78% TTFT — the same shape as at `n_predict=32` (+12.0% for +78%). So
H5 is answered: short constrained output materially improves the *budget* but leaves
the *concurrency regime* exactly where it was.

A grammar-constrained action set was not added. `n_predict=1` already isolates the
regime the brief asked about, and the brief explicitly warns against making benchmark
correctness depend on model semantic quality; a grammar would constrain which token is
emitted, not how long the request takes. [DEFERRED]

## 8. Decoupling prompt threads from generation threads [MEASURED — the largest win]

H4 asked whether tying `-t` and `-tb` together was leaving performance on the table.
It was. The prior launcher never set `-tb`, so it defaulted to `-t`, and prompt
processing — which §7 shows is the *entire* TTFT — ran at the decode width.

All at `b4096 ub4096 np8`, `n_predict=32`, 8 requests per cell.

| t | tb | c | req/s | TTFT p50 | prompt mean | gen mean | total p95 | W |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8 (prior default) | 1 | 0.341 | 2368 | 2367 | 562 | 2964 | 93.4 |
| 4 | 8 | 1 | 0.327 | 2336 | 2329 | 724 | 3089 | 92.4 |
| 6 | 8 | 1 | 0.337 | 2363 | 2352 | 612 | 3011 | 93.0 |
| 6 | 12 | 1 | 0.436 | 1712 | 1712 | 579 | 2306 | 108.9 |
| 4 | 12 | 1 | 0.408 | 1775 | 1756 | 694 | 2492 | 105.4 |
| **4** | **16** | 1 | 0.471 | **1441** | 1440 | 679 | 2154 | **104.9** |
| 6 | 16 | 1 | 0.483 | 1465 | 1465 | 604 | 2102 | 108.9 |
| **8** | **16** | 1 | **0.504** | 1457 | 1448 | 533 | **2009** | 110.6 |
| 6 | 24 | 1 | 0.483 | 1488 | 1484 | 583 | 2078 | 109.4 |

Against the prior `t8/tb8` default, `t8/tb16` is **−38.5% TTFT (2368 -> 1457 ms) and
+47.8% throughput (0.341 -> 0.504 req/s)** for +18.4% power.

Three things this establishes:

1. **`tb` is the binding parameter, `t` is nearly irrelevant.** At `tb=8`, changing `t`
   from 4 to 8 moves TTFT by 1.4% (2336 -> 2368). At `tb=16`, changing `t` from 4 to 8
   moves it by 1.1%. Prompt width sets the controller's latency; decode width does not.
2. **`tb=16` is the optimum and it is exactly the physical core count.** `tb=24` reaches
   into SMT siblings and gives nothing back (1488 vs 1465 ms, inside dispersion). That
   is the expected shape for a bandwidth-bound prefill: a second thread on a shared
   core adds no memory parallelism.
3. **The brief's desired outcome exists.** `t4/tb16` delivers `t8`-like TTFT
   (1441 vs 1457 ms, *better*) at the lowest power of the wide group (104.9 vs
   110.6 W) and only −6.5% throughput, while holding 4 rather than 8 decode threads
   against the co-tenants. Decode is slower (gen 679 vs 533 ms) but §7 showed
   generation is not in the TTFT budget.

At c=2 the `tb=16` variants converge (2825–2890 ms TTFT, 0.546–0.551 req/s), so the
`t` choice is a co-tenancy decision, not a throughput one — which is what Task 10
tests.

**This is the single largest effect measured in this pass, and it is a pure
configuration change: no kernel work, no scheduler work, one flag that was never set.**

## 9. Realistic prefix reuse [MEASURED]

`tools/prefix_reuse_gate.py`. Not the easy version: each request carries a **distinct
volatile suffix**, so this measures prefix reuse rather than deduplication of an
identical prompt. Stable prefix = controller instructions + action schema + fixed
topology; volatile suffix = current state, regenerated per request. Total held near
1954 tokens. Config `t8/tb16 b4096 ub4096 np8`, `n_predict=8`, 10 requests per cell.

| designed reuse | cache | evaluated | reused | TTFT p50 | prompt mean | XDNA nodes offloaded | W |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0% | off | 2063.5 | 0 | 1644.5 | 1618.4 | 1470 | 111.0 |
| 0% | on | 1945.0 | 120 | 1529.4 | 1508.9 | 1470 | 114.4 |
| 50% | off | 1943.4 | 0 | 1452.4 | 1453.4 | 1470 | 113.5 |
| 50% | on | 958.3 | 984 | **1033.2** | 1025.8 | **0** | 113.1 |
| 75% | off | 1941.6 | 0 | 1429.7 | 1440.1 | 1470 | 112.3 |
| 75% | on | 478.5 | 1464 | **581.6** | 579.8 | **0** | 109.7 |
| 90% | off | 1968.6 | 0 | 1461.6 | 1476.4 | 1470 | 114.4 |
| 90% | on | 192.8 | 1776 | **265.1** | 262.8 | **0** | 100.9 |

**At 90% reuse, TTFT falls from 1461.6 ms to 265.1 ms — 5.5x, and −13.5 W.** The
achieved reuse matches the design (1776 of 1969 tokens reused), so this is a genuine
prefix effect, not an artefact.

### XDNA drops out on its own, exactly where the threshold predicts [MEASURED]

The runtime declines the NPU below `kMTile = 1024` evaluated tokens. Watch
`ne11_nodes_offloaded` collapse:

- 0% reuse: 1945 tokens still evaluated -> **1470 nodes offloaded**, NPU engaged
- 50% reuse: 958 evaluated (below 1024) -> **0 nodes offloaded**
- 75% reuse: 478 evaluated -> **0**
- 90% reuse: 193 evaluated -> **0**

This is the natural split the brief anticipated, and it needed no forcing:

| request kind | evaluated tokens | path | TTFT |
|---|---:|---|---:|
| cold / large context miss | ~1950 | CPU + NPU hybrid | ~1460 ms |
| warm prefix / small state delta | ~190 | CPU only | ~265 ms |

The crossover for a 1954-token controller prompt is **~47% reuse**: above it the
uncached suffix falls under 1024 tokens and the offload correctly declines. The NPU
was not forced onto a tiny suffix, and should not be — the CPU-only warm path is
5.5x faster than the hybrid cold path.

**Prefix reuse is worth far more than any hardware or batching change measured in this
pass**: 5.5x against the ~1.15x from lifting the batch ceiling and the ~1.6x from
fixing `-tb`. Note also that all three compose in the same direction — each one
reduces evaluated tokens or widens the threads that process them — and the two that
help most both end with the NPU idle.

### A contamination bug caught before it was published [CORRECTION]

The first run of this experiment reported `eval = 1 token, TTFT 60.8 ms` at 0% reuse
with caching on — a full cache hit where the prompts were supposed to be distinct.
The cause: both arms used the same seeds (`1000 + i`), so the `cache=True` arm re-sent
the exact prompts the `cache=False` arm had just warmed. It was measuring
deduplication of an identical prompt, which is precisely what this task exists to
avoid, and it would have overstated the benefit enormously.

Fixed with disjoint seed ranges per arm, and the harness now records the server's
`cache_n` so achieved reuse is reported rather than assumed. The contaminated run is
preserved as `prefix_reuse_CONTAMINATED.csv` rather than deleted.

## 10. GPU prefill vs GPU decode pressure [MEASURED — the phases differ, the policy does not]

`tools/gpu_phase_gate.py`. The prior conclusion used a mixed workload whose GPU request
had a ~128-token prompt and a long generation — almost entirely GPU *decode*. The
controller was therefore never measured under sustained GPU *prefill*. Here the worker
is driven in a specific phase for the whole cell, with the CPU verifier tenant also
running, against controller widths t4/t6/t8 (all at `tb16`).

| controller | phase | ctrl TTFT p50 | ctrl TTFT p95 | GPU pp tok/s | GPU tg tok/s | W |
|---|---|---:|---:|---:|---:|---:|
| t4 tb16 | idle | 1517.5 | 1530.5 | — | — | 113.2 |
| t4 tb16 | prefill-2k | 1993.6 | 2146.1 | 260.9 | 14.37 | 119.9 |
| t4 tb16 | prefill-8k | 1957.4 | 2141.5 | 271.6 | 14.54 | 120.0 |
| t4 tb16 | **decode** | **2545.7** | 2618.7 | 217.1 | 11.04 | 120.0 |
| t6 tb16 | idle | 1514.9 | 1519.0 | — | — | 115.3 |
| t6 tb16 | prefill-2k | 1944.2 | 2140.3 | 257.6 | 14.36 | 119.9 |
| t6 tb16 | prefill-8k | 2013.8 | 2081.3 | 269.6 | 14.36 | 120.0 |
| t6 tb16 | **decode** | **2492.7** | 2612.8 | 214.0 | 10.95 | 120.0 |
| t8 tb16 | idle | 1520.7 | 1532.7 | — | — | 115.8 |
| t8 tb16 | prefill-2k | 1908.9 | 2001.2 | 255.7 | 14.43 | 119.9 |
| t8 tb16 | prefill-8k | 1873.0 | 2040.8 | 267.4 | 14.47 | 119.9 |
| t8 tb16 | **decode** | **2505.9** | 2660.4 | 215.9 | 11.10 | 119.9 |

### The phases really are different — and the expected direction is wrong [MEASURED]

**GPU decode hurts the controller more than GPU prefill does**, consistently at every
width:

| phase | controller TTFT vs idle |
|---|---|
| prefill-2k | +26% to +31% |
| prefill-8k | +23% to +33% |
| **decode** | **+64% to +68%** |

The intuition that a large dense prefill matmul would be the worse neighbour is wrong
here. GPU decode streams the entire 16.7 GiB of worker weights for every token, so it
holds memory bandwidth continuously; GPU prefill reads those weights once per large
batch and is comparatively compute-dense. The controller is bandwidth-bound (§8), so
the continuously-streaming tenant is the one that hurts. **Isolating the phases was
worth doing: the prior mixed workload happened to be testing the harsher phase, but
for the wrong reason and without knowing it.**

### But no phase favours narrower controller threads [MEASURED]

Within any phase, the spread across t4/t6/t8 is small and does not favour narrow:

- prefill-2k: 1993.6 / 1944.2 / 1908.9 — **t8 best**
- prefill-8k: 1957.4 / 2013.8 / 1873.0 — **t8 best**
- decode: 2545.7 / 2492.7 / 2505.9 — t6 best by 0.5%, inside dispersion
- idle: 1517.5 / 1514.9 / 1520.7 — identical

The between-phase effect is 23–68%; the between-width effect inside a phase is at most
4.5% and has no consistent sign. GPU throughput is likewise unaffected by controller
width (pp 255.7–271.6, tg 10.95–14.54 grouped by phase, not by width).

**PHASE-AWARE THREAD POLICY NOT JUSTIFIED.** This confirms `service-cotenancy`'s
conclusion, but now against the workload it was missing, and with the mechanism
identified. It is also the expected result given §8: at `tb16` every width uses the
same 16 prompt threads, and `t` only sets decode width, which §7 showed is not in the
TTFT budget. A phase-aware policy would be switching the parameter that does not
matter.
