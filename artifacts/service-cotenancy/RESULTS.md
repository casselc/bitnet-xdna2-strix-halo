# Warm persistent service topology under concurrency

The previous passes measured **component throughput** with one-shot benchmark
processes. This measures **request latency** against long-lived services, which
is a different question: model load is excluded, requests queue, and the
single-flight NPU lease becomes a resource that concurrent requests share.

| | |
|---|---|
| branch base | `gpu-cotenancy` @ `fbd8bf00108480c68919752fbb521fafd786d47d` |
| branch | `service-cotenancy` |
| controller | BitNet-b1.58-2B-4T I2_S, promoted XDNA runtime, `llama-server` on :8081 |
| worker | Qwen3.6-27B `UD-Q4_K_XL`, Vulkan/RADV, `llama-server` on :8082 |
| verifier | structured Clojure/SCI graph + invariant checking |

| tag | meaning |
|---|---|
| **[MEASURED]** | observed on this machine |
| **[DERIVED]** | arithmetic over measured quantities |
| **[DEFERRED]** | not done, with the reason |

---

## 1. Method

Three quantities are kept **distinct** throughout, because blending them is how
service measurements mislead:

| | |
|---|---|
| **queue wait** | submitted → the service began working on it |
| **service time** | began → finished (taken from the server's own timings) |
| **total latency** | submitted → response in hand |

Service time comes from `llama-server`'s reported `prompt_ms + predicted_ms`
rather than from the client's wall clock; the difference between the two **is**
the queue wait plus transport, and is recorded rather than assumed to be zero.

Load is **closed-loop**: `c` client threads each pull from a shared counter, so
an overloaded service produces longer latencies rather than an unbounded
backlog — which is what a real caller with a connection pool sees.

Controller prompt: a structured system-state report, **1954 tokens measured**,
identical for every request. Generation 32 tokens (a controller decides; it does
not write essays). Worker prompt: a fixed code-generation task, 128 tokens.

### Process discipline

Every service is tracked and reaped **by explicit PID**. No pattern matching
appears anywhere in the harness. `service_ctl.sh` additionally refuses to start
when the target port is already held, and after starting asserts the listener is
the PID it recorded — see section 7 for why.

---

## 2. Warm single-request baselines [MEASURED]

Warm services, no model load in the measurement. Raw: `baseline_t{4,6,8}.csv`.

| controller width | TTFT p50 | total p50 | total p95 | package |
|---|---:|---:|---:|---:|
| t4 | 3300 ms | 4022 ms | 5135 ms | 63.0 W |
| t6 | 2639 ms | 3250 ms | 4371 ms | 72.1 W |
| **t8** | **2366 ms** | **2914 ms** | **3598 ms** | 92.0 W |

| GPU worker | TTFT p50 | total p50 | total p95 | package |
|---|---:|---:|---:|---:|
| t4 / t6 / t8 (controller idle) | 435.7 / 436.8 / 436.0 ms | 10698 / 10683 / 10682 ms | 10777 / 10684 / 10683 ms | 90.2 / 92.8 / 95.1 W |

Two things worth stating plainly.

**Controller TTFT is 2.4–3.3 seconds.** That is the number the previous pass did
not measure, and it is a constraint on everything downstream: a control plane
whose first token arrives 2.4 s after the request is not interactive. It is
dominated by prefill of the 1954-token prompt, not by generation.

**The worker is completely insensitive to controller width when the controller
is idle** — 435.7 / 436.8 / 436.0 ms TTFT across t4/t6/t8. Any worker
degradation seen later is caused by concurrent controller *activity*, which is
consistent with the previous pass's finding that co-tenancy cost is activity and
not footprint.

---

## 3. Controller concurrency: the service saturates at one [MEASURED]

t6, 16 requests per cell. Raw: `concurrency_t6.csv`.

| concurrency | req/s | TTFT p50 | TTFT p95 | total p50 | total p95 | **queue p95** | **lease wait** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.302 | 2643 | 3803 | 3243 | 4402 | 8 | **0** |
| 2 | 0.320 | 2734 | 5427 | 6254 | 6365 | 2663 | **0** |
| 4 | 0.331 | 2925 | 11854* | 11854 | 18069 | 5900 | **0** |
| 8 | 0.329 | 5379 | 6884 | 24673 | **41167** | 18156 | **0** |

Throughput is **flat**: 0.302 → 0.331 req/s from c=1 to c=4 (+9.6%), then
nothing. Total latency grows **linearly** with concurrency — 3.2 s, 6.3 s,
11.9 s, 24.7 s — which is the signature of a single-server queue, not of a
system with any parallelism to exploit.

At c=8 the **queue wait is 18.2 s of a 41.2 s p95 total**: 44% of the request's
life is spent waiting to be admitted.

### The NPU lease is not the bottleneck [MEASURED]

**Lease wait is exactly zero at every concurrency level, and the contended
fraction is zero.** `llama-server` runs a single compute stream, so two slots
never issue two graphs concurrently; requests queue in the server's own
scheduler long before they reach the NPU.

This is measured, not inferred, and the counters were validated against known
contention first — because an uninstrumented zero and a real zero are
indistinguishable. Running `tests/test_xdna_concurrent`, which deliberately puts
two threads on the lease:

```
acquisitions=257  immediate=2  waited=254  (98.8% contended)
mean wait 5091.2 us   mean hold 7059.0 us   max waiters=1
```

Raw: `lease_instrumentation_validation.csv`.

**One related measurement worth recording for future designs.** The lease is
held for essentially the whole prefill — mean hold **14.87 ms** over ~210
invocations per 1954-token request — because it brackets `accumulate`, the ggml
barrier and the multi-threaded scaling epilogue, not merely the NPU dispatch. So
a future design that *did* issue concurrent graphs would find the lease
serialising a large amount of CPU work, not just device time. It does not today,
and that is why the lease is invisible in these numbers.

Same shape at the other widths. Raw: `concurrency_t{4,6,8}.csv`.

| arm | req/s | TTFT p50 | total p50 | total p95 | queue p95 | lease wait |
|---|---:|---:|---:|---:|---:|---:|
| t4 c1 | 0.243 | 3312 | 4026 | 5176 | 7 | 0 |
| t4 c2 | 0.253 | 3441 | 7883 | 8217 | 3324 | 0 |
| t4 c4 | 0.247 | 4061 | 15450 | 24984 | 8190 | 0 |
| t8 c2 | 0.359 | 2470 | 5589 | 5604 | 2353 | 0 |
| t8 c4 | 0.383 | 2641 | 10476 | 14995 | 4852 | 0 |

**Maximum useful controller concurrency is 1.** Beyond it, throughput is flat and
every extra in-flight request adds its full service time to everyone's latency.

---

## 4. Chained controller → worker [MEASURED]

The simplest real handoff: controller decides, a deterministic payload is built
from its output, the warm worker executes. The payload depends on the
controller's output but **not on it being correct** — a benchmark whose timing
depends on model semantics is not reproducible. c=1, 6 chains per cell. Raw:
`chain_t{4,6,8}.csv`.

| width | chain p50 | chain p95 | package |
|---|---:|---:|---:|
| t4 | 15034 ms | 15765 ms | 85.5 W |
| t6 | 14243 ms | 15180 ms | 89.0 W |
| **t8** | **13818 ms** | **14247 ms** | 95.1 W |

Handoff overhead itself is negligible (sub-millisecond, a local HTTP call). The
chain is ~4 s of controller and ~10 s of worker; the worker's 128-token
generation at ~12.4 t/s dominates, so **controller width moves the chain by only
8%** (15034 → 13818 ms).

---

## 5. Mixed load, per class, with the control-plane tenant [MEASURED]

`C:1,CW:1` at concurrency 2, verifier running for exactly the load window.
Raw: `mixed_policy_t{4,6,8}.csv`, `mixed_C_1_CW_1.csv`, `mixed_C_1_CW_2_W_1.csv`.

| width | req/s | C p50 | C p95 | chain p50 | chain p95 | verifier ops/s | ver p95 | ver p99 | package |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t4 | 0.133 | 9051 | 10638 | 20977 | 22026 | **1226.2** | **1.102** | 1.732 | **104.9 W** |
| t6 | 0.144 | 6277 | 7672 | 19743 | 21188 | 1203.9 | 1.114 | **1.722** | 109.8 W |
| **t8** | **0.163** | **5159** | **6384** | **18974** | **19916** | 1150.2 | 1.238 | 1.867 | 116.9 W |

### The throughput-only answer did not survive queueing [DERIVED]

The previous pass recommended **t6** from component throughput. At the service
level **t8 wins every latency metric and throughput**: controller p50 −18%,
controller p95 −17%, chain p50 −4%, chain p95 −6%, requests/s +13%, all relative
to t6. It loses only on verifier headroom (−4.5% ops/s, +11% p95) and power
(+6.5%).

The mechanism is visible in the numbers: in a service the controller's **prefill
is the long pole** (TTFT 2.4–3.3 s of a ~3–4 s request), and more threads finish
it sooner and release the CPU sooner. That outweighs the GPU-decode headroom a
narrower controller preserves — which was the whole basis for preferring t6.

**Aggregate `total_ms` p50 is deliberately not used for this recommendation.**
With n=8 over a two-class mix its value depends on how many of each class landed
in the sample, which is why t8 appears worst on that column (16595 ms) while
winning both per-class medians. Per-class distributions are the honest view, and
reporting only the aggregate would have inverted the conclusion.

### Verifier / control-plane tenant [MEASURED]

The tenant is started with the load and stopped with it, so its window matches
the load exactly — the previous pass's fixed-duration tenant outlived the
benchmark and made average power incomparable, and that is not repeated.

| width | ops/s | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| standalone (no service load) | ~1300 | 0.73 ms | 0.90 ms | 1.15 ms |
| t4 under mixed load | 1226.2 | 0.766 ms | 1.102 ms | 1.732 ms |
| t6 under mixed load | 1203.9 | 0.781 ms | 1.114 ms | 1.722 ms |
| t8 under mixed load | 1150.2 | 0.825 ms | 1.238 ms | 1.867 ms |

The previous pass's lesson holds and is *milder* here: throughput falls 6–12%
and p50 rises 5–13%, while **p95 rises 22–38% and p99 rises 50–62%**. Even at
t8 the tenant retains 88% of standalone throughput with a p99 of 1.87 ms, which
is comfortable for a verifier. **CPU headroom is not the constraint at any
tested width.**

---

## 6. Soak [MEASURED]

150 mixed requests (`C:1,CW:1,W:1`) at t8, concurrency 2, over **900 s**, with
residency sampled every 5 s by explicit PID. Raw: `soak_t8.csv`,
`soak_residency.csv`.

| | |
|---|---|
| throughput | 0.166 req/s |
| total p50 / p95 | 12220 / 18135 ms |
| verifier | 1249.5 ops/s, p50 0.78 ms, **p95 0.936 ms** |
| failures / timeouts / wrong-backend | **0** |
| package power | **115.7 W** |
| controller RSS, second half | +0.3 MiB |
| worker RSS, second half | +0.1 MiB |
| GTT | 58.4 GiB, **+0.0** |
| temperature | 41.5 → 59.0 °C, max 60.0 |

Controller RSS rises 2771 → 5232 MiB over the first minutes and then plateaus:
that is the ~2.0 GiB of NPU weights uploading lazily on the first requests plus
KV allocation, **not growth** — drift over the second half is +0.3 MiB. No
leak-like behaviour in the controller, the worker, the direct-output arena or
GTT.

### A power-measurement bug the soak exposed [MEASURED]

The **first** soak reported **43.2 W** for a load independently measured at
~117 W. Package RAPL wraps at 65533 J, which at this load is every **~560 s**,
so a 900 s window wraps **1.6 times**. The single `if delta < 0: delta += wrap`
correction cannot fix more than one wrap, and never fires at all when a wrap
still leaves a positive raw delta.

Energy is now accumulated from the existing 0.5 s poll, where each interval is
far shorter than a wrap period and a negative delta unambiguously means exactly
one wrap. The re-run gives **115.7 W**, consistent with the 116.9 W measured over
a short window on the same load.

**Scope of the error:** every other power figure in this pass comes from a window
of 15–60 s, well inside one wrap period, and is unaffected. The two soak runs
agree on everything else — 0.166 req/s both times, p50 12220 vs 12221 ms,
verifier 1249.5 vs 1247.3 ops/s — so only the power number ever differed. The
bad run is preserved as `soak_t8_INVALID_POWER.csv` rather than deleted.

---

## 7. Phase-aware thread policy [MEASURED — NOT JUSTIFIED]

The hypothesis carried in from the previous pass: widen the controller when the
GPU is idle, narrow it when the GPU is prefilling.

**The measurements refute its premise directly, so no oracle was needed.** The
policy can only pay if some width other than the widest wins in some phase. It
does not:

| phase | best width | evidence |
|---|---|---|
| GPU idle | **t8** | baseline TTFT 2366 ms vs 2639 (t6), 3300 (t4) |
| GPU active, mixed load | **t8** | controller p50 5159 vs 6277, 9051; chain p95 19916 vs 21188, 22026 |

t8 is best in **both** phases, so there is no phase in which switching to a
narrower controller improves service latency. A phase-aware policy has nothing
to exploit. Recorded as **NOT JUSTIFIED**, on measurement rather than on a
cost/benefit guess, and no scheduler was built.

The one thing a narrower controller does buy is verifier headroom (+6.6% ops/s
and −11% p95 at t4 versus t8) and power (−10.3%). If a future control plane
acquires a hard tail-latency deadline, that is the trade to revisit — but it is
a *static* trade, not a phase-aware one.

## 8. What was not done [DEFERRED]

- **Steady-rate arrival (Task 7B).** The concurrency sweep is closed-loop, which
  is burst-shaped: `c` requests are submitted together and a new one is
  submitted only as one completes. It answers the divergence question the brief
  asks — p95/p50 goes 1.36, 1.02, 1.52, 1.67 at c=1,2,4,8 — but a controlled
  sub-saturation arrival rate was not separately generated. Given that the
  service saturates at c=1, a "50/75/90% of capacity" open-loop test would be
  measuring inter-arrival time against a single-server queue whose behaviour is
  already characterised.
- **Controller decode/TTFT swept under co-tenancy.** Controller decode was
  measured only as part of full requests.
- **ROCm as a second GPU backend.** Unchanged from the previous pass.

---

## 9. Verdict

### **SERVICE VALID — ADMISSION CONTROL NEEDED**

The topology works: two warm services and a control-plane tenant run
concurrently for 900 s with **zero failures, zero timeouts, zero fallbacks and
no memory growth**, at 115.7 W. Chained controller→worker requests complete in
~13.8 s p50. Correctness holds — the controller offloads to the NPU on every
request with no silent CPU fallback.

**What it needs is admission control, not more hardware.** The controller
service saturates at **concurrency 1**. Every additional in-flight request adds
its full service time to everyone else's latency: p95 total goes 4.4 s → 6.4 s →
18.1 s → 41.2 s at c=1/2/4/8, while throughput stays flat at ~0.33 req/s. A
caller that pipelines requests makes the service strictly worse for every
caller. The fix is a queue with a depth limit in front of the controller, which
is a small piece of work and not attempted here.

### Answers to the specific questions

| question | answer |
|---|---|
| **recommended static controller thread count** | **t8** |
| **phase-aware thread policy** | **NOT JUSTIFIED** — t8 is best in both GPU-idle and GPU-active phases, so no phase exists for a narrower setting to win |
| **max tested concurrency before tails become unattractive** | **1.** p95 already doubles at c=2 (4.4 → 6.4 s) with +6% throughput; by c=4 it is 4.1x the c=1 p95 |
| **verifier p95 / p99 at the recommended point** | **0.936 ms / ~1.9 ms** under soak; 1.238 / 1.867 ms in the denser mixed cell |

**t8 rather than t6 is a change from the previous pass**, and it is a change of
*conclusion*, not of data. That pass recommended t6 from component throughput
(GPU pp/tg). At the service level t8 wins every latency metric and throughput,
because the controller's **prefill is the long pole** — TTFT is 2.4–3.3 s of a
3–4 s request — and more threads finish it sooner and release the CPU sooner.
That outweighs the GPU-decode headroom a narrower controller preserves. The
throughput-only answer did not survive queueing.

The cost of t8 is real but small: **−4.5% verifier throughput, +11% verifier
p95, +6.5% package power** versus t6. If a future control plane acquires a hard
tail deadline, t6 or t4 buys it back.

### The bottleneck is not where the hardware work suggested

Three passes of hardware measurement pointed at the NPU, memory bandwidth and
thread balance. At the service level:

- **the NPU lease is never contended** (0 waits in ten cells, counters validated
  against 98.8%-contended ground truth);
- **memory does not grow** and GTT does not move;
- **CPU headroom is not the constraint** — the verifier keeps 88–96% of its
  standalone throughput at every width.

The binding constraint is that **`llama-server` serialises requests through one
compute stream**. That is a property of the serving process, not of XDNA2, Zen 5
or the Radeon, and it is where the next engineering effort belongs if higher
controller concurrency is ever needed.

### Service budget for the controller-specialist phase

The numbers this pass exists to hand forward:

| quantity | measured |
|---|---|
| controller TTFT, warm, uncontended | **2366 ms** (t8, 1954-token prompt) |
| controller total, 32 output tokens | 2914 ms p50 / 3598 ms p95 |
| controller prefill rate | ~500–830 tok/s depending on width and load |
| controller decode rate | ~49–52 tok/s |
| usable concurrency | **1** |
| chain latency, controller → 128-token worker reply | 13.8 s p50 |
| verifier headroom at the service point | 1250 ops/s, p95 0.94 ms |

**TTFT is the number to design against.** 2.4 s to first token is dominated by
prefilling ~2000 tokens of context; a controller specialist that needs less
context, or emits its decision sooner, moves this far more than any further
hardware tuning available here.
