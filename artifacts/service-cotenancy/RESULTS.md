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
