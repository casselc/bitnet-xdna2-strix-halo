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
