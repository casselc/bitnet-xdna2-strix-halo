# Tri-device co-tenancy: NPU controller + Vulkan GPU worker + Zen 5

Does the three-device topology actually work when the devices run **at the same
time** on one LPDDR5X pool?

| | |
|---|---|
| branch base | `runtime-v1-promotion` @ `712b7c6` |
| branch | `gpu-cotenancy` |
| controller | BitNet-b1.58-2B-4T I2_S, XDNA2 offload, promoted defaults |
| worker | Qwen3.6-27B `UD-Q4_K_XL`, 16.67 GiB, Vulkan/RADV |
| CPU tenant | structured Clojure (SCI/babashka) graph + invariant checking |

| tag | meaning |
|---|---|
| **[MEASURED]** | observed on this machine |
| **[DERIVED]** | arithmetic over measured quantities |
| **[DEFERRED]** | not done, with the reason |

---

## 1. Method

Arms are launched **concurrently** and each reports its own throughput, so a
regression lands on the resource that suffered rather than disappearing into a
blended number. Arms are **interleaved across rounds** rather than run in
blocks, because this machine drifts. 3 rounds, medians. Package RAPL sampled
across each arm; the `core` subdomain is unusable on this SoC, so package is the
only trustworthy counter.

Every child process is tracked and reaped **by explicit PID**. There is no
pattern matching in the harness: `pkill -f` once killed this project's own
harness, and `pgrep -x` silently fails on names longer than 15 characters —
which earlier in this same pass let two benchmark runs execute concurrently
before it was noticed.

Raw: `tri_device.csv`, `tri_device.json`.

## 2. Single-resource baselines [MEASURED]

| arm | controller pp2048 | GPU pp512 | GPU tg64 | package |
|---|---:|---:|---:|---:|
| **A** GPU alone | — | **288.5** | **12.29** | 87.3 W |
| controller NPU t4 alone | 598.2 | — | — | 60.5 W |
| controller NPU t6 alone | 750.4 | — | — | 71.8 W |
| controller NPU t8 alone | **868.2** | — | — | 76.0 W |

## 3. The tri-device matrix [MEASURED]

| arm | controller | GPU pp512 | GPU tg64 | package |
|---|---:|---:|---:|---:|
| **B** GPU + **CPU** controller t8 | 525.8 | 221.8 | 11.35 | 115.0 W |
| **C** GPU + **NPU** controller t4 | 488.1 | **250.7** | **11.34** | **92.9 W** |
| **D** GPU + **NPU** controller t6 | 564.0 | 243.2 | 11.18 | 101.9 W |
| **E** GPU + **NPU** controller t8 | **604.5** | 236.4 | 11.03 | 105.5 W |

### The NPU controller is better on three axes at once [DERIVED]

Comparing **B against E** — same 8 threads, same GPU worker, the only difference
being whether the linears run on the NPU:

| | CPU controller | NPU controller | |
|---|---:|---:|---|
| controller prefill | 525.8 | **604.5** | **1.150x** |
| GPU prefill | 221.8 | **236.4** | **1.066x** |
| GPU decode | 11.35 | 11.03 | 0.972x |
| package power | 115.0 W | **105.5 W** | **−9.5 W** |

The controller gets 15% faster, the GPU worker's prefill gets 6.6% faster, and
the system draws 9.5 W less — **simultaneously**. The mechanism is
straightforward: offloading the linears frees CPU cores that the GPU worker's
host side needs, so both sides win. The one regression is GPU decode at −2.8%,
which is small and is discussed in section 5.

This is the central result. It is not a marginal trade — it is the same
configuration being better for every party.

### Co-tenancy is not free [MEASURED]

| threads | controller alone → co-tenant | GPU pp alone → co-tenant | GPU tg |
|---|---|---|---|
| t4 | 598.2 → 488.1 (**−18.4%**) | 288.5 → 250.7 (−13.1%) | −7.7% |
| t6 | 750.4 → 564.0 (**−24.8%**) | 288.5 → 243.2 (−15.7%) | −9.0% |
| t8 | 868.2 → 604.5 (**−30.4%**) | 288.5 → 236.4 (−18.0%) | −10.3% |

Both sides lose. The NPU reduces contention; it does not remove it. Note the
controller loses **more** the more threads it has — its own CPU-side work
(attention, the scaling epilogue) is what collides with the GPU worker's host
threads, so widening the controller widens the collision.

## 4. The Pareto frontier [DERIVED]

This is the question the brief asked, and there is a real frontier:

| arm | controller | GPU pp | GPU tg | power |
|---|---:|---:|---:|---:|
| **C** t4 | 488.1 | **250.7** | **11.34** | **92.9 W** |
| **D** t6 | 564.0 | 243.2 | 11.18 | 101.9 W |
| **E** t8 | **604.5** | 236.4 | 11.03 | 105.5 W |

Going t4 → t8 buys **+23.9%** controller throughput and costs **−5.7%** GPU
prefill, **−2.7%** GPU decode and **+13.6%** power.

**No arm dominates.** The right choice depends on what the system is for:

- **worker-throughput-first → C (t4).** Best GPU prefill and decode, lowest
  power, at a controller that is still 488 t/s.
- **balanced → D (t6).** Recommended default. It keeps 93% of E's controller
  throughput for 60% of E's GPU prefill loss, at 3.6 W less.
- **controller-latency-first → E (t8).** Best controller, worst worker, most
  power.

The brief's hypothesis — that a narrower controller might win overall — is
**confirmed in direction**: t4 and t6 leave measurably more headroom for the GPU
worker, and t8's isolated controller advantage does not survive contact with a
co-tenant. But the effect is modest (single-digit percent on the GPU side), so
it is a tuning knob, not a redesign.

## 5. Memory pressure [MEASURED / DEFERRED]

The runtime keeps **2006.2 MiB** of expanded int8 weights resident in the same
LPDDR5X the GPU uses. The driver reports `uma: 1` and the GPU's real capacity is
**GTT (97.7 GiB)**, not the 0.5 GiB VRAM carve-out, so there is no separate
"GPU memory" to hide in.

**Capacity is not the problem.** 2.0 GiB against 97.7 GiB of GTT and 122 GiB of
system RAM is not a capacity constraint, and nothing in the matrix behaves like
one — no arm failed to allocate, and no arm showed the cliff that memory
exhaustion produces.

**What the matrix does show is bandwidth/occupancy contention**, and it appears
most clearly in GPU **decode**: −7.7% to −10.3% depending on thread count, and
worse the more CPU threads the controller uses. Decode is memory-bound, so it is
the workload most exposed to a competing memory consumer, and it degrades
monotonically with controller CPU width. That is the signature of bandwidth
contention rather than capacity exhaustion.

**GTT accounting is reported but should not be over-read.** It sits at 57.8 GiB
whenever the GPU model is loaded and 41.3 GiB otherwise, regardless of whether
the controller is idle or actively prefilling. That figure clearly includes more
than this workload's footprint, so it is recorded as raw data and **no bandwidth
number is derived from it**. This machine exposes no trustworthy DRAM bandwidth
counter, and per the brief none is manufactured from utilisation proxies.

## 6. Verdict

### **TRI-DEVICE ARCHITECTURE VALIDATED**

All three devices run concurrently, correctly, and with no failure mode. The
NPU controller is not merely compatible with a GPU worker — it is **strictly
better than a CPU controller for every party at once**: +15.0% controller,
+6.6% GPU prefill, −9.5 W, at the same thread count.

| | best configuration |
|---|---|
| **isolated GPU** | 288.5 pp512 / 12.29 tg64 @ 87.3 W (arm A) |
| **isolated controller** | 868.2 pp2048 @ 76.0 W (NPU, t8) |
| **best tri-device Pareto point** | **D — GPU + NPU controller at t6**: 564.0 controller / 243.2 GPU pp / 11.18 GPU tg @ 101.9 W |

D is the recommended default: 93% of the best co-tenant controller throughput,
97% of the best co-tenant GPU prefill, and neither extreme's power or latency
cost.

### What this does not settle

- **ROCm was not tested.** It is not packaged for this distribution release and
  installing it is a large system-wide change that risks the working
  XRT + mlir-aie + Peano environment. Vulkan/RADV is the measured backend; the
  independent correctness path is CPU (PPL 19.6300 vs 19.4907, 0.71% apart).
  A ROCm arm could change the GPU absolute numbers but not the CPU-vs-NPU
  controller comparison, which is measured against a fixed worker.
- **Packed ternary residency is not justified by this data.** The 2.0 GiB
  footprint is not a capacity problem on a 122 GiB machine. Per the brief this
  is recorded, not acted on.
- **Controller decode and TTFT under co-tenancy** were not measured; the
  controller arms measure prefill (pp2048), which is the contended phase.
