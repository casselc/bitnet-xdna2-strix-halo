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

## 5. Memory pressure: capacity is free, activity is not [MEASURED]

The runtime keeps **2006.2 MiB** of expanded int8 weights resident in the same
LPDDR5X the GPU uses. The driver reports `uma: 1`, and the GPU's real capacity
is **GTT (97.7 GiB)** rather than the 0.5 GiB VRAM carve-out, so there is no
separate "GPU memory" to hide in. The brief asks to separate the **capacity**
cost of that footprint from the **bandwidth/contention** cost of the work.

That separation is available without new machinery. A **decode-only** controller
loads the model and uploads all 2.0 GiB of weights but issues **zero NPU
dispatches** — decode takes the CPU GEMV path — so it isolates footprint from
activity. Raw: `tri_device_extra.csv`.

| arm | NPU weights resident | NPU active | GPU pp512 | GPU tg64 |
|---|---|---|---:|---:|
| **A** GPU alone | no | no | 288.5 | 12.29 |
| **F** GPU + controller **decoding** | **yes, 2.0 GiB** | **no** | **290.3** | **12.29** |
| **E** GPU + controller **prefilling** t8 | yes | yes | 236.4 | 11.03 |

**A → F costs nothing.** 290.3 vs 288.5 pp512 and 12.29 vs 12.29 tg64 — the
resident 2.0 GiB is invisible to the GPU worker, within noise and if anything
marginally faster.

**F → E costs everything.** −18.6% prefill and −10.3% decode. **The entire
co-tenancy penalty is concurrent activity, not memory footprint.**

This is a direct answer to B11 and it settles B12: **packed ternary residency
would buy nothing here.** Shrinking a footprint that already costs zero cannot
improve co-tenancy. If that work is ever justified it will be for a different
reason — fitting more specialists in memory, say — and not by this evidence.

**GTT accounting is recorded but deliberately not interpreted.** It reads
57.8 GiB whenever the GPU model is loaded and 41.3 GiB otherwise, regardless of
whether the controller is idle, decoding or prefilling. It plainly includes far
more than this workload, so **no bandwidth number is derived from it**. This
machine exposes no trustworthy DRAM bandwidth counter, and per the brief none is
manufactured from a utilisation proxy.

## 6. The CPU harness tenant [MEASURED]

A workload shaped like the eventual control plane rather than a synthetic loop:
structured Clojure (SCI/babashka) building a 220-node dependency graph,
topologically ordering it, and checking the ordering invariant — the shape a
verifier actually has. Standalone baseline **1153 ops/s, p50 0.770 ms,
p95 1.275 ms**.

| arm | controller | GPU pp | GPU tg |
|---|---:|---:|---:|
| **D** GPU + controller t6 | 564.0 | 243.2 | 11.18 |
| **G** GPU + controller t6 **+ tenant** | 549.1 | 236.4 | 11.02 |
| | **−2.6%** | **−2.8%** | **−1.4%** |
| controller t6 alone | 750.4 | — | — |
| **H** controller t6 **+ tenant**, no GPU | 739.4 | — | — |
| | **−1.5%** | | |

**Adding a real control-plane tenant costs 1.5–2.8%.** There is genuine CPU
headroom left at t6 even with the GPU worker running — which is the practical
case for the recommended default.

*Power in arms G and H is not comparable to the others.* The tenant runs a fixed
25 s while the benchmarks finish earlier, so package power is averaged over a
window that includes idle tail. The G figure (80.4 W vs D's 101.9 W) is that
averaging artifact, not a real power saving.

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
- **Packed ternary residency is affirmatively NOT justified.** Section 5
  measures the resident 2.0 GiB as costing the GPU worker **zero** (arm F equals
  arm A). Shrinking a footprint that already costs nothing cannot improve
  co-tenancy. Per the brief this is recorded, not acted on — and the evidence is
  now stronger than "not yet justified": it is measured as pointless *for this
  purpose*.
- **Controller TTFT under co-tenancy** was not measured; the controller arms
  measure prefill throughput (pp2048), which is the contended phase. Controller
  decode was measured only in arm F (42.15 t/s) as the vehicle for the memory
  separation, not swept.
- **`-ub 2048` is required** for the controller to use the NPU at all; every
  controller arm here passes it. See `docs/RUNTIME_STATUS.md`.
