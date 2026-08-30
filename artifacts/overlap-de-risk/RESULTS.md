# Overlap de-risk: is cross-microbatch pipelining actually justified?

Purpose of this pass: decide, with measurement and a dependency-aware model,
whether the invasive cross-microbatch pipeline proposed at the end of
`artifacts/next-pass/RESULTS.md` is worth building -- and de-risk the concurrency
mechanism with a smaller experiment first. **The pipeline is deliberately not
implemented here.**

Base commit: `fb4493e9241cf82b3d1a3b03a0780aa0dc333585` (`next-pass-results`).
Branch: `overlap-de-risk`. `main` was at `885df0ca` and was not touched.

Every claim below is tagged:

| tag | meaning |
|---|---|
| **[MEASURED]** | observed on this machine, raw data in this directory |
| **[SIMULATED]** | output of the offline scheduler, from measured inputs |
| **[UPPER BOUND]** | theoretical limit, not achievable as stated |
| **[DEFERRED]** | not done in this pass, with the reason |

---

## 0. Correction to the previous pass: the NPU duty cycle was overstated

**Retracting a published number before building on it.**

`artifacts/next-pass/RESULTS.md` section 8 reported NPU duty cycles of 10.4-44.6%
and a per-prefill NPU busy time of ~973 ms at 15 threads. Those came from
`tools/duty_cycle.sh`, which divided *cumulative* NPU device time by the number of
timed reps. **llama-bench runs one warmup prefill in addition to the `-r` timed
reps**, and that warmup dispatches to the NPU exactly like a timed rep. The
divisor therefore charged a whole extra prefill's device time across the reps.

Confirmed directly **[MEASURED]** -- dispatch count against `-r`, tiles=2, 15
threads:

| `-r` | dispatches | device ms |
|---:|---:|---:|
| 1 | 1284 | 1495.8 |
| 2 | 1926 | 2233.3 |
| 4 | 3210 | 3698.0 |

Exactly `642 * (r + 1)`. The slope is one prefill (642 dispatches); the intercept
is the warmup. `tools/duty_cycle2.py` now regresses device time on `r` and takes
the **slope** as per-prefill NPU busy time, and pairs it with llama-bench's own
tok/s, which already excludes the warmup -- so both sides of the ratio are finally
consistent.

Corrected duty cycle, pp2048 `-ub 2048`, idle machine **[MEASURED]**
(`duty_cycle.csv`):

| threads | tiles | tok/s | wall/prefill | NPU busy | **duty** | previously reported |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 431.9 | 4741 ms | 370 ms | **7.8%** | 10.4% |
| 4 | 2 (all) | 541.5 | 3782 ms | 734 ms | **19.4%** | 25.7% |
| 8 | 1 | 830.0 | 2467 ms | 366 ms | **14.8%** | 20.2% |
| 8 | 2 (all) | 753.1 | 2719 ms | 730 ms | **26.9%** | 36.4% |
| 15 | 1 | 1222.9 | 1675 ms | 410 ms | **24.5%** | 32.0% |
| 15 | 2 (all) | 927.5 | 2208 ms | 742 ms | **33.6%** | 44.6% |

**The NPU is idle for 75.5% of prefill at the winning operating point, not 68%.**
The direction of the previous conclusion is unchanged and in fact strengthened;
the magnitude was wrong.

A second, independent check confirms the correction rather than the original.
With corrected timing the NPU sustains **11.0 TFLOPS of useful arithmetic
(11.6 TOPS issued)** in-model -- which matches the standalone kernel benchmark of
11.6-13.2 TOPS. The previous pass reported an in-model 8.37 TFLOPS and could not
explain why it sat 1.6x below the same kernel measured standalone. That gap was
the artefact, and it is now closed.

### What this also invalidates

The previous pass solved three configurations for a "CPU-side residue" of 1209 ms
and concluded **Zen 5 at 15 threads does 10.71 TFLOPS of BitNet linear algebra
against the NPU's 8.37**. That arithmetic used the inflated NPU time. Redone with
corrected numbers it implies a CPU linear rate of ~16 TFLOPS, which is ~89% of the
AVX512-VNNI peak and not credible -- so the residue is **not** identical between
CPU-only and hybrid, and the simple three-equation solve is invalid.

The most likely reason: in CPU-only mode the f32 epilogue is fused into the same
pass as accumulation, whereas in hybrid mode it is a separate pass over the NPU's
int32 output. Hybrid therefore carries *more* CPU work, not the same amount.

**The "CPU is the faster GEMM engine by ~1.3x" claim is therefore withdrawn
pending direct measurement.** The weaker and still-supported statement is that the
two engines are of broadly comparable throughput on this workload. Section 2
measures the residue directly instead of inferring it.

---

## 1. Baseline reproduction [MEASURED]

pp2048, `-ub 2048`, 3 interleaved repetitions of 3 inner reps each, idle machine.
Raw: `baseline.csv`.

| threads | mode | tok/s (median) | spread | vs CPU-only |
|---:|---|---:|---:|---:|
| 15 | CPU-only | 1023.9 | 1018.0-1025.1 | — |
| 15 | hybrid (auto) | 1216.2 | 1211.6-1232.2 | **1.19x** |
| 8 | CPU-only | 639.7 | 637.3-641.9 | — |
| 8 | hybrid (auto) | 839.7 | 830.7-840.5 | **1.31x** |

Matches the recorded baseline (1020.7 / 1219.7 / 638.8 / 823.7) within 2%. **No
drift.** Dispersion is under 2% across all four configurations.

Also verified:
- **CPU-only emits exactly zero NPU dispatches** in every run.
- Hybrid dispatch count is `642 * (r+1)`, which decomposes exactly:
  `30 layers x 11 dispatches x 2 token tiles x (147/150 offloaded tensors) = 646`.
- Governor `powersave` (amd-pstate EPP default), boost enabled; package 54-73 C
  across the batch; load average recorded per row.
