# Direct mapped-output epilogue (stage_out elimination)

Targets the largest cheap measured inefficiency in the existing NPU path: the
mapped-C -> `g_acc` copy that thread 0 performs before the multi-threaded scaling
epilogue can run. The epilogue is made to consume NPU output directly.

**This is not "NPU epilogue fusion".** The scaling epilogue stays on the CPU,
multi-threaded, with the same arithmetic. Only the intermediate host copy goes.

Base: `3dff59bb3af4d641aa6c65d54b45d48c38d238a7` (`overlap-de-risk`), which is
that branch's tip after this pass's bookkeeping correction. `main` untouched at
`885df0ca`; `next-pass-results` untouched at `fb4493e9`.

Tags used below, never blurred:

| tag | meaning |
|---|---|
| **[MEASURED]** | observed on this machine; raw data in this directory |
| **[DERIVED]** | arithmetic over measured quantities |
| **[PREDICTED]** | a projection, not yet confirmed |
| **[DEFERRED]** | not done, with the reason |

---

## 0. Baseline reproduction [MEASURED]

pp2048 `-ub 2048`, 3 interleaved reps of 3 inner reps, idle machine.
Raw: `baseline.csv`.

| threads | mode | tok/s (median) | spread | vs `overlap-de-risk` |
|---:|---|---:|---:|---:|
| 15 | CPU-only | 1020.5 | 1018.4-1020.7 | 1023.9 |
| 15 | hybrid auto | 1221.7 | 1215.0-1228.8 | 1216.2 |
| 8 | hybrid auto | 837.6 | 828.7-839.0 | 839.7 |
| 4 | hybrid auto | 543.1 | 541.7-545.0 | 542.7 |

Within 1% of the previous branch. CPU-only emits **zero** NPU dispatches. Shape
tests bit-exact. Perplexity identical in both modes (307.5806 +/- 27.85495).

---

## 1. Where C-output time actually goes [MEASURED]

The aggregate `stage_out` counter could not answer this, and it sits inside the
`k_chunks == 1` branch, so it never measured `ffn_down`'s deep-K path at all.
Per-logical-shape counters now separate every stage. Raw:
`output_cost_by_shape.csv`.

**pp2048 `-ub 2048` auto, 15 threads, per prefill (wall 1654 ms):**

| shape | n | wait | sync_out | stage_in | stage_out | partacc | partcopy | epi wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `attn_q` + `attn_out` | 60 | 68.2 | 6.0 | 7.9 | **42.9** | 0 | 0 | 9.0 |
| `ffn_gate` + `ffn_up` | 174 | 193.1 | 23.1 | 8.5 | **120.2** | 0 | 0 | 29.0 |
| `ffn_down` | 87 | 95.7 | 11.0 | 15.5 | 0 | **31.9** | **10.9** | 4.4 |
| **total** | | 357.0 | 40.1 | 31.9 | **163.2** | 31.9 | 10.9 | 42.3 |

- thread-0 serial output work: **237.8 ms, 14.4% of wall**
- `stage_out` alone: **163.2 ms, 9.9%** -- what this change removes
- **deep-K path: 42.8 ms, 2.6% -- previously uninstrumented and unknown**

Forced all-NPU amplifies `stage_out` to **308.2 ms (14.1%)** as expected, since
every token tile goes to the device. pp3968 gives **333.6 ms (8.2%)**.

So the ceiling for `k_chunks == 1` direct output is 9.9% (~1.11x) at 2K
**[DERIVED]**, and `ffn_down`'s 2.6% is a separate target this pass does not
touch.

---

## 2. Implementation

```
before:  NPU -> reused mapped C -> thread-0 memcpy -> g_acc -> barrier
              -> all threads read g_acc, scale into dst

after:   NPU -> persistent per-(token tile, N chunk) slot -> barrier
              -> all threads read that slot directly, scale into dst
```

Scope: `k_chunks == 1` -- `attn_q`, `attn_out`, `ffn_gate`, `ffn_up`. `ffn_down`
(`k_chunks == 3`) keeps the deep-K accumulation path unchanged, per the brief.
Behind `BITNET_XDNA_DIRECT_OUT=1`, default off.

Decisions worth recording:

- **Separate XRT BOs, not sub-buffers of an arena.** Simplest robust mechanism
  first; rebinding a different BO per dispatch was previously measured at
  0.000-0.007 ms, so binding cost is not a reason to prefer sub-buffers.
- **Held in a `std::deque`, not a `std::vector`.** Growing the pool must never
  move an `xrt::bo` whose mapped host pointer has already been handed out.
- **The slot index carries both dimensions** -- token tile *and* N chunk -- so a
  later token tile cannot clobber an earlier tile's results before the epilogue
  has read them. Sizing for one token tile would have been the obvious bug.
- **Lifetime is bounded by the existing ggml barrier after the epilogue**, not by
  timing. A slot is rewritten only on the next graph node, which cannot begin
  until every epilogue reader has finished.

Arena: **30 MiB** (3 slots x 10 MiB) at the deployed split, **60 MiB** at all-NPU.

### Mechanism confirmed by counter, not inferred [MEASURED]

```
BITNET_XDNA_DIRECT_OUT=0   stage_out = 324.9 ms over 4.54 GB
BITNET_XDNA_DIRECT_OUT=1   stage_out =   0.0 ms over 0.00 GB
```

with `ffn_down`'s `partacc`/`partcopy` unchanged, confirming the scope held.

---

## 3. Correctness [MEASURED]

All twelve direct shape cases bit-exact in **all four** combinations of
`direct` x `async`, including T=1536 and T=2048 which exercise the multi-token-
tile lifetime hazard the arena introduces:

```
                       direct=0 async=0   direct=0 async=1
                       direct=1 async=0   direct=1 async=1
attn_q / attn_out      T=1024/1536/2048   bit-exact in all four
ffn_gate / ffn_up      T=1024/1536/2048   bit-exact in all four   (3 N-chunks)
ffn_down               T=1024/1536/2048   bit-exact in all four   (3 K-chunks)
```

End-to-end perplexity identical across every mode:

| mode | PPL |
|---|---|
| CPU-only | 307.5806 +/- 27.85495 |
| hybrid, `g_acc` | 307.5806 +/- 27.85495 |
| hybrid, direct | 307.5806 +/- 27.85495 |
| hybrid, direct + async | 307.5806 +/- 27.85495 |

**No silent fallback:** 147 resident tensors and no declined tensors under direct
output -- the same coverage as the `g_acc` path.

---

## 4. Four-variant matrix, both paths at R=10 [MEASURED]

Interleaved round-robin, 5 reps x 3 prefills. Raw: `direct_output_ab.csv`.
Both arms use the old R=10 split here, which is why section 5 re-measures with
each path at its own calibration.

| cell | A sync+`g_acc` | B async+`g_acc` | C sync+direct | D async+direct | C vs A |
|---|---:|---:|---:|---:|---:|
| pp2048 ub2048 t15 | 1215.7 | 1249.6 | **1265.0** | 1268.5 | 1.041x |
| pp2048 ub2048 t8 | 817.1 | 829.2 | 822.5 | 824.6 | 1.007x |
| pp2048 ub2048 t4 | 541.0 | 560.6 | **584.1** | 585.8 | 1.080x |
| pp2048 all-NPU t15 | 926.6 | 1002.4 | **1085.9** | 1080.2 | **1.172x** |
| pp3968 ub2048 t15 | 969.7 | 1000.5 | **1016.9** | 1013.0 | 1.049x |
| pp2048 ub1024 t15 | 924.3 | 1008.2 | **1082.8** | 1082.2 | **1.172x** |

**The gain tracks how much of thread 0's staging is *exposed* rather than hidden
behind the CPU threads' own token share.** This corrects a framing from the
previous pass: "while the other 14 threads are parked at the barrier" is true at
all-NPU, but at the deployed auto split those threads are computing, and thread
0's staging is partly or wholly hidden behind them. Hence 1.172x at all-NPU and
at `-ub 1024`, 1.080x at 4 threads (where the cost model already gives the NPU
every tile), and only 1.007x at 8 threads, where 7 CPU workers are the critical
path in that barrier.

The epilogue does get dearer, as expected when reading a freshly-synced device
buffer instead of a host copy: 623 -> 714 thread-ms at t15 (+6 ms wall),
1210 -> 1304 at all-NPU. Far smaller than the staging removed.

### Task 3: is async subsumed? **Yes.**

D is never better than C beyond noise, and is slightly *worse* at all-NPU
(1080.2 vs 1085.9) and pp3968 (1013.0 vs 1016.9). Async existed to overlap
evacuation with the next dispatch; direct output removes the evacuation, so there
is nothing left to hide. **Async is superseded** -- see section 9.

---

## 5. The cost model moves [MEASURED]

R in `f = R/(R + n_threads - 1)` measures the cost of NPU-assigned work, and the
`g_acc` path charged every NPU token tile an extra single-threaded staging copy.
Removing it makes NPU tiles cheaper and moves the balance point toward the
device. Raw: `cost_model.csv`, `cost_model_recal.csv`.

pp2048 `-ub 2048`, exhaustive tile sweep:

| threads | path | tiles=0 | tiles=1 | tiles=2 | auto | pick | best | regret |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | `g_acc` | 346.4 | 429.6 | 539.5 | 542.4 | 2 | 2 | 0.995x |
| 4 | direct | 347.7 | 430.2 | 588.2 | 586.7 | 2 | 2 | 1.003x |
| 8 | `g_acc` | 637.5 | 817.9 | 755.1 | 824.6 | 1 | 1 | 0.992x |
| 8 | direct | 631.0 | 820.9 | **844.9** | 816.5 | 1 | **2** | **1.035x** |
| 15 | `g_acc` | 1012.8 | 1220.4 | 924.2 | 1228.3 | 1 | 1 | 0.994x |
| 15 | direct | 1008.4 | 1260.4 | 1077.7 | 1264.4 | 1 | 1 | 0.997x |

**At 8 threads the optimum moves from 1 tile to 2**, and R=10 no longer finds it.

Recalibrated against a wider grid (pp2048 and pp3072 x threads 4/6/8/10/12/15,
exhaustive vs auto):

| | R = 10 | R = 25 |
|---|---:|---:|
| mean regret | 1.026x | **1.005x** |
| worst regret | **1.147x** (6 threads) | 1.017x |

R=10 picks 1 tile at 6 threads where 2 is worth 733.1 vs 639.0 -- a 14.7% loss.
R in [21, 41] reproduces all three pp2048 optima; **R=25** sits mid-range.

Implemented as a default that follows the active output path -- `kR_GACC = 10`,
`kR_DIRECT = 25` -- with `BITNET_XDNA_NPU_THREADS` still overriding both.
Verified: at 8 threads `g_acc` picks 1 tile, direct picks 2, and an explicit
override of 10 returns direct to 1. **No extra model complexity was added**;
only the constant moved, per the brief.

---

## 6. Deployment comparison, each path at its own calibration [MEASURED]

5 interleaved reps. Raw: `deployment_ab.csv`.

| prompt | threads | CPU-only | `g_acc` R=10 | direct R=25 | direct/`g_acc` | direct/CPU |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 4 | 346.1 | 538.7 | 583.3 | **1.083x** | 1.685x |
| 2048 | 6 | 498.5 | 635.9 | 732.8 | **1.152x** | 1.470x |
| 2048 | 8 | 639.6 | 831.0 | 836.6 | 1.007x | 1.308x |
| 2048 | 15 | 1014.2 | 1211.8 | 1274.1 | **1.051x** | 1.256x |
| 3968 | 4 | 279.2 | 364.3 | 375.5 | 1.031x | 1.345x |
| 3968 | 6 | 410.5 | 508.7 | 536.5 | **1.055x** | 1.307x |
| 3968 | 8 | 512.6 | 638.4 | 642.8 | 1.007x | 1.254x |
| 3968 | 15 | 837.6 | 962.1 | 1006.6 | **1.046x** | 1.202x |

Never worse anywhere measured. Largest gains at 4-6 threads -- the thread-starved
regime a resident controller actually runs in -- and the 6-thread case (1.152x)
combines an exposed staging cost with the recalibrated split.

---

## 7. Energy [MEASURED]

Package RAPL only; the `core` subdomain is unusable on this SoC (non-monotonic,
not a wrap). Alternating arms, 5 reps, pp2048 `-ub 2048`. Raw: `energy.csv`.

| threads | path | tok/s | avg W | mJ/token | vs `g_acc` |
|---:|---|---:|---:|---:|---:|
| 6 | `g_acc` | 636.6 | 83.7 | 131.54 | — |
| 6 | direct | 728.2 | **70.6** | 96.94 | **0.737x** |
| 8 | `g_acc` | 826.8 | 93.5 | 113.16 | — |
| 8 | direct | 837.0 | **74.9** | 89.40 | **0.790x** |

**The energy win is larger than the throughput win, and at 8 threads it is almost
entirely energy: 1.007x throughput but 0.790x energy per token, at 18.6 W lower
average package power.** That is the clearest evidence that the staging copy was
not merely idle time -- a single thread hammering 2.27 GB through memory while
others waited was burning power without producing throughput. Removing it shows
up as power even where it does not show up as speed.

---

## 8. Whole-system Pareto [MEASURED]

Controller (2048-token prompt, 32-token answer) against 8 synthetic CPU
co-tenant workers, 3 interleaved reps. Raw: `whole_system.csv`.

| arm | TTFT | total | co-tenant it/s |
|---|---:|---:|---:|
| 15T CPU-only | 3368 ms | 3921 ms | 1062.8 |
| 8T hybrid `g_acc` | 2861 ms | 3505 ms | 1070.5 |
| **8T hybrid direct** | **2803 ms** | **3460 ms** | **1078.9** |
| 4T hybrid direct | 3830 ms | 4668 ms | 1088.1 |

**8T direct dominates 8T `g_acc` on both axes** -- 2.1% faster TTFT *and* 0.8%
more co-tenant throughput -- and continues to dominate 15T CPU-only (17% faster
TTFT, 1.5% more co-tenant headroom). The co-tenant gain is the same effect as
section 7: less memory traffic monopolised by thread 0 leaves more for everyone
else.

So the answer to the question the task actually asked -- does this improve the
*already-useful* Pareto point, not just an isolated benchmark -- is yes, on both
axes simultaneously.

---

## 9. Decision

### **DIRECT OUTPUT: PROMOTE**

| criterion | finding |
|---|---|
| correctness | Unchanged. Bit-exact in all four `direct` x `async` combinations across all twelve shape cases including the multi-token-tile paths; perplexity identical; 147 resident tensors, no declines, no silent fallback. |
| complexity | Modest and contained. ~150 lines inside `runtime/`; no ggml, graph, barrier or quantization change. One new lifetime rule, bounded by the existing barrier. |
| end-to-end win | Repeatable and material: 1.007-1.152x throughput, **0.737-0.790x energy per token**, and a Pareto improvement on both axes under co-tenancy. |
| regressions | None measured. The epilogue costs ~6 ms wall more at t15; memory grows 30 MiB at the deployed split (60 MiB at all-NPU, and it would be 120 MiB at `-ub 4096`). |

Promoted to the default. `BITNET_XDNA_DIRECT_OUT=0` restores the `g_acc` path,
which stays as reference and as the fallback for `ffn_down`'s deep-K case.

### **ASYNC: SUPERSEDED**

Kept behind its flag, default off, **not recommended**. Its entire purpose was
overlapping evacuation with the next dispatch, and direct output removes the
evacuation. It is never better than direct output alone beyond noise and is
slightly worse in two cells. It should be deleted whenever the `g_acc` path is
retired; it is retained now only because that path is still live for `ffn_down`.

### What is left on the output path [MEASURED]

| cost | per prefill (2K, deployed) | status |
|---|---:|---|
| `stage_out` | 163.2 ms -> **0** | eliminated |
| deep-K `partacc` + `partcopy` (`ffn_down`) | 42.8 ms | **untouched, now measured** |
| `stage_in` | 31.9 ms | untouched; activations must still reach the device |
| epilogue | ~42 ms wall | inherent; slightly dearer reading device memory |

`ffn_down`'s deep-K path is the obvious remaining target at 2.6% of wall, and it
is the natural scope for a follow-up. It is deliberately excluded here.
