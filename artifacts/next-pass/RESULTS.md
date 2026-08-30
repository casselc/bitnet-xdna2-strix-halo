# Next-pass results: where the next real end-to-end win is

Continuation of the working MVP. Everything below was measured on the Strix Halo
box.

**Measurement conditions.** Sections 1-3 were measured while the machine carried
background load (~2 of 16 cores); every sweep row records the background CPU it
saw, and all comparisons are interleaved rather than blocked. **Sections 6-8 were
re-measured on an idle machine** after that load was removed -- absolute numbers
there are higher, and they supersede the earlier ones where they overlap.

Builds: `refs/BitNet/build-xdna2` (sections 1-3), `build-xdna3` (sections 6-10;
adds the thread-aware cost model). Both llamafile-off with the XDNA runtime linked.

Raw data in this directory: `sweep.csv` (123 runs), `whole_system.csv` (48 runs).
Harnesses: `tools/sweep.py`, `tools/whole_system.py`, `tools/duty_cycle.sh`,
`tools/energy_probe2.sh`, `tools/energy_per_token.sh`.

**The short answer to "where is the next real end-to-end win?"** -- The NPU is
idle for 68% of prefill because it is serialized against the CPU between ops, and
60% of prefill is work it never touches. Overlapping the two engines across
micro-batches is worth a further 1.39x and needs no new kernel. Section 10.

---

## 1. The headline: the NPU's value is set by how many CPU threads it is competing with

The most important *deployment* result of the pass. (The most important
*diagnostic* result is section 8: the NPU is idle 68% of prefill.)

Sweeping NPU tile share against CPU thread count (medians of 2 interleaved reps):

| prompt | -ub | threads | CPU-only | best hybrid | share | **gain** |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 2048 | 4 | 337 | 505 | all NPU | **1.50x** |
| 2048 | 2048 | 8 | 623 | 743 | half | **1.19x** |
| 2048 | 2048 | 15 | 814 | 1018 | half | **1.25x** |
| 2048 | 1024 | 4 | 352 | 535 | all NPU | **1.52x** |
| 2048 | 1024 | 8 | 651 | 739 | all NPU | **1.13x** |
| 2048 | 1024 | 15 | 987 | 981 | none | 0.99x |
| 3968 | 2048 | 4 | 268 | 385 | all NPU | **1.44x** |
| 3968 | 2048 | 8 | 511 | 617 | half | **1.21x** |
| 3968 | 2048 | 15 | 805 | 874 | half | **1.09x** |

**At 4 threads the NPU is worth ~1.5x. At 15 threads it is worth ~1.0-1.25x.**

The previously reported "1.12x" was measured at 15 threads, which is the *worst*
case for the NPU and the least likely deployment. For the intended topology --
where the CPU is running Samizdat/Jolt/SCI and the controller gets a slice -- the
relevant number is closer to **1.5x**.

Two corollaries worth stating plainly:

- **`-ub` must be >= the NPU tile (1024) or the NPU cannot be used at all.** Every
  `ub=512` row is 1.00x, because a 512-token micro-batch never reaches the tile
  size. This is a deployment footgun: the llama.cpp default is 512.
- **`tiles=0` reproduces CPU-only within noise** in every row, confirming the
  offload check itself costs nothing when it declines.

---

## 2. A cost model replaces the global split (Workstream 2)

`BITNET_XDNA_SPLIT` was a single global fraction rounded to whole tiles. It cannot
be right at both ends, because **thread 0 is consumed driving the device**: the
CPU's share is served by `nth-1` workers, so the balance point moves with thread
count.

Model, fitted to the sweep:

```
f = R / (R + (n_threads - 1)),   R = 10        snapped to a whole NPU tile
```

`R` is the NPU's throughput expressed in Zen 5 threads. R in [9, 12] reproduces
the measured optimum in **all six** swept (tiles-available, thread-count) cases,
including the awkward one where 15 threads with only one tile available should
decline the NPU entirely:

| tiles available | threads | model picks | measured best |
|---:|---:|---:|---:|
| 2 | 4 | 2 | 2 |
| 2 | 8 | 1 | 1 |
| 2 | 15 | 1 | 1 |
| 1 | 4 | 1 | 1 |
| 1 | 8 | 1 | 1 |
| 1 | 15 | 0 | 0 |

Implemented as `bitnet_xdna_token_split_nt()`; `BITNET_XDNA_SPLIT` and
`BITNET_XDNA_TILES` remain as overrides for controlled benchmarking.
`R` is not a constant of nature -- it will move with kernel quality and with the
CPU's thread scaling -- so it is overridable via `BITNET_XDNA_NPU_THREADS`.

**Granularity is now the binding constraint, not the ratio.** At T=2048 with a
1024-token tile there are only three reachable partitions (0, half, all). The
model frequently wants something in between and cannot express it.

---

## 3. Redundant activation traffic: real, and worth ~2% (Workstream 3)

The accumulate loop was `[token tile][N chunk][K chunk]`, which refilled and
re-flushed the mapped activation buffer once per N chunk even though every N
chunk of a K slice consumes the *same* activations. Predicted waste: 608 MB of
memcpy+CLFLUSH per 2048-token prefill, 36% of all activation traffic.

Swapped to `[token tile][K chunk][N chunk]`, with the device sync split from the
dispatch so A is flushed once per K slice.

| | N-outer (before) | K-outer (after) |
|---|---:|---:|
| `sync_in` | 60 ms | **39 ms (-35%)** |
| pp2048 @ t=8 | 717.5 t/s | 731.7 t/s |

**The traffic reduction is exactly as predicted (36% vs 35% measured). The
end-to-end gain is ~1.02x**, because `sync_in` was only ~4% of dispatch time to
begin with. The brief's warning not to assume this was large merely because the
code looked redundant was correct. Kept -- it is strictly better and costs
nothing -- but it is not the win.

---

## 4. Correctness

`tests/test_xdna_shapes.cpp` now drives `bitnet_xdna_accumulate` -- the same entry
point ggml uses -- for all three real shapes against the scalar oracle with real
GGUF weights, at T = 1024 / 1536 / 2048 (one tile, one-and-a-half tiles exercising
the M zero-padding, two tiles):

- `attn_q`/`attn_output` 2560x2560 -- 1 N-chunk, 1 K-chunk
- `ffn_gate`/`ffn_up` 2560x6912 -- 3 N-chunks, N padded 6912 -> 7680
- `ffn_down` 6912x2560 -- 3 K-chunks with int32 accumulation

Previously only 2560x2560 had direct bit-exact coverage; the chunking, padding and
K-accumulation paths -- the ones most likely to be wrong -- had none.

End-to-end perplexity is unchanged at every configuration tested:
`307.5806 +/- 27.85495`, identical to CPU-only.

---

## 5. Rejected hypotheses

Kept deliberately, because several of these were confidently believed at some
point in this project and three of them were briefly *reported* as findings.

| Hypothesis | Verdict | Evidence |
|---|---|---|
| int4 weights give ~2x MAC throughput | **FALSE — 1.035x** | Matched int8xint8 / int8xint4 kernels differing in one line; two `REPEAT` values solve out dispatch cost. `mac_4x16_16x16` really does 1024 MACs, and really does issue at half the rate. Bandwidth win only. |
| "The NPU is clearly the faster engine" | **Retracted** | Rested on a 777 ms / 24% figure measured at `-ub 512`, where pp2048 is *four* micro-batches. Correct figure is 1555 ms / 48%. The engines are roughly comparable. |
| Context switch is 0.10-0.22 ms, far better than published | **Retracted** | Measured on trivial `add_one`/`add_two` kernels. Real designs cost +1.2 to +2.4 ms. |
| Redundant activation traffic is a major cost | **True but ~2%** | 36% of activation bytes predicted, 35% measured — but `sync_in` was only ~4% of dispatch time. See section 3. |
| `xrt::run` construction per dispatch is a real cost | **FALSE** | Persistent run + `set_arg` vs fresh `kern(...)`, interleaved: 0.000-0.007 ms. |
| CPU memory-bandwidth contention starves the NPU | **FALSE** | 1 -> 16 ggml threads of concurrent load: 5.61 -> 5.55 TOPS. Flat. |
| More resident weight buffers degrade the NPU | **FALSE** | 1 -> 60 buffers (375 MiB): 10.8 -> 11.7 TOPS. Flat. |
| "Run reuse saves nothing"; "BO rebinding costs 0.266 ms" | **Both false positives** | Artifacts of block-ordered benchmarking against 32% between-run drift. Disappeared under round-robin interleaving. |
| ~50 TOPS is the yardstick | **Wrong yardstick** | That is the marketing int8 figure. Published int8->**int32** (which is what we need — the accumulator must not saturate) is 25.31 TOPS. |

Two process lessons that cost real time and are worth carrying forward:

- **Block-ordered A/B is not safe on this machine.** Between-run drift is ~32%,
  larger than most effects being measured. Everything here is interleaved.
- **A stray background process is indistinguishable from a real regression.** A
  leftover NPU load of mine degraded the device 5.9x and produced a plausible,
  entirely false headline (0.87x / 0.75x / 0.64x). The tell was that the CPU-only
  arm swung 35% while the hybrid arm was stable-but-low. Every run now records
  the background load it saw.

---

## 6. Whole-system best configuration

The controller does not own the machine. `tools/whole_system.py` runs the
controller workload (2048-token structured prompt, 32-token answer) against a
co-tenant CPU load, and records what the co-tenant still achieves. Medians of 2
interleaved reps, idle machine.

**Idle (no co-tenant):**

| threads | config | TTFT | vs CPU-only |
|---:|---|---:|---:|
| 2 | CPU-only | 11184 ms | — |
| 2 | + NPU | 5870 ms | **1.91x** |
| 4 | CPU-only | 5885 ms | — |
| 4 | + NPU | 3794 ms | **1.55x** |
| 8 | CPU-only | 3194 ms | — |
| 8 | + NPU | 2460 ms | **1.30x** |
| 15 | CPU-only | 2013 ms | — |
| 15 | + NPU | 1665 ms | **1.21x** |

**With 8 co-tenant workers — the Pareto frontier:**

| threads | config | TTFT | co-tenant it/s |
|---:|---|---:|---:|
| 15 | + NPU | **2476 ms** | 1147.2 |
| 8 | + NPU | **2855 ms** | **1170.0** |
| 8 | CPU-only | 3636 ms | 1173.0 |
| 15 | CPU-only | 3380 ms | 1156.5 |
| 4 | + NPU | 4151 ms | 1177.5 |
| 4 | CPU-only | 6485 ms | 1181.5 |

**The whole-system answer: 8 threads + NPU strictly dominates 15 threads CPU-only
on both axes** -- 2855 ms vs 3380 ms TTFT (18% faster) *and* 1170.0 vs 1156.5
co-tenant it/s (1.2% more headroom left for the other tenant). Spending seven more
CPU cores buys less than the NPU does, and costs the co-tenant more.

That is the configuration to deploy: **`-t 8 -ub 2048`, cost-model split.** If TTFT
is worth more than co-tenant throughput, 15 threads + NPU reaches 2476 ms for a
3.4% co-tenant cost.

Note the trend: the NPU is worth **1.91x at 2 threads and 1.21x at 15**. The
scarcer the CPU, the more the NPU is worth -- which is exactly the regime a
resident controller sharing a box with Samizdat/Jolt/SCI actually runs in.

---

## 7. Energy: the hybrid is faster *and* cheaper

Previously unmeasured. RAPL on this SoC exposes `package-0` and a `core`
subdomain; **the `core` domain is unusable** -- it reads +15.1 W for one busy
thread but +7.2 W for sixteen, which is non-monotonic and not a counter wrap
(both domains wrap at 65.5 kJ, ~500 s at full load). Package-only below.

Paired alternating idle/load windows, 10 s x 4 reps, idle machine:

| load | package delta | sd |
|---|---:|---:|
| 1 busy Zen 5 thread | +26.73 W | 0.14 |
| 16 busy Zen 5 threads | +110.99 W | 0.61 |
| **NPU, sustained GEMM** | **+11.14 W** | 1.30 |

**A saturated NPU costs less than half of one busy CPU core.** The measured `core`
delta for the NPU load is +0.02 W, consistent with the host thread blocking in
`wait()` rather than spinning -- so the +11.14 W is the device, not its driver.

Energy per prefill token during the real workload (pp2048, 5 reps, package energy
over the timed run):

| threads | mode | tok/s | avg W | mJ/token | vs CPU-only |
|---:|---|---:|---:|---:|---:|
| 4 | CPU-only | 350.8 | 66.8 | 190.3 | — |
| 4 | hybrid | 542.7 | 59.8 | 110.2 | **0.58x** |
| 8 | CPU-only | 638.8 | 96.9 | 151.8 | — |
| 8 | hybrid | 823.7 | 94.1 | 114.3 | **0.75x** |
| 15 | CPU-only | 1020.7 | 117.3 | 115.0 | — |
| 15 | hybrid | 1219.7 | 112.4 | 92.1 | **0.80x** |

**The hybrid is faster and lower-energy at every thread count, and draws less
average power while doing it** (59.8 W vs 66.8 W at 4 threads). Prefill energy
falls 20-42%, with the largest saving in the thread-starved regime. This is the
strongest single argument for the NPU in this design, and it is independent of
the throughput argument.

---

## 8. Where the time actually goes

The decisive measurement of this pass. `BITNET_XDNA_STATS` reports NPU device
time; comparing it against prefill wall time gives the duty cycle. pp2048,
`-ub 2048`, idle machine, 3 reps (`tools/duty_cycle.sh`):

| threads | tiles | tok/s | wall/rep | NPU busy | **duty** |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 (half) | 429.6 | 4768 ms | 497 ms | **10.4%** |
| 4 | 2 (all) | 541.8 | 3780 ms | 972 ms | **25.7%** |
| 8 | 1 | 829.6 | 2469 ms | 498 ms | **20.2%** |
| 8 | 2 | 752.5 | 2722 ms | 990 ms | **36.4%** |
| 15 | 1 | 1214.9 | 1686 ms | 539 ms | **32.0%** |
| 15 | 2 | 938.5 | 2182 ms | 973 ms | **44.6%** |

**At the operating point that wins, the NPU computes for 32% of prefill. Even
when it is given 100% of the offloaded linear algebra, it computes for 45%.**

Two things fall out, and they matter more than any kernel number:

1. **NPU busy time is constant** (972 / 990 / 973 ms) across 4, 8 and 15 threads,
   as it must be -- same work, same device. **CPU-side time is what varies**
   (2808 / 1732 / 1209 ms). The two add to the wall time. The engines are
   *serialized*, not overlapped: the hook dispatches, blocks in `wait()`, and
   crosses a `ggml_barrier` once per matmul, 330 times per token-tile.
2. **The kernel is not the bottleneck.** While it runs, the NPU sustains
   **~8.4 TFLOPS of useful arithmetic** (8.9 TOPS issued; 92% shape efficiency --
   the 8% is N 6912->7680 padding and K 6912->7680 on `ffn_down`). Dispatch
   accounting confirms it exactly: 147/150 tensors x 11 dispatches/layer x 30
   layers x 2 tiles x 2 reps = 1284, the measured count.

So the arithmetic that matters:

```
wall(15T, all-NPU) = 973 ms NPU  +  1209 ms CPU  =  2182 ms      (serialized)
                     max(973, 1209)              =  1209 ms      (if overlapped)
```

**Perfect overlap is worth 1.80x at this operating point -- 938 -> 1694 tok/s --
without touching the kernel, the weight format, or the split.**

And the composition argument decides the priority. A kernel tuned from 8.4 to
25 TFLOPS would cut NPU busy time 973 -> 327 ms, giving `327 + 1209 = 1536 ms`
(1.42x). But *combined* with overlap it gives `max(327, 1209) = 1209 ms` -- the
same as overlap alone. **Once the engines overlap, the NPU kernel stops being on
the critical path entirely.** Kernel work done first is largely wasted; overlap
done first makes the kernel irrelevant and moves the bottleneck to the CPU-side
residue (`attn_k`/`attn_v`, attention, norms, the f32 epilogue, the f16 lm_head).

### 8b. The time model, and a correction to the milestone-A economics

Solving the three 15-thread configurations for the CPU-side residue (the work the
NPU never touches: `attn_k`/`attn_v`, attention itself, RMSNorms and sub-norms,
RoPE, relu^2, the f32 epilogue, the f16 tied lm_head):

```
all-NPU:    2182 ms  =  973 ms NPU        +  1209 ms residue    (measured directly)
CPU-only:   2006 ms  =  797 ms CPU linear +  1209 ms residue
split:      1686 ms  =  max(539, 399)     +  1209 ms residue
            predicted 1748 ms vs measured 1686 ms -- 3.7% error
```

The model reproduces the split configuration to within 3.7% from independently
measured quantities, so the decomposition is trustworthy. Two consequences:

**1. 60% of prefill is work the NPU cannot help with.** Not 5%, not 20%. Any
NPU-side improvement is capped by Amdahl at **1.66x**, and the current hybrid
already captures 1.19x of that.

**2. Milestone A's economics were wrong, and in the NPU's favour.** That analysis
divided the *entire* CPU prefill by the linear FLOPs and got "CPU 4.29 TFLOPS vs
NPU 6.53 TFLOPS", concluding the NPU was ~1.5x the CPU at linear algebra. It had
attributed the 1209 ms residue to the CPU's linear kernel. Corrected:

| engine | BitNet linear algebra |
|---|---:|
| Zen 5, 15 threads (AVX512-VNNI I2_S kernel) | **10.71 TFLOPS** |
| XDNA2, 32 cores, tuned int8 kernel | **8.37 TFLOPS** |

**The NPU is not the faster engine at this GEMM -- the CPU is, by ~1.3x.** (10.71
TFLOPS is ~55% of the Zen 5 `vpdpbusd` peak, which is a plausible figure for a
well-tuned kernel and an independent sanity check on the decomposition.)

The NPU earns its place for two different reasons, both measured here: it is
**additional** capacity that costs almost no CPU (section 6: 8 threads + NPU
dominates 15 threads CPU-only on both TTFT and co-tenant throughput), and it does
its share at **1/10th the power** (section 7).

---

## 9. Measured contribution of each optimization

| change | measured effect | kept? |
|---|---|---|
| Tuned xclbin (`m=128`, single-dispatch N=6912) vs stock `whole_array` | 9.08 -> **13.17 TOPS** (2560x2560); 5.76 -> **11.62** (2560x6912) | yes |
| Concurrent CPU+NPU split vs NPU-only | 938 -> **1215 tok/s** @15T (**1.29x**) | yes |
| `-ub` >= 1024 (NPU tile size) | the difference between the NPU being used and **not being used at all** | mandatory |
| Thread-aware cost model, `f = R/(R+nth-1)`, R=10 | auto-selects the measured optimum at 4, 8 and 15 threads: **1.55x / 1.30x / 1.19x** | yes |
| K-outer loop reorder (Workstream 3) | `sync_in` 60 -> 39 ms (-35%); **1.02x** end-to-end | yes |
| int8xint4 mixed-precision kernel | **1.035x** compute; halves weight DMA | not for compute |
| Persistent `xrt::run` + `set_arg` | 0.000-0.007 ms | no effect |
| Relaxed atomic instead of mutex in `bitnet_xdna_available()` | CPU-only pp512 878 -> **1277 tok/s** (removed a tax on the *baseline*) | yes |

End to end, versus CPU-only on the same machine: **1.19x at 15 threads, 1.30x at
8, 1.55x at 4, 1.91x at 2** -- at **0.80x / 0.75x / 0.58x** the energy per token.

---

## 10. Recommended next direction

**One recommendation: pipeline the NPU across micro-batches, so its linear work
overlaps the CPU's 1209 ms of non-offloadable work.**

Everything in section 8 points here and nowhere else:

- The NPU computes for **32%** of prefill at the winning operating point. The
  device is idle for two thirds of the time it is nominally "in use".
- The engines are **serialized**, not overlapped, across ops: the hook dispatches,
  blocks in `wait()`, and crosses a `ggml_barrier` once per matmul, ~330 times per
  token tile. Within a matmul the two engines do run concurrently -- that already
  works and is worth 1.29x -- but *between* matmuls the NPU has nothing to do.
- Hiding NPU work under the residue takes wall from 1686 -> **1209 ms**, i.e.
  1215 -> **1694 tok/s, a 1.39x further gain** at 15 threads, and more at lower
  thread counts where the NPU is worth more.
- **It requires no new kernel, no new weight format, and no accuracy risk.**

The composition argument is what makes this the *only* sensible next step rather
than one option among several. A kernel tuned from 8.4 to 25 TFLOPS -- a large,
speculative effort -- would give `327 + 1209 = 1536 ms` (1.10x). Combined with
overlap it gives `max(327, 1209) = 1209 ms`: **identical to overlap alone.** Once
the engines overlap, the NPU kernel is off the critical path and further kernel
work buys nothing. Doing the kernel first is largely wasted effort; doing overlap
first makes the kernel question moot and moves the bottleneck onto the CPU
residue, which is then the honest next target.

**Why it is feasible.** llama.cpp already splits prefill into micro-batches
(`-ub`), and the dependency structure permits a two-stage pipeline: micro-batch
`i+1`'s layer-0 q/k/v projections depend only on token embeddings, which are
available immediately -- only its *attention* needs micro-batch `i`'s KV. So the
NPU can run `i+1`'s projections while the CPU finishes `i`'s attention, norms and
epilogue.

**Honest cost.** This is the first change in this project that cannot be made
inside a single guarded `ggml-cpu.c` hunk. It needs a prefill loop that owns
micro-batch scheduling, which means either a custom prefill path or real changes
to `llama_decode`'s batching. That is a materially larger change than anything
attempted so far, and it is the reason to de-risk it first.

**First step, ~half a day, before committing to the above:** instrument the
1209 ms residue and break it down (attention vs norms vs epilogue vs `attn_k`/`v`
vs lm_head). Two outcomes change the plan:
- If the f32 epilogue is a large share, **fusing it into the NPU kernel** is a much
  cheaper win and should be done first.
- If attention dominates and grows quadratically toward the 4K controller
  contexts, the residue is the real ceiling and pipelining buys less than 1.39x --
  in which case prefill attention, not the linears, is the thing to move.

### What not to do next, and why

| | |
|---|---|
| Tune the kernel toward 25 TFLOPS | Off the critical path once overlapped (above). Also: the CPU already does this GEMM at 10.71 TFLOPS, so kernel work chases an engine that is not the faster one. |
| 2-bit / int4 weights over DMA | Measured **1.035x** compute. Real, but a *bandwidth* optimization -- and `sync_in` is 4% of dispatch time. It matters for NPU **decode** (28.5 -> 113.8 tok/s ceiling), which is explicitly not a goal. |
| Deep-K accumulation, fusion, packed residency | All target data movement, which section 3 measured at ~2% end-to-end. |
| Finer split granularity | The 0/50/100% tile quantization looked like a problem, but the optimum sits *at* 50% and the curve is flat near its peak; the loss is second-order. |
