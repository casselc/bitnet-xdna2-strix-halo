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

---

## 2. What the CPU-side residue actually is [MEASURED]

Instrumented with `runtime/ggml_node_profile.c`, which brackets every ggml graph
node on thread 0 between "before compute" and "after `ggml_barrier`". Because
ggml barriers after every node those intervals tile the graph exactly, and the
measurement confirms it: **node durations sum to 99.6-100.0% of the graph span**
in every configuration. There is no pool of hidden scheduler overhead.

Enabled at runtime by `BITNET_PROFILE=<path>`; with it unset the same binary
reproduces the baseline exactly (1224/1029 tok/s vs 1216/1024), so profiling does
not perturb what it measures.

NPU device time is attributed per node from the XDNA dispatch-time counter, so
"NPU time" and "CPU work inside an offloaded node" are separately measured.
Raw: `residue_2k.csv`, `residue_4k.csv`, `residue_*_allnpu.csv`.

### Whole prefill, 15 threads, `-ub 2048` (ms)

| category | pp2048 hybrid | pp2048 CPU-only | pp3968 hybrid | pp3968 CPU-only |
|---|---:|---:|---:|---:|
| **attention** (`FLASH_ATTN_EXT`) | **599** | 594 | **1988** | 1975 |
| ffn_gate | 209 (111 NPU) | 302 | 405 (223 NPU) | 584 |
| ffn_up | 213 (110 NPU) | 301 | 417 (222 NPU) | 580 |
| ffn_down | 228 (110 NPU) | 295 | 437 (219 NPU) | 566 |
| attn_q_proj | 79 (37 NPU) | 120 | 152 (74 NPU) | 231 |
| attn_out_proj | 80 (38 NPU) | 114 | 155 (74 NPU) | 219 |
| norm (121 nodes) | 90 | 96 | 171 | 187 |
| ffn_activation (relu^2) | 50 | 55 | 94 | 104 |
| attn_k_proj | 42 | 39 | 74 | 77 |
| attn_v_proj | 42 | 43 | 74 | 83 |
| residual_add | 21 | 23 | 40 | 45 |
| rope | 17 | 19 | 33 | 37 |
| lm_head | 6 | 7 | 12 | 14 |
| kv_cache_write | 3 | 3 | 5 | 6 |
| **total** | **1679** | 2010 | **4056** | 4709 |
| of which NPU device time | 406 (24.2%) | 0 | 811 (20.0%) | 0 |

### Attention dominates, and it is growing

**Attention is 599 ms (35.7%) of hybrid prefill at 2K and 1988 ms (49.0%) at 4K**
-- 3.3x more time for 1.94x more tokens, i.e. the expected O(T^2). At 4K it is
larger than every NPU-eligible projection combined, and it runs entirely on CPU.

This directly answers the question the pass was set to ask. The perfect-overlap
bound is `total / max(NPU, CPU-residue)`, and the residue is dominated by a term
that grows quadratically while NPU work grows linearly:

| context | NPU work | CPU residue | **[UPPER BOUND]** perfect overlap |
|---|---:|---:|---:|
| pp2048 | 406 ms | 1273 ms | **1.32x** |
| pp3968 | 811 ms | 3245 ms | **1.25x** |

**The upper bound falls as context grows, and it is well below the 1.39x claimed
last pass** (which used the inflated NPU time corrected in section 0). This is a
theoretical ceiling assuming zero dependency constraints; section 4 computes what
the real DAG permits.

### The micro-batch structure, which is what pipelining would exploit

At `-ub 1024`, pp2048 (2 micro-batches, per-position medians):

| category | mb0 | mb1 |
|---|---:|---:|
| attention | 159.4 ms | **414.9 ms** |
| ffn_gate/up/down | 540.8 ms | 542.2 ms |
| attn_q + attn_out | 146.1 ms | 145.0 ms |
| everything else | 122.6 ms | 120.9 ms |
| **span** | **970.9 ms** | **1224.6 ms** |

Attention is the only category that differs, and it differs by 2.6x -- mb1
attends to roughly three times the keys mb0 does. Everything else is flat, as it
must be. Sum across micro-batches reproduces the whole-prefill figure (159+415 =
574 ms vs 599 ms measured with a single micro-batch).

**But note what `-ub 1024` costs to obtain**: 922.9 tok/s against 1215.7 at
`-ub 2048`. Cross-micro-batch pipelining at 2K requires at least two
micro-batches, so it must start from a configuration that is **24% slower**, and
win that back before it shows any gain at all.

### The largest single inefficiency is not overlap [MEASURED]

Forcing every token tile to the NPU (`BITNET_XDNA_TILES=2`) isolates the cost of
*using* the device from the cost of its arithmetic. Per prefill at 2K:

| | all-NPU | auto split (deployed) |
|---|---:|---:|
| NPU device time | 745 ms | 399 ms |
| **`stage_out`** (mapped C buffer -> `g_acc`) | **309 ms**, 4.54 GB | **162 ms**, 2.27 GB |
| `stage_in` (activations -> mapped A buffer) | 58 ms | 32 ms |
| epilogue (int32->f32), wall | ~80 ms | ~40 ms |

`stage_out` is a `memcpy` of the NPU's int32 results out of the mapped output
buffer, running **single-threaded on thread 0 at 14 GB/s while the other 14
threads are parked at the `ggml_barrier`**. At the deployed configuration that
is **~194 ms per prefill, 11.4% of the 1697 ms wall**, spent moving bytes that
the epilogue is about to read again.

It exists because thread 0 issues every N-chunk dispatch into one reused mapped C
buffer, so results must be evacuated before the next dispatch overwrites them.
Giving the dispatches distinct output regions would let the epilogue -- which
already touches every one of these elements, across all threads -- read the
mapped buffer directly, removing the copy rather than parallelising it.

**This is a bigger, cheaper and lower-risk win than anything overlap offers at
2K, and unlike overlap it needs no change to llama.cpp's scheduler.**

### Correction: the shape tests were not exercising the chunked paths

`tests/test_xdna_shapes.cpp` allocated its weight blob inside `run_case` and
freed it on return. `bitnet_xdna` caches uploaded weights keyed by the tensor's
data pointer -- stable in production because GGUF tensors are mmap'd -- so the
allocator handed a later shape an address a freed earlier shape had used, and
the cache matched on a stale pointer.

The runtime **failed safe**: `get_resident` validates the cached K/N and declined
rather than returning wrong data. But every chunked shape was skipped, so a suite
that reported "NPU declined this shape" was recorded as passing coverage. This
failure is present at the base commit `fb4493e` and is not a regression from this
pass.

Fixed by loading each tensor once in `main` and holding it for the whole run,
which reproduces production pointer semantics. All twelve cases now pass
bit-exact, including the paths most likely to be wrong:

```
attn_q / attn_output (1 N-chunk, 1 K-chunk)      T=1024/1536/2048  bit-exact
ffn_gate / ffn_up    (3 N-chunks, 6912->7680)    T=1024/1536/2048  bit-exact
ffn_down             (3 K-chunks, int32 accum)   T=1024/1536/2048  bit-exact
```

Perplexity unchanged and identical across backends: `307.5806 +/- 27.85495` for
both `BITNET_XDNA=0` and `=1`, matching the recorded baseline.

---

## 3. Dependency trace [MEASURED]

`trace.jsonl` -- one full prefill (pp2048, `-ub 1024`, 2 micro-batches, 1150
nodes, 269 KB). Per node: micro-batch, graph index, op, tensor name (carrying the
layer), source tensor names, rebased start/end in us, NPU device time and
dispatch count. Timestamps are rebased to the prefill start; absolute monotonic
values carry no information and would differ every run.

This is sufficient to reconstruct the operation DAG, and it is what
`tools/pipeline_sim.py` consumes.

---

## 4. Dependency-constrained schedule [SIMULATED]

`tools/pipeline_sim.py` builds the real transformer DAG from the measured
per-operation times and list-schedules it over one NPU and one CPU resource.

Model assumptions, stated because they bound the claim:

- **One CPU resource.** Operation times are measured at 15 threads, so a node
  already uses the whole CPU. Treating the CPU as one resource claims no gain
  from running two CPU nodes at once -- conservative, and it keeps the question
  on CPU/NPU overlap, which is what the pipeline proposal is about.
- **One NPU resource** -- the device is single-tenant.
- **Offloading does not free the CPU.** An op on the NPU still costs the CPU its
  measured staging + epilogue (section 2). Modelling NPU placement as free for
  the CPU would overstate the pipeline.
- **The one cross-micro-batch edge is included**: `attn(m,L)` waits on
  `kv_write(m-1,L)`, since causal attention reads every earlier micro-batch's KV.
  That edge is exactly what permits a wavefront, and what limits it.

**Validation.** The simulator's serial schedule reproduces measurement without
being fitted to it: CPU-only 2001 ms simulated vs **2010 ms measured** (0.4%);
all-NPU 2214 ms simulated vs **2193 ms measured** (1.0%).

| config | A serial CPU-only | A serial +NPU | **B perfect [UPPER BOUND]** | **C dep-constrained** | NPU duty (C) | CPU util (C) |
|---|---:|---:|---:|---:|---:|---:|
| pp2048 ub2048 (1 mb) | 2001 | 2214 | **1482** (1.49x) | **1603** (1.38x) | 45.7% | **92.5%** |
| pp2048 ub1024 (2 mb) | 1996 | 2231 | **1488** (1.50x) | **1607** (1.39x) | 46.2% | **92.6%** |
| pp3968 ub2048 (2 mb) | 4692 | 5186 | **3698** (1.40x) | **3995** (1.30x) | 37.3% | **92.6%** |
| pp3968 ub1024 (4 mb) | 4801 | 5166 | **4040** (1.28x) | **4222** (1.22x) | 26.7% | **95.7%** |

Perfect overlap and the dependency-constrained bound are **not** the same number:
dependencies cost 8-12% of the theoretical gain at every context and micro-batch
count.

### The comparison that decides it

The speedups above are against the *all-NPU serial* schedule. That is not what is
deployed. The deployed configuration already overlaps CPU and NPU **within** each
matmul by splitting tokens, and it is measured. Against that:

| config | deployed hybrid [MEASURED] | C dep-constrained [SIMULATED] | **gain** |
|---|---:|---:|---:|
| pp2048 `-ub 2048` | **1685 ms** | 1603 ms | **1.05x** |
| pp3968 `-ub 2048` | **4090 ms** | 3995 ms | **1.02x** |
| pp3968 `-ub 1024` | 5130 ms | 4222 ms | 0.97x vs the ub2048 deployed point |

**Full dependency-aware cross-micro-batch overlap is worth 1.02-1.05x over what
is already running.** At 4K the four-micro-batch schedule (4222 ms) is *worse*
than today's two-micro-batch deployed configuration (4090 ms).

Three measured reasons, all pointing the same way:

1. **The CPU is already saturated: 92.5-95.7% utilised in the overlapped
   schedule.** Overlap helps when a resource is idle. The idle resource here is
   the NPU, and the bottleneck is the CPU, so shifting work onto the NPU is
   limited by what the CPU must still do for it.
2. **The critical path is attention** -- 594 ms of the 2K critical path and
   1378 ms of the 4K one, more than every projection on the path combined. No
   amount of NPU scheduling shortens it, because attention never runs on the NPU.
3. **More micro-batches make it worse, not better.** Going from 2 to 4
   micro-batches at 4K drops NPU duty from 37.3% to 26.7% and pushes CPU
   utilisation to 95.7%: smaller tiles mean proportionally more per-op CPU
   staging for the same arithmetic.
