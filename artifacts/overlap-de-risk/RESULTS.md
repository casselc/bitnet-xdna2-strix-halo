# Overlap de-risk: is cross-microbatch pipelining actually justified?

Purpose of this pass: decide, with measurement and a dependency-aware model,
whether the invasive cross-microbatch pipeline proposed at the end of
`artifacts/next-pass/RESULTS.md` is worth building -- and de-risk the concurrency
mechanism with a smaller experiment first. **The pipeline is deliberately not
implemented here.**

Base commit: `fb4493e9241cf82b3d1a3b03a0780aa0dc333585` (`next-pass-results`).
Branch: `overlap-de-risk`, ending at `5eba847943a05c6a0e55b7cbdf15bf44ed49307e`. `main` was at `885df0ca` and was not touched.

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

### Also withdrawn: "data movement is worth ~2%"

`artifacts/next-pass/RESULTS.md` section 10 dismissed deep-K accumulation,
fusion and packed residency together, on the grounds that they "all target data
movement, which section 3 measured at ~2% end-to-end".

That measurement was narrower than the conclusion drawn from it. Section 3 of
that document measured exactly one effect -- a redundant activation copy caused
by an N-outer loop order -- and found it worth ~2%. It did not measure data
movement in general.

This pass measured a *different* data-movement cost on the same path,
`stage_out`, at **162 ms per prefill, 11.4% of wall** (section 2). So the
generalisation was wrong, and the specific 2% figure remains correct for what it
actually measured. Deep-K accumulation, fusion and packed residency have **not**
been disproven; they are unmeasured.

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

**Definition, used consistently below.** "CPU-side residue" means *work the
CURRENT NPU offload path does not execute* -- not work the NPU is incapable of.
Some of it (attention above all) may be eligible for NPU execution or fusion
later; nothing here shows otherwise, and section 10 treats attention as an open
target rather than a closed one.

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

---

## 5. Asynchronous dispatch spike [MEASURED]

The brief proposes proving the concurrency mechanism on FFN gate/up before
touching llama.cpp's scheduler. The measurements in section 2 point at a better
target for the same mechanism, at a smaller implementation disturbance: the
evacuation of each N-chunk's results runs **immediately after waiting for that
chunk, while the device is idle**. Submitting the next chunk first turns that
serial copy into overlapped work -- the same submit-without-wait / join-at-the-
consumer structure the gate/up spike would have tested, entirely inside
`runtime/`, with no change to ggml, the graph, or the barrier structure.

Implemented as `Program::submit_async` / `wait_pending` with two alternating
output buffers, behind `BITNET_XDNA_ASYNC=1`. At most one dispatch is
outstanding; submitting with one pending throws. Every N-chunk of a K-slice
reads the same activations, so the A buffer is stable across the submits being
pipelined -- a K-slice boundary drains before A is restaged.

**Correctness:** bit-exact with the flag on and off, all twelve shape cases
including 3-way N-chunking, 6912->7680 padding and 3-way K-accumulation.

**Performance**, interleaved round-robin, 5 reps of 3 inner reps, 15 threads:

| config | sync | async | **gain** | sd (sync/async) |
|---|---:|---:|---:|---|
| pp2048 `-ub 2048` | 1221.9 | 1262.5 | **1.033x** | 8.6 / 8.8 |
| pp3968 `-ub 2048` | 973.5 | 1005.5 | **1.033x** | 5.2 / 3.6 |
| pp2048 `-ub 1024` | 923.0 | 1001.2 | **1.085x** | 2.5 / 7.8 |

Real and reproducible -- the 2K gain is roughly 5 standard deviations -- but
small, and it does not change any deployment choice: async at `-ub 1024` (1001)
still loses to synchronous at `-ub 2048` (1222).

**Why only 3.3% when staging is 11% of wall.** The pipeline currently applies
only where `n_chunks > 1`, which is `ffn_gate` and `ffn_up`. `ffn_down` is
`n_chunks=1, k_chunks=3` and `attn_q`/`attn_out` are 1x1, so they still run
synchronously. Of gate/up's staging, chunks 1 and 2 overlap but chunk 0 cannot,
so roughly 2/3 of 64% of staging is hidden -- about 70 ms of 1685 ms, or 4.2%
predicted against 3.3% measured.

Extending the pipeline across K-chunks would cover `ffn_down`, but K-chunks each
need different activations, so it additionally requires double-buffering the A
buffer -- the device is still reading A when the next chunk would be staged. That
is a larger change and is not attempted here.

**Verdict: the asynchronous execution mechanism is de-risked and works.** It is
bit-exact, it is cheap, and it confirms the simulation's prediction that overlap
of this kind is worth only a few percent. It is left **off by default** in this
pass so the branch stays directly comparable to the baseline; it is ready to be
promoted.

---

## 6. Decision on cross-micro-batch pipelining

### **PIPELINE NOT JUSTIFIED**

Not blocked -- the dependency structure genuinely permits a wavefront, and
section 5 shows the asynchronous mechanism works and is bit-exact. It is not
justified because the measured and simulated gain does not repay an invasive
change to llama.cpp's micro-batch scheduler.

The evidence, in the order it decides the question:

1. **Dependency-aware simulation predicts 1.02-1.05x over the deployed
   configuration** [SIMULATED, validated against measurement to within 1%].
   Against an all-NPU serial schedule it looks like 1.30-1.39x, but that is not
   what is deployed; the runtime already overlaps CPU and NPU inside each matmul
   by splitting tokens.

2. **The CPU is the bottleneck, not the idle NPU.** In the overlapped schedule
   CPU utilisation is 92.5-95.7%. Overlap pays when a resource idles; here the
   idle resource is the NPU, and every operation moved to it still costs the CPU
   its staging and epilogue.

3. **Attention is the critical path and it grows quadratically** -- 594 ms of
   the 2K critical path, 1378 ms of the 4K one. It is 35.7% of prefill at 2K and
   49.0% at 4K. No NPU scheduling shortens it.

4. **More micro-batches make it worse.** 2 -> 4 micro-batches at 4K drops
   simulated NPU duty from 37.3% to 26.7% and raises CPU utilisation to 95.7%.
   The 4-micro-batch schedule (4222 ms) loses to today's deployed 2-micro-batch
   configuration (4090 ms).

5. **The entry price is real.** At 2K, pipelining needs `-ub 1024`, which costs
   24% before any pipeline gain (922.9 vs 1215.7 tok/s).

6. **The mechanism was tried and is worth a few percent** [MEASURED]. The async
   spike -- the same submit-without-wait structure a pipeline would use, at a
   smaller granularity -- returns 1.033-1.085x. That is the empirical check on
   the simulation, and it agrees with it.

**The perfect-overlap upper bound of 1.32x (2K) / 1.25x (4K) is a ceiling that
dependencies, CPU saturation and the `-ub` entry price erode to a few percent.
It should not be quoted as an achievable figure, and the 1.39x from the previous
pass is withdrawn** (it additionally used the NPU time corrected in section 0).

---

## 7. Cost-model holdout [MEASURED]

`R = 10` in `f = R/(R + n_threads - 1)` was fitted against a sweep at threads
4/8/15 and prompts 2048/3968. Tested here on thread counts and micro-batch sizes
**not used to establish it**: threads 3/6/10/12 x prompts 1024/1536/3072/3968,
auto against every fixed tile allocation, 2 interleaved reps. The chosen tile
count is read back from the dispatch counter, not assumed.

| threads | prompt | auto pick | auto tok/s | best pick | best tok/s | regret |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1024 | 1 | 606.2 | 1 | 607.7 | 1.003x |
| 3 | 1536 | 1 | 420.6 | 1 | 420.5 | 1.000x |
| 3 | 3072 | 3 | 381.1 | 3 | 382.5 | 1.004x |
| 3 | 3968 | 3 | 289.3 | 3 | 328.5 | *1.135x* |
| 6 | 1024 | 1 | 826.7 | 1 | 827.1 | 1.001x |
| 6 | 1536 | 1 | 850.3 | 1 | 844.1 | 0.993x |
| 6 | 3072 | 2 | 568.3 | 2 | 579.2 | 1.019x |
| 6 | 3968 | 2 | 512.8 | 3 | 517.6 | 1.009x |
| 10 | 1024 | 1 | 943.3 | 1 | 925.6 | 0.981x |
| 10 | 1536 | 1 | 1056.6 | 1 | 1058.6 | 1.002x |
| 10 | 3072 | 2 | 802.7 | 1 | 795.6 | 0.991x |
| 10 | 3968 | 2 | 771.9 | 1 | 765.7 | 0.992x |
| 12 | 1024 | 1 | 963.8 | **0** | 1011.3 | **1.049x** |
| 12 | 1536 | 1 | 1115.9 | 1 | 1138.6 | 1.020x |
| 12 | 3072 | 2 | 882.8 | 1 | 888.9 | 1.007x |
| 12 | 3968 | 2 | 866.2 | 1 | 858.6 | 0.991x |

**Exact match 10/16 (62%), mean regret 1.012x, worst 1.135x.**

Both extremes need reading carefully:

- **The worst regret is not a model error.** At threads=3 / pp3968 auto and the
  exhaustive best chose the *same* tile count (3). The 13.5% gap is run-to-run
  variance at the slowest configuration in the sweep, where each measurement is a
  ~14 s prefill and only 2 reps were taken. It bounds the noise floor of this
  holdout, not the model.
- **Six of the sixteen "mismatches" are noise in the other direction**: three
  have regret **below 1.0**, meaning auto beat the allocation the sweep called
  best, which is only possible if the "best" was noise. Two more are within 0.9%.
- **One genuine error**: threads=12 / pp1024, where the model takes the single
  available tile and CPU-only would have been 4.9% faster. With a 1024-token
  micro-batch there is exactly one tile, so the split is all-or-nothing; that is
  the coarsest case the quantisation produces, and it is where the model should
  be expected to miss.

**Verdict: R=10 performs adequately -- 1.2% mean regret across held-out
configurations. No added model complexity is warranted**, per the brief. The one
worthwhile refinement, if it is ever wanted, is a floor that declines the offload
when only a single tile is available and thread count is high; it is worth ~5% in
one corner and nothing elsewhere.

---

## 8. Real GPU co-tenant [DEFERRED]

The brief permits a bounded GPU co-tenant experiment **only if a stable local
ROCm model/runtime already exists**, and says to defer explicitly rather than
improvise. Surveyed on this machine:

| | |
|---|---|
| ROCm (`rocminfo`, `rocm-smi`, `/opt/rocm*`) | **not installed** |
| `vulkaninfo` / Vulkan loader | **not present** |
| PyTorch (any backend) | **not importable** |
| `ollama` | **not installed** |
| `/usr/bin/lemonade` | present, but its cache is **6 KB (a `config.json`, no model weights)** |
| HuggingFace cache | **53 KB**, holding only the BitNet GGUF metadata |
| Any GGUF > 100 MB on disk | only `models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf` |
| iGPU busy | 0% |

There is no GPU inference stack with a model on this machine. Standing one up
means installing a runtime *and* downloading a model, which the brief rules out,
and the only local model is BitNet I2_S, which no GPU backend supports anyway.

**DEFERRED.** The synthetic CPU co-tenant result from the previous pass
therefore still stands unchallenged, and the open question is unchanged: whether
unified-memory bandwidth contention between a GPU worker and the NPU controller
changes the deployment recommendation. Nothing measured here bears on it.

To run it later, the cheapest honest setup is a Vulkan-backend llama.cpp build
plus one small dense GGUF for the GPU worker, with the controller unchanged.

---

## 9. Recommended next major engineering change

### **FUSE THE NPU EPILOGUE — remove the staging round-trip**

One recommendation, chosen because it is the largest *measured* inefficiency
still on the table, the cheapest to implement, and the only candidate that
carries no correctness risk by construction.

**What is wrong now** [MEASURED, section 2]. Every N-chunk dispatch writes into
one reused mapped output buffer, so thread 0 must evacuate each chunk's int32
results into `g_acc` before the next dispatch overwrites them. That copy is
**162 ms per prefill at the deployed configuration (309 ms at all-NPU), 2.27 GB,
single-threaded at 14 GB/s, while the other 14 threads are parked at the
`ggml_barrier`.** The epilogue then reads `g_acc` again to scale it into `dst`.
With `stage_in` and the epilogue that is ~194 ms, **11.4% of a 1697 ms prefill**,
spent moving bytes that are about to be read again.

**The change.** Give each dispatch its own region of one output arena
(`xrt::bo` sub-buffers over a `M_tile x 7680 x 4` = 31.5 MB parent, the pattern
already used for weights), so nothing is overwritten before it is consumed. The
multi-threaded epilogue then reads the mapped device buffer directly and writes
`dst`, and `g_acc` disappears. Traffic per element drops from
`write g_acc + read g_acc + write dst` to `read mapped + write dst`, and the work
moves off thread 0 onto all threads.

**Predicted gain: ~1.11x at 2K** (162 ms of 1685 ms), of which the async spike
already demonstrates 1.033x. This is a *prediction from measurement*, not a
measured result, and must be verified before it is quoted.

**Why this and not the alternatives:**

| candidate | why not first |
|---|---|
| Cross-micro-batch pipeline | 1.02-1.05x simulated, invasive. Section 6. |
| Kernel tuned toward 25 TOPS | The NPU is idle 75.5% of prefill; making it faster while idle changes little, and the deployed split already balances the engines. |
| int4 / 2-bit weights over DMA | Measured 1.035x compute last pass; a bandwidth optimisation whose payoff is in NPU **decode**, which is not a goal. |
| Deep-K accumulation, block fusion, packed residency | **Unmeasured, not disproven.** The previous pass's "~2% data movement" figure covered one specific activation-copy effect only; this pass found a different data-movement cost at 11.4%, so this family deserves measurement rather than dismissal. |

**It also unblocks the thread-starved regime, which is the deployment target.**
Staging is why the all-NPU assignment (933 tok/s) loses to CPU-only (1015) at 15
threads: 309 ms of single-threaded copying is added on top of the device time.
Removing it makes larger NPU shares viable, and larger NPU shares are worth most
at 2-4 threads, where the previous pass measured the hybrid at 1.55-1.91x and
0.58x the energy per token.

### The larger target after that, stated honestly

**Attention is 35.7% of prefill at 2K and 49.0% at 4K, growing as O(T^2), and is
the critical path in every simulated schedule.** It is the only remaining item
with a ceiling above ~1.15x: removing it entirely would be 1.56x at 2K and 1.96x
at 4K.

It is explicitly **not** recommended as the next change, because nothing measured
here says XDNA2 would run it faster. Prefill attention is bandwidth-bound over
the KV cache rather than MAC-bound, and this pass found the NPU is not the faster
engine on the arithmetic it already does well. The honest next step for attention
is a bounded feasibility measurement -- what a flash-attention-shaped kernel
achieves on aie2p for these shapes -- not an implementation commitment.

---

## 10. Summary

| # | question | answer |
|---|---|---|
| 1 | base / branch | base `fb4493e9241cf82b3d1a3b03a0780aa0dc333585` (`next-pass-results`); branch `overlap-de-risk`. `main` untouched at `885df0ca`. |
| 2 | public-environment cleanup | **Completed** and pushed as an independent first checkpoint. Hostname and ZFS/cmdline identifiers removed from HEAD; `tools/capture_environment.sh` sanitizes by construction; `tools/scan_artifacts.sh` verified against a synthetic file containing all six leak classes. History not rewritten. |
| 3 | CPU-residue decomposition | **[MEASURED]** 2K: attention 599 ms (35.7%), FFN projections 650 ms, attention projections 159 ms, norm 90, relu^2 50, k/v 84, rope 17, residual 21, lm_head 6. 4K: attention **1988 ms (49.0%)**, everything else roughly 2x its 2K value. Node time accounts for 99.6-100.0% of graph span. |
| 4 | current NPU duty cycle | **[MEASURED]** **24.5%** at the deployed point (15 threads, `-ub 2048`); 33.6% if given every token tile. Corrects 32.0%/44.6% from the previous pass (section 0). |
| 5 | perfect-overlap limit | **[UPPER BOUND]** **1.32x** at 2K, **1.25x** at 4K -- and *falling* with context, because the residue grows O(T^2) while NPU work grows O(T). |
| 6 | dependency-constrained overlap | **[SIMULATED]** 1.38x (2K) / 1.30x (4K) against an all-NPU serial schedule, but **1.05x (2K) / 1.02x (4K) against the deployed configuration**. Simulator validated to within 1% of measurement. CPU utilisation 92.5-95.7%. |
| 7 | async sibling spike | **[MEASURED]** Mechanism works, bit-exact: **1.033x** (pp2048 ub2048), **1.033x** (pp3968 ub2048), **1.085x** (pp2048 ub1024), 5 interleaved reps. Off by default, ready to promote. |
| 8 | cost-model holdout | **[MEASURED]** 10/16 exact, **mean regret 1.012x**. One genuine 4.9% miss (12 threads / 1024 tokens, single-tile all-or-nothing). R=10 adequate; no added complexity warranted. |
| 9 | GPU co-tenant | **[DEFERRED]** -- no ROCm, no Vulkan, no torch, no GPU-runnable model on this machine. Survey in section 8. |
| 10 | recommendation | **FUSE THE NPU EPILOGUE** -- remove the staging round-trip. Section 9. |

### Three claims withdrawn this pass

1. **NPU duty cycle / busy time** -- warmup prefill was double-counted. Idle is
   75.5%, not 68%. Independently corroborated by in-model throughput finally
   matching the standalone kernel.
2. **"The CPU is the faster GEMM engine by ~1.3x"** -- the three-equation solve
   assumed an identical residue between CPU-only and hybrid, which the corrected
   numbers make impossible. The supportable statement is that the engines are of
   broadly comparable throughput.
3. **"Data movement is worth ~2%"** -- that figure covered one specific
   activation-copy effect. A different data-movement cost on the same path
   measures 11.4%. Deep-K accumulation, fusion and packed residency are
   **unmeasured, not disproven.**

### Reproduction

```bash
# environment + public-artifact hygiene
bash tools/capture_environment.sh && bash tools/scan_artifacts.sh

# build (llamafile OFF; XDNA runtime + node profiler compiled in)
cmake --build refs/BitNet/build-xdna3 -j24 && make          # tests
BITNET_XDNA_ARTIFACTS=$PWD/artifacts/xclbin-tuned make check

export BITNET_XDNA_ARTIFACTS=$PWD/artifacts/xclbin-tuned
python3 tools/baseline_check.py                             # Task 1
python3 tools/duty_cycle2.py                                # Task 0 correction
python3 tools/residue_breakdown.py --prompt 2048 --ub 2048 1024 \
        --modes hybrid cpu --out artifacts/overlap-de-risk/residue_2k.csv
python3 tools/residue_breakdown.py --prompt 3968 --ub 2048 1024 \
        --modes hybrid cpu --out artifacts/overlap-de-risk/residue_4k.csv
python3 tools/pipeline_sim.py \
        --residue artifacts/overlap-de-risk/residue_{2k,4k}.csv \
        --allnpu  artifacts/overlap-de-risk/residue_{2k,4k}_allnpu.csv
python3 tools/async_ab.py                                   # Task 5
python3 tools/cost_model_holdout.py                         # Task 7
```

Measurement discipline throughout: interleaved round-robin (never blocked),
>= 3 reps exploratory / >= 5 for sub-10% claims, medians with dispersion,
background load and NPU device time recorded per run, and process reaping by
explicit PID only -- `pkill -f <pattern>` matches this harness's own command line
and has killed it repeatedly in earlier passes.
