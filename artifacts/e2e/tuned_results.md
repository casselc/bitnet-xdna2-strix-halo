# Round 2: kernel tuning + eliminating context switches

Three changes, each validated by measurement, with correctness re-verified after
every one (perplexity stayed at `307.5806 +/- 27.85495`, identical to CPU-only).

## What changed

| # | change | effect |
|---|---|---|
| 1 | `--b-col-maj 1` | **+35%** on the 2560x2560 kernel. Also removes the transpose from the weight repack, since column-major B *is* the GGUF's native `[N,K]` layout. Repack 2.7 s -> 1.0 s. |
| 2 | C ObjectFifo depth 2 -> 1 | C is written once per tile and drained, so it does not need double buffering. Frees L1 (`2mk + 2kn + mn*4 <= ~62 KB`) and legalizes `m=128`, which was previously "failing to build". |
| 3 | `tb_max_n_rows` 4 -> 2 | Collapses the C shim DMA's outer repeat dimension, whose stride is `m * n_aie_rows * N`. **This was the real `aie.dma_bd` stride wall** — it blocked both `N=6912` and `m=128`, and it is one parameter, not a hardware limit on N. |

Standalone kernel throughput, M=1024:

| shape | before | after |
|---|---|---|
| 2560 x 2560 | 9.08 TOPS | **13.17 TOPS** |
| 2560 x 6912 | 5.76 TOPS (as 2 x 3456) | **11.62 TOPS** (single dispatch) |
| 6912 x 2560 | 9.35 TOPS | 7.88 TOPS (regressed; deep K prefers `k=128`) |

## Then the real bottleneck appeared

Tuned kernels made the *end-to-end* result **worse** (pp2048 571 -> 509 t/s) even
though every kernel got faster. Per-shape instrumentation showed in-model
throughput at ~40% of standalone, and a four-way split of the dispatch timer
localized it: **87-95% of dispatch time is inside `run.wait()`** — the device is
genuinely busy — while cache flushes are 4% and submission 1%.

Cycling three `xrt::hw_context`s (one per compiled shape) in BitNet's per-layer
order, measured interleaved against the same kernels in isolation:

| shape | alone | cycled | penalty |
|---|---|---|---|
| 2560 x 2560 | 1.159 ms | 3.592 ms | **+210%** |
| 2560 x 6912 | 3.556 ms | 5.926 ms | +67% |
| 6912 x 2560 | 4.756 ms | 7.253 ms | +53% |

Each context claims all 8 columns, so they cannot be co-resident and the array is
reprogrammed on every switch. **This is 10-20x worse than the 0.22 ms measured
with the smaller M=512 designs** — the penalty scales with design size.

Fix: decompose every shape onto **one** program (2560x2560), N-chunking the wide
FFN and K-chunking the deep down-projection with int32 accumulation. 6912 pads to
7680 (11% waste) to make the chunking even. More dispatches, zero switches.

Result: **mean dispatch 4.334 ms -> 1.211 ms**, essentially the isolated figure.

## End-to-end

| prompt | CPU-only | hybrid v1 | hybrid v2 | v2 gain | vs CPU |
|---:|---:|---:|---:|---:|---:|
| 128 | 864.0 | 213.1 | 269.7 | 1.27x | 0.31x |
| 512 | 1255.1 | 642.4 | 710.0 | 1.11x | 0.57x |
| 2048 | 1019.2 | 571.0 | 628.4 | 1.10x | 0.62x |
| 3968 | 799.1 | 484.3 | 539.5 | 1.11x | 0.68x |
| tg32 (decode) | 79.8 | 80.2 | 80.4 | — | 1.01x |

**Still slower than CPU-only.** But the reason has changed completely.

## Where the time goes now

For one pp2048 prefill:

```
NPU device time  :  777 ms  (24%)   642 dispatches x 1.211 ms
serial CPU work  : 2482 ms  (76%)   <- the NPU is idle for all of it
total            : 3259 ms
CPU-only, everything : 2009 ms
```

The NPU now does its share of the arithmetic in **777 ms against the CPU's 2009 ms
for the whole prefill** — it is genuinely the faster engine for that work. The
loss is entirely structural: offload is *exclusive*, so 16 CPU cores idle on a
barrier for 76% of the wall clock, and the hybrid additionally pays for staging
int32 accumulators and the f32 epilogue that CPU-only fuses into its kernel.

**The next lever is concurrency, not the kernel.** Splitting each GEMM's output
rows between NPU and CPU so both work simultaneously is the obvious move, and on
these numbers it is worth substantially more than any remaining kernel tuning.
