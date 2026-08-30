# Milestone A — XDNA2 int8 GEMM feasibility at real BitNet shapes

All numbers measured on this machine (AMD RYZEN AI MAX+ 395, RyzenAI-npu5 / aie2p,
8 columns / 32 compute cores, XRT 2.25.0, mlir-aie v1.4.2, Peano 21.0.0.2026080301).
Kernel is the stock `programming_examples/basic/matrix_multiplication/whole_array`
design at `Xilinx/mlir-aie@v1.4.2`, `dtype_in=i8 dtype_out=i32`, **untuned beyond a
tile-size sweep**.

## Dispatch latency (the number the design hinged on)

| metric | value |
|---|---|
| device open (one-time) | 12.3 ms |
| **warm host->NPU->host round trip, p50** | **0.197-0.200 ms** |
| implied cost of 210 dispatches/prefill | **~42 ms** |

Published XDNA2 figures (Krackan) put this at 0.66 ms round trip / 2.67 ms context
switch. Strix Halo measures **3.3x better** than that. Per-matmul offload is
therefore viable; whole-layer fusion is not required for the MVP.

## Buffer costs (npu_probe, no kernel involved)

| operation | 6.25 MiB | 16.88 MiB | throughput |
|---|---|---|---|
| BO alloc + map | 0.657 ms | 1.785 ms | -- |
| sync TO_DEVICE | 0.111 ms | 0.283 ms | ~59-62 GB/s |
| sync FROM_DEVICE | 0.053 ms | 0.144 ms | ~122 GB/s |

**BO allocation is ~0.1 ms/MiB and dominates a naive per-call design.** This is the
hard evidence for allocating every weight BO once at model load. With weights
resident, the per-dispatch transfer cost is activation-in + result-out only
(0.032 + 0.043 ms at 512 tokens).

## Per-shape kernel throughput

M = prefill token count. Tiles are `-m/-k/-n`; `cols` is `--n-aie-cols`.

| tensor | K x N | tiling | M=512 | M=2048 | TOPS @2048 |
|---|---|---|---|---|---|
| `attn_q`, `attn_output` | 2560 x 2560 | 64/64/64, 8 | 803 us | 2957 us | **9.08** |
| `attn_k`, `attn_v` | 2560 x 640 | 64/64/16, 8 | 654 us | 2363 us | **2.84** |
| `ffn_gate`, `ffn_up` (half) | 2560 x 3456 | 64/64/48, 8 | 1665 us | 6294 us | **5.76** |
| `ffn_down` | 6912 x 2560 | 64/64/64, 8 | 1951 us | 7751 us | **9.35** |

Throughput saturates with batch size: 2560x2560 gives 8.17 / 9.00 / 9.28 / 9.28
TOPS at M = 512 / 1024 / 2048 / 4096. **~9.3 TOPS is the stock kernel's ceiling.**

## Two hard constraints hit, both predicted

1. **`N <= 4096` DMA stride wall.** `ffn_gate`/`ffn_up` have N=6912 and fail to
   compile at *any* column count:
   ```
   error: 'aie.dma_bd' op Stride 3 exceeds the [1:1048576] range.
     aie.dma_bd(... sizes = [2, 27, 256, 32] strides = [1769472, 256, 6912, 1])
   ```
   The offending stride is a function of total N, not of the tiling, so
   `--n-aie-cols 4/2`, `--b-col-maj` and `--c-col-maj` all fail identically.
   Workaround: split N into 2 x 3456 (costs one extra dispatch each, and drops
   throughput from ~9.3 to 5.76 TOPS because n=48 is forced instead of n=64).
2. **L1 is 64 KB/core.** `n=128` and `m=128` fail to build at every K/N tried.
3. Tiling must satisfy `N % (n * n_aie_cols) == 0`, which is what forces n=48
   for N=3456 and n=16 for N=640.

## Aggregate: the actual economics

Summing all 7 linears x 30 layers against the measured CPU oracle:

| prompt | NPU linear kernel time | CPU-only *entire* prefill | ratio |
|---|---|---|---|
| 512 | 345.8 ms | 400.8 ms | 0.86x |
| 2048 | 1307.0 ms | 1988.9 ms | **0.66x** |
| 2048, k/v left on CPU | 1165.2 ms | 1988.9 ms | **0.59x** |

Effective throughput on the same arithmetic (2048 tokens = 8.54 TFLOP):
- **CPU (16 threads): 4.29 TFLOPS**
- **NPU (stock kernel, incl. the weak shapes): 6.53 TFLOPS**

`attn_k`/`attn_v` are 6% of the FLOPs but 21% of NPU time (N=640 is too narrow to
fill 8 columns), so they are better left on the CPU -- a per-shape offload
decision made by measurement, as planned.

## Honest assessment

The stock kernel reaches ~9.3 TOPS against a ~50 TOPS device peak (~19%).
Published hand-tuned XDNA2 int8 GEMM reaches 38-56 TOPS, so there is roughly 4-6x
of kernel headroom left on the table. Even so, at controller-scale prompts the
untuned NPU already does the linear algebra in 59-66% of the CPU's total prefill
budget, which is enough for a real end-to-end win. At 512 tokens the margin is
thin (0.86x) and at 128 tokens dispatch overhead would dominate -- the crossover
this milestone set out to find is real and sits in the hundreds-of-tokens range.
