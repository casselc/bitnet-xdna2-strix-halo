# Decision gate: **MVP PASS** — with a negative performance result

## The gate, item by item

The milestone defines PASS as: *a real BitNet checkpoint performs NPU-assisted
prefill on XDNA2 under Linux, then continues generation using CPU BitNet decode
without recomputing the prompt on CPU; correctness is demonstrated against
CPU-only reference execution; end-to-end timing is recorded.*

| requirement | status | evidence |
|---|---|---|
| Real BitNet checkpoint | ✅ | `BitNet-b1.58-2B-4T`, official GGUF, sha256 `4221b252…`, unmodified |
| NPU-assisted prefill on XDNA2 under Linux | ✅ | 410 dispatches per 2 prefills; `RyzenAI-npu5 / aie2p / 6x8` |
| CPU decode continues | ✅ | `-p 0 -n 128` → **`dispatches=0`**; tg32 unchanged at 1.01x |
| Prompt not recomputed on CPU | ✅ | the hook `return`s from `mul_mat` on success; the CPU SIMD kernel does not run for offloaded tensors |
| Correctness vs CPU-only | ✅ | kernel **bit-exact** (1,310,720/1,310,720 int32 accumulators); **perplexity identical to 4 dp** over 2048 tokens while 830 matmuls ran on NPU |
| End-to-end timing recorded | ✅ | `e2e/results.md`, four prompt lengths + decode |
| CPU-only remains runnable | ✅ | `BITNET_XDNA=0`, same binary and weights |

The gate is met. **"GEMM ran" is explicitly not a pass, and this is not that** —
a real checkpoint runs real prefill on the NPU and real decode on the CPU, and
the model's numerical output is provably unchanged.

## The honest result: the NPU is currently slower

| prompt | CPU-only | hybrid | speedup |
|---:|---:|---:|---:|
| 128 | 148 ms | 601 ms | **0.25x** |
| 512 | 408 ms | 797 ms | **0.51x** |
| 2048 | 2009 ms | 3587 ms | **0.56x** |
| 3968 | 4966 ms | 8193 ms | **0.61x** |

**NPU-assisted prefill is 1.6–4x slower than 16 Zen 5 cores.** The deficit
narrows as prompts grow (dispatch overhead amortizes) but does not cross over
within the model's 4096-token context. For the intended resident-controller role
— 1K–4K tokens in, 10–100 tokens out — the NPU as configured here **costs**
time-to-first-token rather than saving it.

## Why, quantified

| | |
|---|---|
| stock mlir-aie `whole_array` int8 kernel | **9.3 TOPS** of a ~50 TOPS device peak (~19%) |
| published hand-tuned XDNA2 int8 GEMM | 38–56 TOPS → **4–6x headroom untouched** |
| CPU effective throughput on the same arithmetic | 4.29 TFLOPS |
| kernel time alone, weighted over BitNet's 7 linears | 1.52 ms |
| + cycling 3 xclbins per layer | +0.18–0.22 ms |
| + ~1.8 GiB resident weight buffers | +0.21 ms |
| **measured inside llama.cpp** | **2.66 ms** |

At 205 dispatches per 512-token prefill, 2.66 ms each is 545 ms of device time —
more than the CPU's entire 408 ms prefill, before any CPU-side work is counted.

Two structural taxes on top: `ffn_gate`/`ffn_up` (N=6912) exceed the
`aie.dma_bd` stride limit and must be split into 2×3456, which forces tile n=48
instead of 64 and drops those shapes from ~9.3 to 5.8 TOPS; and ternary→int8
expansion inflates weights 4x (461 MiB → 1843 MiB) with a one-time 2.7–3.5 s
repack.

## What was surprising, and favourable

Three published XDNA2 figures — all measured on Strix Point/Krackan, not Strix
Halo — turned out to be substantially pessimistic here:

| | published | measured on this machine |
|---|---|---|
| dispatch round trip | 0.66 ms | **0.197 ms** (3.3x better) |
| xclbin/context switch | 2.67 ms | **0.101–0.22 ms** (12–26x better) |

Dispatch overhead is therefore *not* the blocker, which is what the design was
built to guard against. The blocker is kernel throughput.

## The narrowest next experiment

Not "optimize everything". One question decides whether this topology is viable:

> **Can a purpose-built AIE kernel reach ≥25 TOPS on `K=2560,N=2560` int8 at
> M=512 on this device?**

That is the threshold where NPU linear time (~1.5x faster than CPU on the same
arithmetic) overtakes the CPU's whole prefill with overhead included. It is a
kernel-engineering question in a single file, answerable without touching the
integration — which is finished, correct, and instrumented.

Two cheaper follow-ups worth folding in:
1. **int4 weights** using the native `8b x 4b` `4x16x16` mmul: 1024 MAC/cycle
   (2x) and half the weight DMA. The ternary values fit int4 exactly, so this
   costs no accuracy.
2. **Concurrent split** rather than exclusive offload: give the NPU a fraction of
   the output rows and the CPU the rest simultaneously. Today thread 0 dispatches
   while 15 threads idle on a barrier; measured NPU/CPU concurrency on Strix Halo
   is ~1.7–1.9x, and that headroom is entirely unused here.

## Bottom line for the topology decision

The Linux XDNA2 stack is **ready**: driver, firmware, XRT, mlir-aie and Peano all
work on stock Ubuntu 26.04, and a ternary BitNet matmul runs bit-exactly on the
NPU — as far as I can find, the first public instance of BitNet on XDNA2. The
mechanism is not the risk.

The economics do not yet work. On this machine, 16 Zen 5 cores are a *very*
strong baseline for BitNet prefill (4.29 TFLOPS effective), and the stock NPU
kernel at 19% of device peak does not beat them. Whether the NPU earns its place
as the resident controller depends entirely on closing the 4–6x kernel gap, which
published results say is achievable but which nobody has yet done for ternary
weights on this silicon.
