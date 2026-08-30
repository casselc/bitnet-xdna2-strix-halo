# Decision gate: **MVP PASS** — with a negative performance result

> **STATUS: this document records the ORIGINAL gate (round 1) and its numbers are
> superseded.** Two later rounds changed the result. Current numbers live in
> [`e2e/tuned_results.md`](e2e/tuned_results.md) (kernel tuning + single hardware
> context) and [`e2e/concurrent_results.md`](e2e/concurrent_results.md)
> (concurrent CPU+NPU execution, which turned the loss into a **1.12x win at 2048
> tokens**). The performance tables and cost decomposition below describe the
> round-1 runtime and no longer hold. Corrections applied in place are marked.

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
| xclbin/context switch | 2.67 ms | 0.101–0.22 ms — **RETRACTED, see below** |

**Retraction on the switch-cost row.** That 0.101–0.22 ms was measured on two
trivial elementwise `add_one`/`add_two` kernels at N=4096 via Python `@iron.jit` —
nothing like an 8-column GEMM design. Re-measured with the actual designs, a
hardware-context switch costs **+1.2 to +2.4 ms per dispatch** (+53% to +210%),
i.e. comparable to or worse than the published figure it claimed to beat. It went
on to be the single largest cost in the runtime. See
[`kernels/context_switching.md`](kernels/context_switching.md).

Dispatch *submission* overhead is genuinely small (measured at 1-2% of dispatch
time). Context switching is not, and the original framing conflated them.

## The narrowest next experiment

Not "optimize everything". One question decides whether this topology is viable:

> **Can a purpose-built AIE kernel reach ≥25 TOPS on `K=2560,N=2560` int8 at
> M=512 on this device?**

That is the threshold where NPU linear time (~1.5x faster than CPU on the same
arithmetic) overtakes the CPU's whole prefill with overhead included. It is a
kernel-engineering question in a single file, answerable without touching the
integration — which is finished, correct, and instrumented.

Two cheaper follow-ups worth folding in:
1. **int4 weights.** This entry has been corrected twice; here is the settled
   evidence. On **aie2p (`__AIE_ARCH__ == 21`, our part)**,
   `aie::mmul<4,16,16,int8,int4,32>` compiles to a **single native intrinsic**,
   `mac_4x16_16x16_conf` — 4x16x16 = **1024 MACs per instruction**, against
   `mac_8x8_8x8`'s 512 for int8xint8. The B operand is also twice as wide in bits
   (256 x int4 = 1024b vs 64 x int8 = 512b), which is what a genuinely wider
   datapath looks like.

   The "emulated on top of int8 x int8" description an intermediate review cited
   is real, but it describes **`__AIE_ARCH__ == 22` (aie2ps)** — a different
   generation — whose branch in the same header does
   `interleave_unzip` + **two** `mac_4x8_8x16_conf` calls with `unpack_sign`.
   Reading that branch and attributing it to this part produced the wrong
   correction; the source is
   `.venv/.../mlir_aie/include/aie_api/detail/aie2p/mmul_8_4.hpp:23` (arch 21)
   vs `:46` (arch 22).

   **Measured on hardware, and the compute half is false.** A custom kernel
   (`npu/kernels/mm_i8_i4.cc`) runs int8 x int4 bit-exactly and halves the weight
   bytes on the wire, but a two-point solve against a matched int8 kernel gives a
   compute-only ratio of **1.035x**: the 1024-MAC instruction issues at half the
   rate of the 512-MAC one, so MAC throughput is unchanged. Details in
   `kernels/int4_result.md`.

   int4 is therefore a **bandwidth** optimization, not a compute one — but an
   important one, because halving weight traffic is what makes NPU decode
   arithmetically possible at all (28.5 -> 113.8 tok/s ceiling against the CPU's
   80). Budget it accordingly.
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
