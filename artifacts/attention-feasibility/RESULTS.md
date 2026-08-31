# Attention on XDNA2: bounded feasibility study

Answering one question before any integration work:

> For BitNet-2B's **actual** causal-prefill attention shapes on this Strix Halo,
> can XDNA2 run a flash-attention-like kernel fast enough, **after data movement
> is included**, to beat the Zen 5 implementation?

Branch base: `ed97cfcac564be9f85db415faf076695b871e008` (`direct-output-closeout`,
frozen). No llama.cpp integration in this phase.

| tag | meaning |
|---|---|
| **[MEASURED]** | observed on this machine |
| **[DERIVED]** | arithmetic over measured quantities |
| **[DEFERRED]** | not done, with the reason |

---

## A. Real model geometry [MEASURED]

Read from the GGUF's own metadata and from the live runtime, not from
model-family defaults.

| property | value | source |
|---|---|---|
| architecture | `bitnet-b1.58` | GGUF `general.architecture` |
| hidden dimension | 2560 | GGUF `embedding_length` |
| Q heads | **20** | GGUF `attention.head_count` |
| KV heads | **5** | GGUF `attention.head_count_kv` |
| GQA ratio | **4** | derived |
| head dimension | **128** | derived, 2560 / 20 |
| layers | 30 | GGUF `block_count` |
| context length | 4096 | GGUF `context_length` |
| RoPE dim / freq base | 128 / 500000.0 | GGUF `rope.*` |
| RMS norm eps | 1.0e-5 | GGUF |

Runtime facts, from the recorded execution trace and the pinned llama.cpp:

| property | value |
|---|---|
| op | `FLASH_ATTN_EXT`, one node per layer |
| output shape | `ne = [128, 20, T]` = `[head_dim, n_q_heads, tokens]` |
| Q at attention entry | **f32** (`src0_type = 0`) |
| K/V cache precision | **f16** (llama.cpp default `cache_type_k/v`) |
| output precision | f32 |
| sources | `Qcur (view, permuted)`, `cache_k_l<N> (view, permuted)` |
| CPU implementation | `ggml_compute_forward_flash_attn_ext_f16`, dispatching to `_one_chunk` or a split-KV path |

---

## C. CPU oracle [MEASURED]

The baseline any NPU kernel must beat. Measured in situ with the per-node
profiler, 15 threads, medians with the warmup prefill dropped.
Raw: `cpu_oracle.csv`.

| T | prefill | **attention** | share | attn tok/s | logical KV | logical rate | **physical KV** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 414.0 ms | **50.9 ms** | 12.3% | 10059 | 9.38 GiB | 184 GiB/s | **37.5 MiB** |
| 1024 | 876.2 ms | **165.2 ms** | 18.9% | 6198 | 37.50 GiB | 227 GiB/s | **75.0 MiB** |
| 2048 | 2020.9 ms | **602.0 ms** | 29.8% | 3402 | 150.00 GiB | 249 GiB/s | **150.0 MiB** |
| 3968 | 2722.8 ms | **1388.0 ms** | 51.0% | 2859 | 563.09 GiB | 406 GiB/s | **290.6 MiB** |

### The two KV figures are not interchangeable, and the difference decides the experiment

The **logical** rate reaches 406 GiB/s at T=3968, which **exceeds this machine's
DRAM bandwidth**. That is not a measurement error: one layer's KV at T=3968 is
only 10.2 MiB and fits comfortably in the 64 MiB L3, so the CPU is largely
re-reading cache rather than memory.

The **physical** (compulsory) traffic a tiled kernel must move is each layer's KV
once: **37.5 to 290.6 MiB per prefill**. At the NPU's measured weight-stream rate
of 51.2 GiB/s that is **0.7 to 5.5 ms** -- negligible against the CPU's 51 to
1388 ms.

**So this workload is compute-bound, not KV-bandwidth-bound, at these shapes.**
That removes the objection that killed several earlier NPU ideas on this project,
and it means the feasibility question turns almost entirely on whether aie2p can
execute the arithmetic -- above all a numerically stable softmax.

### The arithmetic the NPU would have to beat [DERIVED]

QK^T and PV, causal so ~T/2 keys per query, over 20 Q heads and 30 layers:

> **Correction.** The first published version of this table gave 0.101 / 0.403 /
> 1.611 / 6.049 TFLOP and rates of 1.98-4.36 TFLOPS. Those were MAC counts
> mislabelled as FLOP -- wrong by 2.5x. The corrected figures are below; the
> conclusion is unchanged in direction and stronger in magnitude.

`2 x (T x T/2 x 128 x 20 x 30)` MACs for QK^T and PV together, causal so ~T/2
keys per query, at 2 FLOP per MAC:

| T | attention work | CPU time | **CPU rate** |
|---:|---:|---:|---:|
| 512 | 0.040 TFLOP | 50.9 ms | **0.79 TFLOPS** |
| 1024 | 0.161 TFLOP | 165.2 ms | **0.97 TFLOPS** |
| 2048 | 0.644 TFLOP | 602.0 ms | **1.07 TFLOPS** |
| 3968 | 2.418 TFLOP | 1388.0 ms | **1.74 TFLOPS** |

For scale: the same CPU sustains ~10 TFLOPS on the int8 I2_S GEMM, and the NPU
sustains ~11 TOPS there. **Attention runs at 0.79-1.74 TFLOPS on the CPU** --
roughly a sixth of what the same silicon does on int8 GEMM -- because it is
f32/f16 and because softmax is not a MAC-bound operation. The headroom is
therefore large **if** aie2p can run this arithmetic in a faster format with a
numerically stable softmax. That "if" is the whole experiment.

---

## B. Implementations studied [MEASURED / verified by reading source]

Commits pinned as studied:

| repo | commit |
|---|---|
| **`amd/IRON`** | **`d9e4ec5fab71d34365befd8127f86c5a676a6ae1`** |
| `atassis/xdna-engine` | `64fbf98e46e51c04a76b8d4d8a5c7eb32787c0e6` |
| `Xilinx/mlir-aie` | `af819a802f4e26251d331b47d4a364e70a9c6c54` |
| `ROCm/FastFlowLM` | `4d39e773499185b64ae4dafa6e16962d9af41ca5` |
| local | `mlir_aie 1.4.2`, `llvm_aie 21.0.0.2026080301+c9c5ecb7` |

### A working causal flash-attention kernel for aie2p exists

Not in any of the three repos originally named -- in **`amd/IRON`**, which
`xdna-engine` points at. `aie_kernels/aie2p/mha.cc` plus
`iron/operators/mha/design.py`: fused, block-tiled, online-softmax, **causal by
default**, GQA-aware, and CI-tested on real XDNA2 silicon
(`@pytest.mark.supported_devices("npu2")`, self-hosted krackan runner) against
`torch...scaled_dot_product_attention(is_causal=True)`. Marked green for AIE2P
and blank for AIE2 -- it is an aie2p-exclusive kernel.

Structure: a three-stage spatial pipeline, one column per pipeline, using 3 of
the 4 compute rows -- `batched_matmul_qk` (row 2) -> `softmax` (row 3) ->
`batched_matmul_pv` (row 4), passing tiles through ObjectFifos. 8 pipelines use
24 of 32 cores. **It does not fuse attention into one core**, which is the direct
answer to the "a cascade cannot span a softmax" objection.

**No performance number is published anywhere for it.** "It works" is verified;
"it is fast" is not.

### The softmax risk is real, and the fix is already installed here

Verified locally in our own venv, not just read about:

- **There is no `aie::exp` on aie2p -- only `aie::exp2`**, and `ElementaryOp`
  enumerates no `Exp`/`Log`/`Pow` at all.
- On XDNA2 `aie::exp2` is **hard-restricted to `bfloat16` output** by its
  `requires` clause; the f32-accurate variant exists only for aie2ps.
- The hardware LUT's accuracy over softmax's domain is documented **in AMD's own
  installed source**, `aie_kernels/aie2p/exp2f_vec.cc`: *"the LUT's max relative
  error runs 6.1% on [-1,0] to 49.1% on [-100,0], softmax's range, where this
  poly holds 8.9e-5."*
- **That accurate polynomial is already in our venv** (5234 bytes, f32 in/out,
  degree-5 minimax). It carries a load-bearing `__attribute__((noinline))` --
  *"Peano -O2 miscompiles the inlined form to NaN under high register pressure."*

AMD's shipped MHA uses the **inaccurate** hardware LUT, keeps its online-softmax
statistics `m`, `l` and the rescale factor in **bf16**, and uses a bf16-rounded
`log2e` (1.4453125 vs 1.4426950, a +0.18% systematic temperature shift). Its test
gate is correspondingly loose: `rel_tol=4e-2`, `abs_tol=1.5e-1`, with **0.5% of
elements allowed to fail outright**. All three are cheap to fix and would be
fixed before trusting it.

### Numeric format for aie2p attention

bf16 buffers compiled with `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`, f32
accumulation, **f32 softmax statistics**. This explains the project's earlier
"bf16 is quarter-rate emulation" measurement mechanically: without the macro,
`mmul<8,8,8,bf16>` issues two `mac_4x8_8x8_bf16` (256 MAC/instr native shape);
with it, operands are converted per-mmul to `v64bfp16ebs8` and a single native
512-MAC `mac_8x8_8x8T_conf` issues. AMD's design asserts the macro is mandatory.

**int8 is not viable here** -- softmax needs dynamic range, and this project
already measured that the wide int instructions issue at half rate (1.035x for
int8xint4), so int8's nominal advantage does not materialise.

### The blocker for BitNet: head_dim

`iron/operators/mha/op.py` raises `ValueError` for `d != 64`, and it is not a
soft limit: `B_q == B_kv == d == 64` is baked into `matmul_PV` and `rescale_O` as
literal C-tile strides (`out + j*64 + k*8 + l*512`). **BitNet's head_dim is 128.**
QK^T is fine (d is only the contraction dim); PV and the rescale need their
output indexing re-derived. `xdna-engine` independently hit the same wall for
Gemma.

Other things that would bite, all with reproductions in xdna-engine's notes:
conditional ObjectFifo acquire/release fails to build (they had to freeze RTP
loop bounds to constants); **16 KiB core program memory** overflows on unrolled
query sweeps; nested `range_` loops built and passed at small T but **silently
produced wrong results** at larger T; `aie::load_v` is an aligned load that
silently truncates. Also: `amd/IRON` pins `llvm-aie 22.x` while this machine has
**21.x** -- a major Peano version behind, which given the documented miscompiles
is a first-order risk.

`Xilinx/mlir-aie` has **no** attention or online-softmax example at all.
`FastFlowLM` ships 219 prebuilt xclbins and closed `.so`s with no AIE sources --
**zero reusable code**, useful only as a self-reported yardstick.

---

## D. The gating economic measurement [MEASURED]

Before writing a kernel, the cheap decisive question: **what does adding a second
hardware context cost?** The current runtime's central achievement is serving
every BitNet shape from ONE program so a prefill performs **zero** context
switches. An attention xclbin would be a second large context, alternating with
the GEMM context about twice per layer.

`tools/npu_two_context.cpp`, two production-sized designs, interleaved, 40 iters:

| context | alone | alternating | penalty |
|---|---:|---:|---:|
| A: M1024 K2560 N2560 | 1.235 ms | 3.821 ms | **+209%** |
| B: M1024 K6912 N2560 | 5.393 ms | 7.911 ms | **+47%** |
| **pair** | 6.628 ms | 11.732 ms | **+77%, +5.10 ms** |

**Alternating twice per layer over 30 layers costs ~153 ms per prefill, before
the attention kernel does any work.**

This reproduces `artifacts/kernels/context_switching.md` (+53% to +210% for
3-context cycling) and confirms it applies to a 2-context rotation. A first run
of the older `npu_switch_cost` probe appeared to *contradict* that finding at
-1%; it had silently loaded the small `M512` designs from `artifacts/xclbin`
because the probe takes its directory from `argv[1]`, not the env var. The
recorded finding explicitly says the penalty scales with design size, so the two
agree -- but the near-miss is why the measurement was repeated on production
designs.

### What that does to the budget

| context | CPU attention | switch tax | budget left for an NPU kernel |
|---:|---:|---:|---:|
| 2048 | 602 ms | ~153 ms | **449 ms** |
| 3968 | 1388 ms | ~153 ms | **1235 ms** |

The tax does **not** kill the idea -- it consumes 25% of the 2K budget and 11% of
the 4K one. But an NPU attention kernel must beat roughly **449 ms at 2K** to be
worth anything at all, against a CPU that takes 602 ms.

---

## Status and verdict

**Tasks A, B, C and the gating economic measurement are complete. Task D -- the
standalone kernel -- is scoped but NOT built, so no verdict is claimed.**

Calling this PROMISING or MARGINAL now would be asserting a measurement that does
not exist. What the evidence supports:

**Arguments for continuing:**
- Attention is **29.8% of prefill at 2K and 51.0% at 4K** by measured time,
  giving an Amdahl ceiling of **1.42x / 2.04x** if it were free. (Note this is
  the *time* share; attention is only ~7.5% of prefill FLOPs at 2K -- it is
  time-expensive precisely because the CPU executes it at 0.79-1.74 TFLOPS
  against ~10 TFLOPS on int8 GEMM.)
- **Data movement is not the blocker**: compulsory KV traffic is 37.5-291 MiB per
  prefill, ~5.5 ms at measured NPU DMA rates.
- A **working, causal, GQA-aware, silicon-tested** starting point exists under
  Apache-2.0.
- The **softmax accuracy risk is solved** and the fix is already installed.

**Arguments against:**
- A second hardware context costs a **measured 153 ms/prefill** before any work.
- The reference kernel is **locked to head_dim=64**; BitNet needs 128, requiring
  the PV and rescale C-tile indexing to be re-derived.
- Peano here is a **major version behind** what the reference pins, against
  documented miscompiles.
- Four distinct, reproduced toolchain hazards stand between here and a working
  kernel.

**Recommended next step, in this order:** port the stock d=64 op and measure it
standalone at BitNet's sequence lengths *before* touching head_dim. If a d=64
kernel cannot clear ~449 ms-equivalent at 2K, the d=128 port is not worth
starting, and that is a cheap negative result. Only if it clears should the
head_dim port and the numeric fixes (f32 statistics, `exp2f_vec`, exact `log2e`)
be attempted.

The alternative that avoids the switch tax entirely -- placing attention in the
**same** xclbin as the GEMM -- is a substantially larger redesign (AMD's MHA
already uses 24 of 32 cores) and is not recommended without the standalone
numbers first.
