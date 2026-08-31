# runtime v1 — executable coordinate

Everything needed to rebuild and re-verify the promoted runtime. Reproduced on
this machine at promotion time; see section 5 for what was actually re-run.

## 1. Source coordinates

| component | coordinate |
|---|---|
| evidence-producing source commit | `ed97cfcac564be9f85db415faf076695b871e008` (`direct-output-closeout`) |
| promotion branch base | the same commit; `runtime-v1-promotion` is created from it |
| BitNet | `0b341e582afbf9e1011f24744b554c96a3477eb5` |
| llama.cpp (pinned fork) | `390c307752ab78fd8189f359d6954c9ba1be74af` |
| runtime source | `runtime/` at the evidence-producing commit |
| required patch | `patches/001-bitnet-xdna.patch` (verified by `make check-patch`) |

The attention investigation that closed alongside this promotion is evidence
only and is **not** part of the runtime:
`attention-geometry-gate` @ `f44ae0215d0503cbc4dbb947341c6d023fa646af`.

## 2. Platform

| item | value |
|---|---|
| NPU | `RyzenAI-npu5`, `aie2p`, 6x8, at `0000:c7:00.1` |
| NPU firmware | `1.1.2.65` |
| kernel | `7.0.0-29-generic` |
| XRT | `2.25.00` |
| kernel toolchain (offline only) | `mlir_aie 1.4.2`, `llvm_aie 21.0.0.2026080301+c9c5ecb7`, Peano |

The mlir-aie/Peano toolchain is an **ahead-of-time kernel compiler only**. The
inference path links the C++ XRT API and never touches Python at runtime.

## 3. NPU artifacts

One program serves every BitNet linear shape, so a prefill performs zero
hardware-context switches. The **instruction stream is the executable
identity**; the xclbin container is recorded separately because it was measured
on this project to be non-reproducible across builds of identical geometry.

| role | bytes | SHA-256 |
|---|---:|---|
| `artifacts/xclbin-tuned/mm_M1024_K2560_N2560.insts.bin` | 5520 | `0cdea8e2932e04affe846293a8d3c30285a33fa5fc29ea1b74c66f7ac07abc24` |
| `artifacts/xclbin-tuned/mm_M1024_K2560_N2560.xclbin` | 109022 | `363319183d5a442eeaa9fd2b2c96f3e9df080ff1fe91a1f65119f96c310cb1e7` |

Per-core tile geometry `m=128, k=64, n=64`, built with `C_FIFO_DEPTH=1` and
`TB_MAX_N_ROWS=2`. (`gemm-tile-resweep` later measured `64x128x64` at 1.038x
with less L1 and lower variance; it is **not** promoted here — see that branch.)

## 4. Runtime defaults

| setting | value | override |
|---|---|---|
| `BITNET_XDNA` | off unless set | `=1` enables offload |
| `BITNET_XDNA_DIRECT_OUT` | **on** | `=0` restores the `g_acc` path |
| `BITNET_XDNA_DIRECT_KREDUCE` | **on** | `=0` restores host `part` accumulation |
| `BITNET_XDNA_ASYNC` | **off** | superseded by direct output |
| cost-model `R` | **25** with direct output (10 with `g_acc`) | `BITNET_XDNA_NPU_THREADS` |
| minimum tokens | `kMTile` = **1024** | `BITNET_XDNA_MIN_TOKENS` (clamped up to 1024) |
| concurrency | single-flight invocation lease, always on | — |
| artifact dir | `artifacts/xclbin-tuned` | `BITNET_XDNA_ARTIFACTS` |

**The micro-batch must be at least 1024 tokens or the NPU never runs.**
`llama-bench` and `llama-perplexity` both default to `-ub 512`, which is below
`kMTile`, so `bitnet_xdna_worth_it()` declines every micro-batch and the run
silently falls back to CPU. Pass `-ub 2048`. This is the single most common way
to produce a "the NPU did nothing" result on this runtime.

## 5. Reproduction

```bash
# build (from the repo root; needs refs/, models/, .venv/, .localdeps/)
cd refs/BitNet
R=$(cd ../.. && pwd)
cmake -B build-xdna -G Ninja -DBITNET_X86_TL2=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_COMMON=ON -DGGML_LLAMAFILE=OFF \
      -DBITNET_XDNA_RUNTIME_DIR=$R/runtime \
      -DBITNET_XDNA_XRT_INCLUDE="/usr/include/xrt;$R/.localdeps/usr/include"
cmake --build build-xdna -j32
cd ../..

make check          # patch reproducibility, CPU tests, 12 shape cases, lease

M=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
B=refs/BitNet/build-xdna/bin

# throughput, promoted defaults
BITNET_XDNA=1 BITNET_XDNA_STATS=1 BITNET_XDNA_SHAPE_CSV=/tmp/shapes.csv \
  $B/llama-bench -m $M -p 2048 -n 0 -t 8 -ngl 0 -ub 2048 -r 3

# equivalence WITH THE NPU ENGAGED (note the batch flags)
for mode in 0 1; do
  BITNET_XDNA=$mode $B/llama-perplexity -m $M \
     -f artifacts/correctness/ppl_input.txt -t 16 -ngl 0 \
     -c 2048 -b 2048 -ub 2048 --chunks 4 2>&1 | grep "Final estimate"
done
```

**Verified at promotion time** (details in `../RUNTIME_STATUS.md` section 4):

| check | result |
|---|---|
| `make check` | green — patch reproduces, 12/12 shapes bit-exact, lease holds |
| CPU-only dispatches | **0** |
| `stage_out` / `partacc` / `partcopy` | **0.000 ms** on every shape |
| direct-output arena | 6 slots x 10.0 MiB = **60.0 MiB**, bounded |
| resident tensors | 147 (bench) / 150 (perplexity) |
| pp2048 t8, 5 interleaved reps | **865.7 t/s** median vs 862.3 recorded — **1.004x** |
| CPU-only pp2048 t8 | 637.5 t/s vs 629.6 recorded — 1.013x |
| perplexity, NPU engaged (1320 dispatches) | **312.7569 +/- 14.92981**, identical CPU vs XDNA |
