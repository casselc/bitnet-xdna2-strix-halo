# BitNet on Strix Halo XDNA2 — hybrid NPU-prefill / CPU-decode MVP

A vertical slice answering one question: **is an XDNA2 NPU worth using as the
resident BitNet controller on Strix Halo?**

**Answer: yes, in the range that matters — 1.12x at 2048 tokens, 1.08x at 3968,
after three rounds of optimization.** See [`artifacts/VERDICT.md`](artifacts/VERDICT.md)
for the original gate and [`artifacts/e2e/concurrent_results.md`](artifacts/e2e/concurrent_results.md)
for the current numbers.

- Real `BitNet-b1.58-2B-4T` runs NPU-assisted prefill and CPU decode.
- The NPU kernel is **bit-exact** against the CPU reference, and end-to-end
  **perplexity is identical to 4 decimal places** while 830 matmuls run on the NPU.
- **NPU and CPU now run concurrently** on disjoint halves of the token batch,
  which is what turned a 0.62x loss into a 1.12x win. Exclusive offload left 16
  cores idle for 76% of a prefill.
- Below one NPU tile (1024 tokens) it declines to offload and matches CPU-only
  exactly, rather than losing 0.59x to padding waste.
- As far as I can find, this is the first public instance of BitNet running on XDNA2.

## Layout

| path | what |
|---|---|
| `runtime/` | I2_S format reference, XRT resident-GEMM runtime, the C shim ggml calls, coordinate contract |
| `npu/` | IRON designs; `npu/ref/whole_array.py` is verbatim from mlir-aie v1.4.2 |
| `patches/` | the single guarded patch to the pinned llama.cpp fork |
| `tools/` | probes and benchmarks that decompose kernel vs transfer vs switching cost |
| `tests/` | packing, real-GGUF layout, NPU-vs-CPU bit-exactness, coordinate compatibility |
| `artifacts/` | all evidence, including the negative result |

## Findings

- [`artifacts/kernels/context_switching.md`](artifacts/kernels/context_switching.md) —
  cycling `xrt::hw_context`s costs **+53% to +210%** per dispatch and dominated
  everything else; seven other hypotheses were measured and eliminated first.
- [`artifacts/e2e/tuned_results.md`](artifacts/e2e/tuned_results.md) — three
  one-parameter kernel fixes worth +42%, and why `aie.dma_bd`'s "N ≤ 4096 wall"
  is a tunable, not a hardware limit.
- [`artifacts/e2e/concurrent_results.md`](artifacts/e2e/concurrent_results.md) —
  concurrent execution, and the micro-batch trap: `ne11` is llama.cpp's `-ub`,
  not the prompt length, so the default 512 both blocked the split and forced
  2x padding waste against a 1024-token tile.

## Quick start

```bash
make check-cpu    # packing / coordinates / real-weight layout, no NPU needed
make check        # adds the NPU bit-exactness test
```

Full setup, benchmarks and reproduction: [`artifacts/reproduce.md`](artifacts/reproduce.md).
Known limitations, including where the evidence is thin: [`artifacts/limitations.md`](artifacts/limitations.md).
