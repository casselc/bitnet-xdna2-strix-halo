# BitNet on Strix Halo XDNA2 — hybrid NPU-prefill / CPU-decode MVP

A vertical slice answering one question: **is an XDNA2 NPU worth using as the
resident BitNet controller on Strix Halo?**

**Answer: the mechanism works and is provably correct; the economics do not yet.**
See [`artifacts/VERDICT.md`](artifacts/VERDICT.md).

- Real `BitNet-b1.58-2B-4T` runs NPU-assisted prefill and CPU decode.
- The NPU kernel is **bit-exact** against the CPU reference, and end-to-end
  **perplexity is identical to 4 decimal places** while 830 matmuls run on the NPU.
- After a round of tuning it is **1.5–3.2x slower** than 16 Zen 5 cores
  (was 1.6–4x). The NPU now does its share of the arithmetic in 777 ms against
  the CPU's 2009 ms for a whole 2048-token prefill — it is the faster engine for
  that work. The loss is structural: offload is *exclusive*, so 16 CPU cores idle
  for 76% of the wall clock.
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

## Quick start

```bash
make check-cpu    # packing / coordinates / real-weight layout, no NPU needed
make check        # adds the NPU bit-exactness test
```

Full setup, benchmarks and reproduction: [`artifacts/reproduce.md`](artifacts/reproduce.md).
Known limitations, including where the evidence is thin: [`artifacts/limitations.md`](artifacts/limitations.md).
