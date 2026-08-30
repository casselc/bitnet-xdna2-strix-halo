# BitNet on Strix Halo XDNA2 — hybrid NPU-prefill / CPU-decode MVP

A vertical slice answering one question: **is an XDNA2 NPU worth using as the
resident BitNet controller on Strix Halo?**

**Answer: the mechanism works and is provably correct; the economics do not yet.**
See [`artifacts/VERDICT.md`](artifacts/VERDICT.md).

- Real `BitNet-b1.58-2B-4T` runs NPU-assisted prefill and CPU decode.
- The NPU kernel is **bit-exact** against the CPU reference, and end-to-end
  **perplexity is identical to 4 decimal places** while 830 matmuls run on the NPU.
- It is **1.6–4x slower** than 16 Zen 5 cores, because the stock mlir-aie kernel
  reaches ~9.3 TOPS of a ~50 TOPS device peak.
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

## Quick start

```bash
make check-cpu    # packing / coordinates / real-weight layout, no NPU needed
make check        # adds the NPU bit-exactness test
```

Full setup, benchmarks and reproduction: [`artifacts/reproduce.md`](artifacts/reproduce.md).
Known limitations, including where the evidence is thin: [`artifacts/limitations.md`](artifacts/limitations.md).
