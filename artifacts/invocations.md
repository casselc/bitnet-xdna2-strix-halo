# Example invocations

`BITNET_XDNA_ARTIFACTS` must point at the compiled xclbins; the rest defaults
sensibly. All of these use the same binary and the same weights — the backend is
a runtime choice, never a model property.

```bash
export BITNET_XDNA_ARTIFACTS=$PWD/artifacts/xclbin
M=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
BIN=refs/BitNet/build-xdna/bin
```

## Backend selection

```bash
# backend = :cpu
BITNET_XDNA=0 $BIN/llama-bench -m $M -p 512 -n 32 -t 16 -ngl 0

# backend = :hybrid-npu-cpu
BITNET_XDNA=1 $BIN/llama-bench -m $M -p 512 -n 32 -t 16 -ngl 0
```

## Tuning knobs

| variable | default | meaning |
|---|---|---|
| `BITNET_XDNA` | unset (= off) | `1` enables NPU offload for eligible prefill GEMMs |
| `BITNET_XDNA_MIN_TOKENS` | 64 | batches smaller than this stay on CPU; decode (1 token) never qualifies |
| `BITNET_XDNA_ARTIFACTS` | `artifacts/xclbin` | where the AOT-compiled xclbin/insts pairs live |
| `BITNET_XDNA_STATS` | unset | print dispatch counters, repack cost and resident bytes at exit |

```bash
# Raise the offload threshold so only large prefills use the NPU
BITNET_XDNA=1 BITNET_XDNA_MIN_TOKENS=512 $BIN/llama-bench -m $M -p 2048 -n 0 -t 16 -ngl 0

# Show what the NPU actually did
BITNET_XDNA=1 BITNET_XDNA_STATS=1 $BIN/llama-bench -m $M -p 512 -n 0 -t 16 -ngl 0
#   [bitnet-xdna] dispatches=410  dispatch_total=1090.6 ms  mean=2.660 ms
#   [bitnet-xdna] weight repack+upload=2669.0 ms  resident int8 weights=1843.1 MiB
```

## Graceful degradation

The NPU path is never required. Any of these silently fall back to CPU with
identical output:

```bash
BITNET_XDNA=1 BITNET_XDNA_ARTIFACTS=/nonexistent $BIN/llama-bench -m $M -p 512 -n 0 -t 16 -ngl 0
```
Shapes without an xclbin (`attn_k`, `attn_v` at N=640) always take the CPU path,
as does every tensor if the device cannot be opened.

## Numerical equivalence check

```bash
for mode in 0 1; do
  BITNET_XDNA=$mode $BIN/llama-perplexity -m $M -f artifacts/correctness/ppl_input.txt \
      -t 16 -ngl 0 --chunks 4 2>&1 | grep "Final estimate"
done
# both: PPL = 307.5806 +/- 27.85495
```

## Kernel-level measurement

```bash
build/npu_probe        # BO alloc / sync costs, no kernel involved
build/npu_gemm_bench   # dispatch cost vs number of resident weight buffers
build/npu_switch_cost  # kernel time vs xclbin switching, per-layer order
make check             # full test suite
```
