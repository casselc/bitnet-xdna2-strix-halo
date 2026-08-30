#!/usr/bin/env bash
# How much of prefill wall-time is the NPU actually computing?
#
# The standalone kernel measures 11.8-13.2 TOPS and the in-model linear-algebra
# rate is 6.53 TFLOPS, yet the fitted cost model says the NPU is worth only ~10
# Zen 5 threads (~0.63x of the 16-thread CPU). Those cannot all be true unless
# the NPU is idle for a large part of the prefill. This measures that directly:
#   duty = (NPU dispatch time) / (prefill wall time)
set -u
M=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
BIN=refs/BitNet/build-xdna3/bin/llama-bench
export BITNET_XDNA_ARTIFACTS=$PWD/artifacts/xclbin-tuned
export BITNET_XDNA_STATS=1
P=${P:-2048}
printf "%-8s %-7s %-10s %-11s %-10s %-7s\n" threads tiles pp_tok/s wall_ms npu_ms duty
for th in 4 8 15; do
  for tiles in 1 2; do
    out=$(BITNET_XDNA=1 BITNET_XDNA_TILES=$tiles timeout 900 \
          $BIN -m $M -p $P -n 0 -t $th -ngl 0 -r 3 -ub $P 2>&1)
    ts=$(echo "$out" | grep -oE "pp$P \|[ ]*[0-9.]+" | grep -oE "[0-9.]+$")
    npu=$(echo "$out" | grep -oE "dispatch_total=[0-9.]+" | grep -oE "[0-9.]+$")
    [ -z "$ts" ] && { echo "  t=$th tiles=$tiles FAILED"; continue; }
    # 3 reps of P tokens; wall per rep = P/ts seconds
    python3 -c "
ts=$ts; npu=${npu:-0}; P=$P; reps=3
wall=P/ts*1000.0
print(f'{$th:<8} {$tiles:<7} {ts:<10.1f} {wall:<11.0f} {npu/reps:<10.0f} {npu/reps/wall*100:5.1f}%')"
  done
done
