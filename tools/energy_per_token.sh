#!/usr/bin/env bash
# Joules per prefill token, CPU-only vs hybrid, measured at the package.
# This is the deployment-relevant energy number: a spin-loop watt figure says
# nothing about what the actual GEMM costs.
set -u
PKG=/sys/class/powercap/intel-rapl:0/energy_uj
WRAP=$(cat /sys/class/powercap/intel-rapl:0/max_energy_range_uj)
M=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
BIN=refs/BitNet/build-xdna3/bin/llama-bench
export BITNET_XDNA_ARTIFACTS=$PWD/artifacts/xclbin-tuned
P=2048; R=5
printf "%-8s %-9s %-9s %-9s %-9s %-10s\n" threads mode tok/s watts J/token rel_J
for th in 4 8 15; do
  base=""
  for mode in cpu hybrid; do
    [ "$mode" = cpu ] && X=0 || X=1
    e0=$(cat $PKG); t0=$(date +%s.%N)
    out=$(BITNET_XDNA=$X timeout 1200 $BIN -m $M -p $P -n 0 -t $th -ngl 0 -r $R -ub $P 2>&1)
    e1=$(cat $PKG); t1=$(date +%s.%N)
    ts=$(echo "$out" | grep -oE "pp$P \|[ ]*[0-9.]+" | grep -oE "[0-9.]+$")
    base=$(python3 -c "
e=(($e1)-($e0))%$WRAP; dt=$t1-$t0; ts=$ts; P=$P; R=$R
W=e/1e6/dt                       # average package power over the whole invocation
J=e/1e6/(P*R)                    # includes model load; see note
# energy attributable to the timed prefills only, using measured power x prefill time
Jt=W*(P/ts)/P
b='$base'
rel=(f'{Jt/float(b):.2f}x' if b else '1.00x')
print(f'{ts:.1f} {W:.1f} {Jt*1000:.2f} {rel} {Jt}')")
    read ts_ W_ Jt_ rel_ raw <<< "$base"
    printf "%-8s %-9s %-9s %-9s %-9s %-10s\n" "$th" "$mode" "$ts_" "$W_" "$Jt_" "$rel_"
    [ "$mode" = cpu ] && base="$raw" || base=""
  done
done
