#!/usr/bin/env bash
# Find the stock whole_array kernel's ceiling on K=2560,N=2560 int8.
#
# The MVP measured 9.3 TOPS against a ~50 TOPS device peak. Published hand-tuned
# XDNA2 int8 GEMM reaches 38-56 TOPS. Before writing a custom kernel, establish
# how much of that gap is reachable by tiling alone.
#
# Constraints that make a config legal:
#   N % (n * n_aie_cols) == 0     -- enforced by the design
#   K % k == 0
#   L1 is 64 KB/core: A(m*k) + B(k*n) int8 + C(m*n) int32, times double buffering
set -u
PY=.venv/bin/python
M=${M:-512}; K=${K:-2560}; N=${N:-2560}
printf "%-5s %-5s %-5s %-5s %-10s %-12s %s\n" m k n cols "NPU us" "GFLOPS" result
for cols in 8 4; do
for m in 32 64 128; do
for k in 32 64 128 256; do
for n in 16 32 64 80; do
  (( N % (n * cols) )) && continue
  (( K % k ))          && continue
  (( M % m ))          && continue
  # L1 estimate, single-buffered; the design double-buffers so budget ~half
  l1=$(( m*k + k*n + m*n*4 ))
  (( l1 > 32768 )) && continue
  out=$(timeout 600 $PY npu/ref/whole_array.py -M $M -K $K -N $N -m $m -k $k -n $n \
        --dtype_in i8 --dtype_out i32 --n-aie-cols $cols 2>&1)
  us=$(grep -oE 'NPU time.*: [0-9.]+' <<<"$out" | grep -oE '[0-9.]+$')
  gf=$(grep -oE 'NPU GFLOPS +: [0-9.]+' <<<"$out" | grep -oE '[0-9.]+$')
  if grep -q PASS <<<"$out"; then res=PASS
  elif grep -q 'Stride' <<<"$out"; then res=STRIDE
  elif grep -q 'exit code 1' <<<"$out"; then res=COMPILE_FAIL
  else res=FAIL; fi
  printf "%-5s %-5s %-5s %-5s %-10s %-12s %s\n" "$m" "$k" "$n" "$cols" "${us:--}" "${gf:--}" "$res"
done; done; done; done
