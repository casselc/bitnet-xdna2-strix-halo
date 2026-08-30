#!/usr/bin/env bash
# Task 6: package energy per prefill token, deployed g_acc path vs direct output.
#
# Package RAPL only. The intel-rapl:0:0 "core" subdomain is NOT usable on this
# SoC -- it reads +15.1 W for one busy thread and +7.2 W for sixteen, which is
# non-monotonic and not a counter wrap (both wrap at 65.5 kJ).
#
# Alternating arms, never blocked, so thermal drift cancels between them.
#
# WARNING: never reap with `pkill -f <pattern>` -- the pattern matches this
# script's own command line. Track PIDs explicitly.
set -u
PKG=/sys/class/powercap/intel-rapl:0/energy_uj
WRAP=$(cat /sys/class/powercap/intel-rapl:0/max_energy_range_uj)
M=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
BIN=refs/BitNet/build-xdna3/bin/llama-bench
export BITNET_XDNA_ARTIFACTS=$PWD/artifacts/xclbin-tuned
P=${P:-2048}; UB=${UB:-2048}; INNER=${INNER:-3}; REPS=${REPS:-5}
OUT=${OUT:-artifacts/direct-output/energy.csv}
echo "rep,threads,mode,tok_s,watts,mj_per_token" > "$OUT"
for TH in ${THREADS:-8 6}; do
  for rep in $(seq 1 $REPS); do
    for mode in gacc direct; do
      [ "$mode" = direct ] && D=1 || D=0
      e0=$(cat $PKG); t0=$(date +%s.%N)
      out=$(BITNET_XDNA=1 BITNET_XDNA_DIRECT_OUT=$D timeout 1800 \
            $BIN -m $M -p $P -n 0 -t $TH -ngl 0 -r $INNER -ub $UB 2>&1)
      e1=$(cat $PKG); t1=$(date +%s.%N)
      ts=$(echo "$out" | grep -oE "pp$P \|[ ]*[0-9.]+" | grep -oE "[0-9.]+$")
      python3 -c "
e=(($e1)-($e0))%$WRAP; dt=$t1-$t0; ts=$ts
W=e/1e6/dt                 # average package power over the whole invocation
mj=W/ts*1000               # mJ per prefill token at that power and rate
print(f'$rep,$TH,$mode,{ts:.1f},{W:.1f},{mj:.2f}')" | tee -a "$OUT"
    done
  done
done
echo
python3 - "$OUT" <<'PY'
import csv, sys, statistics as st
from collections import defaultdict
rows=list(csv.DictReader(open(sys.argv[1])))
g=defaultdict(list)
for r in rows: g[(r["threads"],r["mode"])].append(r)
print(f"{'threads':>8}{'mode':>9}{'tok/s':>9}{'avg W':>8}{'mJ/token':>10}{'vs g_acc':>10}")
for th in sorted({r["threads"] for r in rows}, key=int):
    base=st.median([float(x["mj_per_token"]) for x in g[(th,"gacc")]])
    for m in ("gacc","direct"):
        v=g[(th,m)]
        mj=st.median([float(x["mj_per_token"]) for x in v])
        print(f"{th:>8}{m:>9}{st.median([float(x['tok_s']) for x in v]):>9.1f}"
              f"{st.median([float(x['watts']) for x in v]):>8.1f}{mj:>10.2f}{mj/base:>9.3f}x")
PY
