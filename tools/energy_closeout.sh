#!/usr/bin/env bash
# Package energy per prefill token for the closeout configuration.
#
#   arena    = the frozen direct-output-arena reference: direct output, host
#              `part` accumulation for deep-K
#   closeout = this branch's default: direct output + direct K-reduce
#
# Package RAPL only; the core subdomain is unusable on this SoC. Alternating
# arms so thermal drift cancels. Reap by explicit PID, never `pkill -f`.
set -u
PKG=/sys/class/powercap/intel-rapl:0/energy_uj
WRAP=$(cat /sys/class/powercap/intel-rapl:0/max_energy_range_uj)
M=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
BIN=refs/BitNet/build-xdna3/bin/llama-bench
export BITNET_XDNA_ARTIFACTS=$PWD/artifacts/xclbin-tuned
P=${P:-2048}; UB=${UB:-2048}; INNER=${INNER:-3}; REPS=${REPS:-5}
OUT=${OUT:-artifacts/direct-output-closeout/energy.csv}
echo "rep,threads,mode,tok_s,watts,mj_per_token" > "$OUT"
for TH in ${THREADS:-8 6}; do
  for rep in $(seq 1 $REPS); do
    for mode in arena closeout; do
      [ "$mode" = closeout ] && KR=1 || KR=0
      e0=$(cat $PKG); t0=$(date +%s.%N)
      out=$(BITNET_XDNA=1 BITNET_XDNA_DIRECT_KREDUCE=$KR timeout 1800 \
            $BIN -m $M -p $P -n 0 -t $TH -ngl 0 -r $INNER -ub $UB 2>&1)
      e1=$(cat $PKG); t1=$(date +%s.%N)
      ts=$(echo "$out" | grep -oE "pp$P \|[ ]*[0-9.]+" | grep -oE "[0-9.]+$")
      python3 -c "
e=(($e1)-($e0))%$WRAP; dt=$t1-$t0; ts=$ts
W=e/1e6/dt; print(f'$rep,$TH,$mode,{ts:.1f},{W:.1f},{W/ts*1000:.2f}')" >> "$OUT"
    done
  done
done
python3 - "$OUT" <<'PY'
import csv, sys, statistics as st
from collections import defaultdict
rows=list(csv.DictReader(open(sys.argv[1]))); g=defaultdict(list)
for r in rows: g[(r["threads"],r["mode"])].append(r)
print(f"{'threads':>8}{'mode':>10}{'tok/s':>9}{'avg W':>8}{'mJ/token':>10}{'vs arena':>10}")
for th in sorted({r['threads'] for r in rows}, key=int):
    base=st.median([float(x["mj_per_token"]) for x in g[(th,"arena")]])
    for m in ("arena","closeout"):
        v=g[(th,m)]; mj=st.median([float(x["mj_per_token"]) for x in v])
        print(f"{th:>8}{m:>10}{st.median([float(x['tok_s']) for x in v]):>9.1f}"
              f"{st.median([float(x['watts']) for x in v]):>8.1f}{mj:>10.2f}{mj/base:>9.3f}x")
PY
