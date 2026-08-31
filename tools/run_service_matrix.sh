#!/usr/bin/env bash
# Drive the remaining service-cotenancy experiments in one pass.
# Each controller width needs a service restart (llama-server fixes -t at
# startup), so widths are the outer loop and the GPU worker stays warm
# throughout -- reloading 16.7 GiB per arm would dominate everything.
set -uo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"; cd "$R"
RUN=/tmp/bitnet-service
O=artifacts/service-cotenancy
PY=.venv/bin/python
LEASE=$RUN/lease.csv

echo "### concurrency at t4 / t8 (t6 already measured)"
for T in 4 8; do
  tools/service_ctl.sh start-ctrl $T >/dev/null 2>&1 || { echo "t$T start FAILED"; continue; }
  timeout 3600 $PY tools/service_bench.py concurrency --threads $T --conc 1 2 4 \
      --n 12 --lease-csv $LEASE --outdir $O --tag t$T 2>&1 | grep -E "c=" 
done

echo "### chained controller -> worker, t4/t6/t8"
for T in 4 6 8; do
  tools/service_ctl.sh start-ctrl $T >/dev/null 2>&1 || continue
  timeout 3600 $PY tools/service_bench.py chain --threads $T --conc 1 --n 6 \
      --ctrl-predict 32 --work-predict 128 --lease-csv $LEASE \
      --outdir $O --tag t$T 2>&1 | grep -E "chain"
done

echo "### mixed load + verifier, t6 (recommended default)"
tools/service_ctl.sh start-ctrl 6 >/dev/null 2>&1
for MIX in "C:1,CW:1" "C:1,CW:2,W:1"; do
  timeout 3600 $PY tools/service_bench.py mixed --threads 6 --conc 2 --n 8 \
      --mix "$MIX" --verifier --lease-csv $LEASE --outdir $O \
      --tag "$(echo "$MIX" | tr ':,' '__')" 2>&1 | grep -E "mixed"
done

echo "### thread-policy comparison at the mixed cell"
for T in 4 8; do
  tools/service_ctl.sh start-ctrl $T >/dev/null 2>&1 || continue
  timeout 3600 $PY tools/service_bench.py mixed --threads $T --conc 2 --n 8 \
      --mix "C:1,CW:1" --verifier --lease-csv $LEASE --outdir $O \
      --tag "policy_t$T" 2>&1 | grep -E "mixed"
done
echo "### matrix complete"
