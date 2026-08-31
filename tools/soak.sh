#!/usr/bin/env bash
# Bounded soak at the recommended service point, with residency tracking.
#
# Task 14/15: watch for RSS growth, arena growth and latency drift under a
# sustained mixed load. Not a repeat of the packed-ternary question -- that is
# settled -- only leak-like behaviour over many requests.
set -uo pipefail
cd "$(dirname "$0")/.."
T=${1:-8}; N=${2:-200}
RUN=/tmp/bitnet-service
O=artifacts/service-cotenancy

tools/service_ctl.sh start-ctrl $T >/dev/null 2>&1 || { echo "start failed"; exit 1; }
CP=$(cat $RUN/ctrl.pid); WP=$(cat $RUN/work.pid)
echo "soak t$T n=$N  controller pid=$CP  worker pid=$WP"

# Sample residency by explicit PID for the duration of the load.
( echo "t_s,ctrl_rss_mib,work_rss_mib,gtt_gib,gpu_busy,degC"
  s=0
  while kill -0 "$CP" 2>/dev/null; do
    printf "%d,%s,%s,%s,%s,%s\n" "$s" \
      "$(awk '/VmRSS/{printf "%.1f", $2/1024}' /proc/$CP/status 2>/dev/null)" \
      "$(awk '/VmRSS/{printf "%.1f", $2/1024}' /proc/$WP/status 2>/dev/null)" \
      "$(awk '{printf "%.2f", $1/1073741824}' /sys/class/drm/card0/device/mem_info_gtt_used 2>/dev/null)" \
      "$(cat /sys/class/drm/card0/device/gpu_busy_percent 2>/dev/null)" \
      "$(awk '{printf "%.1f", $1/1000}' /sys/class/hwmon/hwmon2/temp1_input 2>/dev/null)"
    sleep 5; s=$((s+5))
  done ) > $O/soak_residency.csv &
SAMPLER=$!

timeout 5400 .venv/bin/python tools/service_bench.py soak --threads $T --conc 2 \
    --n $N --mix "C:1,CW:1,W:1" --verifier --lease-csv $RUN/lease.csv \
    --outdir $O --tag "t$T" 2>&1 | grep -E "soak"

kill $SAMPLER 2>/dev/null
echo "### soak complete"
