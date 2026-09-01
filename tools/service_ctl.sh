#!/usr/bin/env bash
# Start/stop the warm persistent services for the service-cotenancy pass.
#
# llama-server fixes -t at startup, so changing controller width means
# restarting the controller. The GPU worker is started once and left warm
# across the whole matrix -- reloading a 16.7 GiB model per arm would dominate
# every measurement.
#
# PIDs are written to files and reaped BY PID. This project has twice killed its
# own harness with pattern matching (`pkill -f`), and `pgrep -x` silently fails
# on names longer than 15 characters, so no pattern matching appears here.
set -uo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
RUN="${SERVICE_RUN_DIR:-/tmp/bitnet-service}"
mkdir -p "$RUN"

CTRL_MODEL="$R/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
WORK_MODEL="$R/models/worker/Qwen3.6-27B-UD-Q4_K_XL.gguf"
CTRL_BIN="$R/refs/BitNet/build-xdna/bin/llama-server"
WORK_BIN="$R/refs/BitNet/build-vulkan/bin/llama-server"

# A previous manual start left an orphaned server holding port 8081: killing the
# launching shell did not kill the server, so the new instance failed to bind
# while /health still answered FROM THE STALE PROCESS. Every measurement would
# then have been attributed to the wrong binary and the wrong lease file. So:
# refuse to start if the port is already held, and after starting, verify the
# listener is the pid we just recorded.
port_owner() {  # port -> pid or empty
    ss -ltnp 2>/dev/null | awk -v p=":$1" '$4 ~ p"$" {print}' \
        | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2
}

assert_port_free() {
    local o; o=$(port_owner "$1")
    if [ -n "$o" ]; then
        echo "port $1 already held by pid $o -- refusing to start (stale service?)"
        exit 1
    fi
}

assert_owner() {  # port, expected_pid
    local o; o=$(port_owner "$1")
    if [ "$o" != "$2" ]; then
        echo "port $1 is served by pid ${o:-none}, not the pid we started ($2)"
        exit 1
    fi
}

wait_health() {  # port, timeout_s
    local p=$1 t=${2:-120} i=0
    while [ $i -lt "$t" ]; do
        if curl -s -m 2 "http://127.0.0.1:$p/health" 2>/dev/null | grep -q '"ok"'; then
            return 0
        fi
        sleep 1; i=$((i+1))
    done
    return 1
}

stop_pidfile() {
    local f=$1
    [ -f "$f" ] || return 0
    local p; p=$(cat "$f")
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
        kill "$p" 2>/dev/null
        for _ in $(seq 1 20); do kill -0 "$p" 2>/dev/null || break; sleep 0.5; done
        kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
    fi
    rm -f "$f"
}

case "${1:-}" in
  start-ctrl)
    T=${2:-6}
    stop_pidfile "$RUN/ctrl.pid"
    assert_port_free 8081
    : > "$RUN/lease.csv"
    # NOTE: a conditional env prefix (${VAR:+NAME=val}) does NOT work here --
    # after expansion bash treats it as the command name, not an assignment.
    # Export explicitly instead.
    if [ -n "${NE11_CSV:-}" ]; then export BITNET_XDNA_NE11_CSV="$NE11_CSV"; : > "$NE11_CSV"; fi
    if [ -n "${NE11_STATS:-}" ]; then export BITNET_XDNA_NE11_STATS=1; fi
    if [ -n "${NE11_EVERY:-}" ]; then export BITNET_XDNA_NE11_EVERY="$NE11_EVERY"; fi
    BITNET_XDNA=1 BITNET_XDNA_STATS=1 \
    BITNET_XDNA_LEASE_STATS=1 \
    BITNET_XDNA_LEASE_CSV="$RUN/lease.csv" \
    BITNET_XDNA_LEASE_EVERY="${LEASE_EVERY:-32}" \
    nohup "$CTRL_BIN" -m "$CTRL_MODEL" -t "$T" -ngl 0 \
        -c "${CTRL_CTX:-20480}" -np "${CTRL_SLOTS:-8}" \
        -b "${CTRL_B:-2048}" -ub "${CTRL_UB:-2048}" \
        ${CTRL_TB:+-tb "$CTRL_TB"} \
        --host 127.0.0.1 --port 8081 --no-webui \
        > "$RUN/ctrl.log" 2>&1 < /dev/null &
    echo $! > "$RUN/ctrl.pid"
    wait_health 8081 180 || { echo "controller FAILED to become healthy"; tail -5 "$RUN/ctrl.log"; exit 1; }
    assert_owner 8081 "$(cat "$RUN/ctrl.pid")"
    # /health returning ok does NOT mean the XDNA weights are resident: they are
    # expanded and uploaded lazily on the first qualifying prefill. Measuring
    # immediately after health contaminates the first cell of every restart, so
    # warm explicitly and prove dispatches actually happened.
    if [ "${CTRL_WARMUP:-1}" = "1" ]; then
        "$R/tools/service_warmup.py" --port 8081 --label "t=$T" \
            --out "${WARMUP_OUT:-$RUN/warmup.json}" \
            || { echo "controller warmup FAILED"; exit 1; }
    fi
    echo "controller up t=$T tb=${CTRL_TB:-<=t} b=${CTRL_B:-2048} ub=${CTRL_UB:-2048}" \
         "np=${CTRL_SLOTS:-8} pid=$(cat "$RUN/ctrl.pid")"
    ;;
  start-work)
    stop_pidfile "$RUN/work.pid"
    assert_port_free 8082
    # sg render: /dev/dri/renderD128 is root:render and this session's group
    # set predates the membership, so without it RADV silently falls back to
    # llvmpipe (a CPU rasteriser) and the "GPU" arm measures the CPU.
    sg render -c "nohup $WORK_BIN -m $WORK_MODEL -ngl 99 -c 8192 -t 4 \
        --host 127.0.0.1 --port 8082 --no-webui > $RUN/work.log 2>&1 < /dev/null &
        echo \$! > $RUN/work.pid"
    wait_health 8082 300 || { echo "worker FAILED to become healthy"; tail -5 "$RUN/work.log"; exit 1; }
    assert_owner 8082 "$(cat "$RUN/work.pid")"
    echo "worker up pid=$(cat "$RUN/work.pid")"
    ;;
  stop)
    stop_pidfile "$RUN/ctrl.pid"; stop_pidfile "$RUN/work.pid"; echo "stopped"
    ;;
  status)
    for s in ctrl work; do
      f="$RUN/$s.pid"
      if [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; then
        echo "$s: pid $(cat "$f") RSS $(awk '/VmRSS/{print $2/1024" MiB"}' /proc/"$(cat "$f")"/status 2>/dev/null)"
      else echo "$s: down"; fi
    done
    ;;
  *) echo "usage: $0 {start-ctrl [threads]|start-work|stop|status}"; exit 2;;
esac
