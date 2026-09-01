#!/usr/bin/env bash
# Start/stop a candidate controller model on its own port, using the SAME
# pinned llama.cpp build as the BitNet controller.
#
# The pinned build (9918 / 390c30775) already carries LLM_ARCH_QWEN35,
# LLM_ARCH_LFM2, LLM_ARCH_LFM2MOE and LLM_ARCH_NEMOTRON_H, so no separate
# worktree is needed and every candidate is measured on the same runtime as the
# incumbent. That matters more than being current: a throughput difference
# between two llama.cpp revisions would otherwise be attributed to the model.
#
# --slot-save-path is always set. It is what makes the per-domain state
# footprint measurable by serialization instead of inferred from RSS.
#
# PIDs are reaped BY PID, never by pattern: `pkill -f` has killed this
# project's own harness before, and `pgrep -x` silently fails on names longer
# than 15 characters.
set -uo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
RUN="${CAND_RUN_DIR:-/tmp/bitnet-candidates}"
mkdir -p "$RUN"
BIN="${CAND_BIN:-$R/refs/BitNet/build-xdna/bin/llama-server}"

port_owner() {
    ss -ltnp 2>/dev/null | awk -v p=":$1" '$4 ~ p"$" {print}' \
        | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2
}

wait_health() {
    local p=$1 t=${2:-300} i=0
    while [ $i -lt "$t" ]; do
        curl -s -m 2 "http://127.0.0.1:$p/health" 2>/dev/null | grep -q '"ok"' && return 0
        # a dead server will never become healthy; fail fast instead of waiting
        [ -f "$RUN/$LABEL.pid" ] && ! kill -0 "$(cat "$RUN/$LABEL.pid")" 2>/dev/null && return 2
        sleep 1; i=$((i+1))
    done
    return 1
}

case "${1:-}" in
  start)
    LABEL="${2:?label}"; MODEL="${3:?model path}"; PORT="${4:-8090}"
    T="${CAND_T:-4}"; CTX="${CAND_CTX:-40960}"; SLOTS="${CAND_SLOTS:-8}"
    B="${CAND_B:-4096}"; UB="${CAND_UB:-4096}"; TB="${CAND_TB:-16}"
    [ -f "$MODEL" ] || { echo "no such model: $MODEL"; exit 1; }
    o=$(port_owner "$PORT")
    [ -n "$o" ] && { echo "port $PORT already held by pid $o -- refusing"; exit 1; }
    SAVE="$RUN/state-$LABEL"; mkdir -p "$SAVE"
    # BITNET_XDNA=0: the NPU path is BitNet-ternary-specific. A warm controller
    # engages it 0% of the time anyway (envelope 9), so leaving it off keeps
    # every candidate on one identical CPU path.
    BITNET_XDNA=0 nohup "$BIN" -m "$MODEL" -t "$T" -ngl 0 \
        -c "$CTX" -np "$SLOTS" -b "$B" -ub "$UB" -tb "$TB" \
        --slot-save-path "$SAVE/" \
        ${CAND_CACHE_RAM:+--cache-ram "$CAND_CACHE_RAM"} \
        --host 127.0.0.1 --port "$PORT" --no-webui \
        > "$RUN/$LABEL.log" 2>&1 < /dev/null &
    echo $! > "$RUN/$LABEL.pid"
    if wait_health "$PORT" "${CAND_WAIT:-300}"; then
        echo "up   $LABEL port=$PORT pid=$(cat "$RUN/$LABEL.pid") save=$SAVE"
        grep -iE "^.*(arch|n_layer|n_embd|n_head|model type|model params|model size)" \
            "$RUN/$LABEL.log" 2>/dev/null | head -8
    else
        echo "FAILED $LABEL -- last log lines:"; tail -25 "$RUN/$LABEL.log"; exit 1
    fi
    ;;
  stop)
    LABEL="${2:?label}"
    f="$RUN/$LABEL.pid"
    [ -f "$f" ] || { echo "no pidfile for $LABEL"; exit 0; }
    p=$(cat "$f")
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
        kill "$p" 2>/dev/null
        for _ in $(seq 1 30); do kill -0 "$p" 2>/dev/null || break; sleep 0.5; done
        kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
    fi
    rm -f "$f"; echo "stopped $LABEL"
    ;;
  status)
    for f in "$RUN"/*.pid; do
        [ -f "$f" ] || continue
        l=$(basename "$f" .pid); p=$(cat "$f")
        if kill -0 "$p" 2>/dev/null; then
            echo "$l: pid $p RSS $(awk '/VmRSS/{printf "%.0f MiB",$2/1024}' /proc/"$p"/status 2>/dev/null)"
        else echo "$l: down"; fi
    done
    ;;
  *) echo "usage: $0 {start LABEL MODEL [PORT]|stop LABEL|status}"; exit 2;;
esac
