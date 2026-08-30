#!/usr/bin/env bash
# Can RAPL resolve the NPU's power at all?
#
# A first attempt with one idle window, one load window, and one idle window was
# inconclusive: the idle baseline drifted 48.5 -> 56.6 W between the two idle
# samples, which is larger than the ~4 W the NPU appeared to add. Drift of that
# size swamps the effect, exactly as block-ordered throughput benchmarks were
# swamped earlier in this project.
#
# So: alternate idle and load windows N times and compare the paired means. Drift
# that is slow relative to the alternation period cancels.
set -u
PKG=/sys/class/powercap/intel-rapl:0/energy_uj
WRAP=$(cat /sys/class/powercap/intel-rapl:0/max_energy_range_uj)
SECS=${SECS:-10}
REPS=${REPS:-4}

watts() {   # watts <seconds>  -> package watts over the window
  local p0 p1 d
  p0=$(cat $PKG); sleep "$1"; p1=$(cat $PKG)
  d=$(( p1 - p0 )); (( d < 0 )) && d=$(( d + WRAP ))
  python3 -c "print(f'{$d/1e6/$1:.2f}')"
}

echo "Alternating idle/load, ${SECS}s windows x ${REPS} reps (paired, to cancel drift)"
for LOAD in "cpu" "npu"; do
  idles=(); loads=()
  for r in $(seq $REPS); do
    idles+=("$(watts $SECS)")
    if [ "$LOAD" = cpu ]; then
      pids=(); for i in $(seq 16); do setsid bash -c 'while true; do :; done' >/dev/null 2>&1 & pids+=($!); done
    else
      pids=(); setsid $NPU_LOAD >/dev/null 2>&1 & pids+=($!)
    fi
    sleep 2
    loads+=("$(watts $SECS)")
    for p in "${pids[@]}"; do kill -9 -"$p" 2>/dev/null; done
    sleep 3
  done
  python3 - "$LOAD" "${idles[*]}" "${loads[*]}" <<'PY'
import sys, statistics as st
label, idle, load = sys.argv[1], [float(x) for x in sys.argv[2].split()], [float(x) for x in sys.argv[3].split()]
d = [l - i for l, i in zip(load, idle)]
print(f"  {label:>4}: idle {st.mean(idle):6.2f} W   load {st.mean(load):6.2f} W"
      f"   delta {st.mean(d):+6.2f} W  (per-pair {[f'{x:+.1f}' for x in d]})")
if len(d) > 1:
    sd = st.stdev(d)
    print(f"        paired sd {sd:.2f} W -> {'RESOLVED' if abs(st.mean(d)) > 2*sd else 'NOT RESOLVED (delta within 2sd)'}")
PY
done
