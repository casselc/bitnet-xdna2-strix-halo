#!/usr/bin/env bash
# Energy probe v2 -- separates CPU-core power from uncore (where the NPU lives).
#
# intel-rapl:0   = package-0  (whole SoC: cores + uncore + NPU)
# intel-rapl:0:0 = core       (CPU cores only)
# => NPU power = d(package) - d(core), which removes the host thread that is
#    busy submitting work and would otherwise be charged to the NPU.
#
# WARNING: never reap with `pkill -f <pattern>` here. The pattern matches this
# script's own command line and kills the harness. Track PIDs explicitly.
set -u
PKG=/sys/class/powercap/intel-rapl:0/energy_uj
CORE=/sys/class/powercap/intel-rapl:0:0/energy_uj
PKG_WRAP=$(cat /sys/class/powercap/intel-rapl:0/max_energy_range_uj)
CORE_WRAP=$(cat /sys/class/powercap/intel-rapl:0:0/max_energy_range_uj)
SECS=${SECS:-12}; REPS=${REPS:-4}
NPU_LOAD=${NPU_LOAD:?set NPU_LOAD}

read_e() { echo "$(cat $PKG) $(cat $CORE)"; }
delta()  { python3 -c "import sys;a,b,w=map(int,sys.argv[1:4]);print((b-a)%w)" "$1" "$2" "$3"; }

# window <secs> -> "pkg_watts core_watts"
window() {
  local s=$1; read p0 c0 <<< "$(read_e)"; local t0=$(date +%s.%N)
  sleep "$s"
  read p1 c1 <<< "$(read_e)"; local t1=$(date +%s.%N)
  python3 -c "
import sys
p0,p1,c0,c1,pw,cw,t0,t1=sys.argv[1:9]
dt=float(t1)-float(t0)
dp=(int(p1)-int(p0))%int(pw); dc=(int(c1)-int(c0))%int(cw)
print(f'{dp/1e6/dt:.3f} {dc/1e6/dt:.3f}')" "$p0" "$p1" "$c0" "$c1" "$PKG_WRAP" "$CORE_WRAP" "$t0" "$t1"
}

echo "Paired idle/load, ${SECS}s x ${REPS} reps; package vs core split"
for LOAD in cpu1 cpu16 npu; do
  dp_l=(); dc_l=()
  for r in $(seq 1 $REPS); do
    read ip ic <<< "$(window $SECS)"           # idle window
    case $LOAD in
      cpu1)  taskset -c 0 bash -c 'while :; do :; done' & LP=$! ;;
      cpu16) LP=""; for i in $(seq 0 15); do taskset -c $i bash -c 'while :; do :; done' & LP="$LP $!"; done ;;
      npu)   $NPU_LOAD >/dev/null 2>&1 & LP=$! ;;
    esac
    sleep 1
    read lp lc <<< "$(window $SECS)"           # load window
    for p in $LP; do kill -9 $p 2>/dev/null; done
    wait 2>/dev/null
    sleep 1
    dp_l+=($(python3 -c "print(f'{$lp-$ip:.2f}')"))
    dc_l+=($(python3 -c "print(f'{$lc-$ic:.2f}')"))
  done
  python3 - "$LOAD" "${dp_l[*]}" "${dc_l[*]}" <<'PY'
import sys, statistics as st
name, dp, dc = sys.argv[1], [float(x) for x in sys.argv[2].split()], [float(x) for x in sys.argv[3].split()]
un = [p-c for p, c in zip(dp, dc)]
def f(lbl, v):
    m, s = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
    print(f"   {name:<6} {lbl:<9} {m:+7.2f} W  sd {s:5.2f}   {[f'{x:+.1f}' for x in v]}")
f("package", dp); f("core", dc); f("UNCORE", un)
PY
done
