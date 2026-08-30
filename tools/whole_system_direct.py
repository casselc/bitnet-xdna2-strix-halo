#!/usr/bin/env python3
"""Task 8: does direct output improve the already-useful Pareto point?

The question is not whether direct output wins an isolated benchmark -- Task 5
answered that -- but whether it improves the deployment configuration the
previous pass recommended: 8 threads + NPU, which strictly dominated 15-thread
CPU-only on both controller TTFT and co-tenant throughput.

Reuses the synthetic co-tenant from tools/whole_system.py unchanged.
"""
import csv, os, statistics as st, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from whole_system import CpuLoad, controller   # same co-tenant and timing as before

ART = os.path.abspath("artifacts/xclbin-tuned")

def run_arm(prompt, gen, threads, backend, ub):
    """backend: 'cpu' | 'gacc' | 'direct'"""
    prev = {k: os.environ.get(k) for k in ("BITNET_XDNA_DIRECT_OUT",)}
    os.environ["BITNET_XDNA_DIRECT_OUT"] = "1" if backend == "direct" else "0"
    try:
        # tiles=None selects CPU-only inside controller(); anything else enables
        # the NPU. We want the cost model, not a forced tile count, so pass a
        # sentinel that controller() maps to BITNET_XDNA=1 and then unset it.
        if backend == "cpu":
            return controller(prompt, gen, threads, None, ub)
        os.environ["BITNET_XDNA"] = "1"
        os.environ.pop("BITNET_XDNA_TILES", None)
        return controller_auto(prompt, gen, threads, ub)
    finally:
        for k, v in prev.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v

def controller_auto(prompt, gen, threads, ub):
    """controller() forces BITNET_XDNA_TILES; we want the cost model's own pick."""
    import re, subprocess
    from whole_system import BIN, MODEL
    env = dict(os.environ, BITNET_XDNA_ARTIFACTS=ART, BITNET_XDNA="1", BITNET_XDNA_STATS="1")
    env.pop("BITNET_XDNA_TILES", None)
    cmd = [BIN, "-m", MODEL, "-p", str(prompt), "-n", str(gen), "-t", str(threads),
           "-ngl", "0", "-r", "2", "-ub", str(ub)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    out = r.stdout + r.stderr
    pp = re.search(rf"pp{prompt}\s*\|\s*([0-9.]+)", out)
    tg = re.search(rf"tg{gen}\s*\|\s*([0-9.]+)", out)
    if not pp: return None
    pp_ts = float(pp.group(1)); tg_ts = float(tg.group(1)) if tg else 0.0
    ttft = prompt / pp_ts * 1000.0
    return {"pp_tok_s": pp_ts, "tg_tok_s": tg_ts, "ttft_ms": round(ttft, 1),
            "total_ms": round(ttft + (gen / tg_ts * 1000.0 if tg_ts else 0.0), 1)}

def main():
    prompt, gen, ub, bg, reps = 2048, 32, 2048, 8, 3
    arms = [("15T CPU-only", 15, "cpu"),
            ("8T hybrid g_acc", 8, "gacc"),
            ("8T hybrid direct", 8, "direct"),
            ("4T hybrid direct", 4, "direct")]
    rows = []
    print(f"controller p{prompt}/n{gen} ub{ub}; {bg} co-tenant workers; "
          f"{len(arms)} arms x {reps} interleaved reps")
    with CpuLoad(bg) as load:
        for rep in range(1, reps + 1):
            for label, th, backend in arms:
                s0 = load.sample()
                r = run_arm(prompt, gen, th, backend, ub)
                s1 = load.sample()
                it = (s1 - s0) if isinstance(s0, float) else 0.0
                rows.append(dict(rep=rep, arm=label, threads=th, backend=backend,
                                 ttft_ms=r["ttft_ms"], total_ms=r["total_ms"],
                                 cotenant_it_s=round(load.sample(), 1)))
                print(f"  [{rep}] {label:<18} TTFT {r['ttft_ms']:>7.0f} ms  "
                      f"total {r['total_ms']:>7.0f} ms  co-tenant {rows[-1]['cotenant_it_s']:.1f} it/s",
                      flush=True)
    out = "artifacts/direct-output/whole_system.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n{'arm':<20}{'TTFT ms':>9}{'total ms':>10}{'co-tenant it/s':>16}")
    for label, th, backend in arms:
        v = [r for r in rows if r["arm"] == label]
        print(f"{label:<20}{st.median([x['ttft_ms'] for x in v]):>9.0f}"
              f"{st.median([x['total_ms'] for x in v]):>10.0f}"
              f"{st.median([x['cotenant_it_s'] for x in v]):>16.1f}")
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
