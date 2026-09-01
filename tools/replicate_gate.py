#!/usr/bin/env python3
"""Task 11 -- interleaved replication of the two candidate configurations.

The prior width comparisons were block-ordered (all of t4, then all of t6, ...)
with per-class samples as small as 5. That cannot separate a configuration
effect from machine drift, and it cannot support a p95.

Here the two finalists alternate ROUND BY ROUND, both see identical workloads
and identical worker state, both are warmed explicitly after each restart, and
each accumulates >= 20 requests before any p95 is reported. Dispersion is
reported alongside the medians, and p99 is deliberately not reported: at n~24
per class the 99th percentile is the maximum, and quoting it would be quoting
one sample.
"""
import argparse, random, statistics as st, subprocess, sys, time

sys.path.insert(0, "tools")
from service_bench import (run_controller, controller_prompt, write_rows, pct,
                           Ne11Window, LeaseWindow, Power, append_jsonl)

NE11 = "/tmp/bitnet-service/ne11.csv"
LEASE = "/tmp/bitnet-service/lease.csv"


def restart(t, tb, b=4096, ub=4096, slots=8):
    env = dict(CTRL_B=str(b), CTRL_UB=str(ub), CTRL_SLOTS=str(slots),
               CTRL_TB=str(tb), NE11_STATS="1", NE11_CSV=NE11, NE11_EVERY="210")
    cmd = ["env"] + [f"{k}={v}" for k, v in env.items()] + \
          ["bash", "tools/service_ctl.sh", "start-ctrl", str(t)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"restart failed: {r.stdout[-400:]} {r.stderr[-400:]}")
    return r.stdout.strip().splitlines()[-1]


def boot_ci(a, b, iters=5000, seed=11):
    rng = random.Random(seed)
    d = []
    for _ in range(iters):
        ma = st.median([rng.choice(a) for _ in a])
        mb = st.median([rng.choice(b) for _ in b])
        if ma:
            d.append(mb / ma - 1.0)
    d.sort()
    return d[int(.025 * len(d))], d[int(.975 * len(d))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="8:16", help="config A as t:tb")
    ap.add_argument("--b", default="4:16", help="config B as t:tb")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--per-round", type=int, default=6)
    ap.add_argument("--n-predict", type=int, default=8)
    ap.add_argument("--out", default="artifacts/service-batching-gate/replicate.csv")
    a = ap.parse_args()

    prompt = controller_prompt(1954)
    cfgs = {"A": tuple(int(x) for x in a.a.split(":")),
            "B": tuple(int(x) for x in a.b.split(":"))}
    acc = {k: {"ttft": [], "total": []} for k in cfgs}
    rows = []
    for rnd in range(a.rounds):
        order = ["A", "B"] if rnd % 2 == 0 else ["B", "A"]
        for k in order:
            t, tb = cfgs[k]
            restart(t, tb)
            pw = Power()
            with Ne11Window(NE11) as nw, LeaseWindow(LEASE) as lw, pw:
                t0 = time.perf_counter()
                out = [run_controller(f"r{rnd}_{i}", t, 1, a.n_predict, prompt).row()
                       for i in range(a.per_round)]
                wall = time.perf_counter() - t0
                time.sleep(0.4)
            ok = [r for r in out if not r.get("err")]
            ttft = [r["client_ttft_ms"] for r in ok if r.get("client_ttft_ms")]
            tot = [r["total_ms"] for r in ok]
            acc[k]["ttft"] += ttft
            acc[k]["total"] += tot
            rec = dict(round=rnd, config=k, t=t, tb=tb, requests=len(ok),
                       req_per_s=round(len(ok) / wall, 3),
                       ttft_p50=pct(ttft, .5), total_p50=pct(tot, .5),
                       watts=pw.watts)
            rec.update(nw.delta()); rec.update(lw.delta())
            rows.append(rec)
            append_jsonl(a.out.replace(".csv", "_requests.jsonl"),
                         [dict(r, config=k, round=rnd, t=t, tb=tb) for r in ok])
            print(f"  round {rnd} {k} t{t}/tb{tb}: n={len(ok)} "
                  f"ttft p50 {rec['ttft_p50']:>8.1f}  {rec['watts']}W", flush=True)
            write_rows(a.out, rows)

    print("\n=== pooled, interleaved ===")
    for k, (t, tb) in cfgs.items():
        v = acc[k]["ttft"]
        print(f"  {k} t{t}/tb{tb}: n={len(v)}  ttft p50 {pct(v,.5):.1f}  "
              f"p95 {pct(v,.95):.1f}  mean {st.mean(v):.1f}  "
              f"sd {st.pstdev(v):.1f}  IQR {pct(v,.75)-pct(v,.25):.1f}  "
              f"min {min(v):.0f}  max {max(v):.0f}")
    A, B = acc["A"]["ttft"], acc["B"]["ttft"]
    lo, hi = boot_ci(A, B)
    rel = st.median(B) / st.median(A) - 1
    print(f"\n  B vs A median TTFT: {rel*100:+.2f}%  (95% CI {lo*100:+.2f}% .. {hi*100:+.2f}%)")
    print("  p99 deliberately not reported: at this n it is the maximum.")
    if lo <= 0 <= hi:
        print("  -> the two configurations are NOT separable at this sample size")
    else:
        print(f"  -> {'B' if rel < 0 else 'A'} is faster, and the CI excludes zero")


if __name__ == "__main__":
    main()
