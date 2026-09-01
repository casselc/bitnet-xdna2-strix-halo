#!/usr/bin/env python3
"""Task 4 -- does the lease instrumentation cost anything measurable?

The runtime appends a cumulative snapshot and fflushes from inside the lease
release path. That is probably cheap, but every service timing on this branch
depends on it being cheap, so it is measured rather than assumed.

The env var is read once at startup, so ON and OFF cannot be interleaved per
request -- they are interleaved per CELL, alternating across several rounds, so
a drift in machine state (thermal, page cache, another tenant) cannot line up
with one arm the way a block-ordered A-then-B run would.

Reports the difference with a bootstrap CI, because a single pair of means
cannot support a "~1%" claim on its own.
"""
import argparse, json, random, statistics as st, subprocess, sys, time

sys.path.insert(0, "tools")
from service_bench import run_controller, controller_prompt, write_rows, pct


def restart(threads, lease_on, run_dir="/tmp/bitnet-service"):
    env = dict(CTRL_B="2048", CTRL_UB="2048", CTRL_SLOTS="8")
    if lease_on:
        env["LEASE_EVERY"] = "32"
    else:
        # service_ctl always sets BITNET_XDNA_LEASE_CSV, which itself turns the
        # stats on. NO_LEASE makes the launcher omit both.
        env["NO_LEASE"] = "1"
    cmd = ["env"] + [f"{k}={v}" for k, v in env.items()] + \
          ["bash", "tools/service_ctl.sh", "start-ctrl", str(threads)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"controller restart failed: {r.stdout[-400:]} {r.stderr[-400:]}")
    return r.stdout


def cell(prompt, n, threads, n_predict):
    lat, ttft = [], []
    for i in range(n):
        row = run_controller(f"lo{i}", threads, 1, n_predict, prompt).row()
        if row.get("err"):
            continue
        lat.append(row["prompt_ms"])
        if row.get("client_ttft_ms"):
            ttft.append(row["client_ttft_ms"])
    return lat, ttft


def boot_ci(a, b, iters=5000, seed=7):
    """Bootstrap CI on the relative difference of means, mean(b)/mean(a) - 1."""
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        ra = [rng.choice(a) for _ in a]
        rb = [rng.choice(b) for _ in b]
        ma, mb = st.mean(ra), st.mean(rb)
        if ma:
            diffs.append(mb / ma - 1.0)
    diffs.sort()
    return diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--per-cell", type=int, default=12)
    ap.add_argument("--n-predict", type=int, default=1)
    ap.add_argument("--out", default="artifacts/service-batching-gate/lease_overhead.csv")
    a = ap.parse_args()

    prompt = controller_prompt(1954)
    acc = {"on": [], "off": []}
    rows = []
    for rnd in range(a.rounds):
        # Alternate which arm leads, so a monotone drift cannot favour one arm.
        order = ["off", "on"] if rnd % 2 == 0 else ["on", "off"]
        for arm in order:
            print(f"  round {rnd} arm {arm}: restarting...", flush=True)
            restart(a.threads, arm == "on")
            lat, ttft = cell(prompt, a.per_cell, a.threads, a.n_predict)
            acc[arm] += lat
            rows.append(dict(round=rnd, arm=arm, n=len(lat),
                             prompt_ms_mean=round(st.mean(lat), 2),
                             prompt_ms_p50=round(pct(lat, .5), 2),
                             prompt_ms_p95=round(pct(lat, .95), 2),
                             client_ttft_p50=round(pct(ttft, .5), 2) if ttft else None))
            print(f"    n={len(lat)} prefill mean {st.mean(lat):.1f} ms "
                  f"p50 {pct(lat,.5):.1f}", flush=True)

    write_rows(a.out, rows)
    on, off = acc["on"], acc["off"]
    m_on, m_off = st.mean(on), st.mean(off)
    rel = m_on / m_off - 1.0
    lo, hi = boot_ci(off, on)
    print(f"\n  OFF n={len(off)} mean {m_off:.2f} ms   p50 {pct(off,.5):.2f}")
    print(f"  ON  n={len(on)} mean {m_on:.2f} ms   p50 {pct(on,.5):.2f}")
    print(f"  relative effect of instrumentation: {rel*100:+.2f}% "
          f"(95% CI {lo*100:+.2f}% .. {hi*100:+.2f}%)")
    verdict = ("LEASE INSTRUMENTATION OVERHEAD NEGLIGIBLE"
               if abs(rel) < 0.01 and abs(lo) < 0.02 and abs(hi) < 0.02
               else "LEASE INSTRUMENTATION OVERHEAD NOT NEGLIGIBLE -- adjust dump frequency")
    print(f"  -> {verdict}")
    with open(a.out.replace(".csv", ".json"), "w") as f:
        json.dump(dict(n_on=len(on), n_off=len(off), mean_on=m_on, mean_off=m_off,
                       rel_effect=rel, ci_lo=lo, ci_hi=hi, verdict=verdict), f, indent=2)


if __name__ == "__main__":
    main()
