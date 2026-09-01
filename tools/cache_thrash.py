#!/usr/bin/env python3
"""Task 8 -- characterize the LRU cliff, do not build an admission controller.

controller-state-scheduler found that a cyclic scan over an oversized working set
collapses to a 0% hit rate rather than degrading proportionally. Reproduce only
enough to confirm it under THIS configuration and cache budget, and to derive the
rule an admission controller would need.

Two access patterns at each working-set size:

  random      -- realistic: independent draws over the working set
  cyclic      -- adversarial: round-robin, the pattern that defeats LRU exactly

Below capacity both should hit. Above capacity the interesting question is
whether random degrades gracefully while cyclic collapses.
"""
import argparse, random, subprocess, sys, time

sys.path.insert(0, "tools")
from multi_domain import make_domains, calibrate_delta, cell
from service_bench import write_rows, pct


def restart(cache_ram, t=4, tb=16, b=4096, ub=4096, slots=8, ctx=40960):
    env = dict(CTRL_B=str(b), CTRL_UB=str(ub), CTRL_SLOTS=str(slots),
               CTRL_CTX=str(ctx), CTRL_TB=str(tb), CACHE_RAM=str(cache_ram))
    r = subprocess.run(["env"] + [f"{k}={v}" for k, v in env.items()] +
                       ["bash", "tools/service_ctl.sh", "start-ctrl", str(t)],
                       capture_output=True, text=True, timeout=2400)
    if r.returncode != 0:
        raise RuntimeError(f"restart failed:\n{r.stdout[-500:]}")
    return r.stdout.strip().splitlines()[-1]


def run_pattern(domains, order, n_delta, n_predict, threads, turn0, jsonl=None):
    """Issue one request per entry of `order` (indices into domains)."""
    from multi_domain import contamination_check
    from service_bench import run_controller, assert_timing_sane, append_jsonl
    rows = []
    t0 = time.perf_counter()
    for k, i in enumerate(order):
        d = domains[i]
        r = run_controller(f"th{k}", threads, 1, n_predict,
                           d.prompt(turn0 + k, n_delta), cache=True,
                           capture_text=True)
        row = r.row(); row.update(domain=i, tag=d.tag)
        rows.append(row)
    wall = time.perf_counter() - t0
    good, bad = assert_timing_sane(rows, "thrash")
    if jsonl:
        append_jsonl(jsonl, rows)
    hits = [r for r in good if (r.get("reused_n") or 0) > 200]
    tt = [r["ttft_ms"] for r in good]
    return dict(requests=len(rows), usable=len(good), excluded=len(bad),
                hit_rate=round(len(hits) / max(len(good), 1), 3),
                ttft_p50=pct(tt, .5), ttft_p95=pct(tt, .95),
                req_per_s=round(len(good) / wall, 3), wall_s=round(wall, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-ram", type=int, default=8192)
    ap.add_argument("--capacity", type=int, required=True,
                    help="measured capacity at this budget (from cache_ram.csv)")
    ap.add_argument("--below", type=float, default=0.75)
    ap.add_argument("--above", type=float, default=1.25)
    ap.add_argument("--requests", type=int, default=120)
    ap.add_argument("--delta-tokens", type=int, default=128)
    ap.add_argument("--n-predict", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--outdir", default="artifacts/controller-state-envelope")
    a = ap.parse_args()

    line = restart(a.cache_ram)
    print(f"== {line}\n   measured capacity {a.capacity} domains\n")
    d0 = make_domains(1)[0]
    lines, got, pre = calibrate_delta(d0, a.delta_tokens)
    rows = []
    rng = random.Random(20260901)
    for frac, tag in ((a.below, "below capacity"), (a.above, "above capacity")):
        ws = max(2, int(round(a.capacity * frac)))
        doms = make_domains(ws)
        cell("warm", doms, ws, lines, 1, a.n_predict, a.threads,
             cache=True, turn0=0)
        for pattern in ("random", "cyclic"):
            if pattern == "cyclic":
                order = [i % ws for i in range(a.requests)]
            else:
                order = [rng.randrange(ws) for _ in range(a.requests)]
            rec = run_pattern(doms, order, lines, a.n_predict, a.threads,
                              turn0=1000,
                              jsonl=f"{a.outdir}/cache_thrash_requests.jsonl")
            rec.update(cache_ram_mib=a.cache_ram, capacity=a.capacity,
                       working_set=ws, frac_of_capacity=frac, regime=tag,
                       pattern=pattern, delta_tokens=got)
            rows.append(rec)
            print(f"  {tag:<16} ws={ws:<4} {pattern:<7}: hit {rec['hit_rate']:>5.1%}  "
                  f"ttft p50 {rec['ttft_p50']:>8.1f} p95 {rec['ttft_p95']:>9.1f}  "
                  f"{rec['req_per_s']:>6.3f} rps", flush=True)
            write_rows(f"{a.outdir}/cache_thrash.csv", rows)
    print(f"\nwrote {a.outdir}/cache_thrash.csv")


if __name__ == "__main__":
    main()
