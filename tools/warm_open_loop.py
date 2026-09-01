#!/usr/bin/env python3
"""Task 5 -- open-loop characterization of the REAL warm multi-domain workload.

Closed-loop tests find saturation but cannot describe a service: with a fixed
number of in-flight requests a slower service simply receives fewer arrivals, so
the offered load depends on the latency being measured (coordinated omission).

Arrival schedule uses the construction corrected on service-batching-gate:
conditioned on N arrivals in [0, T], a Poisson process places them as N i.i.d.
Uniform(0, T) order statistics. Fixing N = round(rate x T) keeps the clustering
while making the offered rate EXACTLY what the label claims -- drawing
exponential gaps instead lets N vary with the seed, which previously mislabelled
an arm by 32%.

Latency is charged from the SCHEDULED arrival, never from when a client thread
woke up.
"""
import argparse, random, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "tools")
from multi_domain import make_domains, calibrate_delta, cell, contamination_check
from service_bench import (run_controller, write_rows, pct, assert_timing_sane,
                           append_jsonl, Power)


def restart(cache_ram, t=4, tb=16, b=4096, ub=4096, slots=8, ctx=40960):
    env = dict(CTRL_B=str(b), CTRL_UB=str(ub), CTRL_SLOTS=str(slots),
               CTRL_CTX=str(ctx), CTRL_TB=str(tb), CACHE_RAM=str(cache_ram))
    r = subprocess.run(["env"] + [f"{k}={v}" for k, v in env.items()] +
                       ["bash", "tools/service_ctl.sh", "start-ctrl", str(t)],
                       capture_output=True, text=True, timeout=2400)
    if r.returncode != 0:
        raise RuntimeError(f"restart failed:\n{r.stdout[-500:]}")
    return r.stdout.strip().splitlines()[-1]


def arm(domains, n_delta, rate, duration, n_predict, threads, turn0, jsonl=None):
    rng = random.Random(4242)
    n = max(1, int(round(rate * duration)))
    sched = sorted(rng.uniform(0.0, duration) for _ in range(n))
    tags = {d.tag for d in domains}
    rows, lock = [], threading.Lock()
    inflight = {"n": 0, "max": 0}
    t0 = time.perf_counter()

    def fire(i, due):
        with lock:
            inflight["n"] += 1
            inflight["max"] = max(inflight["max"], inflight["n"])
        d = domains[i % len(domains)]
        try:
            r = run_controller(f"ol{i}", threads, 0, n_predict,
                               d.prompt(turn0 + i // len(domains), n_delta),
                               cache=True, capture_text=True)
            row = r.row()
            if not row.get("err"):
                row["contaminated"] = int(bool(
                    contamination_check(r.chain.get("text", ""), d.tag, tags)))
        except Exception as e:
            row = {"err": f"{type(e).__name__}"}
        finally:
            with lock:
                inflight["n"] -= 1
        done = time.perf_counter() - t0
        row.update(domain=d.idx, sched_s=round(due, 3), done_s=round(done, 3))
        if not row.get("err"):
            # From the SCHEDULED arrival: a late start is charged to the service.
            row["ol_total_ms"] = round((done - due) * 1000, 2)
            delay = (done - due) * 1000 - (row.get("total_ms") or 0)
            row["ol_ttft_ms"] = round((row.get("ttft_ms") or 0) + max(delay, 0.0), 2)
        with lock:
            rows.append(row)

    with ThreadPoolExecutor(max_workers=max(16, n)) as ex:
        futs = []
        for i, due in enumerate(sched):
            now = time.perf_counter() - t0
            if due > now:
                time.sleep(due - now)
            futs.append(ex.submit(fire, i, due))
        for f in futs:
            f.result()
    wall = time.perf_counter() - t0
    if jsonl:
        append_jsonl(jsonl, rows)
    ok = [r for r in rows if not r.get("err")]
    good, _ = assert_timing_sane(ok, "openloop")
    hits = [r for r in good if (r.get("reused_n") or 0) > 200]
    tt = [r["ol_ttft_ms"] for r in good if r.get("ol_ttft_ms") is not None]
    to = [r["ol_total_ms"] for r in good if r.get("ol_total_ms") is not None]
    return dict(scheduled=n, completed=len(ok), errors=len(rows) - len(ok),
                usable=len(good), wall_s=round(wall, 1),
                completed_rps=round(len(ok) / wall, 3),
                cache_hit_rate=round(len(hits) / max(len(good), 1), 3),
                inflight_max=inflight["max"],
                ttft_p50=pct(tt, .5), ttft_p95=pct(tt, .95),
                total_p50=pct(to, .5), total_p95=pct(to, .95),
                total_max=round(max(to), 1) if to else None,
                contaminated=sum(r.get("contaminated", 0) for r in good))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=float, required=True)
    ap.add_argument("--fracs", default="0.25,0.5,0.75,0.9")
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--domains", type=int, default=32)
    ap.add_argument("--delta-tokens", type=int, default=128)
    ap.add_argument("--n-predict", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cache-ram", type=int, default=8192)
    ap.add_argument("--outdir", default="artifacts/controller-state-envelope")
    a = ap.parse_args()

    line = restart(a.cache_ram)
    print(f"== {line}\n")
    doms = make_domains(a.domains)
    lines, got, pre = calibrate_delta(doms[0], a.delta_tokens)
    cell("warm", doms, a.domains, lines, 1, a.n_predict, a.threads,
         cache=True, turn0=0)
    print(f"   {a.domains} warm domains, prefix {pre} tok, delta {got} tok, "
          f"output {a.n_predict} tok, capacity {a.capacity} rps\n")
    rows, cursor = [], 1
    for frac in [float(x) for x in a.fracs.split(",")]:
        rate = a.capacity * frac
        pw = Power()
        with pw:
            rec = arm(doms, lines, rate, a.duration, a.n_predict, a.threads,
                      cursor, f"{a.outdir}/warm_open_loop_requests.jsonl")
        cursor += rec["scheduled"] // a.domains + 2
        rec.update(frac_of_capacity=frac, offered_rps=round(rate, 3),
                   domains=a.domains, delta_tokens=got, watts=pw.watts)
        rows.append(rec)
        print(f"  {frac:>5.0%}  offered {rate:>6.3f}/s  completed "
              f"{rec['completed_rps']:>6.3f}/s  n={rec['usable']:>4}  "
              f"hit {rec['cache_hit_rate']:>5.1%}  ttft p50 {rec['ttft_p50']:>8.1f} "
              f"p95 {rec['ttft_p95']:>9.1f}  total p95 {rec['total_p95']:>9.1f}  "
              f"inflight_max {rec['inflight_max']:>3}  err {rec['errors']}", flush=True)
        write_rows(f"{a.outdir}/warm_open_loop.csv", rows)
    print(f"\nwrote {a.outdir}/warm_open_loop.csv")


if __name__ == "__main__":
    main()
