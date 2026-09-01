#!/usr/bin/env python3
"""Task 12 -- open-loop arrival characterization on the winning configuration.

The closed-loop tests are right for finding saturation but wrong for describing
a service: with a fixed number of in-flight requests, a slow service simply
receives fewer arrivals, so the load offered depends on the latency being
measured. That is coordinated omission, and it hides exactly the queue growth an
admission-control threshold has to be set against.

Here arrivals follow a schedule fixed BEFORE the run, from a Poisson process at
a chosen fraction of measured capacity. A request that cannot start on time
still counts its wait from its SCHEDULED arrival, not from when a worker picked
it up.
"""
import argparse, json, random, statistics as st, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "tools")
from service_bench import (run_controller, controller_prompt, write_rows, pct,
                           Power, append_jsonl)


def run_arm(prompt, rate, duration, threads, n_predict, timeout_s):
    """Fire requests on a pre-computed Poisson schedule."""
    # Conditioned on N arrivals in [0, T], a Poisson process places them as N
    # i.i.d. Uniform(0, T) order statistics. Drawing exponential gaps instead
    # lets N vary with the seed: at rate 0.3325 over 150 s the mean is 49.5 with
    # sd 8.1, and seed 4242 drew 66 -- so an arm labelled "50% of capacity"
    # actually offered 66%. Fixing N to round(rate*T) keeps the clustering and
    # makes the offered rate exactly what the label claims.
    rng = random.Random(4242)
    n_arrivals = max(1, int(round(rate * duration)))
    sched = sorted(rng.uniform(0.0, duration) for _ in range(n_arrivals))

    rows, lock = [], threading.Lock()
    inflight = {"n": 0, "max": 0}
    t0 = time.perf_counter()

    def fire(i, due):
        with lock:
            inflight["n"] += 1
            inflight["max"] = max(inflight["max"], inflight["n"])
        try:
            r = run_controller(f"o{i}", threads, 0, n_predict, prompt).row()
        except Exception as e:
            r = {"err": f"{type(e).__name__}"}
        finally:
            with lock:
                inflight["n"] -= 1
        done = time.perf_counter() - t0
        # Latency from the SCHEDULED arrival, so a late start is charged to the
        # service rather than silently removed from the distribution.
        r["sched_s"] = round(due, 3)
        r["done_s"] = round(done, 3)
        r["ol_total_ms"] = round((done - due) * 1000, 2)
        if r.get("client_ttft_ms") is not None and r.get("total_ms") is not None:
            start_delay = (done - due) * 1000 - r["total_ms"]
            r["ol_ttft_ms"] = round(r["client_ttft_ms"] + max(start_delay, 0.0), 2)
        with lock:
            rows.append(r)

    with ThreadPoolExecutor(max_workers=max(8, int(rate * timeout_s) + 8)) as ex:
        futs = []
        for i, due in enumerate(sched):
            now = time.perf_counter() - t0
            if due > now:
                time.sleep(due - now)
            futs.append(ex.submit(fire, i, due))
        for f in futs:
            f.result()
    return rows, len(sched), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=float, required=True,
                    help="measured sustainable req/s of the winning config")
    ap.add_argument("--fracs", default="0.5,0.75,0.9,1.0,1.1")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-predict", type=int, default=8)
    ap.add_argument("--timeout-s", type=float, default=60.0)
    ap.add_argument("--out", default="artifacts/service-batching-gate/open_loop.csv")
    a = ap.parse_args()

    prompt = controller_prompt(1954)
    rows = []
    for frac in [float(x) for x in a.fracs.split(",")]:
        rate = a.capacity * frac
        pw = Power()
        with pw:
            rr, offered, wall = run_arm(prompt, rate, a.duration, a.threads,
                                        a.n_predict, a.timeout_s)
        ok = [r for r in rr if not r.get("err")]
        errs = len(rr) - len(ok)
        ttft = [r["ol_ttft_ms"] for r in ok if r.get("ol_ttft_ms") is not None]
        tot = [r["ol_total_ms"] for r in ok if r.get("ol_total_ms") is not None]
        rec = dict(frac_of_capacity=frac, offered_rps=round(rate, 3),
                   scheduled=offered, completed=len(ok), errors=errs,
                   wall_s=round(wall, 1),
                   completed_rps=round(len(ok) / wall, 3),
                   ttft_p50=pct(ttft, .5), ttft_p95=pct(ttft, .95),
                   ttft_p99=pct(ttft, .99) if len(ttft) >= 100 else None,
                   total_p50=pct(tot, .5), total_p95=pct(tot, .95),
                   total_p99=pct(tot, .99) if len(tot) >= 100 else None,
                   total_max=round(max(tot), 1) if tot else None,
                   watts=pw.watts)
        rows.append(rec)
        append_jsonl(a.out.replace(".csv", "_requests.jsonl"),
                     [dict(r, frac=frac) for r in ok])
        print(f"  {frac:>5.0%} of capacity  offered {rate:.3f}/s  "
              f"completed {rec['completed_rps']:.3f}/s  n={len(ok)}  "
              f"ttft p50 {rec['ttft_p50']:>8.1f} p95 {rec['ttft_p95']:>9.1f}  "
              f"total p95 {rec['total_p95']:>9.1f} max {rec['total_max']:>9.1f}  "
              f"err {errs}", flush=True)
        write_rows(a.out, rows)
    write_rows(a.out, rows)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
