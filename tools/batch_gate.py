#!/usr/bin/env python3
"""Tasks 5 and 6 -- is controller concurrency ~1 caused by -b/-ub 2048?

service-cotenancy measured concurrency at -b 2048 -ub 2048 with 1954-token
controller prompts. Two such prompts are 3908 prompt tokens and cannot share one
2048-token physical batch, so "concurrency 1" may be a statement about that
ceiling rather than about the hardware.

Task 5 sweeps the batch ceiling against slot count and client concurrency.
Task 6 adds a mechanism oracle: the pinned /completion accepts a LIST of
prompts, so an explicit two-prompt batch can be compared against two independent
simultaneous requests. If explicit batching wins and independent requests do
not, the limit is server batch formation, not the graph or the runtime.

The decisive instrument is the ne11 histogram, not throughput: it shows directly
whether offloaded work ever appears in the [2048, 4096) bucket, i.e. whether two
requests were ever combined into one graph.
"""
import argparse, json, statistics as st, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "tools")
from service_bench import (run_controller, controller_prompt, write_rows, pct,
                           Ne11Window, LeaseWindow, SlotSampler, Power, CTRL, post)

NE11 = "/tmp/bitnet-service/ne11.csv"
LEASE = "/tmp/bitnet-service/lease.csv"


def restart(threads, b, ub, np_slots, tb=None):
    env = dict(CTRL_B=str(b), CTRL_UB=str(ub), CTRL_SLOTS=str(np_slots),
               NE11_STATS="1", NE11_CSV=NE11, NE11_EVERY="210")
    if tb:
        env["CTRL_TB"] = str(tb)
    cmd = ["env"] + [f"{k}={v}" for k, v in env.items()] + \
          ["bash", "tools/service_ctl.sh", "start-ctrl", str(threads)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"restart failed: {r.stdout[-500:]} {r.stderr[-500:]}")
    return r.stdout.strip().splitlines()[-1]


def closed_loop(prompt, n, conc, threads, n_predict):
    """n requests through a fixed number of in-flight slots."""
    rows, t0 = [], time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(run_controller, f"b{i}", threads, conc, n_predict, prompt)
                for i in range(n)]
        for f in futs:
            rows.append(f.result().row())
    return rows, time.perf_counter() - t0


def multiprompt(prompts, n_predict, timeout=900):
    """Task 6 arm B: ONE request carrying several independent prompts."""
    t0 = time.perf_counter()
    d = post(CTRL, "/completion",
             dict(prompt=prompts, n_predict=n_predict, temperature=0,
                  seed=42, cache_prompt=False), timeout=timeout)
    return d, (time.perf_counter() - t0) * 1000


def measure(label, fn, threads, conc, extra=None):
    """Run one cell inside every instrument at once."""
    pw = Power()
    with Ne11Window(NE11) as nw, LeaseWindow(LEASE) as lw, \
         SlotSampler() as ss, pw:
        out = fn()
        time.sleep(0.5)          # let the runtime flush a final snapshot
    rec = dict(cell=label, threads=threads, concurrency=conc)
    rec.update(nw.delta()); rec.update(lw.delta()); rec.update(ss.summary())
    rec.update(watts=pw.watts, gpu_busy_med=pw.gpu_busy_med)
    if extra:
        rec.update(extra)
    return rec, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--per-cell", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=1954)
    ap.add_argument("--configs", default="2048:2048,4096:4096",
                    help="comma list of b:ub")
    ap.add_argument("--slots", default="1,2,8")
    ap.add_argument("--conc", default="1,2,4")
    ap.add_argument("--out", default="artifacts/service-batching-gate/batch_gate.csv")
    ap.add_argument("--oracle", action="store_true",
                    help="also run the Task 6 explicit multi-prompt oracle")
    a = ap.parse_args()

    prompt = controller_prompt(a.tokens)
    configs = [tuple(int(x) for x in c.split(":")) for c in a.configs.split(",")]
    slots = [int(x) for x in a.slots.split(",")]
    concs = [int(x) for x in a.conc.split(",")]
    rows = []

    for (b, ub) in configs:
        for ns in slots:
            line = restart(a.threads, b, ub, ns)
            print(f"\n== b={b} ub={ub} np={ns} :: {line}", flush=True)
            for c in concs:
                if c > ns:
                    continue          # more in-flight than slots is a queue test, not a batch test
                label = f"b{b}-ub{ub}-np{ns}-c{c}"
                (rec, (rr, wall)) = measure(
                    label, lambda: closed_loop(prompt, a.per_cell, c, a.threads, a.n_predict),
                    a.threads, c)
                ok = [r for r in rr if not r.get("err")]
                ttft = [r["client_ttft_ms"] for r in ok if r.get("client_ttft_ms")]
                tot = [r["total_ms"] for r in ok]
                pm = [r["prompt_ms"] for r in ok if r.get("prompt_ms")]
                gm = [r["gen_ms"] for r in ok if r.get("gen_ms")]
                rec.update(b=b, ub=ub, slots=ns, requests=len(ok), errors=len(rr) - len(ok),
                           wall_s=round(wall, 2), req_per_s=round(len(ok) / wall, 3),
                           ttft_p50=pct(ttft, .5), ttft_p95=pct(ttft, .95),
                           total_p50=pct(tot, .5), total_p95=pct(tot, .95),
                           prompt_ms_mean=round(st.mean(pm), 1) if pm else None,
                           gen_ms_mean=round(st.mean(gm), 1) if gm else None,
                           mode="independent")
                rows.append(rec)
                print(f"   {label:<26} {rec['req_per_s']:>6.3f} req/s  "
                      f"ttft p50 {rec['ttft_p50']:>8.1f}  total p95 {rec['total_p95']:>9.1f}  "
                      f"ne11_max {rec.get('ne11_max_offloaded_bucket')}  "
                      f"hist {rec.get('ne11_offloaded_hist')}", flush=True)
                write_rows(a.out, rows)

            if a.oracle and ns >= 2:
                # Task 6: two independent prompts in ONE request vs two requests.
                for k in (2, 4):
                    if k > ns:
                        continue
                    label = f"b{b}-ub{ub}-np{ns}-ORACLE{k}"
                    (rec, (d, ms)) = measure(
                        label, lambda: multiprompt([prompt] * k, a.n_predict),
                        a.threads, k)
                    # A multi-prompt /completion returns a LIST, one result per
                    # prompt, each with its own timings block.
                    res = d if isinstance(d, list) else [d]
                    pms = [x.get("timings", {}).get("prompt_ms")
                           for x in res if x.get("timings", {}).get("prompt_ms")]
                    pns = [x.get("timings", {}).get("prompt_n")
                           for x in res if x.get("timings", {}).get("prompt_n")]
                    rec.update(b=b, ub=ub, slots=ns, requests=len(res), errors=0,
                               wall_s=round(ms / 1000, 2),
                               req_per_s=round(len(res) / (ms / 1000), 3),
                               total_p50=round(ms, 1),
                               prompt_ms_mean=round(st.mean(pms), 1) if pms else None,
                               prompt_ms_max=round(max(pms), 1) if pms else None,
                               prompt_n_total=sum(pns) if pns else None,
                               mode="multiprompt")
                    rows.append(rec)
                    print(f"   {label:<26} {rec['req_per_s']:>6.3f} req/s  "
                          f"wall {ms:>9.1f} ms  ne11_max "
                          f"{rec.get('ne11_max_offloaded_bucket')}  "
                          f"hist {rec.get('ne11_offloaded_hist')}", flush=True)
                    write_rows(a.out, rows)

    write_rows(a.out, rows)
    print(f"\nwrote {a.out} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
