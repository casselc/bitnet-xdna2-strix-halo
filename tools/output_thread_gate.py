#!/usr/bin/env python3
"""Tasks 7 and 8 -- controller output length, and prompt vs generation threads.

Task 7: the prior benchmark generated 32 tokens. A controller emitting a short
constrained action may sit in a different regime entirely, so 1/4/8/32 output
tokens are compared at the winning batch config. Sampling is deterministic; the
benchmark deliberately does not depend on the model choosing a *good* action.

Task 8: the launcher tied -t and -tb together (-tb defaults to -t), but prompt
processing is the long pole -- ~2.35 s of a ~2.9 s request. Wide prompt threads
with narrower decode threads should give t8-like TTFT without t8-like sustained
CPU pressure, which matters because the controller shares this box with a GPU
worker and a CPU verifier.

Widths are chosen against the measured topology (16C/32T), not swept blindly.
"""
import argparse, statistics as st, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "tools")
from service_bench import (run_controller, controller_prompt, write_rows, pct,
                           Ne11Window, LeaseWindow, SlotSampler, Power)

NE11 = "/tmp/bitnet-service/ne11.csv"
LEASE = "/tmp/bitnet-service/lease.csv"


def restart(t, tb, b, ub, np_slots):
    env = dict(CTRL_B=str(b), CTRL_UB=str(ub), CTRL_SLOTS=str(np_slots),
               NE11_STATS="1", NE11_CSV=NE11, NE11_EVERY="210")
    if tb:
        env["CTRL_TB"] = str(tb)
    cmd = ["env"] + [f"{k}={v}" for k, v in env.items()] + \
          ["bash", "tools/service_ctl.sh", "start-ctrl", str(t)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"restart failed: {r.stdout[-500:]} {r.stderr[-500:]}")
    return r.stdout.strip().splitlines()[-1]


def cell(label, prompt, n, conc, threads, n_predict, extra=None):
    pw = Power()
    with Ne11Window(NE11) as nw, LeaseWindow(LEASE) as lw, SlotSampler() as ss, pw:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=conc) as ex:
            rows = [f.result().row() for f in
                    [ex.submit(run_controller, f"x{i}", threads, conc, n_predict, prompt)
                     for i in range(n)]]
        wall = time.perf_counter() - t0
        time.sleep(0.4)
    ok = [r for r in rows if not r.get("err")]
    ttft = [r["client_ttft_ms"] for r in ok if r.get("client_ttft_ms")]
    tot = [r["total_ms"] for r in ok]
    pm = [r["prompt_ms"] for r in ok if r.get("prompt_ms")]
    gm = [r["gen_ms"] for r in ok if r.get("gen_ms")]
    rec = dict(cell=label, threads=threads, concurrency=conc, n_predict=n_predict,
               requests=len(ok), errors=len(rows) - len(ok), wall_s=round(wall, 2),
               req_per_s=round(len(ok) / wall, 3),
               ttft_p50=pct(ttft, .5), ttft_p95=pct(ttft, .95),
               total_p50=pct(tot, .5), total_p95=pct(tot, .95),
               prompt_ms_mean=round(st.mean(pm), 1) if pm else None,
               gen_ms_mean=round(st.mean(gm), 1) if gm else None,
               watts=pw.watts, gpu_busy_med=pw.gpu_busy_med)
    rec.update(nw.delta()); rec.update(lw.delta()); rec.update(ss.summary())
    if extra:
        rec.update(extra)
    return rec, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("output", "threads"), required=True)
    ap.add_argument("--b", type=int, default=4096)
    ap.add_argument("--ub", type=int, default=4096)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--per-cell", type=int, default=8)
    ap.add_argument("--conc", default="1,2,4")
    ap.add_argument("--predicts", default="1,4,8,32")
    ap.add_argument("--pairs", default="4:8,6:8,6:12,8:8",
                    help="t:tb pairs for --mode threads")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    prompt = controller_prompt(1954)
    concs = [int(x) for x in a.conc.split(",")]
    rows = []

    if a.mode == "output":
        out = a.out or "artifacts/service-batching-gate/short_output.csv"
        line = restart(a.threads, None, a.b, a.ub, a.slots)
        print(f"== {line}\n", flush=True)
        for npd in [int(x) for x in a.predicts.split(",")]:
            for c in concs:
                lbl = f"npred{npd}-c{c}"
                rec, _ = cell(lbl, prompt, a.per_cell, c, a.threads, npd,
                              dict(b=a.b, ub=a.ub, slots=a.slots, tb=None))
                rows.append(rec)
                print(f"  {lbl:<16} {rec['req_per_s']:>6.3f} req/s  "
                      f"ttft p50 {rec['ttft_p50']:>8.1f}  total p50 {rec['total_p50']:>8.1f}  "
                      f"p95 {rec['total_p95']:>9.1f}  gen {rec['gen_ms_mean']}  "
                      f"{rec['watts']}W", flush=True)
                write_rows(out, rows)
    else:
        out = a.out or "artifacts/service-batching-gate/thread_split.csv"
        for pair in a.pairs.split(","):
            t, tb = (int(x) for x in pair.split(":"))
            line = restart(t, tb, a.b, a.ub, a.slots)
            print(f"\n== t={t} tb={tb} :: {line}", flush=True)
            for c in concs:
                lbl = f"t{t}-tb{tb}-c{c}"
                rec, _ = cell(lbl, prompt, a.per_cell, c, t, 32,
                              dict(b=a.b, ub=a.ub, slots=a.slots, tb=tb, t=t))
                rows.append(rec)
                print(f"  {lbl:<16} {rec['req_per_s']:>6.3f} req/s  "
                      f"ttft p50 {rec['ttft_p50']:>8.1f}  prompt {rec['prompt_ms_mean']:>8.1f}  "
                      f"gen {rec['gen_ms_mean']:>7.1f}  total p95 {rec['total_p95']:>9.1f}  "
                      f"{rec['watts']}W", flush=True)
                write_rows(out, rows)
    write_rows(out, rows)
    print(f"\nwrote {out} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
