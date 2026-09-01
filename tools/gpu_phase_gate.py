#!/usr/bin/env python3
"""Task 10 -- does GPU PREFILL pressure differ from GPU DECODE pressure?

service-cotenancy concluded "phase-aware thread policy not justified", but its
mixed workload gave the GPU a ~128-token prompt and a long generation. That is
almost entirely GPU *decode*, so the conclusion was drawn without ever putting
the controller under sustained GPU *prefill*.

The two phases stress different things. GPU decode is latency-bound on a small
weight-streaming loop; GPU prefill is a large dense matmul that competes with
the controller for exactly the resource §8 showed the controller is limited by --
memory bandwidth. If any phase favours narrower controller threads, it is this one.

Arms:
  A  GPU PREFILL pressure -- ~2K and ~8K worker prompts, very short generation
  B  GPU DECODE  pressure -- short worker prompt, long generation
  idle                    -- no worker load, as the control

against controller widths t4/t6/t8 (all at tb16, the measured optimum).
"""
import argparse, statistics as st, subprocess, sys, threading, time

sys.path.insert(0, "tools")
from service_bench import (run_controller, run_worker, controller_prompt, write_rows,
                           pct, Ne11Window, LeaseWindow, Power, start_verifier,
                           stop_verifier)

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


def worker_prompt(approx_tokens):
    """A long coding-shaped prompt, sized by repetition."""
    unit = ("def handler_{i}(request, ctx):\n"
            "    # validate, authorize, then dispatch to the service layer\n"
            "    if not ctx.authorized(request.principal):\n"
            "        raise Forbidden(request.principal)\n"
            "    return dispatch(request, timeout={t})\n\n")
    out, i = ["Review the following module and summarize its structure.\n\n"], 0
    while sum(len(x) for x in out) < approx_tokens * 3:
        out.append(unit.format(i=i, t=100 + (i * 7) % 900)); i += 1
    out.append("\nSummary:")
    return "".join(out)


class WorkerLoad:
    """Keep the GPU busy in one phase for the whole controller cell."""

    def __init__(self, phase, tokens=2048, n_predict=8):
        self.phase, self.tokens, self.n_predict = phase, tokens, n_predict
        self._stop = threading.Event()
        self.rows = []
        self._t = None

    def _run(self):
        p = worker_prompt(self.tokens)
        i = 0
        while not self._stop.is_set():
            try:
                r = run_worker(f"w{i}", 1, self.n_predict, prompt=p).row()
                self.rows.append(r)
            except Exception:
                pass
            i += 1

    def __enter__(self):
        if self.phase != "idle":
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
            time.sleep(6)          # let the GPU reach the phase before measuring
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=90)

    def summary(self):
        ok = [r for r in self.rows if not r.get("err")]
        if not ok:
            return dict(gpu_reqs=0)
        pm = [r["prompt_ms"] for r in ok if r.get("prompt_ms")]
        pn = [r["prompt_n"] for r in ok if r.get("prompt_n")]
        gm = [r["gen_ms"] for r in ok if r.get("gen_ms")]
        gn = [r["gen_n"] for r in ok if r.get("gen_n")]
        return dict(gpu_reqs=len(ok),
                    gpu_prompt_tok_s=round(sum(pn) / (sum(pm) / 1000), 1) if pm and sum(pm) else None,
                    gpu_gen_tok_s=round(sum(gn) / (sum(gm) / 1000), 2) if gm and sum(gm) else None,
                    gpu_prompt_n_mean=round(st.mean(pn), 0) if pn else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="4,6,8")
    ap.add_argument("--tb", type=int, default=16)
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--n-predict", type=int, default=8)
    ap.add_argument("--verifier", action="store_true", default=True)
    ap.add_argument("--out", default="artifacts/service-batching-gate/gpu_phase.csv")
    a = ap.parse_args()

    prompt = controller_prompt(1954)
    phases = [("idle", 0, 0),
              ("prefill-2k", 2048, 4),
              ("prefill-8k", 8192, 4),
              ("decode", 256, 256)]
    rows = []
    for t in [int(x) for x in a.widths.split(",")]:
        line = restart(t, a.tb)
        print(f"\n== t={t} tb={a.tb} :: {line}", flush=True)
        for phase, wtok, wpred in phases:
            v = start_verifier() if a.verifier else None
            pw = Power()
            with WorkerLoad(phase, wtok, wpred) as wl, \
                 Ne11Window(NE11) as nw, LeaseWindow(LEASE) as lw, pw:
                t0 = time.perf_counter()
                out = [run_controller(f"g{i}", t, 1, a.n_predict, prompt).row()
                       for i in range(a.per_cell)]
                wall = time.perf_counter() - t0
                time.sleep(0.4)
            vs = stop_verifier(v) if v else {}
            ok = [r for r in out if not r.get("err")]
            ttft = [r["client_ttft_ms"] for r in ok if r.get("client_ttft_ms")]
            tot = [r["total_ms"] for r in ok]
            rec = dict(t=t, tb=a.tb, phase=phase, worker_prompt_tokens=wtok,
                       worker_n_predict=wpred, requests=len(ok),
                       req_per_s=round(len(ok) / wall, 3),
                       ttft_p50=pct(ttft, .5), ttft_p95=pct(ttft, .95),
                       total_p50=pct(tot, .5), total_p95=pct(tot, .95),
                       watts=pw.watts, gpu_busy_med=pw.gpu_busy_med)
            rec.update(wl.summary()); rec.update(nw.delta()); rec.update(lw.delta())
            rec.update(vs or {})
            rows.append(rec)
            print(f"   {phase:<12} ctrl ttft p50 {rec['ttft_p50']:>8.1f}  p95 "
                  f"{rec['ttft_p95']:>8.1f}  gpu_pp {rec.get('gpu_prompt_tok_s')}  "
                  f"gpu_tg {rec.get('gpu_gen_tok_s')}  {rec['watts']}W", flush=True)
            write_rows(a.out, rows)
    write_rows(a.out, rows)
    print(f"\nwrote {a.out} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
