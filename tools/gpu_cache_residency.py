#!/usr/bin/env python3
"""Task 7 -- does buying more warm controller domains with RAM harm the GPU worker?

This is why the cache-RAM experiment belongs on THIS machine: the Radeon 8060S is
an iGPU on unified memory, so a large host-side prompt cache and the worker's GTT
allocation come out of the same 122 GiB. A capacity knob that quietly starves the
GPU is not a capacity knob.

The cache FILL phase and the RESIDENT steady-state phase are measured separately.
Filling hundreds of domains is itself a heavy prefill workload; judging GPU
steady-state decode during the fill would measure the fill, not the residency.
"""
import argparse, glob, statistics as st, subprocess, sys, threading, time

sys.path.insert(0, "tools")
from multi_domain import make_domains, calibrate_delta, cell
from service_bench import (run_worker, write_rows, pct, Power, read_int)

RUN = "/tmp/bitnet-service"


def restart_ctrl(cache_ram, t=4, tb=16, b=4096, ub=4096, slots=8, ctx=40960):
    env = dict(CTRL_B=str(b), CTRL_UB=str(ub), CTRL_SLOTS=str(slots),
               CTRL_CTX=str(ctx), CTRL_TB=str(tb), CACHE_RAM=str(cache_ram))
    r = subprocess.run(["env"] + [f"{k}={v}" for k, v in env.items()] +
                       ["bash", "tools/service_ctl.sh", "start-ctrl", str(t)],
                       capture_output=True, text=True, timeout=2400)
    if r.returncode != 0:
        raise RuntimeError(f"ctrl restart failed:\n{r.stdout[-500:]}")
    return r.stdout.strip().splitlines()[-1]


def mem():
    m = {}
    for line in open("/proc/meminfo"):
        p = line.split(":")
        if p[0] in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            m[p[0]] = int(p[1].split()[0]) // 1024
    out = dict(mem_avail_mib=m.get("MemAvailable"),
               swap_used_mib=m.get("SwapTotal", 0) - m.get("SwapFree", 0))
    for p in glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used"):
        try:
            out["gtt_used_mib"] = int(open(p).read().strip()) // (1024 * 1024)
            break
        except Exception:
            pass
    return out


def rss(pidfile):
    try:
        pid = open(pidfile).read().strip()
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS"):
                return round(int(line.split()[1]) / 1024, 1)
    except Exception:
        return None


class GpuDecode:
    """Keep the worker decoding: short prompt, long generation."""

    def __init__(self, active, n_predict=256):
        self.active, self.n_predict = active, n_predict
        self._stop = threading.Event()
        self.rows, self._t = [], None

    def _run(self):
        i = 0
        while not self._stop.is_set():
            try:
                self.rows.append(run_worker(f"gw{i}", 1, self.n_predict).row())
            except Exception:
                pass
            i += 1

    def __enter__(self):
        if self.active:
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
            time.sleep(8)          # reach steady-state decode before measuring
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=120)

    def summary(self):
        ok = [r for r in self.rows if not r.get("err")]
        if not ok:
            return dict(gpu_reqs=0, gpu_gen_tok_s=None)
        gm = [r["gen_ms"] for r in ok if r.get("gen_ms")]
        gn = [r["gen_n"] for r in ok if r.get("gen_n")]
        return dict(gpu_reqs=len(ok),
                    gpu_gen_tok_s=round(sum(gn) / (sum(gm) / 1000), 2)
                    if gm and sum(gm) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="8192,32768")
    ap.add_argument("--capacities", default="58,251",
                    help="measured warm capacity per budget, same order")
    ap.add_argument("--steady-requests", type=int, default=40)
    ap.add_argument("--delta-tokens", type=int, default=128)
    ap.add_argument("--n-predict", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--outdir", default="artifacts/controller-state-envelope")
    a = ap.parse_args()

    budgets = [int(x) for x in a.budgets.split(",")]
    caps = [int(x) for x in a.capacities.split(",")]
    rows = []
    for cr, cap in zip(budgets, caps):
        line = restart_ctrl(cr)
        d0 = make_domains(1)[0]
        lines, got, pre = calibrate_delta(d0, a.delta_tokens)
        doms = make_domains(cap)
        print(f"\n== --cache-ram {cr} MiB, filling {cap} domains :: {line}", flush=True)

        # ---- FILL phase, measured separately ----
        pw = Power()
        with pw:
            t0 = time.perf_counter()
            cell("fill", doms, cap, lines, 1, a.n_predict, a.threads,
                 cache=True, turn0=0)
            fill_s = time.perf_counter() - t0
        m = mem()
        rows.append(dict(cache_ram_mib=cr, domains=cap, phase="fill",
                         gpu="idle", wall_s=round(fill_s, 1),
                         fill_rate_domains_s=round(cap / fill_s, 3),
                         watts=pw.watts, ctrl_rss_mib=rss(f"{RUN}/ctrl.pid"), **m))
        print(f"   FILL: {cap} domains in {fill_s:.0f}s "
              f"({cap/fill_s:.2f} dom/s), {pw.watts}W, "
              f"RSS {rows[-1]['ctrl_rss_mib']} MiB, GTT {m.get('gtt_used_mib')} MiB, "
              f"avail {m['mem_avail_mib']} MiB", flush=True)
        write_rows(f"{a.outdir}/gpu_cache_residency.csv", rows)

        # ---- RESIDENT steady state, GPU idle then decoding ----
        cursor = 1
        for gpu_active in (False, True):
            gd = GpuDecode(gpu_active)
            pw = Power()
            with gd, pw:
                rec, _ = cell(f"steady-{cr}-{'dec' if gpu_active else 'idle'}",
                              doms, a.steady_requests, lines, 1, a.n_predict,
                              a.threads, cache=True, turn0=cursor)
            cursor += a.steady_requests // max(cap, 1) + 2
            m = mem()
            hit = rec["reused_mean"] and rec["reused_mean"] > 200
            row = dict(cache_ram_mib=cr, domains=cap, phase="steady",
                       gpu="decode" if gpu_active else "idle",
                       ttft_p50=rec["ttft_p50"], ttft_p95=rec["ttft_p95"],
                       total_p50=rec["total_p50"], req_per_s=rec["req_per_s"],
                       eval_mean=rec["eval_mean"], reused_mean=rec["reused_mean"],
                       cache_hit=int(bool(hit)), contaminated=rec["contaminated"],
                       watts=pw.watts, ctrl_rss_mib=rss(f"{RUN}/ctrl.pid"), **m)
            row.update(gd.summary())
            rows.append(row)
            print(f"   STEADY gpu={row['gpu']:<6}: ctrl ttft p50 {row['ttft_p50']:>8.1f} "
                  f"p95 {row['ttft_p95']:>8.1f}  reused {row['reused_mean']:>7}  "
                  f"gpu_tg {row.get('gpu_gen_tok_s')}  {row['watts']}W  "
                  f"GTT {m.get('gtt_used_mib')}  avail {m['mem_avail_mib']}  "
                  f"swap {m['swap_used_mib']}", flush=True)
            write_rows(f"{a.outdir}/gpu_cache_residency.csv", rows)
    print(f"\nwrote {a.outdir}/gpu_cache_residency.csv")


if __name__ == "__main__":
    main()
