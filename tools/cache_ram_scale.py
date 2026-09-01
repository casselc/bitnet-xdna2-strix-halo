#!/usr/bin/env python3
"""Task 6 -- does --cache-ram scale warm-domain capacity as the KV model predicts?

The mechanism is already understood: residency is bounded by the server's RAM
prompt cache (--cache-ram, default 8192 MiB), not by slot count, and the implied
per-state size agrees with KV geometry to ~1%. What is NOT yet validated is the
DEPLOYMENT KNOB: does raising the budget actually buy proportionally more warm
domains on this box, and where does something else become the limit?

Capacity is predicted BEFORE each run from the measured per-state size, then
probed around the prediction rather than scanned exhaustively.

Probe order matters. Walking domains in warm order collapses to a 0% hit rate at
any oversized working set -- that is LRU thrash, not capacity, because
revisiting an evicted domain evicts the one you were about to visit. Probing
NEWEST -> OLDEST and stopping at the first run of misses measures the resident
set with almost no cascade.

If every probed domain stays warm the result is reported as "CAPACITY > N", never
as "capacity = N": treating the largest tested N as the capacity is an error this
project has already made once.
"""
import argparse, json, math, subprocess, sys, time

sys.path.insert(0, "tools")
from multi_domain import make_domains, calibrate_delta, cell
from service_bench import write_rows, pct, count_tokens

RUN = "/tmp/bitnet-service"
KV_KIB_PER_TOKEN = 5 * 128 * 30 * 2 * 2 / 1024


def restart(cache_ram, t=4, tb=16, b=4096, ub=4096, slots=8, ctx=40960):
    env = dict(CTRL_B=str(b), CTRL_UB=str(ub), CTRL_SLOTS=str(slots),
               CTRL_CTX=str(ctx), CTRL_TB=str(tb), CACHE_RAM=str(cache_ram))
    r = subprocess.run(["env"] + [f"{k}={v}" for k, v in env.items()] +
                       ["bash", "tools/service_ctl.sh", "start-ctrl", str(t)],
                       capture_output=True, text=True, timeout=2400)
    if r.returncode != 0:
        raise RuntimeError(f"restart failed:\n{r.stdout[-600:]}\n{r.stderr[-400:]}")
    return r.stdout.strip().splitlines()[-1]


def rss_mib():
    try:
        pid = open(f"{RUN}/ctrl.pid").read().strip()
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS"):
                return round(int(line.split()[1]) / 1024, 1)
    except Exception:
        return None


def meminfo():
    m = {}
    for line in open("/proc/meminfo"):
        p = line.split(":")
        if p[0] in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            m[p[0]] = int(p[1].split()[0]) // 1024
    return dict(mem_avail_mib=m.get("MemAvailable"),
                swap_used_mib=m.get("SwapTotal", 0) - m.get("SwapFree", 0))


def gtt_mib():
    """amdgpu GTT in use, if the debug node is readable."""
    import glob
    for p in glob.glob("/sys/kernel/debug/dri/*/amdgpu_gtt_mm"):
        try:
            for line in open(p):
                if "used" in line.lower():
                    return line.strip()[:80]
        except Exception:
            pass
    return None


def probe(domains, n_delta, n_predict, threads, stop_after=3):
    """Warm every domain, then probe newest -> oldest until a run of misses."""
    n = len(domains)
    cell("warm", domains, n, n_delta, 1, n_predict, threads, cache=True, turn0=0)
    rows, miss = [], 0
    for i in range(n - 1, -1, -1):
        rec, rr = cell(f"p{i}", [domains[i]], 1, n_delta, 1, n_predict, threads,
                       cache=True, turn0=1)
        r = rr[0]
        warm = (r.get("reused_n") or 0) > 200      # spine restored, not rebuilt
        rows.append(dict(age_from_newest=n - 1 - i, idx=i,
                         eval_n=r.get("prompt_n"), reused_n=r.get("reused_n"),
                         ttft_ms=r.get("ttft_ms"), warm=int(warm)))
        miss = 0 if warm else miss + 1
        if miss >= stop_after:
            break
    return rows, (miss >= stop_after)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="8192,16384,32768")
    ap.add_argument("--delta-tokens", type=int, default=128)
    ap.add_argument("--n-predict", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--headroom", type=float, default=1.25,
                    help="probe this multiple of predicted capacity")
    ap.add_argument("--max-domains", type=int, default=320)
    ap.add_argument("--outdir", default="artifacts/controller-state-envelope")
    a = ap.parse_args()

    rows = []
    for cr in [int(x) for x in a.budgets.split(",")]:
        line = restart(cr)
        d0 = make_domains(1)[0]
        lines, got, pre = calibrate_delta(d0, a.delta_tokens)
        total_tok = pre + got
        per_domain = total_tok * KV_KIB_PER_TOKEN / 1024
        predicted = cr / per_domain
        n_probe = min(a.max_domains, max(8, int(predicted * a.headroom)))
        print(f"\n== --cache-ram {cr} MiB :: {line}")
        print(f"   state {total_tok} tok -> {per_domain:.1f} MiB/domain, "
              f"PREDICTED capacity {predicted:.1f}; probing {n_probe} domains", flush=True)

        doms = make_domains(n_probe)
        t0 = time.perf_counter()
        pr, found_knee = probe(doms, lines, a.n_predict, a.threads)
        wall = round(time.perf_counter() - t0, 1)
        warm_n = sum(x["warm"] for x in pr)
        wt = [x["ttft_ms"] for x in pr if x["warm"] and x["ttft_ms"]]
        ct = [x["ttft_ms"] for x in pr if not x["warm"] and x["ttft_ms"]]
        rec = dict(cache_ram_mib=cr, state_tokens=total_tok,
                   mib_per_domain=round(per_domain, 1),
                   predicted_capacity=round(predicted, 1),
                   probed=n_probe, warm_run=warm_n, knee_found=int(found_knee),
                   capacity=(warm_n if found_knee else None),
                   capacity_gt=(None if found_knee else n_probe),
                   warm_ttft_p50=pct(wt, .5), cold_ttft_p50=pct(ct, .5),
                   rss_mib=rss_mib(), wall_s=wall, **meminfo())
        rows.append(rec)
        if found_knee:
            print(f"   OBSERVED capacity {warm_n} (predicted {predicted:.1f}, "
                  f"ratio {warm_n/predicted:.2f})", flush=True)
        else:
            print(f"   CAPACITY > {n_probe} -- every probed domain stayed warm; "
                  f"the knee was NOT reached, so this is a lower bound", flush=True)
        print(f"   warm TTFT p50 {rec['warm_ttft_p50']}  cold {rec['cold_ttft_p50']}  "
              f"RSS {rec['rss_mib']} MiB  avail {rec['mem_avail_mib']} MiB  "
              f"swap {rec['swap_used_mib']} MiB", flush=True)
        write_rows(f"{a.outdir}/cache_ram.csv", rows)
        write_rows(f"{a.outdir}/cache_ram_probe_{cr}.csv", pr)
    print(f"\nwrote {a.outdir}/cache_ram.csv")


if __name__ == "__main__":
    main()
