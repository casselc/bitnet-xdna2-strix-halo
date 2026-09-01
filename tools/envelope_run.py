#!/usr/bin/env python3
"""Driver for the multi-domain warm-state controller envelope.

Subcommands map to the overnight tasks:
  config-ab    Task 1  state-scheduler config vs service-batching-gate config
  matrix       Task 3  domains x delta size
  concurrency  Task 4  closed loop over DISTINCT warm domains
"""
import argparse, json, statistics as st, subprocess, sys, time

sys.path.insert(0, "tools")
from multi_domain import make_domains, calibrate_delta, cell
from service_bench import write_rows, pct, read_int, slot_context

RUN = "/tmp/bitnet-service"


def restart(t, tb=None, b=2048, ub=2048, slots=8, ctx=40960, cache_ram=None,
            warmup=True):
    env = dict(CTRL_B=str(b), CTRL_UB=str(ub), CTRL_SLOTS=str(slots),
               CTRL_CTX=str(ctx), CTRL_WARMUP="1" if warmup else "0")
    if tb:
        env["CTRL_TB"] = str(tb)
    if cache_ram is not None:
        env["CACHE_RAM"] = str(cache_ram)
    r = subprocess.run(["env"] + [f"{k}={v}" for k, v in env.items()] +
                       ["bash", "tools/service_ctl.sh", "start-ctrl", str(t)],
                       capture_output=True, text=True, timeout=2400)
    if r.returncode != 0:
        raise RuntimeError(f"restart failed:\n{r.stdout[-600:]}\n{r.stderr[-400:]}")
    return r.stdout.strip().splitlines()[-1]


def ctrl_rss_mib():
    try:
        pid = open(f"{RUN}/ctrl.pid").read().strip()
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS"):
                return round(int(line.split()[1]) / 1024, 1)
    except Exception:
        return None


def sysmem():
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")[0], line.split()[1]
        if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            m[k] = int(v) // 1024
    return dict(mem_avail_mib=m.get("MemAvailable"),
                swap_used_mib=(m.get("SwapTotal", 0) - m.get("SwapFree", 0)))


def warm_domains(domains, n_delta, n_predict, threads, label="warm"):
    """Put every domain's spine in cache once; NOT part of any statistic."""
    t0 = time.perf_counter()
    rec, _ = cell(label, domains, len(domains), n_delta, 1, n_predict, threads,
                  cache=True)
    return round(time.perf_counter() - t0, 1), rec


def cmd_config_ab(a):
    """Task 1. Same workload, two server configurations, interleaved."""
    cfgs = {"A state-scheduler (t8 b2048 ub2048)":
            dict(t=8, tb=None, b=2048, ub=2048),
            "B service-batching (t4 tb16 b4096 ub4096)":
            dict(t=4, tb=16, b=4096, ub=4096)}
    doms = make_domains(a.domains)
    lines, got, pre = calibrate_delta(doms[0], a.delta_tokens)
    print(f"stable prefix {pre} tok, delta {got} tok, output {a.n_predict} tok, "
          f"{a.domains} domain(s)\n")
    acc = {k: [] for k in cfgs}
    rows = []
    for rnd in range(a.rounds):
        order = list(cfgs) if rnd % 2 == 0 else list(cfgs)[::-1]
        for name in order:
            c = cfgs[name]
            line = restart(c["t"], c["tb"], c["b"], c["ub"], a.slots, a.ctx)
            warm_domains(doms, lines, a.n_predict, c["t"])
            rec, rr = cell(f"r{rnd}-{name[0]}", doms, a.per_round, lines, 1,
                           a.n_predict, c["t"],
                           jsonl=f"{a.outdir}/config_ab_requests.jsonl")
            rec.update(config=name, round=rnd, t=c["t"], tb=c["tb"],
                       b=c["b"], ub=c["ub"], server=line, rss_mib=ctrl_rss_mib())
            rows.append(rec)
            acc[name] += [r["ttft_ms"] for r in rr if r.get("ttft_ms")]
            print(f"  round {rnd} {name[0]}: n={rec['usable']} "
                  f"ttft p50 {rec['ttft_p50']:>8.1f}  total p50 {rec['total_p50']:>8.1f}  "
                  f"eval {rec['eval_mean']}  reused {rec['reused_mean']}  "
                  f"{rec['watts']}W", flush=True)
            write_rows(f"{a.outdir}/config_ab.csv", rows)
    print("\n=== pooled ===")
    for name, v in acc.items():
        print(f"  {name:<44} n={len(v):>3}  ttft p50 {pct(v,.5):>8.1f}  "
              f"p95 {pct(v,.95):>8.1f}  mean {st.mean(v):>8.1f}  sd {st.pstdev(v):>6.1f}")
    A, B = list(acc.values())
    if A and B:
        rel = st.median(B) / st.median(A) - 1
        print(f"\n  B vs A median TTFT: {rel*100:+.2f}%")
    return rows


def cmd_matrix(a):
    """Task 3. domains x delta size, warm, output 4 tokens."""
    line = restart(a.t, a.tb, a.b, a.ub, a.slots, a.ctx, a.cache_ram)
    print(f"== {line}\n")
    rows = []
    for nd in [int(x) for x in a.domain_counts.split(",")]:
        doms = make_domains(nd)
        for tgt in [int(x) for x in a.deltas.split(",")]:
            lines, got, pre = calibrate_delta(doms[0], tgt)
            wsec, _ = warm_domains(doms, lines, a.n_predict, a.t)
            rec, _ = cell(f"d{nd}-delta{got}", doms, max(a.per_cell, nd), lines, 1,
                          a.n_predict, a.t,
                          jsonl=f"{a.outdir}/multi_domain_requests.jsonl")
            rec.update(prefix_tokens=pre, delta_tokens=got, warm_s=wsec,
                       rss_mib=ctrl_rss_mib(), cache_ram=a.cache_ram or 8192,
                       **sysmem())
            rows.append(rec)
            print(f"  domains {nd:>3} delta {got:>4}: ttft p50 {rec['ttft_p50']:>8.1f} "
                  f"p95 {rec['ttft_p95']:>8.1f}  total p50 {rec['total_p50']:>8.1f}  "
                  f"eval {rec['eval_mean']:>7}  reused {rec['reused_mean']:>7}  "
                  f"{rec['req_per_s']:>5.3f} rps  contam {rec['contaminated']}  "
                  f"RSS {rec['rss_mib']}", flush=True)
            write_rows(f"{a.outdir}/multi_domain.csv", rows)
    return rows


def cmd_concurrency(a):
    """Task 4. Closed loop over DISTINCT warm domains, not identical requests."""
    line = restart(a.t, a.tb, a.b, a.ub, a.slots, a.ctx, a.cache_ram)
    print(f"== {line}\n")
    rows = []
    for nd in [int(x) for x in a.domain_counts.split(",")]:
        doms = make_domains(nd)
        lines, got, pre = calibrate_delta(doms[0], a.delta_tokens)
        warm_domains(doms, lines, a.n_predict, a.t)
        for c in [int(x) for x in a.conc.split(",")]:
            rec, _ = cell(f"d{nd}-c{c}", doms, max(a.per_cell, nd), lines, c,
                          a.n_predict, a.t,
                          jsonl=f"{a.outdir}/concurrency_requests.jsonl")
            rec.update(prefix_tokens=pre, delta_tokens=got, rss_mib=ctrl_rss_mib())
            rows.append(rec)
            print(f"  domains {nd:>3} c={c}: {rec['req_per_s']:>6.3f} rps  "
                  f"ttft p50 {rec['ttft_p50']:>8.1f} p95 {rec['ttft_p95']:>9.1f}  "
                  f"total p95 {rec['total_p95']:>9.1f}  eval {rec['eval_mean']:>7}  "
                  f"contam {rec['contaminated']}", flush=True)
            write_rows(f"{a.outdir}/concurrency.csv", rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("config-ab", "matrix", "concurrency"))
    ap.add_argument("--outdir", default="artifacts/controller-state-envelope")
    ap.add_argument("--t", type=int, default=4)
    ap.add_argument("--tb", type=int, default=16)
    ap.add_argument("--b", type=int, default=4096)
    ap.add_argument("--ub", type=int, default=4096)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=40960)
    ap.add_argument("--cache-ram", type=int, default=None)
    ap.add_argument("--n-predict", type=int, default=4)
    ap.add_argument("--domains", type=int, default=8)
    ap.add_argument("--domain-counts", default="1,8,32,64")
    ap.add_argument("--deltas", default="32,128,256")
    ap.add_argument("--delta-tokens", type=int, default=128)
    ap.add_argument("--conc", default="1,2,4,8")
    ap.add_argument("--per-cell", type=int, default=32)
    ap.add_argument("--per-round", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    {"config-ab": cmd_config_ab, "matrix": cmd_matrix,
     "concurrency": cmd_concurrency}[a.cmd](a)


if __name__ == "__main__":
    main()
