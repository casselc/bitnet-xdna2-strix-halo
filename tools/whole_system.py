#!/usr/bin/env python3
"""Whole-machine benchmark: controller latency under realistic contention.

The deployment target is not "BitNet as fast as possible in isolation" -- it is a
resident controller sharing a Strix Halo with CPU-side work (Samizdat/Jolt/SCI)
and a larger GPU worker. Giving the controller all 16 cores may be the wrong
answer even when it wins the isolated benchmark, because the cores it takes come
straight out of the other tenants.

So this measures the PARETO FRONTIER: controller TTFT and total response latency
against the throughput the co-tenant workload still achieves, across controller
configurations.

Controller workload is deliberately controller-shaped: a large structured prompt
and a short answer, which is where TTFT dominates.
"""
import argparse, csv, json, os, re, subprocess, sys, time
from pathlib import Path

BIN = "refs/BitNet/build-xdna2/bin/llama-bench"
MODEL = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
ART = os.path.abspath("artifacts/xclbin-tuned")


class CpuLoad:
    """Stand-in for the CPU-side tenant: N processes doing integer/branch work,
    which is what a graph/verification workload looks like to the scheduler."""
    def __init__(self, n):
        self.n = n
        self.procs = []

    def __enter__(self):
        prog = ("import time\n"
                "t=time.time(); n=0\n"
                "while time.time()-t < 3600:\n"
                "    n=(n*1103515245+12345)&0x7fffffff\n")
        for _ in range(self.n):
            self.procs.append(subprocess.Popen(
                [sys.executable, "-c", prog],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True))
        # let them reach steady state before the controller is timed
        time.sleep(3)
        return self

    def __exit__(self, *a):
        for p in self.procs:
            try: os.killpg(os.getpgid(p.pid), 9)
            except Exception: pass
        for p in self.procs:
            try: p.wait(timeout=5)
            except Exception: pass
        time.sleep(2)
        return False


def controller(prompt, gen, threads, tiles, ub):
    env = dict(os.environ, BITNET_XDNA_ARTIFACTS=ART, BITNET_XDNA_STATS="1")
    if tiles is None:
        env["BITNET_XDNA"] = "0"
    else:
        env["BITNET_XDNA"] = "1"; env["BITNET_XDNA_TILES"] = str(tiles)
    cmd = [BIN, "-m", MODEL, "-p", str(prompt), "-n", str(gen),
           "-t", str(threads), "-ngl", "0", "-r", "2", "-ub", str(ub)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    out = r.stdout + r.stderr
    pp = re.search(rf"pp{prompt}\s*\|\s*([0-9.]+)", out)
    tg = re.search(rf"tg{gen}\s*\|\s*([0-9.]+)", out)
    if not pp:
        return None
    pp_ts = float(pp.group(1))
    tg_ts = float(tg.group(1)) if tg else 0.0
    ttft_ms = prompt / pp_ts * 1000.0
    total_ms = ttft_ms + (gen / tg_ts * 1000.0 if tg_ts else 0.0)
    return {"pp_tok_s": pp_ts, "tg_tok_s": tg_ts,
            "ttft_ms": round(ttft_ms, 1), "total_ms": round(total_ms, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=2048)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--ub", type=int, default=2048)
    ap.add_argument("--bg", type=int, nargs="+", default=[0, 8],
                    help="co-tenant CPU worker counts")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="artifacts/next-pass/whole_system.csv")
    a = ap.parse_args()

    # Controller configurations: threads x NPU tile share. tiles=None is CPU-only.
    max_tiles = min(a.ub, a.prompt) // 1024
    configs = []
    for th in (2, 4, 8, 15):
        configs.append((th, None))
        for t in range(1, max_tiles + 1):
            configs.append((th, t))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fields = ["rep", "bg_workers", "threads", "tiles", "pp_tok_s", "tg_tok_s",
              "ttft_ms", "total_ms", "cores_used"]
    print(f"controller p{a.prompt}/n{a.gen} ub{a.ub}; "
          f"{len(configs)} configs x {len(a.bg)} bg levels x {a.reps} reps", flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for rep in range(a.reps):
            for bg in a.bg:
                with CpuLoad(bg):
                    for th, tiles in configs:
                        r = controller(a.prompt, a.gen, th, tiles, a.ub)
                        if not r: continue
                        row = dict(rep=rep, bg_workers=bg, threads=th,
                                   tiles=("cpu" if tiles is None else tiles),
                                   cores_used=th, **r)
                        w.writerow(row); fh.flush()
                        print(f"  [{rep+1}] bg={bg} t={th} tiles={row['tiles']}: "
                              f"TTFT {r['ttft_ms']:.0f} ms  total {r['total_ms']:.0f} ms",
                              flush=True)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
