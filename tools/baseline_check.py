#!/usr/bin/env python3
"""Task 1: confirm this branch has not drifted from the recorded baseline.

Interleaved (round-robin) across configurations, never blocked, because prior
work measured 10-30% between-run drift on this machine and block-ordered A/B
produced false positives. Records background load and NPU dispatch count per run.
"""
import argparse, csv, json, os, re, subprocess, sys, time
from pathlib import Path

BIN   = "refs/BitNet/build-xdna3/bin/llama-bench"
MODEL = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
ART   = os.path.abspath("artifacts/xclbin-tuned")

def bg_load():
    """1-minute load average minus our own contribution is not knowable; report raw."""
    return float(open("/proc/loadavg").read().split()[0])

def cpu_mhz():
    v = [float(l.split(":")[1]) for l in open("/proc/cpuinfo") if l.startswith("cpu MHz")]
    return sum(v)/len(v) if v else 0.0

def temp_c():
    for h in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (h/"name").read_text().strip() in ("k10temp", "zenpower"):
                return int((h/"temp1_input").read_text())/1000.0
        except OSError:
            pass
    return None

def run(prompt, ub, threads, hybrid, reps_inner):
    env = dict(os.environ, BITNET_XDNA="1" if hybrid else "0",
               BITNET_XDNA_ARTIFACTS=ART, BITNET_XDNA_STATS="1")
    t = time.time()
    p = subprocess.run([BIN, "-m", MODEL, "-p", str(prompt), "-n", "0",
                        "-t", str(threads), "-ngl", "0", "-r", str(reps_inner),
                        "-ub", str(ub)],
                       capture_output=True, text=True, env=env, timeout=1800)
    out = p.stdout + p.stderr
    m = re.search(rf"pp{prompt} \|\s*([0-9.]+)", out)
    d = re.search(r"dispatches=(\d+)", out)
    dm = re.search(r"dispatch_total=([0-9.]+)", out)
    return dict(tok_s=float(m.group(1)) if m else None,
                dispatches=int(d.group(1)) if d else 0,
                device_ms=float(dm.group(1)) if dm else 0.0,
                wall_s=time.time()-t)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=2048)
    ap.add_argument("--ub", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--out", default="artifacts/overlap-de-risk/baseline.csv")
    a = ap.parse_args()

    configs = [(15, False), (15, True), (8, True), (4, True)]
    rows = []
    print(f"baseline: pp{a.prompt} ub{a.ub}; {len(configs)} configs x {a.reps} interleaved reps")
    for rep in range(1, a.reps+1):
        for th, hyb in configs:
            r = run(a.prompt, a.ub, th, hyb, a.inner)
            r.update(rep=rep, prompt=a.prompt, ub=a.ub, threads=th,
                     mode="hybrid" if hyb else "cpu",
                     bg_load=bg_load(), cpu_mhz=round(cpu_mhz(),0), temp_c=temp_c())
            rows.append(r)
            print(f"  [{rep}] t={th:<2} {r['mode']:<6} {r['tok_s']:>8.1f} tok/s  "
                  f"disp={r['dispatches']:<6} dev={r['device_ms']:>7.1f} ms  "
                  f"load={r['bg_load']:.2f} {r['temp_c']}C", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
