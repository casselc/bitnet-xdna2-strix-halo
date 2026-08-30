#!/usr/bin/env python3
"""NPU duty cycle, with llama-bench's warmup prefill separated out.

CORRECTION to the previous pass. `tools/duty_cycle.sh` divided the cumulative
NPU device time by the number of timed reps. llama-bench runs ONE warmup prefill
in addition to the -r timed reps, and the warmup dispatches to the NPU exactly
like a timed rep, so that divisor charged a whole extra prefill's device time
across the reps and inflated NPU-busy time by roughly (1 + 1/r).

Verified: dispatch count is 642*(r+1) at tiles=2 -- 1284 / 1926 / 3210 for
r = 1 / 2 / 4. Here we regress cumulative device_ms on r; the SLOPE is the
per-prefill NPU busy time and the INTERCEPT is the warmup.

Wall time per prefill comes from llama-bench's own tok/s, which already excludes
the warmup, so the two sides of the duty ratio are finally consistent.
"""
import argparse, csv, json, os, re, subprocess, statistics as st
from pathlib import Path

BIN   = "refs/BitNet/build-xdna3/bin/llama-bench"
MODEL = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
ART   = os.path.abspath("artifacts/xclbin-tuned")

def one(prompt, ub, threads, tiles, r):
    env = dict(os.environ, BITNET_XDNA="1", BITNET_XDNA_ARTIFACTS=ART,
               BITNET_XDNA_STATS="1")
    if tiles is not None:
        env["BITNET_XDNA_TILES"] = str(tiles)
    p = subprocess.run([BIN, "-m", MODEL, "-p", str(prompt), "-n", "0", "-t", str(threads),
                        "-ngl", "0", "-r", str(r), "-ub", str(ub)],
                       capture_output=True, text=True, env=env, timeout=2400)
    o = p.stdout + p.stderr
    tok = re.search(rf"pp{prompt} \|\s*([0-9.]+)", o)
    dev = re.search(r"dispatch_total=([0-9.]+)", o)
    dsp = re.search(r"dispatches=(\d+)", o)
    return (float(tok.group(1)) if tok else None,
            float(dev.group(1)) if dev else 0.0,
            int(dsp.group(1)) if dsp else 0)

def fit(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs)
    return (sum((x-mx)*(y-my) for x, y in zip(xs, ys))/sxx,          # slope
            my - sum((x-mx)*(y-my) for x, y in zip(xs, ys))/sxx*mx)  # intercept

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=2048)
    ap.add_argument("--ub", type=int, default=2048)
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--out", default="artifacts/overlap-de-risk/duty_cycle.csv")
    a = ap.parse_args()

    rows = []
    print(f"pp{a.prompt} ub{a.ub}; device_ms regressed on reps {a.reps} "
          f"(slope = per-prefill NPU busy, intercept = warmup)")
    print(f"{'th':>3} {'tiles':>5} {'tok/s':>8} {'wall/pf':>8} {'NPU/pf':>8} "
          f"{'warmup':>8} {'duty':>7} {'disp/pf':>8}")
    for th in (4, 8, 15):
        for tiles in (1, 2):
            xs, ys, toks, dsps = [], [], [], []
            for r in a.reps:
                tok, dev, dsp = one(a.prompt, a.ub, th, tiles, r)
                xs.append(r); ys.append(dev); toks.append(tok); dsps.append(dsp)
            slope, icept = fit(xs, ys)
            dslope, _ = fit(xs, [float(d) for d in dsps])
            tok = st.median(toks); wall = a.prompt/tok*1000
            rows.append(dict(threads=th, tiles=tiles, tok_s=round(tok,1),
                             wall_ms=round(wall,1), npu_ms=round(slope,1),
                             warmup_ms=round(icept,1), duty=round(slope/wall,4),
                             disp_per_prefill=round(dslope,1),
                             reps=json.dumps(a.reps), device_ms=json.dumps(ys)))
            print(f"{th:>3} {tiles:>5} {tok:>8.1f} {wall:>7.0f}m {slope:>7.0f}m "
                  f"{icept:>7.0f}m {slope/wall*100:>6.1f}% {dslope:>8.0f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
