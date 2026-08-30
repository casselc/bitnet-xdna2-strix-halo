#!/usr/bin/env python3
"""Scheduling-surface sweep: prompt x microbatch x CPU threads x NPU tile share.

Two disciplines the earlier work learned the hard way are baked in:

  * Configurations are run INTERLEAVED, one rep of each per round, not in blocks.
    Block ordering plus host drift previously produced confident false results in
    both directions.
  * Every run records the background CPU load it saw, because this machine is
    shared and a contaminated CPU-only baseline flatters the hybrid.

Emits one CSV row per (config, rep) so the raw data outlives any conclusion.
"""
import argparse, csv, itertools, json, os, re, subprocess, sys, time
from pathlib import Path

BIN = "refs/BitNet/build-xdna/bin/llama-bench"
MODEL = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
ART = os.path.abspath("artifacts/xclbin-tuned")
TILE = 1024   # kMTile in the runtime


def bg_load():
    """Total %CPU of processes other than ours -- the contamination we must record."""
    try:
        out = subprocess.run(["ps", "-eo", "pcpu,comm", "--no-headers"],
                             capture_output=True, text=True, timeout=10).stdout
        tot = 0.0
        for line in out.splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[1].strip() not in ("llama-bench", "ps"):
                try: tot += float(parts[0])
                except ValueError: pass
        return round(tot, 1)
    except Exception:
        return -1.0


def run_one(prompt, ub, threads, tiles):
    """tiles: NPU tile count, or None for CPU-only."""
    env = dict(os.environ, BITNET_XDNA_ARTIFACTS=ART, BITNET_XDNA_STATS="1")
    if tiles is None:
        env["BITNET_XDNA"] = "0"
    else:
        env["BITNET_XDNA"] = "1"
        env["BITNET_XDNA_TILES"] = str(tiles)
    cmd = [BIN, "-m", MODEL, "-p", str(prompt), "-n", "0",
           "-t", str(threads), "-ngl", "0", "-r", "1", "-ub", str(ub)]
    load = bg_load()
    t0 = time.time()
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return None
    out = r.stdout + r.stderr
    m = re.search(rf"pp{prompt}\s*\|\s*([0-9.]+)", out)
    if not m:
        return None
    row = {"tok_s": float(m.group(1)), "wall_s": round(time.time() - t0, 2),
           "bg_load": load}
    for key, pat in (("dispatches", r"dispatches=(\d+)"),
                     ("device_ms", r"dispatch_total=([0-9.]+)"),
                     ("sync_in_ms", r"sync_in (\d+)"),
                     ("submit_ms", r"submit (\d+)"),
                     ("wait_ms", r"wait (\d+)"),
                     ("sync_out_ms", r"sync_out (\d+)"),
                     ("resident_mib", r"resident int8 weights=([0-9.]+)")):
        mm = re.search(pat, out)
        row[key] = float(mm.group(1)) if mm else 0.0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, nargs="+", default=[512, 2048, 3968])
    ap.add_argument("--threads", type=int, nargs="+", default=[4, 8, 15])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="artifacts/next-pass/sweep.csv")
    a = ap.parse_args()

    configs = []
    for prompt in a.prompts:
        # -ub values worth testing: the microbatch caps the NPU's token batch, so
        # anything below the tile size disables offload entirely.
        ubs = sorted({u for u in (512, 1024, 2048, 4096) if u <= max(prompt, 512)})
        for ub in ubs:
            eff = min(ub, prompt)                       # tokens the graph actually sees
            max_tiles = eff // TILE
            shares = [None] + list(range(0, max_tiles + 1))   # None = CPU-only
            for th in a.threads:
                for tiles in shares:
                    configs.append(dict(prompt=prompt, ub=ub, threads=th, tiles=tiles))

    # Drop duplicates that differ only in a knob with no effect at this size.
    seen, uniq = set(), []
    for c in configs:
        key = (c["prompt"], c["ub"], c["threads"],
               "cpu" if c["tiles"] is None else c["tiles"])
        if key not in seen:
            seen.add(key); uniq.append(c)

    print(f"{len(uniq)} configs x {a.reps} reps, interleaved", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fields = ["rep", "prompt", "ub", "threads", "tiles", "tok_s", "wall_s", "bg_load",
              "dispatches", "device_ms", "sync_in_ms", "submit_ms", "wait_ms",
              "sync_out_ms", "resident_mib"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for rep in range(a.reps):
            for i, c in enumerate(uniq):
                r = run_one(c["prompt"], c["ub"], c["threads"], c["tiles"])
                if r is None:
                    continue
                row = dict(rep=rep, tiles=("cpu" if c["tiles"] is None else c["tiles"]),
                           **{k: c[k] for k in ("prompt", "ub", "threads")}, **r)
                w.writerow(row); fh.flush()
                print(f"  [{rep+1}/{a.reps}] {i+1}/{len(uniq)} "
                      f"p{c['prompt']} ub{c['ub']} t{c['threads']} "
                      f"tiles={row['tiles']}: {r['tok_s']:.1f} t/s (bg {r['bg_load']}%)",
                      flush=True)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
