#!/usr/bin/env python3
"""Task 7: does direct mapped output move the cost-model optimum?

R=10 in f = R/(R + n_threads - 1) was calibrated against the g_acc path, where
each token tile assigned to the NPU also cost thread 0 a staging copy. Direct
output removes that per-tile cost, which makes NPU-assigned work cheaper and
could shift the optimal split toward the NPU. Test before assuming.

Sweeps every tile allocation against the auto pick, both with and without direct
output, interleaved.
"""
import argparse, csv, os, re, statistics as st, subprocess
from pathlib import Path

BIN   = "refs/BitNet/build-xdna3/bin/llama-bench"
MODEL = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
ART   = os.path.abspath("artifacts/xclbin-tuned")
DISP_PER_TILE = 323.4

def run(prompt, ub, threads, tiles, direct, inner):
    env = dict(os.environ, BITNET_XDNA="1", BITNET_XDNA_ARTIFACTS=ART,
               BITNET_XDNA_STATS="1", BITNET_XDNA_DIRECT_OUT=str(direct),
               BITNET_XDNA_ASYNC="0")
    if tiles is not None:
        env["BITNET_XDNA_TILES"] = str(tiles)
    p = subprocess.run([BIN, "-m", MODEL, "-p", str(prompt), "-n", "0", "-t", str(threads),
                        "-ngl", "0", "-r", str(inner), "-ub", str(ub)],
                       capture_output=True, text=True, env=env, timeout=3600)
    o = p.stdout + p.stderr
    tok = re.search(rf"pp{prompt} \|\s*([0-9.]+)", o)
    d   = re.search(r"dispatches=(\d+)", o)
    return (float(tok.group(1)) if tok else None,
            round(int(d.group(1)) / (inner + 1) / DISP_PER_TILE) if d else 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=2048)
    ap.add_argument("--ub", type=int, default=2048)
    ap.add_argument("--threads", type=int, nargs="+", default=[4, 8, 15])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--inner", type=int, default=2)
    ap.add_argument("--out", default="artifacts/direct-output/cost_model.csv")
    a = ap.parse_args()

    max_tiles = a.prompt // 1024
    opts = [None] + list(range(0, max_tiles + 1))
    rows, acc = [], {}
    for rep in range(1, a.reps + 1):
        for th in a.threads:
            for direct in (0, 1):
                for t in opts:
                    tok, pick = run(a.prompt, a.ub, th, t, direct, a.inner)
                    key = (th, direct, "auto" if t is None else t)
                    acc.setdefault(key, []).append(tok)
                    if t is None:
                        acc.setdefault((th, direct, "pick"), []).append(pick)
                    rows.append(dict(rep=rep, prompt=a.prompt, ub=a.ub, threads=th,
                                     direct=direct, tiles=("auto" if t is None else t),
                                     auto_pick=(pick if t is None else ""), tok_s=tok))
            print(f"  [{rep}] t={th} done", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"\n{'th':>3} {'direct':>7} " + "".join(f"{f'tiles={t}':>11}" for t in range(max_tiles+1))
          + f"{'auto':>11}{'pick':>6}{'best':>6}{'regret':>8}")
    for th in a.threads:
        for direct in (0, 1):
            fixed = {t: st.median(acc[(th, direct, t)]) for t in range(max_tiles + 1)}
            auto  = st.median(acc[(th, direct, "auto")])
            pick  = round(st.median(acc[(th, direct, "pick")]))
            bt    = max(fixed, key=fixed.get)
            print(f"{th:>3} {direct:>7} " + "".join(f"{fixed[t]:>11.1f}" for t in range(max_tiles+1))
                  + f"{auto:>11.1f}{pick:>6}{bt:>6}{fixed[bt]/auto:>7.3f}x")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
