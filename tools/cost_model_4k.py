#!/usr/bin/env python3
"""Task 6: independent R=25 holdout at long context.

R=25 was calibrated on pp2048/pp3072 at threads 4/6/8/10/12/15. This tests
pp3968 -ub 2048 at thread counts NOT chosen for that fit, comparing the auto
pick against an exhaustive sweep of valid tile allocations.

At pp3968 with -ub 2048 a prefill is two micro-batches of 1984 tokens, so one
micro-batch offers at most 1 whole NPU tile: the allocation space is {0, 1}.
That is a genuinely different regime from the calibration, which is the point.
"""
import argparse, csv, os, re, statistics as st, subprocess
from pathlib import Path

BIN   = "refs/BitNet/build-xdna3/bin/llama-bench"
MODEL = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
ART   = os.path.abspath("artifacts/xclbin-tuned")
DISP_PER_TILE = 323.4

def run(prompt, ub, threads, tiles, inner):
    env = dict(os.environ, BITNET_XDNA="1", BITNET_XDNA_ARTIFACTS=ART,
               BITNET_XDNA_STATS="1")
    if tiles is not None:
        env["BITNET_XDNA_TILES"] = str(tiles)
    p = subprocess.run([BIN, "-m", MODEL, "-p", str(prompt), "-n", "0", "-t", str(threads),
                        "-ngl", "0", "-r", str(inner), "-ub", str(ub)],
                       capture_output=True, text=True, env=env, timeout=3600)
    o = p.stdout + p.stderr
    tok = re.search(rf"pp{prompt} \|\s*([0-9.]+)", o)
    d   = re.search(r"dispatches=(\d+)", o)
    # a pp3968 prefill is 2 micro-batches; report tiles per micro-batch
    per_pref = int(d.group(1)) / (inner + 1) if d else 0
    mb = max(1, -(-prompt // ub))
    return (float(tok.group(1)) if tok else None,
            round(per_pref / mb / DISP_PER_TILE) if d else 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=3968)
    ap.add_argument("--ub", type=int, default=2048)
    ap.add_argument("--threads", type=int, nargs="+", default=[3, 5, 7, 9, 12, 15])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--inner", type=int, default=2)
    ap.add_argument("--out", default="artifacts/direct-output-closeout/cost_model_4k.csv")
    a = ap.parse_args()

    opts = [None, 0, 1]
    rows, acc = [], {}
    for rep in range(1, a.reps + 1):
        for th in a.threads:
            for t in opts:
                tok, pick = run(a.prompt, a.ub, th, t, a.inner)
                key = (th, "auto" if t is None else t)
                acc.setdefault(key, []).append(tok)
                if t is None: acc.setdefault((th, "pick"), []).append(pick)
                rows.append(dict(rep=rep, prompt=a.prompt, ub=a.ub, threads=th,
                                 tiles=("auto" if t is None else t),
                                 auto_pick=(pick if t is None else ""), tok_s=tok))
            print(f"  [{rep}] t={th} done", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"\n{'th':>3}{'tiles=0':>10}{'tiles=1':>10}{'auto':>10}{'pick':>6}"
          f"{'best':>6}{'regret':>9}")
    regrets = []
    for th in a.threads:
        fx = {t: st.median(acc[(th, t)]) for t in (0, 1)}
        auto = st.median(acc[(th, "auto")]); pick = round(st.median(acc[(th, "pick")]))
        bt = max(fx, key=fx.get)
        r = fx[bt] / auto; regrets.append(r)
        print(f"{th:>3}{fx[0]:>10.1f}{fx[1]:>10.1f}{auto:>10.1f}{pick:>6}{bt:>6}{r:>8.3f}x")
    print(f"\n  mean regret {st.mean(regrets):.3f}x   worst {max(regrets):.3f}x")
    print(f"  verdict: {'KEEP R=25' if st.mean(regrets) < 1.02 and max(regrets) < 1.05 else 'R=25 FAILS -- refit'}")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
