#!/usr/bin/env python3
"""Task 7: holdout test of the thread-aware cost model.

R=10 in f = R/(R + n_threads - 1) was fitted against a sweep at threads
4/8/15 and prompts 2048/3968. This tests thread counts and micro-batch sizes
that were NOT used to establish it, and compares the auto-selected tile
allocation against the exhaustively measured best.

Reports exact-match rate and regret = perf(best) / perf(auto). Regret 1.00 means
the model chose the measured optimum.

The chosen tile count is read back from the dispatch counter rather than assumed:
one prefill issues ~323 dispatches per token tile (30 layers x 11 dispatches x
147/150 offloaded tensors), and llama-bench runs one warmup prefill on top of
the -r timed ones.
"""
import argparse, csv, os, re, statistics as st, subprocess
from pathlib import Path

BIN   = "refs/BitNet/build-xdna3/bin/llama-bench"
MODEL = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
ART   = os.path.abspath("artifacts/xclbin-tuned")
DISP_PER_TILE = 323.4

def run(prompt, threads, tiles, inner):
    env = dict(os.environ, BITNET_XDNA="1", BITNET_XDNA_ARTIFACTS=ART,
               BITNET_XDNA_STATS="1")
    if tiles is not None:
        env["BITNET_XDNA_TILES"] = str(tiles)
    p = subprocess.run([BIN, "-m", MODEL, "-p", str(prompt), "-n", "0", "-t", str(threads),
                        "-ngl", "0", "-r", str(inner), "-ub", str(prompt)],
                       capture_output=True, text=True, env=env, timeout=3600)
    o = p.stdout + p.stderr
    tok = re.search(rf"pp{prompt} \|\s*([0-9.]+)", o)
    d   = re.search(r"dispatches=(\d+)", o)
    n_pref = inner + 1                       # + warmup
    chosen = round(int(d.group(1)) / n_pref / DISP_PER_TILE) if d else 0
    return (float(tok.group(1)) if tok else None), chosen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[3, 6, 10, 12])
    ap.add_argument("--prompts", type=int, nargs="+", default=[1024, 1536, 3072, 3968])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--inner", type=int, default=2)
    ap.add_argument("--out", default="artifacts/overlap-de-risk/cost_model_holdout.csv")
    a = ap.parse_args()

    rows, acc = [], {}
    combos = []
    for th in a.threads:
        for p in a.prompts:
            max_tiles = p // 1024
            combos.append((th, p, [None] + list(range(0, max_tiles + 1))))
    print(f"holdout: {len(combos)} (threads,prompt) cells, interleaved x{a.reps}")
    for rep in range(1, a.reps + 1):
        for th, p, tile_opts in combos:
            for t in tile_opts:
                tok, chosen = run(p, th, t, a.inner)
                key = (th, p, "auto" if t is None else t)
                acc.setdefault(key, []).append(tok)
                if t is None:
                    acc.setdefault((th, p, "auto_choice"), []).append(chosen)
                rows.append(dict(rep=rep, prompt=p, threads=th,
                                 tiles=("auto" if t is None else t),
                                 auto_choice=(chosen if t is None else ""),
                                 tok_s=tok))
            print(f"  [{rep}] t={th:<3} pp{p:<5} done", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"\n{'th':>3} {'prompt':>7} {'auto pick':>10} {'auto t/s':>9} "
          f"{'best pick':>10} {'best t/s':>9} {'regret':>8}")
    exact = tot = 0; regrets = []
    for th, p, tile_opts in combos:
        fixed = {t: st.median(acc[(th, p, t)]) for t in tile_opts if t is not None}
        auto = st.median(acc[(th, p, "auto")])
        pick = st.median(acc[(th, p, "auto_choice")])
        best_t = max(fixed, key=fixed.get); best = fixed[best_t]
        r = best / auto
        regrets.append(r); tot += 1; exact += (round(pick) == best_t)
        print(f"{th:>3} {p:>7} {round(pick):>10} {auto:>9.1f} {best_t:>10} {best:>9.1f} {r:>7.3f}x")
    print(f"\n  exact match {exact}/{tot} ({exact/tot*100:.0f}%)   "
          f"mean regret {st.mean(regrets):.3f}x   worst {max(regrets):.3f}x")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
