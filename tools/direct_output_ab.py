#!/usr/bin/env python3
"""Tasks 3 and 5: direct mapped-output vs the g_acc staging path, +/- async.

Four variants, interleaved round-robin (never blocked -- this machine shows
10-30% between-run drift and block-ordered A/B has produced false positives):

  A  sync   + g_acc     the deployed path
  B  async  + g_acc     N-chunk dispatch pipelining (overlap-de-risk)
  C  sync   + direct    persistent per-(tile, n-chunk) output slots
  D  async  + direct    both

Async exists to overlap the evacuation of one chunk with the next dispatch. If
direct output removes the evacuation, async has nothing left to hide, so D is not
assumed to win.

All cumulative counters are divided by (inner + 1) because llama-bench runs one
warmup prefill in addition to the timed reps.
"""
import argparse, csv, os, re, statistics as st, subprocess
from pathlib import Path

BIN   = "refs/BitNet/build-xdna3/bin/llama-bench"
MODEL = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
ART   = os.path.abspath("artifacts/xclbin-tuned")
VARIANTS = [("A sync+g_acc", 0, 0), ("B async+g_acc", 0, 1),
            ("C sync+direct", 1, 0), ("D async+direct", 1, 1)]

def run(prompt, ub, threads, tiles, direct, async_on, inner):
    env = dict(os.environ, BITNET_XDNA="1", BITNET_XDNA_ARTIFACTS=ART,
               BITNET_XDNA_STATS="1",
               BITNET_XDNA_DIRECT_OUT=str(direct), BITNET_XDNA_ASYNC=str(async_on))
    if tiles is not None:
        env["BITNET_XDNA_TILES"] = str(tiles)
    p = subprocess.run([BIN, "-m", MODEL, "-p", str(prompt), "-n", "0", "-t", str(threads),
                        "-ngl", "0", "-r", str(inner), "-ub", str(ub)],
                       capture_output=True, text=True, env=env, timeout=3600)
    o = p.stdout + p.stderr
    def f(pat, d=0.0):
        m = re.search(pat, o); return float(m.group(1)) if m else d
    n_pref = inner + 1
    tok = f(rf"pp{prompt} \|\s*([0-9.]+)", 0.0)
    return dict(
        tok_s=tok,
        wall_ms=(prompt / tok * 1000) if tok else 0.0,
        dispatches=f(r"dispatches=(\d+)") / n_pref,
        npu_ms=f(r"dispatch_total=([0-9.]+)") / n_pref,
        stage_in_ms=f(r"stage_in=([0-9.]+)") / n_pref,
        stage_out_ms=f(r"stage_out=([0-9.]+)") / n_pref,
        stage_out_gb=f(r"stage_out=[0-9.]+ ms over ([0-9.]+) GB") / n_pref,
        epi_thread_ms=f(r"epilogue=([0-9.]+)") / n_pref,
        sync_in_ms=f(r"sync_in (\d+)") / n_pref,
        sync_out_ms=f(r"sync_out (\d+)") / n_pref,
        arena_mib=f(r"arena: \d+ slots x [0-9.]+ MiB = ([0-9.]+) MiB"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--inner", type=int, default=2)
    ap.add_argument("--out", default="artifacts/direct-output/direct_output_ab.csv")
    a = ap.parse_args()

    # (label, prompt, ub, threads, tiles)
    cells = [("pp2048 ub2048 t15",  2048, 2048, 15, None),
             ("pp2048 ub2048 t8",   2048, 2048,  8, None),
             ("pp2048 ub2048 t4",   2048, 2048,  4, None),
             ("pp2048 allNPU t15",  2048, 2048, 15, 2),
             ("pp3968 ub2048 t15",  3968, 2048, 15, None),
             ("pp2048 ub1024 t15",  2048, 1024, 15, None)]
    rows, acc = [], {}
    print(f"{len(cells)} cells x {len(VARIANTS)} variants x {a.reps} interleaved reps")
    for rep in range(1, a.reps + 1):
        for label, prompt, ub, th, tiles in cells:
            for vname, d, y in VARIANTS:
                r = run(prompt, ub, th, tiles, d, y, a.inner)
                r.update(rep=rep, cell=label, prompt=prompt, ub=ub, threads=th,
                         tiles=("auto" if tiles is None else tiles),
                         variant=vname, direct=d, async_on=y)
                rows.append(r)
                acc.setdefault((label, vname), []).append(r)
            print(f"  [{rep}] {label} done", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"\n{'cell':<20}{'variant':<16}{'tok/s':>9}{'sd':>6}{'vs A':>8}"
          f"{'stage_out':>10}{'epi thr':>9}{'npu ms':>8}{'arena':>8}")
    for label, prompt, ub, th, tiles in cells:
        base = st.median([x["tok_s"] for x in acc[(label, "A sync+g_acc")]])
        for vname, _, _ in VARIANTS:
            xs = acc[(label, vname)]
            m = st.median([x["tok_s"] for x in xs])
            print(f"{label:<20}{vname:<16}{m:>9.1f}"
                  f"{st.pstdev([x['tok_s'] for x in xs]):>6.1f}{m/base:>7.3f}x"
                  f"{st.median([x['stage_out_ms'] for x in xs]):>10.1f}"
                  f"{st.median([x['epi_thread_ms'] for x in xs]):>9.0f}"
                  f"{st.median([x['npu_ms'] for x in xs]):>8.0f}"
                  f"{st.median([x['arena_mib'] for x in xs]):>8.0f}")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
