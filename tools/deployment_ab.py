#!/usr/bin/env python3
"""Final deployment comparison, each path at ITS OWN calibrated split.

Comparing direct output against g_acc while both use R=10 understates direct
output, because removing the per-tile staging cost moves the optimal NPU share
(artifacts/direct-output/cost_model_recal.csv). Each arm therefore runs at its
own calibrated R -- g_acc at 10, direct at 25 -- which is how each would actually
be deployed. Interleaved round-robin, medians reported with dispersion.
"""
import argparse, csv, os, re, statistics as st, subprocess
from pathlib import Path

BIN   = "refs/BitNet/build-xdna3/bin/llama-bench"
MODEL = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
ART   = os.path.abspath("artifacts/xclbin-tuned")

def run(prompt, ub, threads, mode, inner):
    env = dict(os.environ, BITNET_XDNA_ARTIFACTS=ART, BITNET_XDNA_STATS="1",
               BITNET_XDNA_ASYNC="0")
    env["BITNET_XDNA"] = "0" if mode == "cpu" else "1"
    env["BITNET_XDNA_DIRECT_OUT"] = "1" if mode == "direct" else "0"
    p = subprocess.run([BIN, "-m", MODEL, "-p", str(prompt), "-n", "0", "-t", str(threads),
                        "-ngl", "0", "-r", str(inner), "-ub", str(ub)],
                       capture_output=True, text=True, env=env, timeout=3600)
    o = p.stdout + p.stderr
    def f(pat, d=0.0):
        m = re.search(pat, o); return float(m.group(1)) if m else d
    n = inner + 1
    tok = f(rf"pp{prompt} \|\s*([0-9.]+)")
    return dict(tok_s=tok, wall_ms=(prompt/tok*1000) if tok else 0,
                npu_ms=f(r"dispatch_total=([0-9.]+)")/n,
                dispatches=f(r"dispatches=(\d+)")/n,
                stage_out_ms=f(r"stage_out=([0-9.]+)")/n,
                epi_thread_ms=f(r"epilogue=([0-9.]+)")/n,
                arena_mib=f(r"= ([0-9.]+) MiB"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--inner", type=int, default=2)
    ap.add_argument("--out", default="artifacts/direct-output/deployment_ab.csv")
    a = ap.parse_args()
    cells = [(p, 2048, th) for p in (2048, 3968) for th in (4, 6, 8, 15)]
    modes = ["cpu", "gacc", "direct"]
    rows, acc = [], {}
    for rep in range(1, a.reps+1):
        for prompt, ub, th in cells:
            for m in modes:
                r = run(prompt, ub, th, m, a.inner)
                r.update(rep=rep, prompt=prompt, ub=ub, threads=th, mode=m)
                rows.append(r); acc.setdefault((prompt, th, m), []).append(r)
            print(f"  [{rep}] pp{prompt} t{th} done", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n{'prompt':>7}{'th':>4}{'CPU-only':>10}{'g_acc R10':>11}{'direct R25':>12}"
          f"{'d/g':>8}{'d/cpu':>8}{'npu ms':>8}{'tiles':>7}{'arena':>7}")
    for prompt, ub, th in cells:
        g = lambda m, k="tok_s": st.median([x[k] for x in acc[(prompt, th, m)]])
        c, ga, d = g("cpu"), g("gacc"), g("direct")
        print(f"{prompt:>7}{th:>4}{c:>10.1f}{ga:>11.1f}{d:>12.1f}"
              f"{d/ga:>7.3f}x{d/c:>7.3f}x{g('direct','npu_ms'):>8.0f}"
              f"{g('direct','dispatches')/323.4:>7.1f}{g('direct','arena_mib'):>7.0f}")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
