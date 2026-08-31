#!/usr/bin/env python3
"""Task 4: deep-K direct reduction vs the host `part` accumulation.

ffn_down (k_chunks == 3) currently evacuates each K chunk into a host `part`
buffer on thread 0, adds them there, copies part -> g_acc, and only then lets the
workers scale. Direct K-reduce gives each K chunk its own persistent slot and has
the CPU workers do the int32 reduction, the scale and the store in one parallel
pass -- no `part`, no g_acc.

Measured ceiling is small: partacc 31.9 ms + partcopy 10.9 ms = 42.8 ms, about
2.6% of a pp2048 prefill. Interleaved round-robin, medians, because an effect
this size is well inside this machine's between-run drift.
"""
import argparse, csv, os, re, statistics as st, subprocess
from pathlib import Path

BIN   = "refs/BitNet/build-xdna3/bin/llama-bench"
MODEL = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
ART   = os.path.abspath("artifacts/xclbin-tuned")

def run(prompt, ub, threads, kreduce, inner, shape_csv=None):
    env = dict(os.environ, BITNET_XDNA="1", BITNET_XDNA_ARTIFACTS=ART,
               BITNET_XDNA_STATS="1", BITNET_XDNA_DIRECT_KREDUCE=str(kreduce))
    if shape_csv: env["BITNET_XDNA_SHAPE_CSV"] = shape_csv
    p = subprocess.run([BIN, "-m", MODEL, "-p", str(prompt), "-n", "0", "-t", str(threads),
                        "-ngl", "0", "-r", str(inner), "-ub", str(ub)],
                       capture_output=True, text=True, env=env, timeout=3600)
    o = p.stdout + p.stderr
    def f(pat, d=0.0):
        m = re.search(pat, o); return float(m.group(1)) if m else d
    n = inner + 1
    tok = f(rf"pp{prompt} \|\s*([0-9.]+)")
    part = {}
    if shape_csv and os.path.exists(shape_csv):
        for r in csv.DictReader(open(shape_csv)):
            if (int(r["K"]), int(r["N"])) == (6912, 2560):
                part = dict(partacc_ms=float(r["partacc_ms"]) / n,
                            partcopy_ms=float(r["partcopy_ms"]) / n,
                            epi_thread_ms=float(r["epi_thread_ms"]) / n)
    return dict(tok_s=tok, wall_ms=(prompt / tok * 1000) if tok else 0.0,
                epi_all_thread_ms=f(r"epilogue=([0-9.]+)") / n, **part)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--out", default="artifacts/direct-output-closeout/kreduce_ab.csv")
    a = ap.parse_args()
    cells = [(2048, 2048, th) for th in (4, 6, 8, 15)] + \
            [(3968, 2048, th) for th in (8, 15)]
    rows, acc = [], {}
    for rep in range(1, a.reps + 1):
        for prompt, ub, th in cells:
            for kr in (0, 1):
                sc = f"/tmp/kred_{prompt}_{th}_{kr}.csv"
                r = run(prompt, ub, th, kr, a.inner, sc)
                r.update(rep=rep, prompt=prompt, ub=ub, threads=th, kreduce=kr)
                rows.append(r); acc.setdefault((prompt, th, kr), []).append(r)
            print(f"  [{rep}] pp{prompt} t{th} done", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in keys})
    print(f"\n{'prompt':>7}{'th':>4}{'part off':>10}{'part on':>9}"
          f"{'partacc':>9}{'partcopy':>9}{'epi thr off':>12}{'epi thr on':>11}{'gain':>8}")
    for prompt, ub, th in cells:
        g = lambda kr, k: st.median([x.get(k, 0.0) for x in acc[(prompt, th, kr)]])
        off, on = g(0, "tok_s"), g(1, "tok_s")
        print(f"{prompt:>7}{th:>4}{off:>10.1f}{on:>9.1f}"
              f"{g(1,'partacc_ms'):>9.1f}{g(1,'partcopy_ms'):>9.1f}"
              f"{g(0,'epi_all_thread_ms'):>12.0f}{g(1,'epi_all_thread_ms'):>11.0f}"
              f"{on/off:>7.3f}x")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
