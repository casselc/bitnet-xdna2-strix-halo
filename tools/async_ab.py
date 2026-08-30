#!/usr/bin/env python3
"""Task 5: async-dispatch spike, measured interleaved.

Pipelines N-chunk dispatches so the host-side evacuation of one chunk's results
overlaps the device executing the next, instead of running while the NPU is
idle. BITNET_XDNA_ASYNC=1 selects it; the synchronous path is unchanged.

Interleaved round-robin, never blocked: this machine shows 10-30% between-run
drift and block-ordered A/B has produced false positives here before.
"""
import argparse, csv, os, re, statistics as st, subprocess, time
from pathlib import Path

BIN   = "refs/BitNet/build-xdna3/bin/llama-bench"
MODEL = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
ART   = os.path.abspath("artifacts/xclbin-tuned")

def run(prompt, ub, threads, async_on, inner):
    env = dict(os.environ, BITNET_XDNA="1", BITNET_XDNA_ARTIFACTS=ART,
               BITNET_XDNA_ASYNC="1" if async_on else "0", BITNET_XDNA_STATS="1")
    p = subprocess.run([BIN, "-m", MODEL, "-p", str(prompt), "-n", "0", "-t", str(threads),
                        "-ngl", "0", "-r", str(inner), "-ub", str(ub)],
                       capture_output=True, text=True, env=env, timeout=3600)
    o = p.stdout + p.stderr
    tok = re.search(rf"pp{prompt} \|\s*([0-9.]+)", o)
    so  = re.search(r"stage_out=([0-9.]+) ms", o)
    dev = re.search(r"dispatch_total=([0-9.]+)", o)
    return (float(tok.group(1)) if tok else None,
            float(so.group(1)) if so else 0.0,
            float(dev.group(1)) if dev else 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=15)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--out", default="artifacts/overlap-de-risk/sibling_overlap.csv")
    a = ap.parse_args()

    cases = [(2048, 2048), (3968, 2048), (2048, 1024)]
    rows, acc = [], {}
    for rep in range(1, a.reps + 1):
        for prompt, ub in cases:
            for async_on in (False, True):
                tok, so, dev = run(prompt, ub, a.threads, async_on, a.inner)
                rows.append(dict(rep=rep, prompt=prompt, ub=ub, threads=a.threads,
                                 async_on=int(async_on), tok_s=tok,
                                 stage_out_ms=so, device_ms=dev))
                acc.setdefault((prompt, ub, async_on), []).append(tok)
                print(f"  [{rep}] pp{prompt} ub{ub} async={int(async_on)}  "
                      f"{tok:8.1f} tok/s  stage_out={so:7.1f} ms", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n{'config':<18}{'sync':>10}{'async':>10}{'gain':>8}{'sync sd':>9}{'async sd':>10}")
    for prompt, ub in cases:
        s = acc[(prompt, ub, False)]; y = acc[(prompt, ub, True)]
        print(f"pp{prompt} ub{ub:<11}{st.median(s):>10.1f}{st.median(y):>10.1f}"
              f"{st.median(y)/st.median(s):>7.3f}x{st.pstdev(s):>9.1f}{st.pstdev(y):>10.1f}")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
