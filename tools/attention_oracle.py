#!/usr/bin/env python3
"""Attention Task C: matched CPU reference measurements for BitNet-2B prefill.

This is the baseline any NPU attention kernel must beat, measured in situ with
the per-node profiler rather than derived from a FLOP model.

Geometry, read from the GGUF and the runtime (not from family defaults):
  20 Q heads, 5 KV heads (GQA 4), head_dim 128, 30 layers, context 4096
  Q enters attention as f32; KV cache is f16 (llama.cpp default)
  output is f32, ne = [128, 20, T]; op is FLASH_ATTN_EXT

KV traffic is reported two ways, because conflating them would mislead the NPU
comparison:

  LOGICAL  what the algorithm reads if every query re-reads every key it attends
           to: 2 (K,V) * 5 heads * 128 dims * T * (T/2) * 2 bytes * 30 layers.
           Divided by attention time this yields 182-411 GiB/s, which EXCEEDS
           this machine's DRAM bandwidth. That is not a measurement error: one
           layer's KV at T=3968 is only 10.2 MiB and fits comfortably in the
           64 MiB L3, so the CPU is largely re-reading cache, not DRAM.

  PHYSICAL the compulsory traffic a tiled kernel must actually move from memory:
           each layer's KV read once, 2 * 5 * 128 * T * 2 bytes * 30 layers.
           This is the figure an NPU kernel would have to stream, and it is far
           smaller than the logical one.

Reporting only the logical rate would make the NPU's job look impossible;
reporting only the physical one would flatter it.
"""
import argparse, collections, csv, json, os, re, statistics as st, subprocess
from pathlib import Path

BIN   = "refs/BitNet/build-xdna3/bin/llama-bench"
MODEL = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
ART   = os.path.abspath("artifacts/xclbin-tuned")
N_LAYER, N_KV_HEAD, HEAD_DIM, N_Q_HEAD = 30, 5, 128, 20

def run(prompt, threads, reps, trace, cpu_only):
    env = dict(os.environ, BITNET_XDNA="0" if cpu_only else "1",
               BITNET_XDNA_ARTIFACTS=ART, BITNET_PROFILE=trace)
    ub = max(prompt, 1)
    p = subprocess.run([BIN, "-m", MODEL, "-p", str(prompt), "-n", "0", "-t", str(threads),
                        "-ngl", "0", "-r", str(reps), "-ub", str(ub)],
                       capture_output=True, text=True, env=env, timeout=3600)
    o = p.stdout + p.stderr
    m = re.search(rf"pp{prompt} \|\s*([0-9.]+)", o)
    return float(m.group(1)) if m else None

def attn_ms(trace):
    """Median FLASH_ATTN_EXT wall time per prefill, warmup graph dropped."""
    per = collections.defaultdict(float)
    span = {}
    for line in open(trace):
        r = json.loads(line)
        if r["op"] == "FLASH_ATTN_EXT":
            per[r["graph"]] += r["dur_us"]
        lo, hi = span.get(r["graph"], (r["t0_us"], r["t1_us"]))
        span[r["graph"]] = (min(lo, r["t0_us"]), max(hi, r["t1_us"]))
    gs = sorted(per)[1:]                       # drop llama-bench's warmup prefill
    return (st.median([per[g] for g in gs]) / 1000.0,
            st.median([(span[g][1] - span[g][0]) for g in gs]) / 1000.0)

def kv_bytes_logical(T):
    # every query re-reads the ~T/2 keys it attends to
    return 2 * N_KV_HEAD * HEAD_DIM * T * (T / 2.0) * 2 * N_LAYER

def kv_bytes_physical(T):
    # compulsory: each layer's KV read once by a tiled kernel
    return 2 * N_KV_HEAD * HEAD_DIM * T * 2 * N_LAYER

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[8, 15])
    ap.add_argument("--prompts", type=int, nargs="+", default=[512, 1024, 2048, 3968])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="artifacts/attention-feasibility/cpu_oracle.csv")
    a = ap.parse_args()
    rows = []
    print(f"{'T':>6}{'thr':>5}{'prefill ms':>12}{'attn ms':>10}{'attn %':>8}"
          f"{'attn tok/s':>12}{'logi GiB':>10}{'logi GiB/s':>11}{'phys MiB':>10}")
    for th in a.threads:
        for T in a.prompts:
            trace = f"/tmp/attn_{T}_{th}.jsonl"
            tok = run(T, th, a.reps, trace, cpu_only=True)
            att, wall = attn_ms(trace)
            kvb = kv_bytes_logical(T); kvp = kv_bytes_physical(T)
            rows.append(dict(prompt=T, threads=th, prefill_tok_s=round(tok,1),
                             prefill_ms=round(wall,1), attn_ms=round(att,1),
                             attn_frac=round(att/wall,4),
                             attn_tok_s=round(T/(att/1000.0),1),
                             kv_bytes_logical=int(kvb),
                             kv_bytes_physical=int(kvp),
                             kv_logical_gib_s=round(kvb/1073741824.0/(att/1000.0),2),
                             kv_physical_mib=round(kvp/1048576.0,1)))
            r = rows[-1]
            print(f"{T:>6}{th:>5}{wall:>12.1f}{att:>10.1f}{r['attn_frac']*100:>7.1f}%"
                  f"{r['attn_tok_s']:>12.1f}{kvb/1073741824.0:>10.2f}"
                  f"{r['kv_logical_gib_s']:>11.2f}{r['kv_physical_mib']:>10.1f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
