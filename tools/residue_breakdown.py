#!/usr/bin/env python3
"""Task 2: decompose prefill into dependency-meaningful categories.

Uses the per-node profiler (runtime/ggml_node_profile.c, BITNET_PROFILE=<path>),
which brackets every ggml graph node on thread 0 between "before compute" and
"after ggml_barrier". Because ggml barriers every node, those intervals tile the
graph exactly -- verified: the node durations sum to within 0.1% of the graph
span, so there is no unattributed scheduler overhead to hunt for.

NPU device time is attributed per node from the XDNA dispatch-time counter
sampled around each node, so the split between "NPU device time" and "CPU work
inside an offloaded node" is measured, not modelled.

llama-bench runs one warmup graph before the timed reps; the warmup carries
one-time weight repack/upload and is excluded here (see the duty-cycle
correction in artifacts/overlap-de-risk/RESULTS.md section 0).
"""
import argparse, collections, csv, json, os, re, statistics as st, subprocess, sys
from pathlib import Path

BIN   = "refs/BitNet/build-xdna3/bin/llama-bench"
MODEL = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
ART   = os.path.abspath("artifacts/xclbin-tuned")

# Ordered: first match wins. Keyed on (op, role) where role is the node name with
# its trailing layer index stripped.
RULES = [
    ("attention",        lambda op, r: op == "FLASH_ATTN_EXT"),
    ("attn_q_proj",      lambda op, r: op == "MUL_MAT" and r == "Qcur"),
    ("attn_k_proj",      lambda op, r: op == "MUL_MAT" and r == "Kcur"),
    ("attn_v_proj",      lambda op, r: op == "MUL_MAT" and r == "Vcur"),
    ("attn_out_proj",    lambda op, r: op == "MUL_MAT" and r == "attn_out"),
    ("ffn_gate",         lambda op, r: op == "MUL_MAT" and r == "ffn_gate"),
    ("ffn_up",           lambda op, r: op == "MUL_MAT" and r == "ffn_up"),
    ("ffn_down",         lambda op, r: op == "MUL_MAT" and r == "ffn_down"),
    ("lm_head",          lambda op, r: op == "MUL_MAT" and r.startswith("result")),
    ("norm",             lambda op, r: op in ("RMS_NORM", "NORM")),
    ("ffn_activation",   lambda op, r: op == "GLU"),
    ("rope",             lambda op, r: op == "ROPE"),
    ("residual_add",     lambda op, r: op == "ADD"),
    ("kv_cache_write",   lambda op, r: op == "SET_ROWS"),
    ("embedding",        lambda op, r: op == "GET_ROWS"),
]
NPU_ELIGIBLE = {"attn_q_proj", "attn_out_proj", "ffn_gate", "ffn_up", "ffn_down"}

def classify(op, role):
    for name, f in RULES:
        if f(op, role):
            return name
    return "other"

def role_of(name):
    return re.sub(r"-\d+$", "", name) or "?"

def run(prompt, ub, threads, hybrid, reps, trace, tiles=None):
    env = dict(os.environ, BITNET_XDNA="1" if hybrid else "0",
               BITNET_XDNA_ARTIFACTS=ART, BITNET_PROFILE=trace)
    if tiles is not None:
        env["BITNET_XDNA_TILES"] = str(tiles)
    p = subprocess.run([BIN, "-m", MODEL, "-p", str(prompt), "-n", "0", "-t", str(threads),
                        "-ngl", "0", "-r", str(reps), "-ub", str(ub)],
                       capture_output=True, text=True, env=env, timeout=3600)
    o = p.stdout + p.stderr
    m = re.search(rf"pp{prompt} \|\s*([0-9.]+)", o)
    return float(m.group(1)) if m else None

def aggregate(trace, n_mb):
    """-> (per_position, spans) keyed by micro-batch position within a prefill.

    Each ggml graph evaluation is one micro-batch. A prefill of `n_mb`
    micro-batches is therefore `n_mb` consecutive graphs, and llama-bench's
    warmup is a whole prefill -- i.e. the first `n_mb` graphs, not the first one.
    Positions must be kept separate: under causal attention mb1 attends to twice
    the keys mb0 does, so averaging them hides exactly the structure that decides
    whether pipelining can work.
    """
    per   = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0.0, 0]))
    span  = {}
    for line in open(trace):
        r = json.loads(line)
        g = r["graph"]
        c = classify(r["op"], role_of(r["name"]))
        e = per[g][c]
        e[0] += r["dur_us"] - r["npu_us"]
        e[1] += r["npu_us"]
        e[2] += 1
        lo, hi = span.get(g, (r["t0_us"], r["t1_us"]))
        span[g] = (min(lo, r["t0_us"]), max(hi, r["t1_us"]))

    graphs = sorted(per)
    graphs = graphs[n_mb:]                       # drop the whole warmup prefill
    by_pos = collections.defaultdict(list)       # position -> [graph ids]
    for i, g in enumerate(graphs):
        by_pos[i % n_mb].append(g)
    spans = {pos: [span[g][1] - span[g][0] for g in gs] for pos, gs in by_pos.items()}
    return {pos: [per[g] for g in gs] for pos, gs in by_pos.items()}, spans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, required=True)
    ap.add_argument("--ub", type=int, nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=15)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--modes", nargs="+", default=["hybrid", "cpu"])
    ap.add_argument("--tiles", type=int, default=None,
                    help="force BITNET_XDNA_TILES (2 = give the NPU every token tile), "
                         "used to measure a pure-NPU assignment for the scheduler")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for ub in a.ub:
        for mode in a.modes:
            trace = f"/tmp/bnp_{a.prompt}_{ub}_{mode}{'_t'+str(a.tiles) if a.tiles is not None else ''}.jsonl"
            tok = run(a.prompt, ub, a.threads, mode == "hybrid", a.reps, trace, a.tiles)
            n_mb = max(1, -(-a.prompt // ub))
            per, spans = aggregate(trace, n_mb)
            print(f"\n== pp{a.prompt} ub{ub} {mode} t{a.threads}: {tok:.1f} tok/s, "
                  f"{n_mb} micro-batch(es)/prefill, "
                  f"{sum(len(v) for v in per.values())} timed graphs ==")
            prefill_total = 0.0
            for pos in sorted(per):
                graphs = per[pos]
                cats = sorted({c for g in graphs for c in g})
                summary = {c: (st.median([g[c][0] for g in graphs])/1000,
                               st.median([g[c][1] for g in graphs])/1000,
                               st.median([g[c][2] for g in graphs])) for c in cats}
                tot = sum(cpu+npu for cpu, npu, _ in summary.values())
                prefill_total += tot
                gs = st.median(spans[pos])/1000
                print(f"  -- micro-batch {pos} of {n_mb}: span {gs:.1f} ms "
                      f"(nodes {tot:.1f} ms, {tot/gs*100:.1f}% accounted)")
                print(f"     {'category':<18}{'CPU ms':>9}{'NPU ms':>9}{'total':>9}{'%':>7}{'n':>6}")
                for c, (cpu, npu, cnt) in sorted(summary.items(), key=lambda x: -(x[1][0]+x[1][1])):
                    if cpu + npu < 0.05: continue
                    print(f"     {c:<18}{cpu:>9.1f}{npu:>9.1f}{cpu+npu:>9.1f}"
                          f"{(cpu+npu)/tot*100:>6.1f}%{cnt:>6.0f}")
                    rows.append(dict(prompt=a.prompt, ub=ub, threads=a.threads, mode=mode,
                                     tok_s=tok, n_micro_batches=n_mb, mb_pos=pos,
                                     category=c, cpu_ms=round(cpu,2), npu_ms=round(npu,2),
                                     count=int(cnt), span_ms=round(gs,1),
                                     npu_eligible=int(c in NPU_ELIGIBLE)))
            print(f"  == prefill total across micro-batches: {prefill_total:.1f} ms "
                  f"(llama-bench wall {a.prompt/tok*1000:.1f} ms)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")

if __name__ == "__main__":
    main()
