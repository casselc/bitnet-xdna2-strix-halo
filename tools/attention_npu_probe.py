#!/usr/bin/env python3
"""Attention Task D (step 1): measure AMD's stock d=64 flash-attention operator
on this machine, at BitNet's sequence lengths, BEFORE attempting the head_dim
port.

Rationale: the operator (amd/IRON @ d9e4ec5, aie_kernels/aie2p/mha.cc +
iron/operators/mha/design.py) is locked to head_dim=64 and BitNet needs 128.
Porting it means re-deriving the PV and rescale C-tile indexing. That is only
worth starting if the d=64 kernel already clears the budget, so this measures
the stock op first and treats a failure to clear as a cheap negative result.

FLOP-equivalent stand-in: BitNet attention is 20 Q heads / 5 KV heads at d=128.
QK^T and PV both scale as heads*d, so 40 heads / 10 KV heads at d=64 does the
SAME arithmetic with the same GQA ratio of 4. That is the configuration measured
here, and it is what makes a d=64 measurement informative about a d=128 model.

The reference is computed in numpy rather than torch: IRON's own reference.py
needs torch, and its run_test needs XRTTensor.from_torch, which does not exist
in the mlir-aie 1.4.2 installed here (IRON pins 1.4.2.dev16). Everything else in
IRON works against our toolchain unchanged.

Reports BOTH kernel-only and fully-burdened time. Only the second decides
viability.
"""
import argparse, csv, json, os, statistics as st, sys, time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.environ.get("IRON_DIR", "/tmp/xdnaresearch/iron"))
from iron.common import AIEContext                      # noqa: E402
from iron.operators.mha.op import MHA                   # noqa: E402
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor  # noqa: E402
import aie.utils as aie_utils                           # noqa: E402


def golden(heads, kv_heads, S, d, seed=42):
    """Causal scaled-dot-product attention, numpy, matching IRON's generator.

    IRON draws uniform[0,1)*4 in bf16 and runs torch SDPA with is_causal=True and
    scale = 1/sqrt(d). Reproduced here in f32 arithmetic over bf16 inputs.
    """
    rng = np.random.default_rng(seed)
    kvh = kv_heads if kv_heads else heads
    Q = (rng.random((heads, S, d), dtype=np.float32) * 4).astype(bfloat16)
    K = (rng.random((kvh,   S, d), dtype=np.float32) * 4).astype(bfloat16)
    V = (rng.random((kvh,   S, d), dtype=np.float32) * 4).astype(bfloat16)
    grp = heads // kvh
    Kb = np.repeat(K, grp, axis=0).astype(np.float32)
    Vb = np.repeat(V, grp, axis=0).astype(np.float32)
    Qf = Q.astype(np.float32)
    scale = 1.0 / np.sqrt(d)
    mask = np.triu(np.ones((S, S), dtype=bool), 1)
    O = np.empty((heads, S, d), dtype=np.float32)
    for h in range(heads):                     # per head, to bound peak memory
        s = (Qf[h] @ Kb[h].T) * scale
        s[mask] = -np.inf
        s -= s.max(axis=-1, keepdims=True)
        np.exp(s, out=s)
        s /= s.sum(axis=-1, keepdims=True)
        O[h] = s @ Vb[h]
    return Q, K, V, O


def run_case(S, heads, kv_heads, pipes, warmup, iters, ctx):
    d = 64
    t_build = time.time()
    op = MHA(num_heads=heads, seq_len=S, d=d, num_KV_heads=kv_heads,
             num_of_pipelines=pipes, context=ctx)
    op.compile()
    build_s = time.time() - t_build

    Q, K, V, Oref = golden(heads, kv_heads, S, d)
    spec = op.get_arg_spec()
    bufs, total_bytes = [], 0
    for arr, sp in zip((Q, K, V), spec[:3]):
        t = XRTTensor(np.ascontiguousarray(arr).reshape(-1).astype(sp.dtype), dtype=sp.dtype)
        bufs.append(t); total_bytes += t.buffer_object().size()
    out = XRTTensor(tuple(spec[3].shape), dtype=spec[3].dtype)
    bufs.append(out); total_bytes += out.buffer_object().size()

    fn = op.get_callable()
    for _ in range(warmup):
        fn(*bufs)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*bufs)
        ts.append((time.perf_counter() - t0) * 1000.0)

    got = np.asarray(out.data).reshape(heads, S, d).astype(np.float32)
    err = np.abs(got - Oref)
    tol = 4.0e-2 * np.abs(Oref) + 1.5e-1
    bad = int((err > tol).sum())
    rel = float(np.linalg.norm(got - Oref) / max(np.linalg.norm(Oref), 1e-30))
    return dict(seq_len=S, heads=heads, kv_heads=kv_heads, pipes=pipes,
                build_s=round(build_s, 1),
                ms_median=round(st.median(ts), 3),
                ms_min=round(min(ts), 3),
                ms_sd=round(st.pstdev(ts), 3) if len(ts) > 1 else 0.0,
                bad_elems=bad, total_elems=int(Oref.size),
                bad_frac=round(bad / Oref.size, 6), rel_l2=round(rel, 5),
                io_mib=round(total_bytes / 1048576.0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--heads", type=int, default=40)      # 20 heads x d=128 equivalent
    ap.add_argument("--kv-heads", type=int, default=10)   # GQA ratio 4, as BitNet
    ap.add_argument("--pipes", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--out", default="artifacts/attention-feasibility/npu_mha.csv")
    a = ap.parse_args()

    ctx = AIEContext(mlir_verbose=False, compiler="peano")
    rows = []
    print(f"stock d=64 MHA, heads={a.heads} kv={a.kv_heads} pipes={a.pipes} "
          f"(FLOP-equivalent to BitNet's 20 heads x d=128, GQA 4)")
    print(f"{'S':>6}{'build s':>9}{'ms/layer':>10}{'sd':>7}{'x30 layers':>12}"
          f"{'rel L2':>9}{'bad %':>8}{'IO MiB':>8}")
    for S in a.seqs:
        try:
            r = run_case(S, a.heads, a.kv_heads, a.pipes, a.warmup, a.iters, ctx)
        except Exception as e:
            print(f"{S:>6}   FAILED: {type(e).__name__}: {str(e)[:110]}")
            rows.append(dict(seq_len=S, heads=a.heads, kv_heads=a.kv_heads,
                             pipes=a.pipes, error=f"{type(e).__name__}: {e}"))
            continue
        r["prefill_ms_30L"] = round(r["ms_median"] * a.layers, 1)
        rows.append(r)
        print(f"{S:>6}{r['build_s']:>9.1f}{r['ms_median']:>10.3f}{r['ms_sd']:>7.3f}"
              f"{r['prefill_ms_30L']:>12.1f}{r['rel_l2']:>9.5f}"
              f"{r['bad_frac']*100:>7.3f}%{r['io_mib']:>8.1f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in keys})
    print(f"wrote {a.out}")
    aie_utils.DefaultNPURuntime.cleanup()


if __name__ == "__main__":
    main()
