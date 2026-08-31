#!/usr/bin/env python3
"""Measure AMD's stock d=64 flash-attention operator on this machine at BitNet's
sequence lengths.

Rationale: the operator (amd/IRON @ d9e4ec5, aie_kernels/aie2p/mha.cc +
iron/operators/mha/design.py) is locked to head_dim=64 and BitNet needs 128.
Porting it means re-deriving the PV and rescale C-tile indexing. That is only
worth starting if the d=64 kernel already clears the budget, so this measures
the stock op first and treats a failure to clear as a cheap negative result.

FLOP-equivalent stand-in: BitNet attention is 20 Q heads / 5 KV heads at d=128.
QK^T and PV both scale as heads*d, so 40 heads / 10 KV heads at d=64 does the
SAME arithmetic with the same GQA ratio of 4. That is the configuration measured
here, and it is what makes a d=64 measurement informative about a d=128 model.

It is FLOP-equivalent, NOT softmax-equivalent: the proxy runs 40 head-wise
softmax rows per query position where real d=128 BitNet runs 20, each over the
same number of keys -- so it does 2x BitNet's softmax work and 1x its QK/PV work.
Whether that difference could matter is settled in
artifacts/attention-feasibility/FINAL_GATE.md section 4, by bounding rather than
by measuring: halving the ENTIRE kernel is a lower bound on any d=128 gain, and
even that bound loses to the CPU.

The reference is computed in numpy rather than torch: IRON's own reference.py
needs torch, and its run_test needs XRTTensor.from_torch, which does not exist
in the mlir-aie 1.4.2 installed here (IRON pins 1.4.2.dev16). Everything else in
IRON works against our toolchain unchanged.

Three timings are reported separately, because only the last decides viability:
  kernel  -- XRT dispatch..wait, as the runtime itself measures it
  staging -- explicit host->device sync of Q/K/V plus device->host sync of O
  burden  -- kernel + host-side dispatch overhead + staging

Artifact identity is printed and recorded for every case. IRON names xclbins
only from repr=True dataclass fields, and num_of_pipelines is repr=False, so two
different pipeline counts collide on one filename. --fresh clears the build
directory per case, which is mandatory whenever the pipeline count varies.
"""
import argparse, csv, hashlib, json, os, shutil, statistics as st, sys, time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.environ.get("IRON_DIR", "/tmp/xdnaresearch/iron"))
from iron.common import AIEContext                      # noqa: E402
from iron.operators.mha.op import MHA                   # noqa: E402
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor  # noqa: E402
import aie.utils as aie_utils                           # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    scale = 1.0 / np.sqrt(d)
    mask = np.triu(np.ones((S, S), dtype=bool), 1)
    O = np.empty((heads, S, d), dtype=np.float32)
    for h in range(heads):                     # per head, to bound peak memory
        kv = h // grp
        s = (Q[h].astype(np.float32) @ K[kv].astype(np.float32).T) * scale
        s[mask] = -np.inf
        s -= s.max(axis=-1, keepdims=True)
        np.exp(s, out=s)
        s /= s.sum(axis=-1, keepdims=True)
        O[h] = s @ V[kv].astype(np.float32)
    return Q, K, V, O


def run_case(S, heads, kv_heads, pipes, warmup, iters, build_dir, fresh, verify=True):
    d = 64
    if fresh and build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    ctx = AIEContext(build_dir=build_dir, mlir_verbose=False, compiler="peano")

    t_build = time.time()
    op = MHA(num_heads=heads, seq_len=S, d=d, num_KV_heads=kv_heads,
             num_of_pipelines=pipes, context=ctx)
    op.compile()
    build_s = time.time() - t_build

    xcl = Path(op.xclbin_artifact.filename)
    ident = dict(xclbin=xcl.name, xclbin_bytes=xcl.stat().st_size,
                 xclbin_sha256=sha256(xcl))

    # The design pads the sequence to a multiple of B_q * pipelines and executes
    # the padded length. A request for 3968 at 8 pipelines runs 4096 of work.
    S_pad = op._calculate_seq_padding(S, pipes)

    Q, K, V, Oref = (golden(heads, kv_heads, S, d) if verify
                     else (np.zeros((heads, S, d), bfloat16),
                           np.zeros((kv_heads, S, d), bfloat16),
                           np.zeros((kv_heads, S, d), bfloat16), None))

    spec = op.get_arg_spec()
    bufs, in_bytes = [], 0
    for arr, sp in zip((Q, K, V), spec[:3]):
        padded = op._pack_compact_to_padded(arr, arr.shape[0], S, S_pad, d)
        t = XRTTensor(np.ascontiguousarray(padded).reshape(-1).astype(sp.dtype),
                      dtype=sp.dtype)
        bufs.append(t); in_bytes += t.buffer_object().size()
    out = XRTTensor(tuple(spec[3].shape), dtype=spec[3].dtype)
    bufs.append(out)
    out_bytes = out.buffer_object().size()

    fn = op.get_callable()
    for _ in range(warmup):
        fn(*bufs)

    # Kernel + host dispatch overhead. Inputs are already resident on the device
    # in this loop, so to("npu") is a no-op and this excludes staging by design.
    kern_ms, disp_ms = [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        r = fn(*bufs)
        disp_ms.append((time.perf_counter() - t0) * 1000.0)
        kern_ms.append(r.npu_time / 1e6)

    # Staging, measured on its own: every input pushed and the output pulled, as
    # a real per-layer integration would have to do.
    stage_ms = []
    for _ in range(max(iters // 2, 5)):
        t0 = time.perf_counter()
        for t in bufs[:3]:
            t.storage.sync_to_device(0, t.buffer_object().size())
        out.storage.sync_from_device(0, out_bytes)
        stage_ms.append((time.perf_counter() - t0) * 1000.0)

    row = dict(seq_len=S, seq_pad=S_pad, heads=heads, kv_heads=kv_heads,
               pipes=pipes, d=d, build_s=round(build_s, 1), iters=iters,
               kernel_ms=round(st.median(kern_ms), 3),
               kernel_ms_min=round(min(kern_ms), 3),
               kernel_ms_sd=round(st.pstdev(kern_ms), 3),
               dispatch_ms=round(st.median(disp_ms), 3),
               staging_ms=round(st.median(stage_ms), 3),
               in_mib=round(in_bytes / 1048576.0, 1),
               out_mib=round(out_bytes / 1048576.0, 1), **ident)
    row["burdened_ms"] = round(row["dispatch_ms"] + row["staging_ms"], 3)

    if verify:
        got = np.asarray(out.data).reshape(heads, S_pad, d)[:, :S, :].astype(np.float32)
        err = np.abs(got - Oref)
        tol = 4.0e-2 * np.abs(Oref) + 1.5e-1
        bad = int((err > tol).sum())
        row.update(bad_elems=bad, total_elems=int(Oref.size),
                   bad_frac=round(bad / Oref.size, 6),
                   rel_l2=round(float(np.linalg.norm(got - Oref) /
                                      max(np.linalg.norm(Oref), 1e-30)), 5))
    for t in bufs:
        del t
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--heads", type=int, default=40)      # 20 heads x d=128 equivalent
    ap.add_argument("--kv-heads", type=int, default=10)   # GQA ratio 4, as BitNet
    ap.add_argument("--pipes", type=int, nargs="+", default=[8])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the numpy golden (it is O(heads*S^2*d) and slow at 4K)")
    # Fresh by default, and it must stay that way whenever the pipeline count
    # varies: IRON names xclbins only from repr=True fields and num_of_pipelines
    # is not one, so a stale build directory silently reuses the wrong xclbin.
    ap.add_argument("--no-fresh", dest="fresh", action="store_false", default=True)
    ap.add_argument("--build-dir",
                    default=os.environ.get("MHA_BUILD_DIR", "/tmp/bitnet-mha-build"))
    ap.add_argument("--out", default="artifacts/attention-feasibility/npu_mha.csv")
    a = ap.parse_args()

    rows = []
    print(f"stock d=64 MHA, heads={a.heads} kv={a.kv_heads} "
          f"(FLOP-equivalent to BitNet's 20 heads x d=128, GQA 4)")
    print(f"build dir {a.build_dir}  fresh={a.fresh}")
    print(f"{'S':>6}{'Spad':>6}{'pipe':>5}{'build':>7}{'kernel':>9}{'stage':>8}"
          f"{'burden':>9}{'x30L':>10}{'relL2':>9}{'bad%':>8}")
    for pipes in a.pipes:
        for S in a.seqs:
            try:
                r = run_case(S, a.heads, a.kv_heads, pipes, a.warmup, a.iters,
                             Path(a.build_dir), a.fresh, verify=not a.no_verify)
            except Exception as e:
                print(f"{S:>6}{'':>6}{pipes:>5}   FAILED: {type(e).__name__}: {str(e)[:90]}")
                rows.append(dict(seq_len=S, pipes=pipes,
                                 error=f"{type(e).__name__}: {e}"))
                continue
            r["prefill_kernel_ms_30L"] = round(r["kernel_ms"] * a.layers, 1)
            r["prefill_burdened_ms_30L"] = round(r["burdened_ms"] * a.layers, 1)
            rows.append(r)
            print(f"{S:>6}{r['seq_pad']:>6}{pipes:>5}{r['build_s']:>7.1f}"
                  f"{r['kernel_ms']:>9.3f}{r['staging_ms']:>8.3f}"
                  f"{r['burdened_ms']:>9.3f}{r['prefill_burdened_ms_30L']:>10.1f}"
                  f"{r.get('rel_l2', float('nan')):>9.5f}"
                  f"{r.get('bad_frac', float('nan'))*100:>7.3f}%")
            print(f"       xclbin {r['xclbin']}  "
                  f"{r['xclbin_bytes']}B  sha {r['xclbin_sha256'][:16]}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in keys})
    print(f"wrote {a.out}")
    aie_utils.DefaultNPURuntime.cleanup()


if __name__ == "__main__":
    main()
