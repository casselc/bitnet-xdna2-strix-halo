#!/usr/bin/env python3
"""The REAL GEMM <-> MHA hardware-context-switch cost.

`tools/npu_two_context.cpp` measured this with a surrogate: two production-sized
GEMM designs alternating, giving +5.10 ms per pair and ~153 ms per prefill. That
was the best available answer before an MHA xclbin existed. One now does, so this
replaces the surrogate with the actual pair the integration would run:

  A -- the production BitNet GEMM context, exactly the xclbin the frozen runtime
       loads (artifacts/xclbin-tuned/mm_M1024_K2560_N2560)
  B -- the stock IRON MHA context at the FLOP-equivalent BitNet geometry
       (40 Q heads / 10 KV heads / d=64), built by this branch

Both contexts are held open simultaneously, which is the condition being
measured: XDNA2 hardware contexts cannot be co-resident, so every alternation
forces the array to be reconfigured.

Measurement structure follows npu_two_context.cpp: alone and alternating are
INTERLEAVED within each cycle rather than run as separate blocks, because this
machine drifts 10-30% between runs. The "alone" samples are taken after three
same-context dispatches so the context is certainly resident.

Every artifact's absolute path, size and SHA-256 is printed before timing. This
project has twice been fooled by loading the wrong xclbin (the M512 designs via
argv[1], and the pipes=4 name collision), so identity is evidence, not decoration.
"""
import argparse, csv, hashlib, json, os, statistics as st, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("IRON_DIR", "/tmp/xdnaresearch/iron"))
import pyxrt                                             # noqa: E402
from iron.common import AIEContext                       # noqa: E402
from iron.operators.mha.op import MHA                    # noqa: E402
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor  # noqa: E402
import aie.utils as aie_utils                            # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(label, *paths):
    """Record what was actually loaded. Paths are reported relative to the repo
    when they live in it, and by name alone when they are build scratch: the
    SHA-256 is the identity, and an absolute scratch path is only noise."""
    out = []
    repo = Path.cwd().resolve()
    for p in paths:
        p = Path(p).resolve()
        try:
            shown = str(p.relative_to(repo))
        except ValueError:
            shown = f"<build>/{p.name}"
        rec = dict(role=label, path=shown, bytes=p.stat().st_size,
                   sha256=sha256(p))
        print(f"  {label:5s} {shown}")
        print(f"        {rec['bytes']:>9} B   sha256 {rec['sha256']}")
        out.append(rec)
    return out


class GemmContext:
    """The production BitNet GEMM xclbin, driven exactly as the C++ runtime does:
    kern(3, insts_bo, insts_bytes, A, B, C) with HOST_ONLY buffers."""

    def __init__(self, dev, stem, M, K, N):
        self.M, self.K, self.N = M, K, N
        self.xclbin_path = f"{stem}.xclbin"
        self.insts_path = f"{stem}.insts.bin"
        xclbin = pyxrt.xclbin(self.xclbin_path)
        dev.register_xclbin(xclbin)
        self.ctx = pyxrt.hw_context(dev, xclbin.get_uuid())
        kname = next(k.get_name() for k in xclbin.get_kernels()
                     if k.get_name().startswith("MLIR_AIE"))
        self.kern = pyxrt.kernel(self.ctx, kname)
        insts = np.fromfile(self.insts_path, dtype=np.uint32)
        self.insts_bytes = insts.nbytes
        self.bo_insts = pyxrt.bo(dev, self.insts_bytes, pyxrt.bo.cacheable,
                                 self.kern.group_id(1))
        self.bo_insts.write(insts.tobytes(), 0)
        self.bo_insts.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.bo_a = pyxrt.bo(dev, M * K,     pyxrt.bo.host_only, self.kern.group_id(3))
        self.bo_b = pyxrt.bo(dev, K * N,     pyxrt.bo.host_only, self.kern.group_id(4))
        self.bo_c = pyxrt.bo(dev, M * N * 4, pyxrt.bo.host_only, self.kern.group_id(5))
        # Ternary weights, uploaded once and resident, as in the real runtime.
        w = (np.arange(K * N, dtype=np.int64) % 3 - 1).astype(np.int8)
        self.bo_b.write(w.tobytes(), 0)
        self.bo_b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        a = (np.arange(M * K, dtype=np.int64) % 255 - 127).astype(np.int8)
        self.bo_a.write(a.tobytes(), 0)
        self.bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def dispatch(self):
        r = self.kern(3, self.bo_insts, self.insts_bytes,
                      self.bo_a, self.bo_b, self.bo_c)
        r.wait()


class MhaContext:
    def __init__(self, heads, kv_heads, S, pipes, build_dir):
        ctx = AIEContext(build_dir=Path(build_dir), compiler="peano")
        self.op = MHA(num_heads=heads, seq_len=S, d=64, num_KV_heads=kv_heads,
                      num_of_pipelines=pipes, context=ctx)
        self.op.compile()
        self.xclbin_path = self.op.xclbin_artifact.filename
        self.insts_path = self.op.insts_artifact.filename
        S_pad = self.op._calculate_seq_padding(S, pipes)
        spec = self.op.get_arg_spec()
        self.bufs = []
        for h, sp in zip((heads, kv_heads, kv_heads), spec[:3]):
            arr = np.zeros(h * S_pad * 64, dtype=np.dtype(sp.dtype))
            self.bufs.append(XRTTensor(arr, dtype=sp.dtype))
        self.bufs.append(XRTTensor(tuple(spec[3].shape), dtype=spec[3].dtype))
        self.fn = self.op.get_callable()

    def dispatch(self):
        self.fn(*self.bufs)


def p50(v):
    return st.median(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemm-stem",
                    default="artifacts/xclbin-tuned/mm_M1024_K2560_N2560")
    ap.add_argument("--gemm-mkn", type=int, nargs=3, default=[1024, 2560, 2560])
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=40)
    ap.add_argument("--kv-heads", type=int, default=10)
    ap.add_argument("--pipes", type=int, default=8)
    ap.add_argument("--cycles", type=int, default=40)
    ap.add_argument("--build-dir",
                    default=os.environ.get("MHA_BUILD_DIR", "/tmp/bitnet-mha-switch-build"))
    ap.add_argument("--out",
                    default="artifacts/attention-feasibility/gemm_mha_switch.json")
    a = ap.parse_args()

    print("npu_gemm_mha_switch -- real GEMM context vs real MHA context\n")
    print("artifact identity:")
    ids = identity("GEMM", f"{a.gemm_stem}.xclbin", f"{a.gemm_stem}.insts.bin")

    # IRON's runtime opens the device; share it so both contexts live on one
    # device handle, as they would inside one inference process.
    from aie.utils.hostruntime.xrtruntime.device import acquire_device
    dev = acquire_device()

    mha = MhaContext(a.heads, a.kv_heads, a.seq, a.pipes, a.build_dir)
    ids += identity("MHA", mha.xclbin_path, mha.insts_path)
    print(f"\n  GEMM  M={a.gemm_mkn[0]} K={a.gemm_mkn[1]} N={a.gemm_mkn[2]}")
    print(f"  MHA   heads={a.heads} kv={a.kv_heads} d=64 S={a.seq} pipes={a.pipes}\n")

    gemm = GemmContext(dev, a.gemm_stem, *a.gemm_mkn)

    for _ in range(3):
        gemm.dispatch(); mha.dispatch()

    aloneG, aloneM, altG, altM = [], [], [], []
    for _ in range(a.cycles):
        for _ in range(3):
            gemm.dispatch()
        t0 = time.perf_counter(); gemm.dispatch()
        aloneG.append((time.perf_counter() - t0) * 1e3)

        for _ in range(3):
            mha.dispatch()
        t0 = time.perf_counter(); mha.dispatch()
        aloneM.append((time.perf_counter() - t0) * 1e3)

        # Alternating: each dispatch follows one on the OTHER context.
        t0 = time.perf_counter(); gemm.dispatch()
        altG.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter(); mha.dispatch()
        altM.append((time.perf_counter() - t0) * 1e3)

    aG, aM, xG, xM = p50(aloneG), p50(aloneM), p50(altG), p50(altM)
    pair_alone, pair_alt = aG + aM, xG + xM
    incr = pair_alt - pair_alone

    print(f"  {'context':<28}{'alone':>10}{'alternating':>14}{'penalty':>10}")
    print(f"  {'GEMM M1024 K2560 N2560':<28}{aG:>10.3f}{xG:>14.3f}"
          f"{(xG/aG-1)*100:>9.0f}%")
    print(f"  {'MHA S=%d' % a.seq:<28}{aM:>10.3f}{xM:>14.3f}"
          f"{(xM/aM-1)*100:>9.0f}%")
    print(f"  {'pair':<28}{pair_alone:>10.3f}{pair_alt:>14.3f}"
          f"{(pair_alt/pair_alone-1)*100:>9.0f}%")
    print(f"\n  incremental switch cost: {incr:+.3f} ms per GEMM<->MHA pair")
    print(f"  30 layers x 1 pair/layer:  {incr*30:.0f} ms per prefill")
    print(f"  30 layers x 2 pairs/layer: {incr*60:.0f} ms per prefill")

    rec = dict(artifacts=ids, gemm_mkn=a.gemm_mkn, seq=a.seq, heads=a.heads,
               kv_heads=a.kv_heads, pipes=a.pipes, cycles=a.cycles,
               gemm_alone_ms=round(aG, 4), gemm_alt_ms=round(xG, 4),
               mha_alone_ms=round(aM, 4), mha_alt_ms=round(xM, 4),
               gemm_alone_sd=round(st.pstdev(aloneG), 4),
               mha_alone_sd=round(st.pstdev(aloneM), 4),
               gemm_alt_sd=round(st.pstdev(altG), 4),
               mha_alt_sd=round(st.pstdev(altM), 4),
               pair_alone_ms=round(pair_alone, 4), pair_alt_ms=round(pair_alt, 4),
               incremental_pair_ms=round(incr, 4),
               prefill_tax_1pair_per_layer_ms=round(incr * 30, 1),
               prefill_tax_2pair_per_layer_ms=round(incr * 60, 1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=2) + "\n")
    print(f"\nwrote {a.out}")
    aie_utils.DefaultNPURuntime.cleanup()


if __name__ == "__main__":
    main()
