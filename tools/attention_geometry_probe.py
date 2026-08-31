#!/usr/bin/env python3
"""Does d=128 geometry make the AIE QK/PV primitives enough more efficient to
reopen the attention economic case?

The final gate rejected the stock AMD/IRON MHA on a work-proportional bound:
pretend d=128 halves the whole d=64 kernel, and it still loses. That bound is
favourable but it is NOT a lower bound if changing d=64 -> d=128 materially
improves execution efficiency per unit of QK/PV work. This measures exactly that
residual possibility, at the primitive level, without building d=128 attention.

Why IRON's GEMM operator is the right probe. It compiles the SAME kernel source
with the SAME flags that MHA's own op.py passes:

    aie_kernels/aie2p/mm.cc
    -Dbf16_bf16_ONLY -DDIM_M -DDIM_K -DDIM_N -DROUND_CONV_EVEN
    -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16  [-DB_COL_MAJ]

and DIM_M/DIM_K/DIM_N are precisely the per-core tile shape that d changes:

    QK^T : [B_q x d] @ [d x B_kv]   -> d is the CONTRACTION dim (tile_k)
    PV   : [B_q x B_kv] @ [B_kv x d] -> d is the OUTPUT width   (tile_n)

so QK d=64 vs d=128 is tile_k 64 vs 128 with B col-major, and PV d=64 vs d=128
is tile_n 64 vs 128 with B row-major. MHA's B_q = B_kv = 64 throughout.

The comparison is controlled: every variant runs the IDENTICAL whole-problem
(M, K, N), so useful MACs and total bytes moved are identical and only the tile
geometry differs. Efficiency is therefore MACs/second, directly comparable, and
a ratio needs no normalisation argument.

Artifact identity (this project has been fooled three times). IRON names
artifacts only from repr=True dataclass fields, and GEMM declares
emulate_bf16_mmul_with_bfp16, round_conv_even, prio_accuracy, dtype_in/out and
use_scalar as repr=False -- all of which change generated code. So each variant
gets its own build directory keyed on a hash of EVERY field, cleared before use,
and both the xclbin and the instruction-stream digests are recorded. The
instruction stream is the executable identity: the xclbin container was measured
on the previous branch to be non-reproducible across builds of one geometry.
"""
import argparse, csv, hashlib, json, os, shutil, statistics as st, subprocess, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("IRON_DIR", "/tmp/xdnaresearch/iron"))
from iron.common import AIEContext                      # noqa: E402
from iron.operators.gemm.op import GEMM                 # noqa: E402
from iron.operators.softmax.op import Softmax           # noqa: E402
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor  # noqa: E402
import aie.utils as aie_utils                           # noqa: E402

BUILD_ROOT = Path(os.environ.get("GEOM_BUILD_DIR", "/tmp/bitnet-geom-build"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def toolchain():
    def ver(pkg):
        try:
            import importlib.metadata as md
            return md.version(pkg)
        except Exception:
            return "unknown"
    iron_dir = Path(os.environ.get("IRON_DIR", "/tmp/xdnaresearch/iron"))
    try:
        sha = subprocess.run(["git", "-C", str(iron_dir), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = "unknown"
    return dict(iron_commit=sha, mlir_aie=ver("mlir_aie"), llvm_aie=ver("llvm_aie"))


def build_key(kind, cfg):
    """Full-config identity. Deliberately NOT IRON's artifact name: fields that
    change generated code are repr=False there and would collide."""
    blob = json.dumps({"kind": kind, **cfg}, sort_keys=True)
    return f"{kind}_" + hashlib.sha256(blob.encode()).hexdigest()[:16]


def artifact_identity(op, build_dir):
    xcl = Path(op.xclbin_artifact.filename)
    insts = Path(op.insts_artifact.filename)
    rec = dict(xclbin_name=xcl.name, xclbin_bytes=xcl.stat().st_size,
               xclbin_sha256=sha256(xcl),
               insts_name=insts.name, insts_bytes=insts.stat().st_size,
               insts_sha256=sha256(insts))
    # Per-core program memory, when aiecc leaves the ELFs behind. The core ELF
    # size is the closest observable proxy for program-memory pressure, which
    # matters because AIE cores have only 16 KiB of it.
    elfs = sorted(build_dir.rglob("core_*.elf"))
    if elfs:
        sizes = [e.stat().st_size for e in elfs]
        rec.update(core_elf_count=len(sizes), core_elf_max_bytes=max(sizes),
                   core_elf_median_bytes=int(st.median(sizes)))
    objs = sorted(build_dir.glob("*.o"))
    if objs:
        rec["kernel_obj_bytes"] = max(o.stat().st_size for o in objs)
    return rec


def l1_bytes(tm, tk, tn, dtype_bytes=2, stack=3328, rtp=8):
    """Peano's per-core L1 requirement for this tile geometry, double-buffered.

    Reproduces the MemoryMap the allocator prints on failure: A, B and C each get
    two buffers (ObjectFifo depth 2), plus stack and RTP. AIE cores have 64 KiB
    of L1 total, so this predicts which geometries can be built at all -- which
    turned out to be the first real constraint on d=128."""
    return stack + rtp + 2 * dtype_bytes * (tm * tk + tk * tn + tm * tn)


L1_CAPACITY = 65536


class GemmVariant:
    """One (stage, d, block) point: a compiled GEMM whose tile geometry is the
    shape the MHA pipeline would give that stage."""

    def __init__(self, label, stage, d, M, K, N, cols, b_q=64, b_kv=64, verify=True):
        self.label, self.stage, self.d = label, stage, d
        self.b_q, self.b_kv = b_q, b_kv
        self.M, self.K, self.N, self.cols = M, K, N, cols
        # The MHA pipeline's two matmuls, in the GEMM operator's tile terms:
        #   QK^T : [B_q x d] @ [d x B_kv]  -> d is the CONTRACTION dim, B col-major
        #   PV   : [B_q x B_kv] @ [B_kv x d] -> d is the OUTPUT width, B row-major
        if stage == "qk":
            tm, tk, tn, bcm = b_q, d, b_kv, True
        elif stage == "pv":
            tm, tk, tn, bcm = b_q, b_kv, d, False
        else:
            raise ValueError(stage)
        self.l1 = l1_bytes(tm, tk, tn)
        self.fits_l1 = self.l1 <= L1_CAPACITY
        self.build_error = None
        self.cfg = dict(M=M, K=K, N=N, tile_m=tm, tile_k=tk, tile_n=tn,
                        b_col_maj=bcm, c_col_maj=False, num_aie_columns=cols,
                        # held at MHA's values; all repr=False in IRON, so they
                        # must be in OUR key or variants would collide
                        emulate_bf16_mmul_with_bfp16=True, round_conv_even=True,
                        prio_accuracy=False, dtype_in="bf16", dtype_out="bf16",
                        use_scalar=False)
        self.verify = verify
        self.build_dir = BUILD_ROOT / build_key("gemm", self.cfg)

    def compile(self):
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        self.build_dir.mkdir(parents=True, exist_ok=True)
        ctx = AIEContext(build_dir=self.build_dir, compiler="peano")
        t0 = time.time()
        self.op = GEMM(context=ctx, **self.cfg)
        self.op.compile()
        self.build_s = time.time() - t0
        self.ident = artifact_identity(self.op, self.build_dir)

        spec = self.op.get_arg_spec()
        rng = np.random.default_rng(7)
        self.A = rng.standard_normal(tuple(spec[0].shape)).astype(np.dtype(spec[0].dtype))
        self.B = rng.standard_normal(tuple(spec[1].shape)).astype(np.dtype(spec[1].dtype))
        self.bufs = [
            XRTTensor(np.ascontiguousarray(self.A).reshape(-1), dtype=spec[0].dtype),
            XRTTensor(np.ascontiguousarray(self.B).reshape(-1), dtype=spec[1].dtype),
            XRTTensor(tuple(spec[2].shape), dtype=spec[2].dtype),
        ]
        self.out_shape = tuple(spec[2].shape)
        self.in_bytes = sum(b.buffer_object().size() for b in self.bufs[:2])
        self.out_bytes = self.bufs[2].buffer_object().size()
        self.fn = self.op.get_callable()
        self.kern_ms, self.disp_ms = [], []

    def dispatch(self):
        t0 = time.perf_counter()
        r = self.fn(*self.bufs)
        return (time.perf_counter() - t0) * 1e3, r.npu_time / 1e6

    def sample(self, warm, timed):
        for _ in range(warm):
            self.fn(*self.bufs)
        for _ in range(timed):
            d, k = self.dispatch()
            self.disp_ms.append(d); self.kern_ms.append(k)

    def check(self):
        """A fast kernel that computes the wrong thing is worse than a slow one.
        bf16 in and bf16 out over a K=2048 contraction accumulates real error, so
        this is a relative-L2 sanity gate, not a bit-exactness claim."""
        if not self.verify:
            return None
        got = np.asarray(self.bufs[2].data).reshape(self.out_shape).astype(np.float32)
        # iron.operators.gemm.reference is C = A @ (B.T if b_col_maj else B).
        # Reproduced in numpy: IRON's own reference imports torch, which is not
        # installed here for the same reason the MHA probe carries its own golden.
        Bf = self.B.astype(np.float32)
        ref = self.A.astype(np.float32) @ (Bf.T if self.cfg["b_col_maj"] else Bf)
        return float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30))

    def row(self):
        macs = self.M * self.K * self.N
        kern = st.median(self.kern_ms)
        return dict(
            label=self.label, stage=self.stage, d=self.d,
            M=self.M, K=self.K, N=self.N, cols=self.cols, cores=self.cols * 4,
            tile_m=self.cfg["tile_m"], tile_k=self.cfg["tile_k"],
            tile_n=self.cfg["tile_n"], b_col_maj=int(self.cfg["b_col_maj"]),
            build_s=round(self.build_s, 1), reps=len(self.kern_ms),
            kernel_ms=round(kern, 4),
            kernel_ms_min=round(min(self.kern_ms), 4),
            kernel_ms_sd=round(st.pstdev(self.kern_ms), 4),
            dispatch_ms=round(st.median(self.disp_ms), 4),
            macs=macs, gflop=round(2 * macs / 1e9, 3),
            tflops=round(2 * macs / (kern / 1e3) / 1e12, 4),
            tmacs=round(macs / (kern / 1e3) / 1e12, 4),
            b_q=self.b_q, b_kv=self.b_kv,
            in_bytes=self.in_bytes, out_bytes=self.out_bytes,
            l1_bytes_est=self.l1, l1_fits=int(self.fits_l1),
            rel_l2=self.rel_l2, **self.ident)


def run_gemm_geometry(M, K, N, cols, blocks, warm, timed, out_csv, verify=True):
    # Six points, not four. d=128 at MHA's native 64x64 block does not fit L1
    # (see l1_bytes), so a real d=128 port is FORCED to a smaller query block.
    # Measuring d=128 only at tile_m=32 would then confound "the effect of d"
    # with "the effect of a smaller block", so tile_m=32 at d=64 is carried as
    # the control that separates them. The two infeasible native-block d=128
    # points are still attempted, so the build failure is recorded evidence
    # rather than a claim.
    # d=128 at MHA's native 64x64 block does not fit L1 (see l1_bytes), so a real
    # d=128 port is FORCED to shrink a block. There are exactly two ways to do
    # that, and picking only one would make the result an artifact of my choice,
    # so BOTH are measured:
    #     route A: halve the query block   B_q  64 -> 32
    #     route B: halve the key block     B_kv 64 -> 32
    # Each route carries its own d=64 control at the same block shape, which is
    # what separates "the effect of d" from "the effect of a smaller block".
    # The infeasible native-block d=128 points are still attempted so that the
    # build failure is recorded evidence rather than an assertion.
    blocks_cfg = [("64x64", 64, 64), ("32x64", 32, 64), ("64x32", 64, 32)]
    variants = []
    for stage in ("qk", "pv"):
        for tag, bq, bkv in blocks_cfg:
            for d in (64, 128):
                variants.append(GemmVariant(
                    f"{stage.upper()} d{d} {tag}", stage, d, M, K, N, cols,
                    bq, bkv, verify))
    print(f"\nGEMM geometry probe   M={M} K={K} N={N} cols={cols} ({cols*4} cores)")
    print(f"  identical useful arithmetic in every variant: "
          f"{M*K*N/1e9:.3f} GMAC, {2*M*K*N/1e9:.3f} GFLOP")
    built = []
    for v in variants:
        tile = f"{v.cfg['tile_m']}x{v.cfg['tile_k']}x{v.cfg['tile_n']}"
        print(f"  building {v.label:12s} tile {tile:>12} bcm="
              f"{int(v.cfg['b_col_maj'])}  L1 est {v.l1:>6}B "
              f"{'OK' if v.fits_l1 else 'OVER 65536'} ...", flush=True)
        try:
            v.compile()
        except Exception as e:
            v.build_error = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"      BUILD FAILED (L1 est {v.l1} B): {v.build_error[:120]}")
            continue
        built.append(v)
        print(f"      {v.build_s:5.1f}s  insts {v.ident['insts_bytes']:>7}B "
              f"sha {v.ident['insts_sha256'][:16]}  xclbin sha "
              f"{v.ident['xclbin_sha256'][:16]}")
    variants_all, variants = variants, built

    # Round-robin the variants across blocks rather than running each to
    # completion: this machine drifts 10-30% between runs, and a block-ordered
    # sweep would charge that drift to whichever variant ran last. Each block
    # re-warms its own context first, so no sample includes a context switch.
    for b in range(blocks):
        for v in variants:
            v.sample(warm, timed)
    for v in variants:
        v.rel_l2 = v.check()

    rows = [v.row() for v in variants]
    for v in variants_all:
        if v.build_error is not None:
            rows.append(dict(label=v.label, stage=v.stage, d=v.d, M=v.M, K=v.K,
                             N=v.N, cols=v.cols, cores=v.cols * 4,
                             tile_m=v.cfg["tile_m"], tile_k=v.cfg["tile_k"],
                             tile_n=v.cfg["tile_n"],
                             b_col_maj=int(v.cfg["b_col_maj"]),
                             l1_bytes_est=v.l1, l1_fits=int(v.fits_l1),
                             build_error=v.build_error))
    print(f"\n  {'variant':12s}{'tile':>14}{'cores':>6}{'kernel ms':>11}{'sd':>8}"
          f"{'TFLOPS':>9}{'relL2':>9}")
    for r in rows:
        tile = f"{r['tile_m']}x{r['tile_k']}x{r['tile_n']}"
        if r.get("build_error"):
            print(f"  {r['label']:12s}{tile:>14}{r['cores']:>6}"
                  f"{'DID NOT BUILD -- L1 needs ' + str(r['l1_bytes_est']) + ' B':>46}")
            continue
        rel = r.get("rel_l2")
        rel = float("nan") if rel is None else rel
        print(f"  {r['label']:12s}{tile:>14}{r['cores']:>6}"
              f"{r['kernel_ms']:>11.4f}{r['kernel_ms_sd']:>8.4f}"
              f"{r['tflops']:>9.3f}{rel:>9.5f}")

    eff = {r["label"]: r.get("tflops") for r in rows}

    def ratio(a, b):
        if eff.get(a) and eff.get(b):
            return round(eff[a] / eff[b], 4)
        return None

    ratios = {}
    for stage in ("QK", "PV"):
        low = stage.lower()
        # Not buildable at the native block, so no ratio exists there.
        ratios[f"{low}_d128_over_d64_at_64x64"] = ratio(
            f"{stage} d128 64x64", f"{stage} d64 64x64")
        for tag in ("32x64", "64x32"):
            t = tag.replace("x", "_")
            # Pure geometry effect: d128 vs d64 at the SAME block shape.
            ratios[f"{low}_d128_over_d64_at_{t}"] = ratio(
                f"{stage} d128 {tag}", f"{stage} d64 {tag}")
            # Cost of the forced block reduction alone, d held at 64.
            ratios[f"{low}_block_{t}_over_64_64_at_d64"] = ratio(
                f"{stage} d64 {tag}", f"{stage} d64 64x64")
            # NET: what a d=128 port would actually get, against the d=64
            # native-block baseline the economic gate was measured against.
            ratios[f"{low}_NET_d128_{t}_over_d64_64x64"] = ratio(
                f"{stage} d128 {tag}", f"{stage} d64 64x64")

    print("\n  normalized efficiency ratios (equal useful arithmetic):")
    for k, v in ratios.items():
        shown = "n/a (does not build)" if v is None else f"{v:.3f}"
        print(f"    {k:40s} {shown}")
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        keys, seen = [], set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k); keys.append(k)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows: w.writerow({k: r.get(k, "") for k in keys})
        print(f"  wrote {out_csv}")
    return rows, ratios


def run_softmax_scaling(cols, row_counts, blocks, warm, timed, out_csv):
    """Softmax cost vs row count, using the SAME softmax.cc that mha.cc includes.

    The 40-head proxy runs twice the head-wise softmax rows of real 20-head
    d=128 BitNet, over the same key count. So the quantity that matters is
    whether softmax time is linear in row count -- if it is, halving the heads
    halves the softmax stage; if a fixed per-block overhead dominates, the ratio
    stays above 0.5 and d=128 gains less than the work argument suggests."""
    print(f"\nSoftmax scaling probe   cols={cols} (= B_kv, MHA's key block)")
    variants = []
    for rows_n in row_counts:
        cfg = dict(rows=rows_n, cols=cols, num_aie_columns=1, num_channels=1)
        bd = BUILD_ROOT / build_key("softmax", cfg)
        if bd.exists():
            shutil.rmtree(bd)
        bd.mkdir(parents=True, exist_ok=True)
        ctx = AIEContext(build_dir=bd, compiler="peano")
        t0 = time.time()
        op = Softmax(context=ctx, **cfg)
        op.compile()
        build_s = time.time() - t0
        ident = artifact_identity(op, bd)
        spec = op.get_arg_spec()
        rng = np.random.default_rng(11)
        x = rng.standard_normal(tuple(spec[0].shape)).astype(np.dtype(spec[0].dtype))
        bufs = [XRTTensor(np.ascontiguousarray(x).reshape(-1), dtype=spec[0].dtype),
                XRTTensor(tuple(spec[1].shape), dtype=spec[1].dtype)]
        variants.append(dict(rows=rows_n, cols=cols, op=op, bufs=bufs,
                             fn=op.get_callable(), build_s=build_s, ident=ident,
                             x=x, kern=[]))
        print(f"  rows={rows_n:>8}  {build_s:5.1f}s  insts sha "
              f"{ident['insts_sha256'][:16]}")

    for b in range(blocks):
        for v in variants:
            for _ in range(warm):
                v["fn"](*v["bufs"])
            for _ in range(timed):
                v["kern"].append(v["fn"](*v["bufs"]).npu_time / 1e6)

    rows_out = []
    print(f"\n  {'rows':>9}{'elems':>11}{'kernel ms':>11}{'sd':>8}"
          f"{'ns/row':>9}{'Melem/s':>10}{'relL2':>9}")
    for v in variants:
        k = st.median(v["kern"])
        got = np.asarray(v["bufs"][1].data).reshape(v["rows"], v["cols"]).astype(np.float32)
        # iron.operators.softmax.reference is a row-wise softmax over the last
        # dim; reproduced in numpy for the same torch-free reason as above.
        z = v["x"].reshape(v["rows"], v["cols"]).astype(np.float32)
        z = z - z.max(axis=-1, keepdims=True)
        np.exp(z, out=z)
        ref = z / z.sum(axis=-1, keepdims=True)
        rel = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30))
        n = v["rows"] * v["cols"]
        rec = dict(rows=v["rows"], cols=v["cols"], elems=n,
                   build_s=round(v["build_s"], 1), reps=len(v["kern"]),
                   kernel_ms=round(k, 4), kernel_ms_sd=round(st.pstdev(v["kern"]), 4),
                   ns_per_row=round(k * 1e6 / v["rows"], 2),
                   melem_per_s=round(n / (k / 1e3) / 1e6, 1),
                   rel_l2=round(rel, 5), **v["ident"])
        rows_out.append(rec)
        print(f"  {rec['rows']:>9}{rec['elems']:>11}{rec['kernel_ms']:>11.4f}"
              f"{rec['kernel_ms_sd']:>8.4f}{rec['ns_per_row']:>9.2f}"
              f"{rec['melem_per_s']:>10.1f}{rec['rel_l2']:>9.5f}")

    # Linear fit t = a + b*rows: 'a' is the fixed per-dispatch overhead that
    # would keep a 20/40-head ratio above 0.5.
    xs = np.array([r["rows"] for r in rows_out], float)
    ys = np.array([r["kernel_ms"] for r in rows_out], float)
    b_slope, a_int = np.polyfit(xs, ys, 1)
    print(f"\n  linear fit  t(ms) = {a_int:.4f} + {b_slope*1e6:.4f} ns/row")
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys())); w.writeheader()
            for r in rows_out: w.writerow(r)
        print(f"  wrote {out_csv}")
    return rows_out, dict(fixed_ms=round(float(a_int), 5),
                          ns_per_row=round(float(b_slope) * 1e6, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["qk_pv", "softmax", "all"])
    ap.add_argument("--mkn", type=int, nargs=3, default=[2048, 2048, 2048])
    ap.add_argument("--cols", type=int, nargs="+", default=[8, 1])
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--timed", type=int, default=5)
    ap.add_argument("--softmax-cols", type=int, default=64)
    ap.add_argument("--softmax-rows", type=int, nargs="+",
                    default=[1024, 4096, 16384, 65536])
    ap.add_argument("--outdir", default="artifacts/attention-feasibility")
    a = ap.parse_args()

    tc = toolchain()
    print("toolchain:", json.dumps(tc))
    print(f"build root: {BUILD_ROOT}  (cleared per variant)")
    summary = dict(toolchain=tc, mkn=a.mkn, blocks=a.blocks, timed=a.timed)

    if a.stage in ("qk_pv", "all"):
        summary["gemm"] = {}
        for cols in a.cols:
            rows, ratios = run_gemm_geometry(
                *a.mkn, cols, a.blocks, a.warm, a.timed,
                f"{a.outdir}/geometry_qk_pv_c{cols}.csv")
            summary["gemm"][f"cols{cols}"] = dict(ratios=ratios, rows=rows)
    if a.stage in ("softmax", "all"):
        rows, fit = run_softmax_scaling(
            a.softmax_cols, a.softmax_rows, a.blocks, a.warm, a.timed,
            f"{a.outdir}/geometry_softmax.csv")
        summary["softmax"] = dict(fit=fit, rows=rows)

    out = Path(a.outdir) / "geometry_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out}")
    aie_utils.DefaultNPURuntime.cleanup()


if __name__ == "__main__":
    main()
