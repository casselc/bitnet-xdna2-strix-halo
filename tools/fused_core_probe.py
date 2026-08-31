#!/usr/bin/env python3
"""Measure a FUSED attention core: does one core do a (q,kv) pair in ~16.1 us?

The decisive test of the utilisation hypothesis, and it needs one core, not 32.

AMD's three-stage spatial pipeline spends 31.98 us of core-time per pair to do
16.147 us of work (QK 3.938 + softmax 9.412 + PV 2.797), because a pipeline runs
at its slowest stage while paying for all three cores. If one fused core does a
pair in about the SUM of the stage times, the utilisation argument holds and
scaling to 32 cores is arithmetic. If it takes materially longer, fusing costs
something the stage times do not capture and the idea is dead -- which is worth
finding out from one core rather than from a full rewrite.

Method: build the fused core for several kv-block counts and fit

    t(n_kv) = intercept + slope * n_kv

The slope is the steady-state per-pair service time, directly comparable to the
16.147 us the stage model predicts. The intercept is the per-q-block cost --
Q load, init_scale_buffer, rescale_O, O drain -- directly comparable to the
C_qblock = 68.38 us the pipeline model fitted independently from the stock
kernel. Two independent cross-checks from one sweep.

Correctness is verified against a numpy golden every time. A fast kernel that
computes the wrong thing would otherwise look like a win.
"""
import argparse, csv, hashlib, json, os, shutil, statistics as st, subprocess, sys, time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.environ.get("IRON_DIR", "/tmp/xdnaresearch/iron"))
from iron.common import AIEContext, PythonGeneratedMLIRArtifact, DesignGenerator  # noqa: E402
from iron.operators.mha.op import MHA                    # noqa: E402
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor  # noqa: E402
import aie.utils as aie_utils                            # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "npu" / "experiments"))
from mha_fused_design import tile_major, from_tile_major, R, T  # noqa: E402

BUILD_ROOT = Path(os.environ.get("FUSED_BUILD_DIR", "/tmp/bitnet-fused-probe"))
PROGMEM = 16384


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class FusedMHA(MHA):
    """MHA's kernel artifacts (mha.o with AMD's own flags), our fused design."""

    n_kv_blocks: int = 64
    relayout_traffic: bool = False

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                REPO / "npu" / "experiments" / "mha_fused_design.py",
                "fused_mha", (),
                dict(B_q=64, B_kv=64, d=64, n_kv_blocks=self.n_kv_blocks,
                     relayout_traffic=self.relayout_traffic),
            ),
        )


def golden(Q, K, V, d):
    """Full (non-causal) attention of one q block over all kv blocks, f32.

    Non-causal because the design sets idx_buffer[1] far above every kv index,
    making every block a full off-diagonal block -- the steady-state pair the
    stage model is built from."""
    s = (Q.astype(np.float32) @ K.astype(np.float32).T) / np.sqrt(d)
    s -= s.max(axis=-1, keepdims=True)
    np.exp(s, out=s)
    s /= s.sum(axis=-1, keepdims=True)
    return s @ V.astype(np.float32)


def run_case(n_kv, B_q, B_kv, d, warm, iters, relayout=False,
             keep_build=False):
    build = BUILD_ROOT / f"nkv{n_kv}_rl{int(relayout)}"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True, exist_ok=True)
    ctx = AIEContext(build_dir=build, compiler="peano")
    op = FusedMHA(num_heads=1, seq_len=64, d=d, num_KV_heads=1,
                  num_of_pipelines=1, context=ctx)
    op.n_kv_blocks = n_kv
    op.relayout_traffic = relayout
    t0 = time.time()
    op.compile()
    build_s = time.time() - t0

    xcl, insts = Path(op.xclbin_artifact.filename), Path(op.insts_artifact.filename)
    elf = next(iter(sorted(build.rglob("elfs_main_core_*/*.elf"))), None)
    text = None
    if elf is not None:
        readelf = Path(aie_utils.config.peano_install_dir()) / "bin" / "llvm-readelf"
        out = subprocess.run([str(readelf), "-S", str(elf)],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            p = line.split()
            if ".text" in p:
                text = int(p[p.index(".text") + 4], 16)
                break

    rng = np.random.default_rng(5)
    Q = (rng.random((B_q, d), dtype=np.float32) * 4).astype(bfloat16)
    Kb = (rng.random((n_kv * B_kv, d), dtype=np.float32) * 4).astype(bfloat16)
    Vb = (rng.random((n_kv * B_kv, d), dtype=np.float32) * 4).astype(bfloat16)

    # Host-side layout transform, interleaved K,V per block into one stream.
    kv_stream = np.concatenate([
        np.concatenate([tile_major(Kb[i * B_kv:(i + 1) * B_kv], T, 8),
                        tile_major(Vb[i * B_kv:(i + 1) * B_kv], 8, T)])
        for i in range(n_kv)])

    bufs = [XRTTensor(tile_major(Q, R, 8).copy(), dtype=bfloat16),
            XRTTensor(np.ascontiguousarray(kv_stream), dtype=bfloat16),
            XRTTensor((B_q * B_kv,), dtype=bfloat16)]
    fn = op.get_callable()
    for _ in range(warm):
        fn(*bufs)
    ts = [fn(*bufs).npu_time / 1e6 for _ in range(iters)]

    got = from_tile_major(np.asarray(bufs[2].data).astype(np.float32),
                          B_q, d, R, T)
    ref = golden(Q, Kb, Vb, d)
    # A NaN/Inf output makes rel-L2 NaN, which reads as "no error" in a table.
    # It is a hard failure: it is what a kernel that skipped its work returns.
    n_bad_float = int((~np.isfinite(got)).sum())
    rel = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30))
    tol = 4.0e-2 * np.abs(ref) + 1.5e-1
    bad = int((np.abs(got - ref) > tol).sum())

    row = dict(n_kv_blocks=n_kv, relayout=int(relayout),
               B_q=B_q, B_kv=B_kv, d=d, cores=1,
               build_s=round(build_s, 1), reps=iters,
               kernel_ms=round(st.median(ts), 5),
               kernel_ms_min=round(min(ts), 5),
               kernel_ms_sd=round(st.pstdev(ts), 5),
               us_per_pair=round(st.median(ts) * 1e3 / n_kv, 4),
               text_bytes=text, rel_l2=round(rel, 5),
               nonfinite_out=n_bad_float,
               bad_elems=bad, total_elems=int(ref.size),
               bad_frac=round(bad / ref.size, 6),
               xclbin_sha256=sha256(xcl), insts_sha256=sha256(insts),
               insts_bytes=insts.stat().st_size)
    for t in bufs:
        del t
    if not keep_build:
        shutil.rmtree(build, ignore_errors=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-kv", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--bq", type=int, default=64)
    ap.add_argument("--bkv", type=int, default=64)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--model",
                    default="artifacts/attention-feasibility/d128_pipeline_model.json")
    ap.add_argument("--out",
                    default="artifacts/attention-feasibility/fused_core.csv")
    a = ap.parse_args()

    stages = json.loads(Path(a.model).read_text())
    st64 = stages["d64_stage_us"]
    predicted = st64["t_qk"] + st64["t_sm"] + st64["t_pv"]
    stock_service = stages["fit"]["service_us"]
    stock_cq = stages["fit"]["c_qblock_us"]

    print(f"fused core probe   B_q={a.bq} B_kv={a.bkv} d={a.d}, 1 core")
    print(f"  stage model: QK {st64['t_qk']:.3f} + softmax {st64['t_sm']:.3f} + "
          f"PV {st64['t_pv']:.3f} = {predicted:.3f} us/pair predicted")
    print(f"  stock pipeline: {stock_service:.3f} us/pair using 3 cores "
          f"= {3*stock_service:.3f} us of core-time\n")
    print(f"  {'n_kv':>6}{'build':>7}{'kernel ms':>11}{'sd':>8}{'us/pair':>9}"
          f"{'.text':>8}{'relL2':>9}{'bad%':>8}")
    rows = []
    for relayout in (False, True):
        print(f"\n  -- {'with' if relayout else 'without'} inter-stage relayout"
              f" traffic --")
        for n in a.n_kv:
            try:
                r = run_case(n, a.bq, a.bkv, a.d, a.warm, a.iters, relayout)
            except Exception as e:
                print(f"  {n:>6}   FAILED: {type(e).__name__}: {str(e)[:100]}")
                rows.append(dict(n_kv_blocks=n, relayout=int(relayout),
                                 error=f"{type(e).__name__}: {e}"))
                continue
            rows.append(r)
            print(f"  {r['n_kv_blocks']:>6}{r['build_s']:>7.1f}"
                  f"{r['kernel_ms']:>11.5f}{r['kernel_ms_sd']:>8.5f}"
                  f"{r['us_per_pair']:>9.3f}{r['text_bytes']:>8}"
                  f"{r['rel_l2']:>9.5f}{r['bad_frac']*100:>7.3f}%"
                  f"{'  NONFINITE!' if r['nonfinite_out'] else ''}")

    nonfinite = [r for r in rows if r.get("nonfinite_out")]
    if nonfinite:
        print(f"\n  REFUSING TO FIT: {len(nonfinite)} case(s) returned "
              f"non-finite output -- that is what a kernel that SKIPPED its "
              f"work returns, so its timing is meaningless.")
        return

    def fit(sel):
        x = np.array([r["n_kv_blocks"] for r in sel], float)
        y = np.array([r["kernel_ms"] for r in sel], float)
        sl, ic = np.polyfit(x, y, 1)
        return sl * 1e3, ic * 1e3

    base = [r for r in rows if r.get("kernel_ms") and not r["relayout"]]
    rel = [r for r in rows if r.get("kernel_ms") and r["relayout"]]
    summary = {}
    if len(base) >= 2:
        slope_us, inter_us = fit(base)
        slope, intercept = slope_us / 1e3, inter_us / 1e3
        slope_us, inter_us = slope * 1e3, intercept * 1e3
        print(f"\n  fit  t(ms) = {intercept:.5f} + {slope:.6f} * n_kv")
        print(f"    steady-state service {slope_us:8.3f} us/pair "
              f"(stage model predicts {predicted:.3f}, ratio {slope_us/predicted:.3f})")
        print(f"    per-q-block cost     {inter_us:8.3f} us "
              f"(pipeline model fitted {stock_cq:.2f} from the stock kernel)")
        total_us = slope_us
        if len(rel) >= 2:
            rl_slope, rl_inter = fit(rel)
            total_us = rl_slope
            print(f"\n  with relayout traffic: {rl_slope:8.3f} us/pair "
                  f"(+{rl_slope-slope_us:.3f} us = +{(rl_slope/slope_us-1)*100:.1f}%)")
            print(f"    that is a LOWER BOUND on the two per-pair layout "
                  f"transforms the stock\n    design gets free from its "
                  f"inter-core DMA: traffic only, no shuffle.")
            summary["relayout_slope_us_per_pair"] = round(rl_slope, 4)
            summary["relayout_intercept_us"] = round(rl_inter, 3)
            summary["relayout_cost_us"] = round(rl_slope - slope_us, 4)

        core_time_ratio = (3 * stock_service) / total_us
        print(f"\n  core-time per pair:  stock {3*stock_service:.3f} us (3 cores)"
              f"   fused {total_us:.3f} us (1 core)")
        print(f"  ==> per-core throughput gain {core_time_ratio:.3f}x")
        print(f"  ==> at 32 fused cores vs 24 pipelined: "
              f"{(32/total_us)/(8/stock_service):.3f}x")
        summary.update(slope_us_per_pair=round(slope_us, 4),
                       intercept_us=round(inter_us, 3),
                       predicted_us_per_pair=round(predicted, 4),
                       slope_over_predicted=round(slope_us / predicted, 4),
                       stock_service_us=stock_service,
                       stock_core_time_us=round(3 * stock_service, 4),
                       stock_c_qblock_us=stock_cq,
                       per_core_gain=round(core_time_ratio, 4),
                       fused_total_us_per_pair=round(total_us, 4),
                       gain_32_fused_vs_24_pipelined=round(
                           (32 / total_us) / (8 / stock_service), 4))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in keys})
    Path(a.out).with_suffix(".json").write_text(
        json.dumps(dict(summary=summary, rows=rows), indent=2) + "\n")
    print(f"\nwrote {a.out}")
    aie_utils.DefaultNPURuntime.cleanup()


if __name__ == "__main__":
    main()
