#!/usr/bin/env python3
"""What element layout does the QK matmul actually emit?

Settles, by measurement rather than by reading mm.cc, whether a "row-major C"
variant exists that would let a fused attention core skip the inter-stage
relayout. It is the evidence behind FUSED_CORE.md section 7.

Background: AMD's MHA pipeline puts layout transforms on its inter-core fifos --
memA carries a_dims, turning the QK matmul's output into the row-major layout
partial_softmax indexes with A[i*B_kv + j]; memP carries q_dims, turning P back
for the PV matmul. A fused core has no inter-stage DMA and must pay for both.
mm.cc has a c_row_maj template parameter, and MHA's QK matmul ALREADY uses
is_c_row_maj = true (it does not define -DC_COL_MAJ), yet still needs the
transform -- which suggests c_row_maj selects tile ORDERING, not element layout.

Rather than conclude that from source, this builds a single core that emits the
QK matmul's C tile verbatim (mha_fused_design.fused_mha with dump_a=True) and
compares it against Q @ K^T under both interpretations. Whichever matches is the
answer.
"""
import argparse, json, shutil, os, sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.environ.get("IRON_DIR", "/tmp/xdnaresearch/iron"))
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "npu" / "experiments"))
sys.path.insert(0, str(REPO / "tools"))

from iron.common import AIEContext, PythonGeneratedMLIRArtifact, DesignGenerator  # noqa: E402
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor  # noqa: E402
import aie.utils as aie_utils                            # noqa: E402
from fused_core_probe import FusedMHA                    # noqa: E402
from mha_fused_design import tile_major, R, T            # noqa: E402


class DumpQK(FusedMHA):
    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                REPO / "npu" / "experiments" / "mha_fused_design.py",
                "fused_mha", (),
                dict(B_q=self.bq, B_kv=self.bkv, d=self.d, n_kv_blocks=1,
                     dump_a=True)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bq", type=int, default=64)
    ap.add_argument("--bkv", type=int, default=64)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--build-dir",
                    default=os.environ.get("QK_LAYOUT_BUILD_DIR",
                                           "/tmp/bitnet-qk-layout"))
    ap.add_argument("--out",
                    default="artifacts/attention-feasibility/qk_layout.json")
    a = ap.parse_args()

    bd = Path(a.build_dir)
    shutil.rmtree(bd, ignore_errors=True)
    bd.mkdir(parents=True, exist_ok=True)
    op = DumpQK(num_heads=1, seq_len=a.bq, d=a.d, num_KV_heads=1,
                num_of_pipelines=1,
                context=AIEContext(build_dir=bd, compiler="peano"))
    op.bq, op.bkv = a.bq, a.bkv
    op.n_kv_blocks, op.relayout_traffic = 1, False
    op.compile()

    rng = np.random.default_rng(3)
    Q = (rng.random((a.bq, a.d), dtype=np.float32) * 4).astype(bfloat16)
    K = (rng.random((a.bkv, a.d), dtype=np.float32) * 4).astype(bfloat16)
    V = np.zeros((a.bkv, a.d), bfloat16)          # unused by dump_a
    kv = np.concatenate([tile_major(K, T, 8), tile_major(V, 8, T)])

    bufs = [XRTTensor(tile_major(Q, R, 8).copy(), dtype=bfloat16),
            XRTTensor(np.ascontiguousarray(kv), dtype=bfloat16),
            XRTTensor((a.bq * a.bkv,), dtype=bfloat16)]
    op.get_callable()(*bufs)
    got = np.asarray(bufs[2].data).astype(np.float32)

    ref = Q.astype(np.float32) @ K.astype(np.float32).T
    interp = {
        "flat_row_major": ref.reshape(-1),
        "tile_major_8x8": ref.reshape(a.bq // R, R, a.bkv // T, T)
                             .transpose(0, 2, 1, 3).reshape(-1),
    }
    rel = {k: float(np.linalg.norm(got - v) / np.linalg.norm(v))
           for k, v in interp.items()}
    best = min(rel, key=rel.get)

    print(f"QK matmul output, {a.bq}x{a.bkv} d={a.d}, vs Q @ K^T:")
    for k, v in rel.items():
        print(f"  {k:>18}  rel-L2 {v:.5f}{'   <-- MATCH' if k == best else ''}")
    print(f"\n  c_row_maj selects tile ORDERING, not element layout."
          if best == "tile_major_8x8" else
          f"\n  output is flat row-major.")
    print(f"  -> a fused core {'MUST' if best == 'tile_major_8x8' else 'need not'}"
          f" relayout between QK and softmax.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        dict(B_q=a.bq, B_kv=a.bkv, d=a.d, mmul_r=R, mmul_t=T,
             rel_l2=rel, best_match=best), indent=2) + "\n")
    print(f"\nwrote {a.out}")
    aie_utils.DefaultNPURuntime.cleanup()


if __name__ == "__main__":
    main()
