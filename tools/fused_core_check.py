#!/usr/bin/env python3
"""Does a fused attention core fit AIE2P's 16 KiB program memory?

The gating feasibility question for a single-core-per-pair attention kernel.
AMD's MHA spends 3 x 9.412 us of core-time per (q,kv) pair to do 16.147 us of
work -- 50.5% compute efficiency -- because a spatial pipeline runs at its
slowest stage while paying for all three cores. Fusing the stages onto one core
recovers that, and lets all 32 cores be used instead of 24, but only if the code
fits. This project has already seen 16 KiB program memory overflow.

The stock build's three cores measure 1696 + 5328 + 3728 = 10752 B of .text, so
a sum suggests it fits. A sum is not a build: the linker adds per-core setup, and
fusing changes register pressure, which can force spills that change code size.
So this compiles a single core that actually calls every kernel a fused core
needs and reports the real .text.

Reuses IRON's MHA operator for the kernel artifacts, so mha.o is compiled with
exactly the flags AMD's own operator uses; only the MLIR design is swapped.
"""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

sys.path.insert(0, os.environ.get("IRON_DIR", "/tmp/xdnaresearch/iron"))
from iron.common import AIEContext, PythonGeneratedMLIRArtifact, DesignGenerator  # noqa: E402
from iron.operators.mha.op import MHA                    # noqa: E402
import aie.utils as aie_utils                            # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PROGMEM_BYTES = 16384          # AIE2P per-core program memory


class FusedProbe(MHA):
    """MHA's kernel artifacts (so mha.o is built identically), our design."""

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                REPO / "npu" / "experiments" / "fused_core_progmem.py",
                "fused_progmem_probe", (),
                dict(B_q=64, B_kv=64, d=64, n_kv_blocks=4),
            ),
        )


def text_sizes(build_dir):
    """.text of every core ELF aiecc left behind, keyed by (col,row)."""
    readelf = Path(aie_utils.config.peano_install_dir()) / "bin" / "llvm-readelf"
    out = {}
    for elf in sorted(build_dir.rglob("elfs_main_core_*/*.elf")):
        name = elf.parent.name.replace("elfs_main_core_", "")
        r = subprocess.run([str(readelf), "-S", str(elf)],
                           capture_output=True, text=True)
        for line in r.stdout.splitlines():
            parts = line.split()
            # readelf -S columns: [Nr] Name Type Address Off Size ...
            # "[ 1]" splits into two tokens, so locate .text and index from it.
            if ".text" in parts:
                i = parts.index(".text")
                out[name] = int(parts[i + 4], 16)
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir",
                    default=os.environ.get("FUSED_BUILD_DIR",
                                           "/tmp/bitnet-fused-build"))
    ap.add_argument("--stock-build-dir", default="/tmp/bitnet-mha-build",
                    help="an existing stock MHA build, for the per-stage baseline")
    ap.add_argument("--out",
                    default="artifacts/attention-feasibility/fused_core_progmem.json")
    a = ap.parse_args()

    rec = {"progmem_bytes": PROGMEM_BYTES}

    stock = Path(a.stock_build_dir)
    if stock.exists():
        sizes = text_sizes(stock)
        # Stock MHA places QK on row 2, softmax on row 3, PV on row 4.
        by_row = {}
        for k, v in sizes.items():
            by_row.setdefault(k.split("_")[1], []).append(v)
        stage = {"2_QK": None, "3_softmax": None, "4_PV": None}
        for row, label in (("2", "2_QK"), ("3", "3_softmax"), ("4", "4_PV")):
            if row in by_row:
                stage[label] = max(by_row[row])
        rec["stock_stage_text_bytes"] = stage
        known = [v for v in stage.values() if v]
        rec["stock_sum_text_bytes"] = sum(known)
        print("stock MHA, .text per stage core:")
        for k, v in stage.items():
            print(f"  {k:12s} {v if v else '-':>8}")
        print(f"  {'sum':12s} {sum(known):>8} B\n")

    build = Path(a.build_dir)
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True, exist_ok=True)
    ctx = AIEContext(build_dir=build, compiler="peano")
    op = FusedProbe(num_heads=1, seq_len=64, d=64, num_KV_heads=1,
                    num_of_pipelines=1, context=ctx)
    print("building a single core that calls zero, matmul_QK, partial_softmax,")
    print("matmul_PV, rescale_O and init_scale_buffer ...", flush=True)
    try:
        op.compile()
    except Exception as e:
        rec["build_error"] = f"{type(e).__name__}: {str(e)[:400]}"
        print(f"\nBUILD FAILED: {rec['build_error'][:300]}")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rec, indent=2) + "\n")
        return 1

    sizes = text_sizes(build)
    rec["fused_core_text_bytes"] = sizes
    biggest = max(sizes.values()) if sizes else None
    rec["fused_max_text_bytes"] = biggest
    print("\nfused core, .text:")
    for k, v in sorted(sizes.items()):
        print(f"  core {k:8s} {v:>8} B")
    if biggest:
        head = PROGMEM_BYTES - biggest
        rec["headroom_bytes"] = head
        rec["fits"] = bool(head >= 0)
        print(f"\n  largest core .text  {biggest:>8} B")
        print(f"  program memory      {PROGMEM_BYTES:>8} B")
        print(f"  headroom            {head:>8} B  "
              f"({head/PROGMEM_BYTES*100:+.1f}%)")
        print(f"\n  {'FITS' if head >= 0 else 'DOES NOT FIT'}"
              f" -- program memory is {'not' if head >= 0 else ''} the blocker")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=2) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
