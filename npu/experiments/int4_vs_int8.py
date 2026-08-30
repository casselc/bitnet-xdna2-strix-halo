#!/usr/bin/env python3
"""Does int8 x int4 issue at 2x the MAC rate of int8 x int8 on aie2p?

Both kernels live in npu/kernels/mm_i8_i4.cc with an identical loop nest; only
the mmul operand type and (r,s,t) differ. Each does REPEAT full 64x64x64 GEMMs
per dispatch so compute dominates the ~0.2 ms dispatch floor. Equal MACs, equal
machinery -- so the wall-time ratio is the MAC-rate ratio.

Expected if the 1024-MAC mac_4x16_16x16 issues in one cycle: int4 ~2x faster.
If it issues in two: ~1.0x, and int4 is a bandwidth win only.
"""
import os, sys, time
import numpy as np
import aie.iron as iron
from aie.iron import (ExternalFunction, In, ObjectFifo, Out, Program,
                      Runtime, Worker)
from aie.iron.device import NPU2

M = K = N = 64
REPEAT = int(os.environ.get("REPEAT", 64))
SRC = open("npu/kernels/mm_i8_i4.cc").read()


def make(sym, b_elems):
    a_ty = np.ndarray[(M * K,), np.dtype[np.int8]]
    b_ty = np.ndarray[(b_elems,), np.dtype[np.int8]]
    c_ty = np.ndarray[(M * N,), np.dtype[np.int32]]

    @iron.jit
    def design(A: In, B: In, C: Out):
        mm = ExternalFunction(
            sym, source_string=SRC, arg_types=[a_ty, b_ty, c_ty],
            compile_flags=[f"-DDIM_M={M}", f"-DDIM_K={K}", f"-DDIM_N={N}",
                           f"-DREPEAT={REPEAT}"])
        inA, inB = ObjectFifo(a_ty, name="inA"), ObjectFifo(b_ty, name="inB")
        outC = ObjectFifo(c_ty, name="outC")

        def core(of_a, of_b, of_c, f):
            ea, eb, ec = of_a.acquire(1), of_b.acquire(1), of_c.acquire(1)
            f(ea, eb, ec)
            of_a.release(1); of_b.release(1); of_c.release(1)

        w = Worker(core, [inA.cons(), inB.cons(), outC.prod(), mm], stack_size=0xD00)

        def seq(A_h, B_h, C_h, ia, ib, oc):
            ia.fill(A_h); ib.fill(B_h); oc.drain(C_h, wait=True)

        rt = Runtime(seq, [a_ty, b_ty, c_ty, inA.prod(), inB.prod(), outC.cons()])
        return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()
    return design


def bench(design, a, b, c, iters=60):
    design(a, b, c)                       # compile + warm
    for _ in range(10):
        design(a, b, c)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); design(a, b, c)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    rng = np.random.default_rng(11)
    A = rng.integers(-128, 128, size=M * K, dtype=np.int64).astype(np.int8)
    a = iron.tensor(A, dtype=np.int8)
    c = iron.zeros(M * N, dtype=np.int32)

    macs = M * K * N * REPEAT
    print(f"{M}x{K}x{N} GEMM x{REPEAT} per dispatch = {macs/1e6:.1f} MMAC, single core\n")
    # One variant per process: the JIT caches on the design function, and both
    # variants use the same function name, so in-process they collide and the
    # second silently reuses the first's kernel.
    want = sys.argv[1] if len(sys.argv) > 1 else None
    res = {}
    for sym, b_elems, label in (("matmul_i8_i8_i32", K * N,     "int8 x int8"),
                                ("matmul_i8_i4_i32", K * N // 2, "int8 x int4")):
        if want and want not in sym:
            continue
        b = iron.tensor(rng.integers(-1, 2, size=b_elems, dtype=np.int64).astype(np.int8),
                        dtype=np.int8)
        ms = bench(make(sym, b_elems), a, b, c)
        gops = 2 * macs / (ms * 1e-3) / 1e9
        res[label] = ms
        print(f"  {label}:  {ms:7.3f} ms   {gops:8.1f} GOPS   B on wire {b_elems} B")

    if len(res) == 2:
        r = res["int8 x int8"] / res["int8 x int4"]
        print(f"\n  int4 speedup: {r:.2f}x")


if __name__ == "__main__":
    sys.exit(main())
