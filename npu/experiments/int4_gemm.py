#!/usr/bin/env python3
"""int8 x int4 GEMM on aie2p: correctness, then throughput vs int8 x int8.

Why this exists: BitNet weights are {-1,0,+1} and fit int4 exactly. Storing them
as int4 halves the bytes crossing DMA, and disassembly shows the 4-bit expansion
happens in the load-store unit (`vldb.unpack`), costing no MAC slots. The open
question is whether aie2p's single native mac_4x16_16x16 (1024 MACs/instr, vs
mac_8x8_8x8's 512) also ISSUES at twice the rate -- which decides whether int4 is
worth 2x or only saves bandwidth.

Single core, one tile, linear DMAs: the point is the MAC pipeline, not tiling.
Both variants use identical M/K/N and identical dispatch machinery, so the wall
time ratio is the answer.
"""
import sys, time
import numpy as np
import aie.iron as iron
from aie.iron import (CompileTime, ExternalFunction, In, ObjectFifo, Out,
                      Program, Runtime, Worker, kernels)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorTiler2D

import os as _os
M = int(_os.environ.get("DM", 64))
K = int(_os.environ.get("DK", 64))
N = int(_os.environ.get("DN", 64))
R, S, T = 4, 16, 16     # aie2p int8 x int4 mmul geometry

KERNEL_SRC = open("npu/kernels/mm_i8_i4.cc").read()


def pack_A(A):
    """A -> mmul tile order: (M/R) x (K/S) blocks of R x S, row-major within."""
    out = np.zeros((M // R, K // S, R, S), dtype=np.int8)
    for z in range(M // R):
        for i in range(K // S):
            out[z, i] = A[z*R:(z+1)*R, i*S:(i+1)*S]
    return out.reshape(-1)


def pack_B_int4(B):
    """B -> mmul tile order, then 2 signed 4-bit weights per byte (low nibble
    first). Values are ternary so they fit 4 bits with no clipping."""
    tiles = np.zeros((K // S, N // T, S, T), dtype=np.int8)
    for i in range(K // S):
        for j in range(N // T):
            tiles[i, j] = B[i*S:(i+1)*S, j*T:(j+1)*T]
    flat = tiles.reshape(-1)
    import os as _os
    if _os.environ.get("NIBBLE") == "unzip":
        # per 256-element mmul tile
        flat = flat.reshape(-1, 256)
        flat = np.concatenate([np.concatenate([t[:128], t[128:]]) for t in flat])
    import os
    mode = os.environ.get("NIBBLE", "interleave")
    n = flat.size
    if mode == "unzip":
        # vldb.unpack appears to DE-INTERLEAVE: the low nibbles of the byte
        # window expand to the first half of the lanes and the high nibbles to
        # the second half. Evidence: with paired packing, B=e00 is exact but
        # B=I fails on ~half the elements -- precisely the half that would live
        # in high nibbles. So split the tile: first half -> low nibbles,
        # second half -> high nibbles of the SAME bytes.
        half = n // 2
        lo = (flat[:half] & 0x0F).astype(np.uint8)
        hi = (flat[half:] & 0x0F).astype(np.uint8)
    elif mode == "hi_first":
        hi = (flat[0::2] & 0x0F).astype(np.uint8)
        lo = (flat[1::2] & 0x0F).astype(np.uint8)
    else:
        lo = (flat[0::2] & 0x0F).astype(np.uint8)
        hi = (flat[1::2] & 0x0F).astype(np.uint8)
    return ((hi << 4) | lo).astype(np.int8)


def unpack_C(c_flat):
    """mmul tile order -> plain [M,N]."""
    t = c_flat.reshape(M // R, N // T, R, T)
    out = np.zeros((M, N), dtype=np.int32)
    for z in range(M // R):
        for j in range(N // T):
            out[z*R:(z+1)*R, j*T:(j+1)*T] = t[z, j]
    return out


@iron.jit
def gemm_i8i4(A: In, B: In, C: Out):
    a_ty = np.ndarray[(M * K,), np.dtype[np.int8]]
    b_ty = np.ndarray[(K * N // 2,), np.dtype[np.int8]]   # int4: half the bytes
    c_ty = np.ndarray[(M * N,), np.dtype[np.int32]]

    mm = ExternalFunction(
        "matmul_i8_i4_i32", source_string=KERNEL_SRC,
        arg_types=[a_ty, b_ty, c_ty],
        compile_flags=[f"-DDIM_M={M}", f"-DDIM_K={K}", f"-DDIM_N={N}"],
    )

    inA, inB = ObjectFifo(a_ty, name="inA"), ObjectFifo(b_ty, name="inB")
    outC = ObjectFifo(c_ty, name="outC")

    def core(of_a, of_b, of_c, mmf):
        ea, eb, ec = of_a.acquire(1), of_b.acquire(1), of_c.acquire(1)
        mmf(ea, eb, ec)
        of_a.release(1); of_b.release(1); of_c.release(1)

    w = Worker(core, [inA.cons(), inB.cons(), outC.prod(), mm], stack_size=0xD00)

    def sequence(A_h, B_h, C_h, inA_h, inB_h, outC_h):
        inA_h.fill(A_h)
        inB_h.fill(B_h)
        outC_h.drain(C_h, wait=True)

    rt = Runtime(sequence, [a_ty, b_ty, c_ty, inA.prod(), inB.prod(), outC.cons()])
    prog = Program(iron.get_current_device(), rt, workers=[w])
    return prog.resolve_program()


def main():
    import os
    rng = np.random.default_rng(7)
    A = rng.integers(-128, 128, size=(M, K), dtype=np.int64).astype(np.int8)
    if os.environ.get("DIAG") == "b00":
        # B = e_00 only: C must equal A[:,0] in column 0 and zero elsewhere.
        # Isolates B's layout -- A's and C's packings cancel out of the check.
        B = np.zeros((K, N), dtype=np.int8); B[0, 0] = 1
    elif os.environ.get("DIAG") == "eye":
        B = np.eye(K, N, dtype=np.int8)      # C must equal A exactly
    else:
        B = rng.integers(-1, 2, size=(K, N), dtype=np.int64).astype(np.int8)
    ref = A.astype(np.int32) @ B.astype(np.int32)

    a_dev = iron.tensor(pack_A(A), dtype=np.int8)
    b_dev = iron.tensor(pack_B_int4(B), dtype=np.int8)
    c_dev = iron.zeros(M * N, dtype=np.int32)

    print(f"int8 x int4 GEMM, M=K=N={M}, mmul {R}x{S}x{T}")
    print(f"  B bytes on the wire: {K*N//2} (int8 would be {K*N})")
    t0 = time.perf_counter()
    gemm_i8i4(a_dev, b_dev, c_dev)
    print(f"  first call (incl. compile): {(time.perf_counter()-t0)*1e3:.0f} ms")

    got = unpack_C(np.asarray(c_dev))
    if np.array_equal(got, ref):
        print(f"  CORRECT: all {M*N} int32 accumulators bit-exact vs numpy")
    else:
        bad = int((got != ref).sum())
        print(f"  WRONG: {bad}/{M*N} differ")
        print(f"    got[0,:6] {got[0,:6]}   ref[0,:6] {ref[0,:6]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
