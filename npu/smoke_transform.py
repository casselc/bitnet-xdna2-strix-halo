#!/usr/bin/env python3
"""Smallest possible IRON design that actually runs on this NPU.

Purpose is not the arithmetic -- it is to answer two questions before any
BitNet work happens:
  1. does the mlir-aie 1.4.2 + Peano toolchain produce a working xclbin for
     npu2 on THIS device (Strix Halo / npu5), and
  2. what does a host->NPU->host round trip actually cost here?

The published XDNA2 dispatch figures (0.66 ms round trip, 2.67 ms context
switch) were measured on Krackan. If they transfer, 210 dispatches per BitNet
prefill costs ~139 ms and the offload is viable above ~512 tokens. If they are
much worse, per-matmul offload is dead and we need whole-layer fusion. Either
way we need the number from this silicon, not from a blog post.
"""
import sys, time, statistics
import numpy as np
import aie.iron as iron
from aie.iron import In, Out, CompileTime

N_ELEMS = 4096

@iron.jit
def add_one(inp: In, out: Out, *, N: CompileTime[int], dtype: CompileTime[type] = np.int32):
    tensor_ty = np.ndarray[(N,), np.dtype[dtype]]
    return iron.algorithms.transform(lambda x: x + 1, tensor_ty)

def main():
    dev = iron.get_current_device()
    print(f"device: {dev}")

    src = iron.arange(N_ELEMS, dtype=np.int32)
    dst = iron.zeros(N_ELEMS, dtype=np.int32)

    # First call includes compilation; keep it separate from the timing.
    t0 = time.perf_counter()
    add_one(src, dst, N=N_ELEMS)
    first_ms = (time.perf_counter() - t0) * 1e3
    print(f"first call (includes compile): {first_ms:.1f} ms")

    expect = np.arange(N_ELEMS, dtype=np.int32) + 1
    got = np.asarray(dst)
    if not np.array_equal(got, expect):
        bad = int((got != expect).sum())
        print(f"CORRECTNESS FAIL: {bad}/{N_ELEMS} elements wrong")
        print(f"  first few got:    {got[:8]}")
        print(f"  first few expect: {expect[:8]}")
        return 1
    print(f"correctness: OK ({N_ELEMS} int32 elements)")

    # Warm dispatches -- design is compiled and resident, so this is the
    # steady-state round-trip cost we actually care about.
    for _ in range(20):
        add_one(src, dst, N=N_ELEMS)

    times = []
    for _ in range(200):
        t = time.perf_counter()
        add_one(src, dst, N=N_ELEMS)
        times.append((time.perf_counter() - t) * 1e3)

    times.sort()
    print()
    print("warm round trip (host -> NPU -> host), 200 iterations:")
    print(f"  mean   {statistics.mean(times):8.3f} ms")
    print(f"  p50    {times[len(times)//2]:8.3f} ms")
    print(f"  p90    {times[int(len(times)*0.9)]:8.3f} ms")
    print(f"  min    {times[0]:8.3f} ms")
    print(f"  max    {times[-1]:8.3f} ms")
    print()
    p50 = times[len(times)//2]
    print(f"implication: 210 BitNet matmul dispatches/prefill x {p50:.3f} ms"
          f" = {210*p50:.1f} ms of dispatch overhead")
    return 0

if __name__ == "__main__":
    sys.exit(main())
