#!/usr/bin/env python3
"""Cost of alternating between two compiled designs.

Matters because BitNet needs several distinct (K,N) shapes and each one is a
separate xclbin. Published XDNA2 data puts a hwctx/xclbin context switch at
2.67 ms -- 13x a plain dispatch. If that holds here, cycling shapes per layer
would cost more than the arithmetic, and the design must instead pad every
shape onto a single kernel. This measures it directly.
"""
import time, statistics
import numpy as np
import aie.iron as iron
from aie.iron import In, Out, CompileTime

@iron.jit
def add_one(inp: In, out: Out, *, N: CompileTime[int], dtype: CompileTime[type] = np.int32):
    tensor_ty = np.ndarray[(N,), np.dtype[dtype]]
    return iron.algorithms.transform(lambda x: x + 1, tensor_ty)

@iron.jit
def add_two(inp: In, out: Out, *, N: CompileTime[int], dtype: CompileTime[type] = np.int32):
    tensor_ty = np.ndarray[(N,), np.dtype[dtype]]
    return iron.algorithms.transform(lambda x: x + 2, tensor_ty)

N = 4096
a = iron.arange(N, dtype=np.int32); o1 = iron.zeros(N, dtype=np.int32)
b = iron.arange(N, dtype=np.int32); o2 = iron.zeros(N, dtype=np.int32)

# compile + warm both
for _ in range(10):
    add_one(a, o1, N=N); add_two(b, o2, N=N)

def timeit(fn, n=150):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter()-t)*1e3)
    ts.sort(); return ts[len(ts)//2]

same = timeit(lambda: add_one(a, o1, N=N))
# alternating forces a different design (and hwctx) on every call
state = {'i': 0}
def alt():
    if state['i'] % 2: add_two(b, o2, N=N)
    else:              add_one(a, o1, N=N)
    state['i'] += 1
alternating = timeit(alt, 300)

print(f"same design, repeated      p50: {same:7.3f} ms")
print(f"alternating two designs    p50: {alternating:7.3f} ms")
print(f"switch penalty per call        : {alternating - same:7.3f} ms")
print()
print(f"If BitNet cycles 3 shapes/layer x 30 layers = 90 switches/prefill,")
print(f"that is {(alternating-same)*90:.1f} ms of switching overhead.")
