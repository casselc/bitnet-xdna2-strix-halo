"""Program-memory check for a FUSED attention core.

AMD's MHA splits QK -> softmax -> PV across three cores (rows 2, 3, 4). The
measured stage times are 3.938 / 9.412 / 2.797 us, so a spatial pipeline that
runs at its slowest stage spends 3 x 9.412 us of core-time to do 16.147 us of
work: 50.5% compute efficiency, with the QK core idle 63% and the PV core idle
74% of the time.

Putting all three stages on ONE core recovers that, but only if the code fits.
AIE2P cores have 16 KiB of program memory, and this project has already seen it
overflow on unrolled query sweeps. The .text of the stock build's three cores
sums to 1696 + 5328 + 3728 = 10752 B, which suggests headroom -- but a sum is
not a build. This builds a single core that actually CALLS every kernel a fused
core would need, so the compiler must emit all their bodies, and reports the
resulting .text.

It is deliberately not a working attention kernel: the loop structure and buffer
plumbing are the cheapest thing that forces all five kernel bodies to be
emitted and kept. The only question asked here is code size.
"""
import sys
from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, Buffer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_


def fused_progmem_probe(B_q=64, B_kv=64, d=64, n_kv_blocks=4):
    dtype = bfloat16
    # Same constant design.py uses: 1/sqrt(d) folded with log2(e) so the kernel
    # can use aie::exp2. A plain Python float, as there.
    inv_scale = (1 / np.sqrt(d)) * 1.4453125
    q_ty = np.ndarray[(B_q, d), np.dtype[dtype]]
    k_ty = np.ndarray[(d, B_kv), np.dtype[dtype]]
    qk_ty = np.ndarray[(B_q, B_kv), np.dtype[dtype]]
    s_ty = np.ndarray[(4 * B_q,), np.dtype[dtype]]
    idx_ty = np.ndarray[(2,), np.dtype[np.int32]]

    # Exactly the symbols mha.o exports and a fused core would call.
    zero = Kernel("zero_bf16", "mha.o", [qk_ty])
    matmul_QK = Kernel("matmul_bf16_bf16_wrapper", "mha.o",
                       [q_ty, k_ty, qk_ty, idx_ty])
    partial_softmax = Kernel("partial_softmax", "mha.o",
                             [qk_ty, qk_ty, s_ty, idx_ty, dtype,
                              np.int32, np.int32, np.int32, np.int32])
    matmul_PV = Kernel("matmul_PV", "mha.o",
                       [qk_ty, k_ty, qk_ty, s_ty, np.int32, np.int32, idx_ty])
    rescale_O = Kernel("rescale_O", "mha.o", [qk_ty, s_ty, np.int32, idx_ty])
    init_scale = Kernel("init_scale_buffer", "mha.o", [s_ty, np.int32])

    # A compute tile has only 2 input DMA channels. The stock design gets away
    # with 3-input cores because its inter-stage fifos connect VERTICALLY
    # ADJACENT tiles and map to shared local memory, costing no DMA channel. A
    # fused core has no neighbour to share with: Q, K and V all arrive from the
    # memtile, which is 3 channels and does not build.
    #
    # K and V for one kv block are always consumed together, so they share one
    # stream and are taken two at a time. That is 2 in (Q, KV) / 1 out (O).
    of_q = ObjectFifo(q_ty, name="inQ", depth=1)
    of_kv = ObjectFifo(k_ty, name="inKV", depth=4)
    of_o = ObjectFifo(qk_ty, name="outO", depth=1)

    # Core-local, because a fused core hands nothing to a neighbour: the A and P
    # ObjectFifos and the scale fifo of the stock design all disappear here.
    # A and P alias deliberately -- partial_softmax_alias_bf16 writes its output
    # over its input, which is what the "alias" in its name means.
    a_buf = Buffer(np.ndarray[(B_q, B_kv), np.dtype[dtype]], name="A_local")
    scale_buf = Buffer(np.ndarray[(4 * B_q,), np.dtype[dtype]], name="scale_local")
    idx_buf = Buffer(np.ndarray[(2,), np.dtype[np.int32]], name="idx_local")

    def fused_task(of_q, of_kv, of_o, a_buf, scale_buf, idx_buf,
                   zero, matmul_QK, partial_softmax, matmul_PV, rescale_O,
                   init_scale):
        # inv_scale is captured from the enclosing scope, as in design.py.
        for _ in range_(sys.maxsize):
            elem_q = of_q.acquire(1)
            elem_o = of_o.acquire(1)
            zero(elem_o)
            init_scale(scale_buf, B_q)
            idx_buf[0] = 0
            idx_buf[1] = 0
            for _ in range_(n_kv_blocks):
                kv = of_kv.acquire(2)          # kv[0] = K block, kv[1] = V block
                zero(a_buf)
                matmul_QK(elem_q, kv[0], a_buf, idx_buf)
                partial_softmax(a_buf, a_buf, scale_buf, idx_buf,
                                inv_scale, B_q, B_kv, B_q, B_kv)
                matmul_PV(a_buf, kv[1], elem_o, scale_buf, B_q, 0, idx_buf)
                of_kv.release(2)
                idx_buf[0] += 1
            rescale_O(elem_o, scale_buf, B_q, idx_buf)
            of_o.release(1)
            of_q.release(1)

    worker = Worker(fused_task,
                    [of_q.cons(), of_kv.cons(), of_o.prod(),
                     a_buf, scale_buf, idx_buf, zero, matmul_QK,
                     partial_softmax, matmul_PV, rescale_O, init_scale])

    Q_h = np.ndarray[(B_q * d,), np.dtype[dtype]]
    K_h = np.ndarray[(2 * n_kv_blocks * d * B_kv,), np.dtype[dtype]]
    O_h = np.ndarray[(B_q * B_kv,), np.dtype[dtype]]

    inq, inkv, outo = of_q.prod(), of_kv.prod(), of_o.cons()

    def sequence(q, kv, o, inq, inkv, outo):
        inq.fill(q)
        inkv.fill(kv)
        outo.drain(o, wait=True)

    rt = Runtime(sequence, [Q_h, K_h, O_h, inq, inkv, outo])
    return Program(NPU2(), rt, workers=[worker]).resolve_program()
