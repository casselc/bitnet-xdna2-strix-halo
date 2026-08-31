"""A FUSED attention core: QK -> softmax -> PV all on one AIE core.

AMD's MHA is a three-stage spatial pipeline, one stage per core, one column per
pipeline. Measured stage times per (q,kv) pair are QK 3.938 us, softmax 9.412 us,
PV 2.797 us -- a 3.4:1 imbalance. A spatial pipeline runs at its slowest stage
but pays for all three cores, so it spends 31.98 us of core-time to do 16.147 us
of work: 50.5% compute efficiency, QK idle 63%, PV idle 74%.

This design tests the alternative: one core does all three stages for its own q
block, so nothing waits on a neighbour. If a fused core takes ~16.1 us per pair
-- the sum of the stage times -- the utilisation argument holds and the rest is
arithmetic. If it takes materially longer, fusing has costs the stage times do
not capture (lost DMA/compute overlap, register pressure, ObjectFifo overhead)
and the idea is dead.

Everything the stock design passes between cores becomes core-local here: the A
fifo (QK->softmax), the P fifo and the scale fifo (softmax->PV), and the
passThrough kernel that copies the scale buffer, all disappear.

Two constraints shape the design, both found by building rather than reading:

  * A compute tile has 2 input DMA channels. Q, K and V is three. The stock
    design has 3-input cores only because its inter-stage fifos connect
    vertically adjacent tiles and map to shared local memory, costing no
    channel. Here K and V -- always consumed together -- share one stream and
    are acquired two at a time.
  * K and V need DIFFERENT dims_to_stream transforms (K feeds a col-major
    matmul, V a row-major one), so they cannot share a transformed stream. The
    transforms are applied host-side instead and the stream is raw. They are
    shim/memtile work, not core work, so this does not affect what is being
    measured; at 16 KiB per pair against ~16 us of compute the stream runs at
    ~1 GB/s, far below the fabric's limit.

A and P alias one buffer. partial_softmax_alias_bf16 is written for that: its
first pass only reads the input, and its second reads position i then writes
position i.
"""
import sys

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, Buffer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_

# aie::mmul shape for bf16 on npu2 with AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
R, S, T = 8, 8, 8


def tile_major(x, r, c):
    """Row-major (rows, cols) -> the tile-major order aie::mmul expects.

    This is what the stock design's dims_to_stream performs in the DMA; done
    here on the host so K and V can share one untransformed stream."""
    rows, cols = x.shape
    return x.reshape(rows // r, r, cols // c, c).transpose(0, 2, 1, 3).reshape(-1)


def from_tile_major(flat, rows, cols, r, c):
    """Inverse of tile_major, for reading O back."""
    return (flat.reshape(rows // r, cols // c, r, c)
                .transpose(0, 2, 1, 3).reshape(rows, cols))


def fused_mha(B_q=64, B_kv=64, d=64, n_kv_blocks=64, kv_depth=4,
              relayout_traffic=False):
    """One fused core, one q block, n_kv_blocks key blocks.

    Every block must be a FULL off-diagonal block -- the steady-state pair the
    stage model is built from -- with no masking making the work per pair
    uneven. partial_softmax derives its masks from idx_buffer and the effective
    sequence extents, so those have to be mutually consistent:

        q_block_idx  = n_kv_blocks     strictly above every kv index, so
                                       neither the block-causal skip
                                       (kv > q) nor the diagonal triangular
                                       mask (kv == q) fires
        S_q_eff  = (n_kv+1) * B_q      valid_q_rows  = S_q_eff - q_idx*B_q  = B_q
        S_kv_eff =  n_kv    * B_kv     valid_kv_cols = S_kv_eff - kv_idx*B_kv >= B_kv

    Getting this wrong is silent and looks like a win: an earlier version set
    q_block_idx and S_eff both to 1<<20, which made valid_q_rows go negative,
    took the "fully padded block contributes nothing" path, and zeroed every
    block. The kernel ran 1.7x faster than the stage model predicts and returned
    NaN.

    relayout_traffic adds the memory traffic of the two per-pair LAYOUT
    TRANSFORMS the stock design gets for free from its inter-core DMA. Its
    memA fifo carries a_dims, turning the QK matmul's tile-major C into the
    row-major layout partial_softmax indexes with A[i*B_kv + j]; its memP fifo
    carries q_dims, turning P back into the tile-major operand layout the PV
    matmul wants. A fused core has no inter-stage DMA, so it must do both
    itself. With this flag the core performs two full-block passes with
    passThroughLine, which is the READ+WRITE TRAFFIC of a relayout without the
    shuffle, and therefore a LOWER BOUND on what a real one costs.
    """
    q_block_bias = n_kv_blocks
    S_q_eff = (n_kv_blocks + 1) * B_q
    S_kv_eff = n_kv_blocks * B_kv
    dtype = bfloat16
    q_ty = np.ndarray[(B_q, d), np.dtype[dtype]]
    kv_ty = np.ndarray[(d, B_kv), np.dtype[dtype]]
    qk_ty = np.ndarray[(B_q, B_kv), np.dtype[dtype]]
    s_ty = np.ndarray[(4 * B_q,), np.dtype[dtype]]
    idx_ty = np.ndarray[(2,), np.dtype[np.int32]]

    # Same constant as design.py: 1/sqrt(d) folded with log2(e) so the kernel
    # can use aie::exp2. A plain Python float, as there.
    inv_scale = (1 / np.sqrt(d)) * 1.4453125

    zero = Kernel("zero_bf16", "mha.o", [qk_ty])
    matmul_QK = Kernel("matmul_bf16_bf16_wrapper", "mha.o",
                       [q_ty, kv_ty, qk_ty, idx_ty])
    partial_softmax = Kernel("partial_softmax", "mha.o",
                             [qk_ty, qk_ty, s_ty, idx_ty, dtype,
                              np.int32, np.int32, np.int32, np.int32])
    matmul_PV = Kernel("matmul_PV", "mha.o",
                       [qk_ty, kv_ty, qk_ty, s_ty, np.int32, np.int32, idx_ty])
    rescale_O = Kernel("rescale_O", "mha.o", [qk_ty, s_ty, np.int32, idx_ty])
    init_scale = Kernel("init_scale_buffer", "mha.o", [s_ty, np.int32])
    # BIT_WIDTH=16, so this moves bf16 elements. In place: same traffic, no
    # extra L1, and it cannot perturb the result.
    passthru = Kernel("passThroughLine", "mha_passThrough.o",
                      [qk_ty, qk_ty, np.int32])

    of_q = ObjectFifo(q_ty, name="inQ", depth=1)
    of_kv = ObjectFifo(kv_ty, name="inKV", depth=kv_depth)
    of_o = ObjectFifo(qk_ty, name="outO", depth=1)

    a_buf = Buffer(qk_ty, name="A_local")          # A and P, aliased
    scale_buf = Buffer(s_ty, name="scale_local")
    idx_buf = Buffer(idx_ty, name="idx_local")

    def fused(of_q, of_kv, of_o, a_buf, scale_buf, idx_buf,
              zero, matmul_QK, partial_softmax, matmul_PV, rescale_O,
              init_scale, passthru):
        for _ in range_(sys.maxsize):
            elem_q = of_q.acquire(1)
            elem_o = of_o.acquire(1)
            zero(elem_o)
            init_scale(scale_buf, B_q)
            idx_buf[0] = 0
            idx_buf[1] = q_block_bias

            # First kv block: first_iter=0, so O_{i-1} is not rescaled (the
            # running max starts at -inf and the factor would be inf).
            kv = of_kv.acquire(2)               # kv[0] = K block, kv[1] = V block
            zero(a_buf)
            matmul_QK(elem_q, kv[0], a_buf, idx_buf)
            if relayout_traffic:
                passthru(a_buf, a_buf, B_q * B_kv)      # a_dims, on-core
            partial_softmax(a_buf, a_buf, scale_buf, idx_buf, inv_scale,
                            B_q, B_kv, S_q_eff, S_kv_eff)
            if relayout_traffic:
                passthru(a_buf, a_buf, B_q * B_kv)      # q_dims, on-core
            matmul_PV(a_buf, kv[1], elem_o, scale_buf, B_q, 0, idx_buf)
            of_kv.release(2)
            idx_buf[0] += 1

            for _ in range_(n_kv_blocks - 1):
                kv = of_kv.acquire(2)
                zero(a_buf)
                matmul_QK(elem_q, kv[0], a_buf, idx_buf)
                if relayout_traffic:
                    passthru(a_buf, a_buf, B_q * B_kv)
                partial_softmax(a_buf, a_buf, scale_buf, idx_buf, inv_scale,
                                B_q, B_kv, S_q_eff, S_kv_eff)
                if relayout_traffic:
                    passthru(a_buf, a_buf, B_q * B_kv)
                matmul_PV(a_buf, kv[1], elem_o, scale_buf, B_q, 1, idx_buf)
                of_kv.release(2)
                idx_buf[0] += 1

            rescale_O(elem_o, scale_buf, B_q, idx_buf)
            of_o.release(1)
            of_q.release(1)

    worker = Worker(fused, [of_q.cons(), of_kv.cons(), of_o.prod(),
                            a_buf, scale_buf, idx_buf, zero, matmul_QK,
                            partial_softmax, matmul_PV, rescale_O, init_scale,
                            passthru])

    Q_h = np.ndarray[(B_q * d,), np.dtype[dtype]]
    KV_h = np.ndarray[(2 * n_kv_blocks * d * B_kv,), np.dtype[dtype]]
    O_h = np.ndarray[(B_q * B_kv,), np.dtype[dtype]]

    inq, inkv, outo = of_q.prod(), of_kv.prod(), of_o.cons()

    def sequence(q, kv, o, inq, inkv, outo):
        inq.fill(q)
        inkv.fill(kv)
        outo.drain(o, wait=True)

    rt = Runtime(sequence, [Q_h, KV_h, O_h, inq, inkv, outo])
    return Program(NPU2(), rt, workers=[worker]).resolve_program()
