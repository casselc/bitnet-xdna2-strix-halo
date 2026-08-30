/* bitnet_i2s.h -- I2_S ternary weight format, reference (golden) implementations.
 *
 * This header is the single source of truth for what the NPU kernel must
 * reproduce. Every claim below was read out of the BitNet sources at the pinned
 * commits, not inferred from names or papers:
 *
 *   packing      microsoft/BitNet@0b341e58 src/ggml-bitnet-mad.cpp:51-95
 *                (quantize_i2_s, x86 ACT_PARALLEL path)
 *   dispatch     isHuangXin/llama.cpp@390c307 ggml/src/ggml-cpu/ggml-cpu.c:1491-1545
 *   act. quant   same file :1418 (quantize_row_i8_s)
 *
 * FORMAT
 * ------
 * Ternary weights are stored as a 2-bit UNSIGNED code with a +1 offset:
 *
 *     code 0 -> -1     code 1 -> 0     code 2 -> +1
 *
 * On x86 the block is QK_I2_S = 128 weights -> 32 bytes. Within a block,
 * weight j goes to byte (j % 32) at bit position (6 - 2*(j / 32)):
 *
 *     byte b holds weights { b, b+32, b+64, b+96 }
 *            at bits        [7:6]  [5:4]  [3:2]  [1:0]
 *
 * That is a 32-lane SIMD interleave -- it is NOT four consecutive weights per
 * byte. Getting this wrong yields plausible-looking garbage, so
 * i2s_unpack_row() is tested against a from-scratch reimplementation of
 * quantize_i2_s in tests/test_i2s_packing.c.
 *
 * A SINGLE f32 scale covers the ENTIRE tensor. It lives immediately after the
 * packed data at byte offset (K*N)/4, followed by 32 bytes of alignment padding.
 * There is no per-block and no per-row scale.
 *
 * THE OPERATION
 * -------------
 *     acc_u[n,t] = sum_k  w_code[n,k] * a_q[k,t]     w_code in {0,1,2} u8
 *                                                    a_q    int8
 *     dst[n,t]   = (acc_u[n,t] - act_sums[t]) * (ws / act_scales[t])
 *
 * The `- act_sums[t]` term is what converts the biased unsigned accumulator
 * back into a true ternary dot product, since
 *     sum_k (w_k + 1) * a_k  =  sum_k w_k*a_k  +  sum_k a_k.
 * The NPU computes acc_u only; the epilogue stays on the CPU.
 */
#ifndef BITNET_I2S_H
#define BITNET_I2S_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Block size for the x86 ACT_PARALLEL layout. ARM NEON uses 64; we are x86. */
#define I2S_QK 128
/* Weights per packed byte. */
#define I2S_PER_BYTE 4

/* Bytes of packed weight data for a K*N tensor (excludes the scale + padding). */
static inline size_t i2s_packed_bytes(int64_t K, int64_t N) {
    return (size_t)((K * N) / I2S_PER_BYTE);
}

/* Byte offset of the per-tensor f32 scale within the tensor's data blob. */
static inline size_t i2s_scale_offset(int64_t K, int64_t N) {
    return i2s_packed_bytes(K, N);
}

/* Read the per-tensor scale out of a packed I2_S blob. */
float i2s_tensor_scale(const void *packed, int64_t K, int64_t N);

/* Unpack one row of K weights into `out` as the unsigned codes {0,1,2}.
 * `row` points at the start of that row's packed bytes (stride K/4).
 * This is the exact inverse of quantize_i2_s's packing loop. */
void i2s_unpack_row(const uint8_t *row, int64_t K, uint8_t *out);

/* Unpack a whole [N,K] weight matrix to unsigned codes, row-major, N*K bytes. */
void i2s_unpack_matrix(const void *packed, int64_t K, int64_t N, uint8_t *out);

/* Convert unsigned codes {0,1,2} to signed ternary {-1,0,+1}. */
void i2s_codes_to_signed(const uint8_t *codes, size_t n, int8_t *out);

/* Per-token symmetric int8 activation quantization, matching quantize_row_i8_s.
 *   scale = 127 / max(|x|, 1e-5)   q = clamp(round(x*scale), -128, 127)
 *   sum   = sum(q)
 * `x` is one token's K activations; `q` receives K int8 values. */
void i2s_quantize_activations(const float *x, int64_t K,
                              int8_t *q, float *scale, int32_t *sum);

/* Golden reference for the raw NPU-side accumulator.
 *   acc[t*N + n] = sum_k w_code[n,k] * a_q[k,t]
 * `w_codes` is [N,K] unsigned codes; `a_q` is [T,K] int8 (token-major).
 * Integer arithmetic only -- the NPU must match this BIT-EXACTLY. */
void i2s_ref_accumulate(const uint8_t *w_codes, const int8_t *a_q,
                        int64_t K, int64_t N, int64_t T, int32_t *acc);

/* The CPU-side epilogue that turns accumulators into final f32 outputs. */
void i2s_apply_epilogue(const int32_t *acc, int64_t N, int64_t T,
                        const float *act_scales, const int32_t *act_sums,
                        float ws, float *dst);

#ifdef __cplusplus
}
#endif
#endif /* BITNET_I2S_H */
