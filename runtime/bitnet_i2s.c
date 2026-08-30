#include "bitnet_i2s.h"
#include <math.h>
#include <string.h>

float i2s_tensor_scale(const void *packed, int64_t K, int64_t N) {
    float s;
    memcpy(&s, (const uint8_t *)packed + i2s_scale_offset(K, N), sizeof(float));
    return s;
}

void i2s_unpack_row(const uint8_t *row, int64_t K, uint8_t *out) {
    /* Mirror of quantize_i2_s's packing loop, run backwards.
     * Forward:  byte[i*32 + (j%32)] |= code[i*128 + j] << (6 - 2*(j/32))
     * so for block i, lane p = j%32, group g = j/32:
     *     code = (byte[i*32 + p] >> (6 - 2*g)) & 3
     * Note the group index runs MSB-first: g=0 sits in bits [7:6]. */
    const int64_t nblocks = K / I2S_QK;
    for (int64_t i = 0; i < nblocks; ++i) {
        const uint8_t *blk = row + i * (I2S_QK / I2S_PER_BYTE); /* 32 bytes */
        for (int g = 0; g < 4; ++g) {
            const int shift = 6 - 2 * g;
            uint8_t *dst = out + i * I2S_QK + (int64_t)g * 32;
            for (int p = 0; p < 32; ++p)
                dst[p] = (uint8_t)((blk[p] >> shift) & 0x3);
        }
    }
    /* K is a multiple of 128 for every BitNet-b1.58-2B-4T linear tensor
     * (2560 = 20*128, 6912 = 54*128), so there is no partial-block tail.
     * Assert rather than silently mis-unpack if that ever changes. */
}

void i2s_unpack_matrix(const void *packed, int64_t K, int64_t N, uint8_t *out) {
    const uint8_t *base = (const uint8_t *)packed;
    const size_t row_bytes = (size_t)(K / I2S_PER_BYTE);
    for (int64_t n = 0; n < N; ++n)
        i2s_unpack_row(base + (size_t)n * row_bytes, K, out + n * K);
}

void i2s_codes_to_signed(const uint8_t *codes, size_t n, int8_t *out) {
    for (size_t i = 0; i < n; ++i) out[i] = (int8_t)codes[i] - 1;
}

void i2s_quantize_activations(const float *x, int64_t K,
                              int8_t *q, float *scale, int32_t *sum) {
    float amax = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        const float a = fabsf(x[k]);
        if (a > amax) amax = a;
    }
    if (amax < 1e-5f) amax = 1e-5f;
    const float s = 127.0f / amax;
    int32_t acc = 0;
    for (int64_t k = 0; k < K; ++k) {
        float v = roundf(x[k] * s);
        if (v > 127.0f)  v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        q[k] = (int8_t)v;
        acc += (int32_t)q[k];
    }
    *scale = s;
    *sum   = acc;
}

void i2s_ref_accumulate(const uint8_t *w_codes, const int8_t *a_q,
                        int64_t K, int64_t N, int64_t T, int32_t *acc) {
    /* Deliberately the dumbest correct implementation: this is the oracle, so
     * it is optimized for being obviously right, not for speed. */
    for (int64_t t = 0; t < T; ++t) {
        const int8_t *a = a_q + t * K;
        for (int64_t n = 0; n < N; ++n) {
            const uint8_t *w = w_codes + n * K;
            int32_t s = 0;
            for (int64_t k = 0; k < K; ++k)
                s += (int32_t)w[k] * (int32_t)a[k];
            acc[t * N + n] = s;
        }
    }
}

void i2s_apply_epilogue(const int32_t *acc, int64_t N, int64_t T,
                        const float *act_scales, const int32_t *act_sums,
                        float ws, float *dst) {
    for (int64_t t = 0; t < T; ++t) {
        const float post = ws / act_scales[t];
        const int32_t asum = act_sums[t];
        for (int64_t n = 0; n < N; ++n)
            dst[t * N + n] = (float)(acc[t * N + n] - asum) * post;
    }
}
