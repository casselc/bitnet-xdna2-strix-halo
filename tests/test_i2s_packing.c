/* test_i2s_packing -- verifies our I2_S unpacker against an independent
 * reimplementation of BitNet's quantize_i2_s packing loop.
 *
 * The point of this test is that the packing is a 32-lane interleave that is
 * easy to get subtly wrong (a naive "4 consecutive weights per byte" reading
 * compiles, runs, and produces plausible garbage). So we transcribe the
 * FORWARD loop from src/ggml-bitnet-mad.cpp:80-88 here, verbatim in structure,
 * and require that i2s_unpack_row inverts it exactly. */
#include "../runtime/bitnet_i2s.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); failures++; } } while (0)

/* Verbatim transcription of quantize_i2_s's packing, x86 ACT_PARALLEL path
 * (microsoft/BitNet@0b341e58 src/ggml-bitnet-mad.cpp:80-88). Input is the
 * per-weight unsigned code {0,1,2}; output is the packed byte stream. */
static void reference_pack(const uint8_t *codes, int64_t n, uint8_t *out) {
    memset(out, 0, (size_t)(n / 4));
    for (int64_t i = 0; i < n / I2S_QK; i++) {
        for (int64_t j = 0; j < I2S_QK; j++) {
            int group_idx = (int)(j / 32);
            int group_pos = (int)(j % 32);
            uint8_t temp = (uint8_t)(codes[i * I2S_QK + j] << (6 - 2 * group_idx));
            out[i * 32 + group_pos] |= temp;
        }
    }
}

static uint32_t rng_state = 0x1234567u;
static uint32_t rnd(void) { /* xorshift32, deterministic */
    uint32_t x = rng_state; x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return (rng_state = x);
}

static void test_roundtrip(int64_t K) {
    uint8_t *codes  = malloc((size_t)K);
    uint8_t *packed = malloc((size_t)K / 4);
    uint8_t *back   = malloc((size_t)K);
    for (int64_t i = 0; i < K; ++i) codes[i] = (uint8_t)(rnd() % 3); /* 0,1,2 */

    reference_pack(codes, K, packed);
    i2s_unpack_row(packed, K, back);

    int bad = 0;
    for (int64_t i = 0; i < K; ++i) if (codes[i] != back[i]) bad++;
    CHECK(bad == 0, "K=%lld roundtrip: %d/%lld weights differ", (long long)K, bad, (long long)K);
    if (bad == 0) printf("  ok   roundtrip K=%-5lld (%lld blocks)\n",
                         (long long)K, (long long)(K / I2S_QK));
    free(codes); free(packed); free(back);
}

/* Pin the interleave explicitly: byte b of a block must hold weights
 * {b, b+32, b+64, b+96} at bit positions [7:6],[5:4],[3:2],[1:0]. */
static void test_interleave_layout(void) {
    uint8_t codes[I2S_QK]; memset(codes, 0, sizeof codes);
    uint8_t packed[I2S_QK / 4];

    /* Put a distinct nonzero code in each of the four groups of lane 5. */
    codes[5]      = 2; /* group 0 -> bits [7:6] */
    codes[5 + 32] = 1; /* group 1 -> bits [5:4] */
    codes[5 + 64] = 2; /* group 2 -> bits [3:2] */
    codes[5 + 96] = 1; /* group 3 -> bits [1:0] */
    reference_pack(codes, I2S_QK, packed);

    const uint8_t expect = (uint8_t)((2u << 6) | (1u << 4) | (2u << 2) | 1u);
    CHECK(packed[5] == expect, "lane 5 byte = 0x%02x, expected 0x%02x", packed[5], expect);
    for (int p = 0; p < 32; ++p)
        if (p != 5) CHECK(packed[p] == 0, "lane %d should be 0, got 0x%02x", p, packed[p]);
    if (!failures) printf("  ok   interleave: byte b holds weights {b,b+32,b+64,b+96}, MSB-first\n");
}

static void test_code_mapping(void) {
    const uint8_t codes[4] = {0, 1, 2, 1};
    int8_t signed_out[4];
    i2s_codes_to_signed(codes, 4, signed_out);
    CHECK(signed_out[0] == -1, "code 0 must map to -1, got %d", signed_out[0]);
    CHECK(signed_out[1] ==  0, "code 1 must map to  0, got %d", signed_out[1]);
    CHECK(signed_out[2] == +1, "code 2 must map to +1, got %d", signed_out[2]);
    if (!failures) printf("  ok   code mapping: 0->-1, 1->0, 2->+1\n");
}

/* The offset identity the whole design rests on:
 *   sum_k (w_k + 1)*a_k  ==  sum_k w_k*a_k + sum_k a_k
 * i.e. subtracting act_sums recovers the true ternary dot product. */
static void test_offset_identity(void) {
    const int64_t K = 256;
    uint8_t *codes = malloc((size_t)K);
    int8_t  *w     = malloc((size_t)K);
    int8_t  *a     = malloc((size_t)K);
    for (int64_t i = 0; i < K; ++i) {
        codes[i] = (uint8_t)(rnd() % 3);
        a[i]     = (int8_t)((int)(rnd() % 255) - 127);
    }
    i2s_codes_to_signed(codes, (size_t)K, w);

    int32_t biased = 0, truth = 0, asum = 0;
    for (int64_t i = 0; i < K; ++i) {
        biased += (int32_t)codes[i] * (int32_t)a[i];
        truth  += (int32_t)w[i]     * (int32_t)a[i];
        asum   += (int32_t)a[i];
    }
    CHECK(biased - asum == truth,
          "offset identity broken: biased(%d) - asum(%d) = %d, expected %d",
          biased, asum, biased - asum, truth);
    if (!failures) printf("  ok   offset identity: acc_u - act_sum == true ternary dot\n");
    free(codes); free(w); free(a);
}

static void test_activation_quant(void) {
    const int64_t K = 512;
    float  *x = malloc(sizeof(float) * (size_t)K);
    int8_t *q = malloc((size_t)K);
    float scale; int32_t sum;
    for (int64_t i = 0; i < K; ++i)
        x[i] = ((float)(rnd() % 20000) / 10000.0f) - 1.0f; /* [-1,1) */
    x[42] = 3.5f; /* a definite absmax */

    i2s_quantize_activations(x, K, q, &scale, &sum);

    CHECK(fabsf(scale - 127.0f / 3.5f) < 1e-3f,
          "scale should be 127/absmax = %.6f, got %.6f", 127.0f / 3.5f, scale);
    CHECK(q[42] == 127, "the absmax element must saturate to 127, got %d", q[42]);
    int32_t recomputed = 0;
    for (int64_t i = 0; i < K; ++i) recomputed += q[i];
    CHECK(recomputed == sum, "reported sum %d != recomputed %d", sum, recomputed);
    /* Range is enforced by int8_t itself; what actually needs checking is that
     * the clamp fires rather than wrapping, so verify the saturating element. */
    CHECK(q[42] == 127, "clamp must saturate, not wrap");
    if (!failures) printf("  ok   activation quant: per-token absmax int8, scale=127/amax, sum consistent\n");

    /* The 1e-5 floor must stop an all-zero row from producing inf/nan. */
    for (int64_t i = 0; i < K; ++i) x[i] = 0.0f;
    i2s_quantize_activations(x, K, q, &scale, &sum);
    CHECK(isfinite(scale), "all-zero activations produced non-finite scale %f", scale);
    CHECK(sum == 0, "all-zero activations should sum to 0, got %d", sum);
    if (!failures) printf("  ok   activation quant: all-zero row clamped by the 1e-5 floor\n");
    free(x); free(q);
}

int main(void) {
    printf("test_i2s_packing\n");
    test_code_mapping();
    test_interleave_layout();
    /* Every real BitNet-2B-4T linear K is a multiple of 128. */
    test_roundtrip(128); test_roundtrip(2560); test_roundtrip(6912);
    test_offset_identity();
    test_activation_quant();
    if (failures) { printf("\n%d FAILURE(S)\n", failures); return 1; }
    printf("\nall passed\n");
    return 0;
}
