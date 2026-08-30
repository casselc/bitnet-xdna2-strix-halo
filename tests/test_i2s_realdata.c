/* test_i2s_realdata -- validate the unpacker against REAL BitNet weights.
 *
 * The round-trip test in test_i2s_packing.c checks our unpacker against our own
 * reimplementation of the packer. Both halves could be wrong in the same way.
 * This test uses actual bytes from the shipped GGUF, where the data itself
 * constrains what a correct reading must look like:
 *
 *   - every 2-bit field must decode to a code in {0,1,2}; BitNet never emits 3
 *   - the +1/-1 populations must be roughly balanced (absmean ternarization is
 *     symmetric around zero)
 *   - a real trained tensor must contain all three codes in quantity
 *
 * A wrong bit-order would still yield codes 0..3, so seeing zero occurrences of
 * code 3 across 6.5M weights is strong evidence the field extraction is right.
 */
#include "../runtime/bitnet_i2s.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1]
        : "artifacts/correctness/tensors/attn_q_l0.packed";
    const int64_t K = argc > 3 ? atoll(argv[2]) : 2560;
    const int64_t N = argc > 3 ? atoll(argv[3]) : 2560;

    FILE *f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    const size_t packed = i2s_packed_bytes(K, N);
    uint8_t *blob = malloc(packed + 4);
    if (fread(blob, 1, packed + 4, f) != packed + 4) { printf("short read\n"); return 1; }
    fclose(f);

    printf("test_i2s_realdata  %s  [K=%lld N=%lld]\n", path, (long long)K, (long long)N);

    const float ws = i2s_tensor_scale(blob, K, N);
    printf("  per-tensor scale: %.10f\n", ws);
    if (!(ws > 0.0f) || !isfinite(ws)) { printf("  FAIL: implausible scale\n"); return 1; }

    uint8_t *codes = malloc((size_t)(K * N));
    i2s_unpack_matrix(blob, K, N, codes);

    int64_t hist[4] = {0, 0, 0, 0};
    for (int64_t i = 0; i < K * N; ++i) hist[codes[i] & 3]++;

    const int64_t total = K * N;
    printf("  code 0 (-1): %10lld  %6.2f%%\n", (long long)hist[0], 100.0 * hist[0] / total);
    printf("  code 1 ( 0): %10lld  %6.2f%%\n", (long long)hist[1], 100.0 * hist[1] / total);
    printf("  code 2 (+1): %10lld  %6.2f%%\n", (long long)hist[2], 100.0 * hist[2] / total);
    printf("  code 3 (--): %10lld  %6.2f%%   <- must be exactly 0\n",
           (long long)hist[3], 100.0 * hist[3] / total);

    int fail = 0;
    if (hist[3] != 0) {
        printf("  FAIL: %lld weights decoded to the invalid code 3 -- bit layout is wrong\n",
               (long long)hist[3]);
        fail = 1;
    }
    if (hist[0] == 0 || hist[1] == 0 || hist[2] == 0) {
        printf("  FAIL: a real trained tensor must use all three codes\n");
        fail = 1;
    }
    /* Absmean ternarization is symmetric, so -1 and +1 counts should be close.
     * Allow a generous 10% relative skew before calling it broken. */
    const double skew = fabs((double)hist[0] - (double)hist[2])
                      / ((double)hist[0] + (double)hist[2]);
    printf("  +/-1 skew: %.4f (symmetric ternarization expects ~0)\n", skew);
    if (skew > 0.10) { printf("  FAIL: implausible sign asymmetry\n"); fail = 1; }

    /* Sparsity sanity: BitNet b1.58 tensors are meaningfully sparse but not
     * degenerate. Anything outside 10%..90% zeros suggests a misread. */
    const double zfrac = (double)hist[1] / total;
    if (zfrac < 0.10 || zfrac > 0.90) {
        printf("  FAIL: zero fraction %.3f is implausible for b1.58\n", zfrac);
        fail = 1;
    }

    if (!fail) printf("\n  ok  real GGUF weights decode cleanly under our layout\n");
    free(codes); free(blob);
    return fail;
}
