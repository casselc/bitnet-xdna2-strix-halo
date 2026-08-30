/* test_xdna_shapes -- bit-exact coverage for ALL THREE real BitNet shapes,
 * through the production code path.
 *
 * test_xdna_gemm.cpp drives Program::run() directly on one 2560x2560 tile. That
 * leaves the parts most likely to be wrong untested: N-chunking (2560x6912 is
 * served as 3 chunks with the last one padded 6912->7680), K-chunking with int32
 * partial accumulation (6912x2560), the M zero-padding of a short final tile, and
 * the K-outside-N loop order. This test calls bitnet_xdna_accumulate() itself --
 * the same entry point ggml uses -- and compares against the scalar oracle.
 *
 * Token counts deliberately include tile boundaries and a non-multiple, since a
 * short final tile exercises the zero-padding path.
 */
extern "C" {
#include "../runtime/bitnet_i2s.h"
#include "../runtime/bitnet_xdna.h"
}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint32_t rng = 0xA5A5A5u;
static uint32_t xs32() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

struct Case { const char *name; const char *file; int64_t K, N; };

static int run_case(const Case &c, int64_t T) {
    const size_t packed = i2s_packed_bytes(c.K, c.N);
    std::vector<uint8_t> blob(packed + 4);
    FILE *f = std::fopen(c.file, "rb");
    if (!f) { std::printf("    cannot open %s\n", c.file); return 1; }
    const size_t got = std::fread(blob.data(), 1, packed + 4, f);
    std::fclose(f);
    if (got != packed + 4) { std::printf("    short read\n"); return 1; }

    // Oracle operands: real ternary weights, deterministic int8 activations.
    std::vector<uint8_t> codes((size_t)(c.K * c.N));
    i2s_unpack_matrix(blob.data(), c.K, c.N, codes.data());
    std::vector<int8_t> w_nk((size_t)(c.K * c.N));
    i2s_codes_to_signed(codes.data(), codes.size(), w_nk.data());

    std::vector<int8_t> a((size_t)(T * c.K));
    for (size_t i = 0; i < a.size(); ++i) a[i] = (int8_t)((int)(xs32() % 255) - 127);

    // The NPU path writes f32 through the epilogue, so drive the oracle the same
    // way: signed accumulate, then scale. act_scales are 1.0 so the comparison
    // stays exact and any divergence is the integer path's fault.
    std::vector<float>   act_scales((size_t)T, 1.0f);
    std::vector<int32_t> act_sums((size_t)T, 0);
    std::vector<float>   dst((size_t)(T * c.N), 0.0f);

    if (!bitnet_xdna_accumulate(blob.data(), c.K, c.N, a.data(), T, (size_t)c.K)) {
        std::printf("    NPU declined this shape\n");
        return 1;
    }
    bitnet_xdna_epilogue(c.N, 0, T, act_scales.data(), 1.0f, dst.data(),
                         (size_t)(c.N * sizeof(float)));

    int64_t bad = 0; double worst = 0;
    for (int64_t t = 0; t < T; ++t) {
        const int8_t *arow = a.data() + t * c.K;
        for (int64_t n = 0; n < c.N; ++n) {
            const int8_t *wrow = w_nk.data() + n * c.K;
            int32_t s = 0;
            for (int64_t k = 0; k < c.K; ++k) s += (int32_t)wrow[k] * (int32_t)arow[k];
            const double d = dst[t * c.N + n] - (double)s;
            if (d != 0.0) { ++bad; if (d > worst || -d > worst) worst = d < 0 ? -d : d; }
        }
    }
    if (bad) {
        std::printf("    FAIL T=%-5lld  %lld/%lld differ, worst %.0f\n",
                    (long long)T, (long long)bad, (long long)(T * c.N), worst);
        return 1;
    }
    std::printf("    ok   T=%-5lld  %lld values bit-exact\n",
                (long long)T, (long long)(T * c.N));
    return 0;
}

int main() {
    const Case cases[] = {
        {"attn_q / attn_output  (1 N-chunk, 1 K-chunk)",
         "artifacts/correctness/tensors/attn_q_l0.packed",   2560, 2560},
        {"ffn_gate / ffn_up     (3 N-chunks, N padded 6912->7680)",
         "artifacts/correctness/tensors/ffn_gate_l0.packed", 2560, 6912},
        {"ffn_down              (3 K-chunks, int32 accumulation)",
         "artifacts/correctness/tensors/ffn_down_l0.packed", 6912, 2560},
    };
    // 1024 = exactly one tile; 1536 = one full + one HALF tile (zero-padding);
    // 2048 = two full tiles.
    const int64_t token_counts[] = {1024, 1536, 2048};

    std::printf("test_xdna_shapes  (production path: bitnet_xdna_accumulate + epilogue)\n");
    if (!bitnet_xdna_available()) {
        std::printf("  NPU unavailable (set BITNET_XDNA=1 and BITNET_XDNA_ARTIFACTS)\n");
        return 77;
    }
    int fails = 0;
    for (const auto &c : cases) {
        std::printf("  %s\n", c.name);
        for (int64_t T : token_counts) fails += run_case(c, T);
    }
    if (fails) { std::printf("\n%d FAILURE(S)\n", fails); return 1; }
    std::printf("\nall shapes bit-exact\n");
    return 0;
}
