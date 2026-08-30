/* test_xdna_gemm -- is the NPU's integer GEMM bit-identical to the CPU's?
 *
 * This op is pure integer arithmetic, so there is no floating-point tolerance
 * to hide behind: every accumulator must match exactly. Anything less means the
 * layout, the tiling, or the ternary expansion is wrong.
 *
 * Weights come from the real shipped GGUF (blk.0.attn_q.weight), not from a
 * random generator, so a layout error that happens to be self-consistent in our
 * own packer would still be caught here.
 */
extern "C" {
#include "../runtime/bitnet_i2s.h"
}
#include "../runtime/xdna_gemm.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

static uint32_t rng = 0xC0FFEEu;
static uint32_t xs32() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

int main(int argc, char **argv) {
    const std::string tensor = argc > 1 ? argv[1]
        : "artifacts/correctness/tensors/attn_q_l0.packed";
    const int64_t K = 2560, N = 2560, M = 512;

    std::printf("test_xdna_gemm  [M=%lld K=%lld N=%lld]  weights: %s\n",
                (long long)M, (long long)K, (long long)N, tensor.c_str());

    // --- real weights: I2_S -> signed ternary [N,K] -> transposed [K,N] ------
    const size_t packed = i2s_packed_bytes(K, N);
    std::vector<uint8_t> blob(packed + 4);
    {
        FILE *f = std::fopen(tensor.c_str(), "rb");
        if (!f) { std::printf("  cannot open %s\n", tensor.c_str()); return 1; }
        if (std::fread(blob.data(), 1, packed + 4, f) != packed + 4) {
            std::printf("  short read\n"); return 1;
        }
        std::fclose(f);
    }
    std::vector<uint8_t> codes((size_t)(K * N));
    i2s_unpack_matrix(blob.data(), K, N, codes.data());

    std::vector<int8_t> w_nk((size_t)(K * N));
    i2s_codes_to_signed(codes.data(), (size_t)(K * N), w_nk.data());

    // Kernel consumes B as [K,N] row-major; GGUF stores rows of K per output n.
    std::vector<int8_t> b_kn((size_t)(K * N));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k)
            b_kn[k * N + n] = w_nk[n * K + k];

    // --- deterministic int8 activations -------------------------------------
    std::vector<int8_t> a((size_t)(M * K));
    for (size_t i = 0; i < a.size(); ++i) a[i] = (int8_t)((int)(xs32() % 255) - 127);

    // --- CPU reference (the oracle) -----------------------------------------
    std::vector<int32_t> c_ref((size_t)(M * N));
    auto t0 = std::chrono::steady_clock::now();
    for (int64_t t = 0; t < M; ++t) {
        const int8_t *arow = a.data() + t * K;
        for (int64_t n = 0; n < N; ++n) {
            const int8_t *wrow = w_nk.data() + n * K;
            int32_t s = 0;
            for (int64_t k = 0; k < K; ++k) s += (int32_t)wrow[k] * (int32_t)arow[k];
            c_ref[t * N + n] = s;
        }
    }
    const double cpu_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
    std::printf("  cpu reference: %.1f ms (scalar, single-threaded oracle)\n", cpu_ms);

    // --- NPU -----------------------------------------------------------------
    std::vector<int32_t> c_npu((size_t)(M * N), 0);
    try {
        xdna::Program prog("artifacts/xclbin/mm_M512_K2560_N2560.xclbin",
                           "artifacts/xclbin/mm_M512_K2560_N2560.insts.bin", M, K, N);
        auto w = prog.upload(b_kn.data());
        xdna::Program::reset_counters();
        prog.run(*w, a.data(), c_npu.data());                       // correctness pass
        for (int i = 0; i < 20; ++i) prog.run(*w, a.data(), c_npu.data());  // timing
        std::printf("  npu: %llu dispatches, %.3f ms mean round trip\n",
                    (unsigned long long)xdna::Program::dispatch_count(),
                    xdna::Program::dispatch_ms() / xdna::Program::dispatch_count());
    } catch (const std::exception &e) {
        std::printf("  NPU ERROR: %s\n", e.what());
        return 1;
    }

    // --- bit-exact comparison ------------------------------------------------
    int64_t bad = 0; int32_t worst = 0; int64_t worst_at = -1;
    for (int64_t i = 0; i < M * N; ++i) {
        const int32_t d = c_npu[i] - c_ref[i];
        if (d) {
            ++bad;
            if (std::abs(d) > std::abs(worst)) { worst = d; worst_at = i; }
        }
    }
    if (bad) {
        std::printf("\n  FAIL: %lld / %lld accumulators differ (%.4f%%)\n",
                    (long long)bad, (long long)(M * N), 100.0 * bad / (M * N));
        std::printf("  worst delta %d at [%lld,%lld]: npu=%d ref=%d\n",
                    worst, (long long)(worst_at / N), (long long)(worst_at % N),
                    c_npu[worst_at], c_ref[worst_at]);
        return 1;
    }
    std::printf("\n  ok  all %lld int32 accumulators BIT-EXACT vs CPU reference\n",
                (long long)(M * N));
    return 0;
}
