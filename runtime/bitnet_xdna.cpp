#include "bitnet_xdna.h"
#include "xdna_gemm.h"
extern "C" {
#include "bitnet_i2s.h"
}

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

/* The xclbins are compiled for a fixed token tile. A longer prefill is served by
 * repeated dispatches over 512-token tiles, zero-padded on the tail. Keeping one
 * tile size means one xclbin per (K,N) rather than one per (T,K,N). */
constexpr int64_t kMTile = 512;

struct ShapePlan {
    int64_t n_chunk;   // N per xclbin (N itself, or a split for N > 4096)
    int     n_chunks;
};

/* Shapes we have AOT artifacts for. ffn_gate/ffn_up have N=6912, which trips the
 * aie.dma_bd stride limit ([1:1048576]) at any column count, so they are served
 * as 2 x 3456. See artifacts/kernels/milestone_a.md. */
bool plan_for(int64_t K, int64_t N, ShapePlan *out) {
    if (K == 2560 && N == 2560) { *out = {2560, 1}; return true; }
    if (K == 6912 && N == 2560) { *out = {2560, 1}; return true; }
    if (K == 2560 && N == 6912) { *out = {3456, 2}; return true; }
    return false;
}

std::string artifact_dir() {
    if (const char *e = std::getenv("BITNET_XDNA_ARTIFACTS")) return e;
    return "artifacts/xclbin";
}

/* One resident entry per weight tensor: the Gemm objects hold the uploaded
 * weights for this tensor's life. Keyed by the tensor's data pointer, which is
 * stable for a loaded model (mmap'd GGUF). */
struct Resident {
    std::vector<std::unique_ptr<xdna::Weights>> chunks;  // one per N-chunk
    int64_t K = 0, N = 0;
};

/* Shared per-shape programs. The driver allows only 16 hardware contexts and
 * BitNet-2B has ~150 offloadable tensors, so a context per tensor exhausts them
 * and every later tensor silently falls back to CPU. Keyed by xclbin stem. */
std::map<std::string, std::unique_ptr<xdna::Program>> g_programs;

xdna::Program *program_for(const std::string &stem, int64_t K, int64_t n_chunk) {
    auto it = g_programs.find(stem);
    if (it != g_programs.end()) return it->second.get();
    auto prog = std::make_unique<xdna::Program>(stem + ".xclbin", stem + ".insts.bin",
                                                kMTile, K, n_chunk);
    auto *raw = prog.get();
    g_programs.emplace(stem, std::move(prog));
    return raw;
}

std::mutex g_mu;
/* Accumulator staging for the split accumulate/epilogue path. Grown on demand
 * and reused; the NPU is single-tenant so one buffer suffices. */
std::vector<int32_t> g_acc;
int64_t g_acc_N = 0;
std::map<const void *, std::unique_ptr<Resident>> g_resident;
std::atomic<uint64_t> g_repack_ns{0};
std::atomic<uint64_t> g_resident_bytes{0};

int  g_state = -1;        // -1 unknown, 0 unavailable, 1 available
std::atomic<int> g_state_fast{-1};   // lock-free mirror of g_state
int64_t g_min_tokens = 64;

bool env_truthy(const char *name) {
    const char *v = std::getenv(name);
    return v && *v && std::strcmp(v, "0") != 0;
}

/* Build the [K, n_chunk] int8 weight slab the kernel consumes.
 * GGUF stores I2_S as [N,K] (K contiguous per output feature); the kernel wants
 * B as [K,N] row-major. We also convert the unsigned {0,1,2} codes to signed
 * {-1,0,+1} here, which is what removes the need for the act_sums correction. */
void build_b_kn(const void *src0, int64_t K, int64_t N,
                int64_t n_begin, int64_t n_chunk, int8_t *out) {
    std::vector<uint8_t> codes((size_t)K);
    const uint8_t *base = static_cast<const uint8_t *>(src0);
    const size_t row_bytes = (size_t)(K / 4);
    for (int64_t j = 0; j < n_chunk; ++j) {
        const int64_t n = n_begin + j;
        i2s_unpack_row(base + (size_t)n * row_bytes, K, codes.data());
        for (int64_t k = 0; k < K; ++k)
            out[k * n_chunk + j] = (int8_t)codes[k] - 1;
    }
    (void)N;
}

Resident *get_resident(const void *src0, int64_t K, int64_t N) {
    auto it = g_resident.find(src0);
    if (it != g_resident.end()) return it->second.get();

    ShapePlan plan;
    if (!plan_for(K, N, &plan)) return nullptr;

    const auto t0 = std::chrono::steady_clock::now();
    auto res = std::make_unique<Resident>();
    res->K = K; res->N = N;
    const std::string dir = artifact_dir();

    const std::string stem = dir + "/mm_M" + std::to_string(kMTile) +
                             "_K" + std::to_string(K) +
                             "_N" + std::to_string(plan.n_chunk);
    xdna::Program *prog = program_for(stem, K, plan.n_chunk);
    std::vector<int8_t> slab((size_t)(K * plan.n_chunk));
    for (int c = 0; c < plan.n_chunks; ++c) {
        build_b_kn(src0, K, N, (int64_t)c * plan.n_chunk, plan.n_chunk, slab.data());
        res->chunks.push_back(prog->upload(slab.data()));
        g_resident_bytes.fetch_add((uint64_t)(K * plan.n_chunk), std::memory_order_relaxed);
    }
    g_repack_ns.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - t0).count(),
                          std::memory_order_relaxed);

    auto *raw = res.get();
    g_resident.emplace(src0, std::move(res));
    return raw;
}

} // namespace

extern "C" {

static void print_stats_atexit(void) {
    const uint64_t n = xdna::Program::dispatch_count();
    std::fprintf(stderr,
        "\n[bitnet-xdna] dispatches=%llu  dispatch_total=%.1f ms  mean=%.3f ms\n"
        "[bitnet-xdna] weight repack+upload=%.1f ms  resident int8 weights=%.1f MiB\n",
        (unsigned long long)n, xdna::Program::dispatch_ms(),
        n ? xdna::Program::dispatch_ms() / (double)n : 0.0,
        g_repack_ns.load() / 1e6,
        g_resident_bytes.load() / 1048576.0);
}

int bitnet_xdna_available(void) {
    // Hot path: every I2_S mul_mat on every thread calls this. Taking a mutex
    // here measurably slowed the CPU-only path (pp512 1277 -> 878 t/s), which
    // would have flattered the hybrid comparison. Resolve once, then answer
    // from a relaxed atomic.
    const int cached = g_state_fast.load(std::memory_order_acquire);
    if (cached >= 0) return cached;

    std::lock_guard<std::mutex> lock(g_mu);
    if (g_state >= 0) { g_state_fast.store(g_state, std::memory_order_release); return g_state; }
    g_state = 0;
    if (!env_truthy("BITNET_XDNA")) { g_state_fast.store(0, std::memory_order_release); return 0; }
    if (env_truthy("BITNET_XDNA_STATS")) std::atexit(print_stats_atexit);
    if (const char *m = std::getenv("BITNET_XDNA_MIN_TOKENS")) {
        const long v = std::strtol(m, nullptr, 10);
        if (v > 0) g_min_tokens = v;
    }
    try { g_state = xdna::device_available() ? 1 : 0; }
    catch (...) { g_state = 0; }
    g_state_fast.store(g_state, std::memory_order_release);
    return g_state;
}

int bitnet_xdna_supports(int64_t K, int64_t N) {
    ShapePlan p; return plan_for(K, N, &p) ? 1 : 0;
}

int bitnet_xdna_worth_it(int64_t n_tokens) { return n_tokens >= g_min_tokens; }

int bitnet_xdna_mul_mat(const void *src0_i2s, int64_t K, int64_t N,
                        const int8_t *a_q, int64_t T, size_t a_row_stride,
                        const float *act_scales, const int32_t *act_sums,
                        float ws,
                        float *dst, size_t dst_row_stride) {
    (void)act_sums;  // signed weights -> accumulator is already the true dot product
    try {
        std::lock_guard<std::mutex> lock(g_mu);
        Resident *res = get_resident(src0_i2s, K, N);
        if (!res) return 0;

        ShapePlan plan; plan_for(K, N, &plan);
        const int64_t n_chunk = plan.n_chunk;
        const std::string stem = artifact_dir() + "/mm_M" + std::to_string(kMTile) +
                                 "_K" + std::to_string(K) +
                                 "_N" + std::to_string(n_chunk);
        xdna::Program *prog = program_for(stem, K, n_chunk);
        int8_t  *a_bo = prog->a_map();
        const int32_t *c_bo = prog->c_map();

        for (int64_t t0 = 0; t0 < T; t0 += kMTile) {
            const int64_t rows = (t0 + kMTile <= T) ? kMTile : (T - t0);
            // Write activations straight into the mapped BO. Zero-pad the tail
            // so the fixed-size kernel always sees defined input; the padded
            // rows are computed and discarded.
            if (rows < kMTile)
                std::memset(a_bo + rows * K, 0, (size_t)((kMTile - rows) * K));
            for (int64_t r = 0; r < rows; ++r)
                std::memcpy(a_bo + r * K,
                            a_q + (size_t)(t0 + r) * a_row_stride, (size_t)K);

            for (size_t c = 0; c < res->chunks.size(); ++c) {
                prog->run_mapped(*res->chunks[c]);
                // Epilogue reads the accumulator directly out of the mapped BO.
                const int64_t n_off = (int64_t)c * n_chunk;
                for (int64_t r = 0; r < rows; ++r) {
                    const float post = ws / act_scales[t0 + r];
                    float *drow = (float *)((char *)dst + (size_t)(t0 + r) * dst_row_stride) + n_off;
                    const int32_t *crow = c_bo + r * n_chunk;
                    for (int64_t j = 0; j < n_chunk; ++j) drow[j] = (float)crow[j] * post;
                }
            }
        }
        return 1;
    } catch (...) {
        return 0;   // any failure -> caller falls back to the CPU kernel
    }
}

int bitnet_xdna_accumulate(const void *src0_i2s, int64_t K, int64_t N,
                           const int8_t *a_q, int64_t T, size_t a_row_stride) {
    try {
        std::lock_guard<std::mutex> lock(g_mu);
        Resident *res = get_resident(src0_i2s, K, N);
        if (!res) return 0;

        ShapePlan plan; plan_for(K, N, &plan);
        const int64_t n_chunk = plan.n_chunk;
        const std::string stem = artifact_dir() + "/mm_M" + std::to_string(kMTile) +
                                 "_K" + std::to_string(K) +
                                 "_N" + std::to_string(n_chunk);
        xdna::Program *prog = program_for(stem, K, n_chunk);

        if ((int64_t)g_acc.size() < T * N) g_acc.resize((size_t)(T * N));
        g_acc_N = N;

        int8_t *a_bo = prog->a_map();
        const int32_t *c_bo = prog->c_map();

        for (int64_t t0 = 0; t0 < T; t0 += kMTile) {
            const int64_t rows = (t0 + kMTile <= T) ? kMTile : (T - t0);
            if (rows < kMTile)
                std::memset(a_bo + rows * K, 0, (size_t)((kMTile - rows) * K));
            for (int64_t r = 0; r < rows; ++r)
                std::memcpy(a_bo + r * K,
                            a_q + (size_t)(t0 + r) * a_row_stride, (size_t)K);

            for (size_t c = 0; c < res->chunks.size(); ++c) {
                prog->run_mapped(*res->chunks[c]);
                const int64_t n_off = (int64_t)c * n_chunk;
                for (int64_t r = 0; r < rows; ++r)
                    std::memcpy(g_acc.data() + (t0 + r) * N + n_off,
                                c_bo + r * n_chunk, (size_t)(n_chunk * 4));
            }
        }
        return 1;
    } catch (...) { return 0; }
}

void bitnet_xdna_epilogue(int64_t N, int64_t row_begin, int64_t row_end,
                          const float *act_scales, float ws,
                          float *dst, size_t dst_row_stride) {
    (void)g_acc_N;
    for (int64_t t = row_begin; t < row_end; ++t) {
        const float post = ws / act_scales[t];
        const int32_t *acc = g_acc.data() + t * N;
        float *drow = (float *)((char *)dst + (size_t)t * dst_row_stride);
        for (int64_t j = 0; j < N; ++j) drow[j] = (float)acc[j] * post;
    }
}

uint64_t bitnet_xdna_dispatches(void)     { return xdna::Program::dispatch_count(); }
double   bitnet_xdna_dispatch_ms(void)    { return xdna::Program::dispatch_ms(); }
double   bitnet_xdna_repack_ms(void)      { return g_repack_ns.load() / 1e6; }
uint64_t bitnet_xdna_resident_bytes(void) { return g_resident_bytes.load(); }
void     bitnet_xdna_reset_counters(void) { xdna::Program::reset_counters(); }

} // extern "C"
