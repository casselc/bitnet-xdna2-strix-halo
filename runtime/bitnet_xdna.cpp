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
#include <utility>
#include <memory>
#include <mutex>
#include <algorithm>
#include <string>
#include <vector>

namespace {

/* The xclbins are compiled for a fixed token tile. A longer prefill is served by
 * repeated dispatches over 512-token tiles, zero-padded on the tail. Keeping one
 * tile size means one xclbin per (K,N) rather than one per (T,K,N). */
/* 1024, not 512: the tiling that reaches 13.2 TOPS uses m=128, and the design
 * requires M/m/n_aie_rows to be an even number of transfer-block rows, so
 * m=128 is illegal below M=1024. At M=512 the best legal tiling (m=64) manages
 * only 9.0 TOPS. Prompts shorter than 1024 are zero-padded. */
constexpr int64_t kMTile = 1024;

/* Every shape is decomposed onto a single (kKChunk x kNChunk) program.
 *
 * Three compiled shapes meant three xrt::hw_contexts, and each context holds all
 * 8 columns, so consecutive dispatches to different shapes force the firmware to
 * reprogram the array. Measured penalty: +53% to +210% per dispatch, which is
 * far more than the extra dispatches chunking costs. See
 * artifacts/kernels/context_switching.md.
 *
 * Chunks are padded rather than resized: padded weight columns are discarded and
 * padded K rows are zeroed in the activation buffer, so both contribute nothing. */
constexpr int64_t kKChunk = 2560;
constexpr int64_t kNChunk = 2560;

struct ShapePlan {
    int n_chunks;   // ceil(N / kNChunk)
    int k_chunks;   // ceil(K / kKChunk); >1 means partial sums must be summed
};

/* Shapes we have AOT artifacts for. ffn_gate/ffn_up have N=6912, which trips the
 * aie.dma_bd stride limit ([1:1048576]) at any column count, so they are served
 * as 2 x 3456. See artifacts/kernels/milestone_a.md. */
bool plan_for(int64_t K, int64_t N, ShapePlan *out) {
    /* BitNet-2B's offloadable linears: 2560x2560 (q, o), 2560x6912 (gate, up),
     * 6912x2560 (down). 6912 pads to 7680 = 3 chunks, an 11% padding cost that
     * buys the removal of every context switch. */
    if (K % 64 || N % 64) return false;
    if ((K != 2560 && K != 6912) || (N != 2560 && N != 6912)) return false;
    out->n_chunks = (int)((N + kNChunk - 1) / kNChunk);
    out->k_chunks = (int)((K + kKChunk - 1) / kKChunk);
    return true;
}

std::string single_stem();

std::string artifact_dir() {
    if (const char *e = std::getenv("BITNET_XDNA_ARTIFACTS")) return e;
    return "artifacts/xclbin-tuned";
}

std::string single_stem() {
    return artifact_dir() + "/mm_M" + std::to_string(kMTile) +
           "_K" + std::to_string(kKChunk) + "_N" + std::to_string(kNChunk);
}

/* One resident entry per weight tensor: the Gemm objects hold the uploaded
 * weights for this tensor's life. Keyed by the tensor's data pointer, which is
 * stable for a loaded model (mmap'd GGUF). */
struct Resident {
    /* Row-major over (n_chunk, k_chunk). */
    std::vector<std::unique_ptr<xdna::Weights>> chunks;
    int64_t K = 0, N = 0;
    ShapePlan plan{};
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

/* Per-shape dispatch accounting. The aggregate mean (2.66 ms) sits well above
 * the weighted mean of the three kernels measured standalone (1.44 ms), and an
 * aggregate cannot say which shape is responsible. Keyed by (K,N-chunk). */
struct ShapeStat { std::atomic<uint64_t> n{0}, ns{0}; };
std::map<std::pair<int64_t,int64_t>, ShapeStat*> g_shape_stats;
std::atomic<uint64_t> g_resident_bytes{0};

int  g_state = -1;        // -1 unknown, 0 unavailable, 1 available
std::atomic<int> g_state_fast{-1};   // lock-free mirror of g_state
int64_t g_min_tokens = 64;

bool env_truthy(const char *name) {
    const char *v = std::getenv(name);
    return v && *v && std::strcmp(v, "0") != 0;
}

/* Build the int8 weight slab the kernel consumes.
 *
 * The kernels are compiled with --b-col-maj 1, so B is column-major: logical
 * element (k, n) lives at offset n*K + k. That is [N,K] row-major -- exactly
 * how the GGUF already stores I2_S (K contiguous per output feature). So this
 * is a straight per-row unpack with NO transpose, which is both simpler and
 * far kinder to the cache than the strided writes the row-major layout needed.
 *
 * We also convert the unsigned {0,1,2} codes to signed {-1,0,+1} here, which is
 * what removes the need for the act_sums correction downstream. */
void build_b_kn(const void *src0, int64_t K, int64_t N,
                int64_t n_begin, int64_t k_begin, int8_t *out) {
    std::vector<uint8_t> codes((size_t)K);
    const uint8_t *base = static_cast<const uint8_t *>(src0);
    const size_t row_bytes = (size_t)(K / 4);
    std::memset(out, 0, (size_t)(kNChunk * kKChunk));   // pads stay zero
    for (int64_t j = 0; j < kNChunk; ++j) {
        const int64_t n = n_begin + j;
        if (n >= N) break;
        i2s_unpack_row(base + (size_t)n * row_bytes, K, codes.data());
        int8_t *dst = out + j * kKChunk;               // column-major B == [N,K]
        const int64_t kmax = std::min(kKChunk, K - k_begin);
        for (int64_t i = 0; i < kmax; ++i)
            dst[i] = (int8_t)codes[k_begin + i] - 1;
    }
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

    res->plan = plan;
    xdna::Program *prog = program_for(single_stem(), kKChunk, kNChunk);
    std::vector<int8_t> slab((size_t)(kKChunk * kNChunk));
    for (int nc = 0; nc < plan.n_chunks; ++nc)
        for (int kc = 0; kc < plan.k_chunks; ++kc) {
            build_b_kn(src0, K, N, (int64_t)nc * kNChunk, (int64_t)kc * kKChunk, slab.data());
            res->chunks.push_back(prog->upload(slab.data()));
            g_resident_bytes.fetch_add((uint64_t)(kKChunk * kNChunk), std::memory_order_relaxed);
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
    double si, su, wa, so;
    xdna::Program::breakdown_ms(&si, &su, &wa, &so);
    const double tot = si + su + wa + so;
    if (tot > 0)
        std::fprintf(stderr,
            "[bitnet-xdna] sync_in %.0f ms (%.0f%%)  submit %.0f ms (%.0f%%)  "
            "wait %.0f ms (%.0f%%)  sync_out %.0f ms (%.0f%%)\n",
            si, 100*si/tot, su, 100*su/tot, wa, 100*wa/tot, so, 100*so/tot);
    for (auto &kv : g_shape_stats) {
        const uint64_t cnt = kv.second->n.load();
        if (!cnt) continue;
        const double ms = kv.second->ns.load() / 1e6 / (double)cnt;
        const double tops = 2.0 * (double)kMTile * (double)kv.first.first * (double)kv.first.second
                            / (ms * 1e-3) / 1e12;
        std::fprintf(stderr, "[bitnet-xdna]   K=%-5lld N=%-5lld  n=%-6llu  %6.3f ms  %5.2f TOPS\n",
                     (long long)kv.first.first, (long long)kv.first.second,
                     (unsigned long long)cnt, ms, tops);
    }
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
    /* Single-threaded convenience form: accumulate, then run the whole epilogue
     * here. ggml uses the split form instead so the epilogue can be spread
     * across the threadpool. */
    (void)act_sums;   // signed weights -> accumulator is already the true dot product
    if (!bitnet_xdna_accumulate(src0_i2s, K, N, a_q, T, a_row_stride)) return 0;
    bitnet_xdna_epilogue(N, 0, T, act_scales, ws, dst, dst_row_stride);
    return 1;
}

int bitnet_xdna_accumulate(const void *src0_i2s, int64_t K, int64_t N,
                           const int8_t *a_q, int64_t T, size_t a_row_stride) {
    try {
        std::lock_guard<std::mutex> lock(g_mu);
        Resident *res = get_resident(src0_i2s, K, N);
        if (!res) return 0;
        const ShapePlan plan = res->plan;
        xdna::Program *prog = program_for(single_stem(), kKChunk, kNChunk);

        if ((int64_t)g_acc.size() < T * N) g_acc.resize((size_t)(T * N));
        g_acc_N = N;

        int8_t *a_bo = prog->a_map();
        const int32_t *c_bo = prog->c_map();

        // Only needed when K is chunked: partial sums accumulate here before
        // being written out once.
        std::vector<int32_t> part;
        if (plan.k_chunks > 1) part.resize((size_t)(kMTile * kNChunk));

        for (int64_t t0 = 0; t0 < T; t0 += kMTile) {
            const int64_t rows = std::min<int64_t>(kMTile, T - t0);

            for (int nc = 0; nc < plan.n_chunks; ++nc) {
                const int64_t n_off  = (int64_t)nc * kNChunk;
                const int64_t n_keep = std::min(kNChunk, N - n_off);   // ignore padding
                if (plan.k_chunks > 1) std::fill(part.begin(), part.end(), 0);

                for (int kc = 0; kc < plan.k_chunks; ++kc) {
                    const int64_t k_off  = (int64_t)kc * kKChunk;
                    const int64_t k_keep = std::min(kKChunk, K - k_off);

                    // Activations for this K slice, zero-padded in both
                    // directions so the fixed-size kernel sees defined input.
                    if (rows < kMTile || k_keep < kKChunk)
                        std::memset(a_bo, 0, (size_t)(kMTile * kKChunk));
                    for (int64_t r = 0; r < rows; ++r)
                        std::memcpy(a_bo + r * kKChunk,
                                    a_q + (size_t)(t0 + r) * a_row_stride + k_off,
                                    (size_t)k_keep);

                    prog->run_mapped(*res->chunks[(size_t)nc * plan.k_chunks + kc]);

                    if (plan.k_chunks == 1) {
                        for (int64_t r = 0; r < rows; ++r)
                            std::memcpy(g_acc.data() + (t0 + r) * N + n_off,
                                        c_bo + r * kNChunk, (size_t)(n_keep * 4));
                    } else {
                        // Summing partial products over K is exact in int32:
                        // |sum| <= K * 127 * 1 fits comfortably.
                        for (int64_t r = 0; r < rows; ++r) {
                            int32_t *pr = part.data() + r * kNChunk;
                            const int32_t *cr = c_bo + r * kNChunk;
                            for (int64_t j2 = 0; j2 < n_keep; ++j2) pr[j2] += cr[j2];
                        }
                    }
                }

                if (plan.k_chunks > 1)
                    for (int64_t r = 0; r < rows; ++r)
                        std::memcpy(g_acc.data() + (t0 + r) * N + n_off,
                                    part.data() + r * kNChunk, (size_t)(n_keep * 4));
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
