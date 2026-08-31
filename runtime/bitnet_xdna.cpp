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
/* Single-flight XDNA invocation lease.
 *
 * accumulate() publishes process-global state -- g_direct (which output slots
 * hold this tensor's results), g_acc, g_cur_shape -- and then RETURNS. The
 * epilogue reads that state afterwards, on every ggml worker thread, holding no
 * lock. Within one graph execution the ggml barrier orders the two. Across two
 * independent inference contexts in one process it orders nothing: context B can
 * enter accumulate and overwrite g_direct and reuse the same output slots while
 * context A's workers are still reading them.
 *
 * The lease makes the whole accumulate -> barrier -> epilogue -> barrier
 * lifetime single-flight. It is taken by the thread that drives the NPU and
 * released only after every CPU reader of that invocation has finished, which
 * the caller signals by calling end() after its final barrier.
 *
 * The lease is the same mutex accumulate would otherwise take, so a leased
 * caller must not re-lock it; t_lease records that this thread already holds it.
 * CPU-only contexts never enter this path and are unaffected. */
thread_local bool t_lease = false;
/* One-time availability/config resolution has its OWN mutex, deliberately not
 * g_mu. Sharing it deadlocked: a worker thread still inside its first
 * bitnet_xdna_available() call blocks on g_mu once thread 0 has taken the
 * invocation lease, while thread 0 waits at the ggml barrier for exactly that
 * worker to arrive. Measured as a hard hang at pp2048. */
std::mutex g_init_mu;
/* Accumulator staging for the split accumulate/epilogue path. Grown on demand
 * and reused; the NPU is single-tenant so one buffer suffices. */
std::vector<int32_t> g_acc;
int64_t g_acc_N = 0;
std::map<const void *, std::unique_ptr<Resident>> g_resident;
std::atomic<uint64_t> g_repack_ns{0};
/* CPU-side cost of USING the NPU, as opposed to the NPU's own device time:
 * the int32->f32 epilogue, and staging activations into the mapped buffer.
 * Measured because at an all-NPU assignment these dominate the offloaded nodes
 * even though no GEMM work remains on the CPU. */
std::atomic<uint64_t> g_epilogue_ns{0};
/* Staging: activations INTO the mapped A buffer, and results OUT of the mapped
 * C buffer into g_acc. Both run single-threaded on thread 0 inside accumulate,
 * while every other thread is parked at the ggml barrier. */
std::atomic<uint64_t> g_stage_in_ns{0}, g_stage_out_ns{0}, g_stage_out_bytes{0};
std::atomic<uint64_t> g_epilogue_elems{0};

/* Per-shape dispatch accounting. The aggregate mean (2.66 ms) sits well above
 * the weighted mean of the three kernels measured standalone (1.44 ms), and an
 * aggregate cannot say which shape is responsible. Keyed by (K,N-chunk). */
/* Per-logical-shape output-path accounting. The aggregate stage_out counter
 * cannot answer where C-output time goes, and in particular it lives inside the
 * k_chunks == 1 branch, so it never measured ffn_down's deep-K partial
 * accumulation at all. Everything below is keyed on the LOGICAL tensor shape
 * (K, N) so each of the three real shapes is separable. */
struct ShapeStat {
    std::atomic<uint64_t> n{0}, ns{0};          // dispatches, total dispatch time
    std::atomic<uint64_t> wait_ns{0};           // blocked on the NPU fence
    std::atomic<uint64_t> sync_out_ns{0};       // XRT sync C from device
    std::atomic<uint64_t> submit_ns{0};
    std::atomic<uint64_t> stage_in_ns{0},   stage_in_bytes{0};
    std::atomic<uint64_t> stage_out_ns{0},  stage_out_bytes{0};   // k_chunks == 1
    std::atomic<uint64_t> partacc_ns{0},    partacc_bytes{0};     // k_chunks > 1 accumulate
    std::atomic<uint64_t> partcopy_ns{0},   partcopy_bytes{0};    // part -> g_acc
    std::atomic<uint64_t> epi_ns{0},        epi_elems{0};         // summed over threads
};
std::map<std::pair<int64_t,int64_t>, ShapeStat*> g_shape_stats;
/* Set by accumulate before its barrier; read by the epilogue after it. One
 * mul_mat node is exactly one accumulate followed by one epilogue with a
 * ggml_barrier between, so a single current-shape pointer is well-defined for
 * the epilogue's duration. */
std::atomic<ShapeStat *> g_cur_shape{nullptr};

ShapeStat *shape_stat(int64_t K, int64_t N) {
    auto key = std::make_pair(K, N);
    auto it = g_shape_stats.find(key);
    if (it == g_shape_stats.end()) it = g_shape_stats.emplace(key, new ShapeStat()).first;
    return it->second;
}
std::atomic<uint64_t> g_resident_bytes{0};

int  g_state = -1;        // -1 unknown, 0 unavailable, 1 available
std::atomic<int> g_state_fast{-1};   // lock-free mirror of g_state
/* Default to one full NPU tile. A batch smaller than the tile is zero-padded up
 * to it, so the NPU does a full tile's work for a fraction of the useful output
 * -- at ne11=512 against a 1024 tile that is 2x waste, and measurably worse than
 * leaving the batch on the CPU (728 vs 1241 t/s at pp512). Offload only when the
 * batch can fill a tile. */
int64_t g_min_tokens = 1024;
/* Fraction of the token batch given to the NPU. CPU and NPU are roughly
 * COMPARABLE on this arithmetic -- an earlier comment here claimed the NPU was
 * clearly faster on the strength of a device-time figure that was wrong by 2x.
 * The balance point is therefore near 0.5, and the real constraint is tile
 * granularity, not the ratio. Overridden by BITNET_XDNA_SPLIT. */
double g_split_frac = -1.0;   /* <0 = derive from the thread-aware cost model */
/* The NPU's throughput on BitNet's linears expressed in Zen 5 threads. Measured
 * by sweeping tile shares against thread counts (artifacts/next-pass/sweep.csv):
 * R = 10 is the value that reproduces the measured optimum in ALL SIX swept
 * (tiles-available, thread-count) cases -- including the awkward one where 15
 * threads at ub=1024 should decline the NPU entirely. Not a constant of nature:
 * it will move with kernel quality and with the CPU's thread scaling. */
double g_npu_threads = 10.0;
/* R depends on which output path is live, because R measures the cost of NPU-
 * assigned work and the g_acc path charged every NPU token tile an extra
 * single-threaded staging copy on thread 0. Removing that copy makes NPU tiles
 * cheaper and moves the balance point toward the device.
 *
 * Measured under direct output (artifacts/direct-output/cost_model_recal.csv,
 * pp2048/pp3072 x threads 4/6/8/10/12/15, exhaustive tile sweep vs auto):
 *
 *     R = 10   mean regret 1.026x, worst 1.147x   (picks 1 tile at 6 threads
 *                                                  where 2 is worth 733 vs 639)
 *     R = 25   mean regret 1.005x, worst 1.017x
 *
 * R in [21, 41] reproduces all three pp2048 optima; 25 sits mid-range. R = 10
 * remains correct for the g_acc path and is kept as the default there. */
constexpr double kR_GACC   = 10.0;
constexpr double kR_DIRECT = 25.0;
/* Experimental: pipeline N-chunk dispatches so the host-side evacuation of one
 * chunk overlaps the device executing the next. BITNET_XDNA_ASYNC=1. Off by
 * default -- the synchronous path is the proven one. */
bool g_async = false;
/* Direct mapped-output epilogue (BITNET_XDNA_DIRECT_OUT=1). Each dispatch writes
 * a persistent per-(token tile, N chunk) output slot, and the multi-threaded
 * epilogue reads that slot directly, so the mapped-C -> g_acc copy disappears.
 * Scope: k_chunks == 1 only (attn_q, attn_out, ffn_gate, ffn_up). ffn_down keeps
 * the deep-K accumulation path unchanged.
 *
 * ON BY DEFAULT since artifacts/direct-output/RESULTS.md: 1.007-1.152x
 * throughput, 0.737-0.790x energy per token, and a Pareto improvement on both
 * axes under co-tenancy, with bit-exact results. BITNET_XDNA_DIRECT_OUT=0
 * restores the g_acc path, which remains the reference and is still the live
 * path for ffn_down. */
bool g_direct_out = true;

/* Published by accumulate on thread 0 BEFORE the ggml barrier and read by every
 * thread AFTER it, so the barrier supplies the happens-before edge. Plain fields
 * are correct here for exactly that reason. */
struct DirectOutPlan {
    bool                  active   = false;
    int                   n_chunks = 0;
    int64_t               N        = 0;
    std::vector<int32_t*> slots;   // [token_tile * n_chunks + n_chunk]
};
DirectOutPlan g_direct;
long g_force_tiles = -1;   /* BITNET_XDNA_TILES: exact NPU tile count, -1 = auto */

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

/* Tensors that failed to become resident. Without this, a failure is retried on
 * every prefill forever -- rebuilding and discarding a multi-MB slab each time --
 * and never surfaces. */
std::map<const void *, std::string> g_failed;

Resident *get_resident(const void *src0, int64_t K, int64_t N) {
    if (g_failed.count(src0)) return nullptr;
    auto it = g_resident.find(src0);
    if (it != g_resident.end()) {
        /* Guard against pointer reuse: the cached plan and chunks belong to the
         * shape they were built for, but the caller supplies K/N independently. */
        if (it->second->K != K || it->second->N != N) return nullptr;
        return it->second.get();
    }

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

/* Wrapper so every failure path is recorded exactly once. */
Resident *get_resident_logged(const void *src0, int64_t K, int64_t N) {
    if (g_failed.count(src0)) return nullptr;
    try {
        Resident *r = get_resident(src0, K, N);
        if (!r) g_failed.emplace(src0, "plan/shape rejected");
        return r;
    } catch (const std::exception &e) {
        g_failed.emplace(src0, e.what());
        return nullptr;
    } catch (...) {
        g_failed.emplace(src0, "unknown exception");
        return nullptr;
    }
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
    {
        const double ep_ms = g_epilogue_ns.load() / 1e6;
        const double el    = (double)g_epilogue_elems.load();
        std::fprintf(stderr,
            "[bitnet-xdna] epilogue=%.1f thread-ms over %.0f Melem "
            "(summed across threads; divide by nth for wall)\n", ep_ms, el / 1e6);
        const double in_ms  = g_stage_in_ns.load()  / 1e6;
        const double out_ms = g_stage_out_ns.load() / 1e6;
        const double out_gb = g_stage_out_bytes.load() / 1e9;
        if (g_direct_out) {
            if (xdna::Program *pg = program_for(single_stem(), kKChunk, kNChunk))
                std::fprintf(stderr,
                    "[bitnet-xdna] direct-out arena: %d slots x %.1f MiB = %.1f MiB\n",
                    pg->out_slot_count(), pg->out_slot_bytes()/1048576.0,
                    pg->out_slot_count() * pg->out_slot_bytes()/1048576.0);
        }
        std::fprintf(stderr,
            "[bitnet-xdna] stage_in=%.1f ms  stage_out=%.1f ms over %.2f GB "
            "(%.1f GB/s, SINGLE-THREADED on thread 0 while others wait)\n",
            in_ms, out_ms, out_gb, out_ms > 0 ? out_gb / (out_ms * 1e-3) : 0.0);
    }
    double si, su, wa, so;
    xdna::Program::breakdown_ms(&si, &su, &wa, &so);
    const double tot = si + su + wa + so;
    if (tot > 0)
        std::fprintf(stderr,
            "[bitnet-xdna] sync_in %.0f ms (%.0f%%)  submit %.0f ms (%.0f%%)  "
            "wait %.0f ms (%.0f%%)  sync_out %.0f ms (%.0f%%)\n",
            si, 100*si/tot, su, 100*su/tot, wa, 100*wa/tot, so, 100*so/tot);
    if (!g_failed.empty()) {
        std::fprintf(stderr, "[bitnet-xdna] %zu tensor(s) NOT offloaded:\n", g_failed.size());
        for (auto &kv : g_failed)
            std::fprintf(stderr, "[bitnet-xdna]   %p : %s\n", kv.first, kv.second.c_str());
    }
    std::fprintf(stderr, "[bitnet-xdna] resident tensors: %zu\n", g_resident.size());
    /* Per-logical-shape output-path table. Machine-readable when
     * BITNET_XDNA_SHAPE_CSV=<path> is set, so tools can consume it directly. */
    {
        FILE *csv = nullptr;
        if (const char *cp = std::getenv("BITNET_XDNA_SHAPE_CSV")) {
            csv = std::fopen(cp, "w");
            if (csv) std::fprintf(csv, "K,N,dispatches,dispatch_ms,submit_ms,wait_ms,"
                                       "sync_out_ms,stage_in_ms,stage_in_mb,stage_out_ms,"
                                       "stage_out_mb,partacc_ms,partacc_mb,partcopy_ms,"
                                       "partcopy_mb,epi_thread_ms,epi_melem\n");
        }
        std::fprintf(stderr,
            "[bitnet-xdna] per-shape output path (ms; stage/part are thread-0 serial, "
            "epi is summed over threads)\n"
            "[bitnet-xdna]   %-13s %6s %8s %8s %9s %9s %10s %9s %10s %10s\n",
            "K x N", "n", "disp", "wait", "sync_out", "stage_in", "stage_out",
            "partacc", "partcopy", "epi(thr)");
        for (auto &kv : g_shape_stats) {
            ShapeStat *v = kv.second;
            const uint64_t cnt = v->n.load();
            if (!cnt) continue;
            char shape[32];
            std::snprintf(shape, sizeof shape, "%lldx%lld",
                          (long long)kv.first.first, (long long)kv.first.second);
            std::fprintf(stderr,
                "[bitnet-xdna]   %-13s %6llu %8.1f %8.1f %9.1f %9.1f %10.1f %9.1f %10.1f %10.1f\n",
                shape, (unsigned long long)cnt,
                v->ns.load()/1e6, v->wait_ns.load()/1e6, v->sync_out_ns.load()/1e6,
                v->stage_in_ns.load()/1e6, v->stage_out_ns.load()/1e6,
                v->partacc_ns.load()/1e6, v->partcopy_ns.load()/1e6, v->epi_ns.load()/1e6);
            if (csv)
                std::fprintf(csv, "%lld,%lld,%llu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
                                  "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                    (long long)kv.first.first, (long long)kv.first.second,
                    (unsigned long long)cnt, v->ns.load()/1e6, v->submit_ns.load()/1e6,
                    v->wait_ns.load()/1e6, v->sync_out_ns.load()/1e6,
                    v->stage_in_ns.load()/1e6, v->stage_in_bytes.load()/1048576.0,
                    v->stage_out_ns.load()/1e6, v->stage_out_bytes.load()/1048576.0,
                    v->partacc_ns.load()/1e6, v->partacc_bytes.load()/1048576.0,
                    v->partcopy_ns.load()/1e6, v->partcopy_bytes.load()/1048576.0,
                    v->epi_ns.load()/1e6, v->epi_elems.load()/1e6);
        }
        if (csv) std::fclose(csv);
    }
}

void bitnet_xdna_invocation_begin(void) {
    g_mu.lock();
    t_lease = true;
}

void bitnet_xdna_invocation_end(void) {
    t_lease = false;
    g_mu.unlock();
}

int bitnet_xdna_available(void) {
    // Hot path: every I2_S mul_mat on every thread calls this. Taking a mutex
    // here measurably slowed the CPU-only path (pp512 1277 -> 878 t/s), which
    // would have flattered the hybrid comparison. Resolve once, then answer
    // from a relaxed atomic.
    const int cached = g_state_fast.load(std::memory_order_acquire);
    if (cached >= 0) return cached;

    std::lock_guard<std::mutex> lock(g_init_mu);
    if (g_state >= 0) { g_state_fast.store(g_state, std::memory_order_release); return g_state; }
    g_state = 0;
    if (!env_truthy("BITNET_XDNA")) { g_state_fast.store(0, std::memory_order_release); return 0; }
    if (env_truthy("BITNET_XDNA_STATS")) std::atexit(print_stats_atexit);
    g_async = env_truthy("BITNET_XDNA_ASYNC");
    /* Default on; set BITNET_XDNA_DIRECT_OUT=0 to fall back to the g_acc path. */
    if (const char *dv = std::getenv("BITNET_XDNA_DIRECT_OUT"))
        g_direct_out = !(dv[0] == '0' && dv[1] == '\0');
    g_npu_threads = g_direct_out ? kR_DIRECT : kR_GACC;
    if (const char *ft = std::getenv("BITNET_XDNA_TILES")) {
        g_force_tiles = std::strtol(ft, nullptr, 10);
    }
    if (const char *nt = std::getenv("BITNET_XDNA_NPU_THREADS")) {
        const double v = std::strtod(nt, nullptr);
        if (v > 0.0) g_npu_threads = v;   // explicit override wins over both defaults
    }
    if (const char *sp = std::getenv("BITNET_XDNA_SPLIT")) {
        const double v = std::strtod(sp, nullptr);
        if (v >= 0.0 && v <= 1.0) g_split_frac = v;
    }
    if (const char *m = std::getenv("BITNET_XDNA_MIN_TOKENS")) {
        const long v = std::strtol(m, nullptr, 10);
        if (v > 0) g_min_tokens = v;
    }
    if (g_min_tokens < kMTile) g_min_tokens = kMTile;
    try { g_state = xdna::device_available() ? 1 : 0; }
    catch (...) { g_state = 0; }
    g_state_fast.store(g_state, std::memory_order_release);
    return g_state;
}

int bitnet_xdna_supports(int64_t K, int64_t N) {
    ShapePlan p; return plan_for(K, N, &p) ? 1 : 0;
}

int bitnet_xdna_worth_it(int64_t n_tokens) { return n_tokens >= g_min_tokens; }

int64_t bitnet_xdna_token_split(int64_t n_tokens) {
    if (n_tokens <= 0) return 0;
    /* Benchmarking override: give the NPU exactly this many whole tiles. The
     * fraction knob rounds, so several distinct fractions collapse onto the same
     * partition and a sweep over them measures the same configuration repeatedly.
     * -1 = use the fraction. */
    if (g_force_tiles >= 0) {
        int64_t t = g_force_tiles * kMTile;
        if (t > n_tokens) t = n_tokens;
        return t;
    }
    /* Below one tile there is nothing to divide: the NPU would pad either way,
     * so give it the whole batch rather than paying a dispatch for a fraction. */
    if (n_tokens <= kMTile) return n_tokens;

    const double frac = g_split_frac >= 0.0 ? g_split_frac : 0.5;
    if (frac >= 1.0) return n_tokens;
    if (frac <= 0.0) return 0;

    /* Round to the nearest whole tile so every dispatch is full. */
    const int64_t tiles = (int64_t)((double)n_tokens / (double)kMTile * frac + 0.5);
    int64_t t = tiles * kMTile;
    if (t < kMTile)   t = kMTile;        // always give the NPU at least one tile
    if (t > n_tokens) t = n_tokens;
    return t;
}

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
        /* Already held when the caller took an invocation lease. */
        std::unique_lock<std::mutex> lock(g_mu, std::defer_lock);
        if (!t_lease) lock.lock();
        Resident *res = get_resident_logged(src0_i2s, K, N);
        if (!res) return 0;
        const ShapePlan plan = res->plan;
        xdna::Program *prog = program_for(single_stem(), kKChunk, kNChunk);

        if ((int64_t)g_acc.size() < T * N) g_acc.resize((size_t)(T * N));
        g_acc_N = N;
        ShapeStat *st = shape_stat(K, N);
        g_cur_shape.store(st, std::memory_order_release);

        /* Direct mapped output applies only where one K chunk produces the final
         * int32 result. Deep-K (ffn_down) still needs host-side accumulation
         * across chunks and keeps the g_acc path untouched. */
        const int64_t n_tiles   = (T + kMTile - 1) / kMTile;
        const bool    use_direct = g_direct_out && plan.k_chunks == 1;
        if (use_direct) {
            prog->ensure_out_slots((int)(n_tiles * plan.n_chunks));
            g_direct.active   = true;
            g_direct.n_chunks = plan.n_chunks;
            g_direct.N        = N;
            g_direct.slots.assign((size_t)(n_tiles * plan.n_chunks), nullptr);
        } else {
            g_direct.active = false;
        }

        int8_t *a_bo = prog->a_map();
        const int32_t *c_bo = prog->c_map();

        // K-chunk partial sums, when K exceeds one chunk. Sized for the whole N
        // so the K loop can be OUTSIDE the N loop (see below).
        std::vector<int32_t> part;
        if (plan.k_chunks > 1) part.resize((size_t)(kMTile * N));

        for (int64_t t0 = 0; t0 < T; t0 += kMTile) {
            const int64_t rows = std::min<int64_t>(kMTile, T - t0);
            if (plan.k_chunks > 1) std::fill(part.begin(), part.end(), 0);

            // K OUTSIDE N. Every N-chunk of a given K-slice consumes the SAME
            // activations, so with N outer the identical slice was copied into
            // the mapped buffer and flushed once per N-chunk. For the two
            // 2560x6912 tensors that was 2 redundant copies each, ~608 MB of
            // memcpy+CLFLUSH per 2048-token prefill (36% of all A traffic).
            for (int kc = 0; kc < plan.k_chunks; ++kc) {
                const int64_t k_off  = (int64_t)kc * kKChunk;
                const int64_t k_keep = std::min(kKChunk, K - k_off);

                const auto si0 = std::chrono::steady_clock::now();
                if (rows < kMTile || k_keep < kKChunk)
                    std::memset(a_bo, 0, (size_t)(kMTile * kKChunk));
                for (int64_t r = 0; r < rows; ++r)
                    std::memcpy(a_bo + r * kKChunk,
                                a_q + (size_t)(t0 + r) * a_row_stride + k_off,
                                (size_t)k_keep);
                {
                    const uint64_t dt = (uint64_t)std::chrono::duration_cast<
                        std::chrono::nanoseconds>(std::chrono::steady_clock::now() - si0).count();
                    g_stage_in_ns.fetch_add(dt, std::memory_order_relaxed);
                    st->stage_in_ns.fetch_add(dt, std::memory_order_relaxed);
                    st->stage_in_bytes.fetch_add((uint64_t)(rows * k_keep), std::memory_order_relaxed);
                }
                prog->sync_a();                     // once per K slice, not per N chunk

                /* Evacuate one finished N-chunk from a mapped output slot.
                 * Identical arithmetic in both the synchronous and pipelined
                 * paths -- only WHEN it runs differs. */
                auto evacuate = [&](const int32_t *cbuf, int64_t n_off, int64_t n_keep) {
                    if (plan.k_chunks == 1) {
                        const auto so0 = std::chrono::steady_clock::now();
                        for (int64_t r = 0; r < rows; ++r)
                            std::memcpy(g_acc.data() + (t0 + r) * N + n_off,
                                        cbuf + r * kNChunk, (size_t)(n_keep * 4));
                        const uint64_t dt = (uint64_t)std::chrono::duration_cast<
                            std::chrono::nanoseconds>(std::chrono::steady_clock::now() - so0).count();
                        g_stage_out_ns.fetch_add(dt, std::memory_order_relaxed);
                        g_stage_out_bytes.fetch_add((uint64_t)(rows * n_keep * 4),
                            std::memory_order_relaxed);
                        st->stage_out_ns.fetch_add(dt, std::memory_order_relaxed);
                        st->stage_out_bytes.fetch_add((uint64_t)(rows * n_keep * 4),
                            std::memory_order_relaxed);
                    } else {
                        /* Deep-K path: int32 partial accumulation. Never counted
                         * by the aggregate stage_out counter, which is why
                         * ffn_down's output cost was previously unknown. */
                        const auto pa0 = std::chrono::steady_clock::now();
                        // Summing partials over K is exact in int32:
                        // |sum| <= K * 127 * 1 fits comfortably.
                        for (int64_t r = 0; r < rows; ++r) {
                            int32_t *pr = part.data() + r * N + n_off;
                            const int32_t *cr = cbuf + r * kNChunk;
                            for (int64_t j2 = 0; j2 < n_keep; ++j2) pr[j2] += cr[j2];
                        }
                        st->partacc_ns.fetch_add((uint64_t)std::chrono::duration_cast<
                            std::chrono::nanoseconds>(std::chrono::steady_clock::now() - pa0).count(),
                            std::memory_order_relaxed);
                        st->partacc_bytes.fetch_add((uint64_t)(rows * n_keep * 8),
                            std::memory_order_relaxed);   // read C + read-modify-write part
                    }
                };
                auto account = [&](std::chrono::steady_clock::time_point d0) {
                    const auto dns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - d0).count();
                    st->n.fetch_add(1, std::memory_order_relaxed);
                    st->ns.fetch_add((uint64_t)dns, std::memory_order_relaxed);
                    double sub, wai, so;
                    xdna::Program::last_breakdown_ms(&sub, &wai, &so);
                    st->submit_ns  .fetch_add((uint64_t)(sub * 1e6), std::memory_order_relaxed);
                    st->wait_ns    .fetch_add((uint64_t)(wai * 1e6), std::memory_order_relaxed);
                    st->sync_out_ns.fetch_add((uint64_t)(so  * 1e6), std::memory_order_relaxed);
                };
                const auto chunk_at = [&](int nc) -> const xdna::Weights & {
                    return *res->chunks[(size_t)nc * plan.k_chunks + kc];
                };

                if (use_direct) {
                    /* Every N chunk of this K slice gets its own persistent
                     * output slot, so nothing is overwritten and no evacuation
                     * is needed. The slot index carries BOTH dimensions --
                     * token tile and N chunk -- because a later token tile must
                     * not clobber an earlier tile's results before the epilogue
                     * has read them. */
                    const int64_t tile = t0 / kMTile;
                    for (int nc = 0; nc < plan.n_chunks; ++nc) {
                        const int slot = (int)(tile * plan.n_chunks + nc);
                        const auto d0 = std::chrono::steady_clock::now();
                        if (g_async) {
                            prog->submit_async_slot(chunk_at(nc), slot);
                            prog->wait_pending();
                        } else {
                            prog->run_presynced_slot(chunk_at(nc), slot);
                        }
                        account(d0);
                        g_direct.slots[(size_t)slot] = prog->out_slot_map(slot);
                    }
                } else if (g_async && plan.n_chunks > 1) {
                    /* Software pipeline over N-chunks. Every N-chunk of this
                     * K-slice reads the same activations, so the A buffer is
                     * stable and chunk nc+1 can be submitted before chunk nc's
                     * results are evacuated. The evacuation then overlaps device
                     * time instead of running while the NPU is idle. Two output
                     * slots alternate so the copy cannot race the device. */
                    auto d0 = std::chrono::steady_clock::now();
                    prog->submit_async(chunk_at(0), 0);
                    for (int nc = 0; nc < plan.n_chunks; ++nc) {
                        prog->wait_pending();
                        account(d0);
                        const int32_t *cbuf = prog->c_map_slot(nc & 1);
                        if (nc + 1 < plan.n_chunks) {
                            d0 = std::chrono::steady_clock::now();
                            prog->submit_async(chunk_at(nc + 1), (nc + 1) & 1);
                        }
                        const int64_t n_off  = (int64_t)nc * kNChunk;
                        evacuate(cbuf, n_off, std::min(kNChunk, N - n_off));
                    }
                } else {
                    for (int nc = 0; nc < plan.n_chunks; ++nc) {
                        const int64_t n_off  = (int64_t)nc * kNChunk;
                        const int64_t n_keep = std::min(kNChunk, N - n_off);
                        const auto d0 = std::chrono::steady_clock::now();
                        prog->run_mapped_presynced(chunk_at(nc));
                        account(d0);
                        evacuate(c_bo, n_off, n_keep);
                    }
                }
            }

            if (plan.k_chunks > 1) {
                const auto pc0 = std::chrono::steady_clock::now();
                for (int64_t r = 0; r < rows; ++r)
                    std::memcpy(g_acc.data() + (t0 + r) * N,
                                part.data() + r * N, (size_t)(N * 4));
                st->partcopy_ns.fetch_add((uint64_t)std::chrono::duration_cast<
                    std::chrono::nanoseconds>(std::chrono::steady_clock::now() - pc0).count(),
                    std::memory_order_relaxed);
                st->partcopy_bytes.fetch_add((uint64_t)(rows * N * 4), std::memory_order_relaxed);
            }
        }
        return 1;
    } catch (...) { return 0; }
}

void bitnet_xdna_epilogue(int64_t N, int64_t row_begin, int64_t row_end,
                          const float *act_scales, float ws,
                          float *dst, size_t dst_row_stride) {
    (void)g_acc_N;
    const auto ep_t0 = std::chrono::steady_clock::now();
    if (g_direct.active && g_direct.N == N) {
        /* Same arithmetic as below, reading the NPU's mapped output slot instead
         * of a host copy of it. Column j of token t lives in the slot for
         * (t / kMTile, j / kNChunk), at row (t % kMTile), column (j % kNChunk). */
        const int nch = g_direct.n_chunks;
        for (int64_t t = row_begin; t < row_end; ++t) {
            const float post = ws / act_scales[t];
            const int64_t tile = t / kMTile, row = t % kMTile;
            float *drow = (float *)((char *)dst + (size_t)t * dst_row_stride);
            for (int nc = 0; nc < nch; ++nc) {
                const int64_t n_off  = (int64_t)nc * kNChunk;
                const int64_t n_keep = std::min(kNChunk, N - n_off);
                const int32_t *acc = g_direct.slots[(size_t)(tile * nch + nc)]
                                     + row * kNChunk;
                float *d = drow + n_off;
                for (int64_t j = 0; j < n_keep; ++j) d[j] = (float)acc[j] * post;
            }
        }
    } else {
    for (int64_t t = row_begin; t < row_end; ++t) {
        const float post = ws / act_scales[t];
        const int32_t *acc = g_acc.data() + t * N;
        float *drow = (float *)((char *)dst + (size_t)t * dst_row_stride);
        for (int64_t j = 0; j < N; ++j) drow[j] = (float)acc[j] * post;
    }
    }
    const uint64_t ep_dt = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - ep_t0).count();
    const uint64_t ep_el = (uint64_t)((row_end - row_begin) * N);
    g_epilogue_ns.fetch_add(ep_dt, std::memory_order_relaxed);
    g_epilogue_elems.fetch_add(ep_el, std::memory_order_relaxed);
    if (ShapeStat *st = g_cur_shape.load(std::memory_order_acquire)) {
        st->epi_ns.fetch_add(ep_dt, std::memory_order_relaxed);
        st->epi_elems.fetch_add(ep_el, std::memory_order_relaxed);
    }
}

int64_t bitnet_xdna_token_split_nt(int64_t n_tokens, int n_threads) {
    /* Explicit overrides win, so benchmarking stays controllable. */
    if (g_force_tiles >= 0 || g_split_frac >= 0.0 || n_threads <= 1)
        return bitnet_xdna_token_split(n_tokens);
    if (n_tokens <= 0) return 0;
    if (n_tokens <= kMTile) return n_tokens;

    const double cpu_workers = (double)(n_threads - 1);   /* thread 0 drives the NPU */
    const double f = g_npu_threads / (g_npu_threads + cpu_workers);

    int64_t tiles = (int64_t)((double)n_tokens / (double)kMTile * f + 0.5);
    const int64_t max_tiles = n_tokens / kMTile;
    if (tiles < 0) tiles = 0;
    if (tiles > max_tiles) tiles = max_tiles;
    int64_t t = tiles * kMTile;

    /* Giving the NPU every token means no CPU worker has anything to do, which
     * throws away n_threads-1 cores. Only do that when the model actually says
     * the NPU outruns all of them. */
    if (t >= n_tokens && g_npu_threads < cpu_workers) t = (max_tiles - 1) * kMTile;
    if (t < 0) t = 0;
    return t;
}

uint64_t bitnet_xdna_dispatches(void)     { return xdna::Program::dispatch_count(); }
double   bitnet_xdna_dispatch_ms(void)    { return xdna::Program::dispatch_ms(); }
double   bitnet_xdna_repack_ms(void)      { return g_repack_ns.load() / 1e6; }
double   bitnet_xdna_epilogue_ms(void)    { return g_epilogue_ns.load() / 1e6; }
uint64_t bitnet_xdna_resident_bytes(void) { return g_resident_bytes.load(); }
void     bitnet_xdna_reset_counters(void) { xdna::Program::reset_counters(); }

} // extern "C"
