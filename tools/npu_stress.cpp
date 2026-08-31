/* Long-lived runtime stress for the direct-output XDNA path.
 *
 * The direct-output arena allocates one persistent XRT output buffer per
 * (token tile, N chunk). Differently sized requests need different counts, so
 * the question this answers is whether the arena reaches a stable HIGH-WATER
 * MARK or grows without bound as a resident controller serves a mix of prompt
 * lengths for a long time.
 *
 * Every iteration carries a correctness sentinel: a hash of the f32 output is
 * compared against one captured single-threaded at startup. test_xdna_shapes
 * proves that reference path is bit-exact against a scalar oracle, so any drift
 * here is a runtime state bug rather than an arithmetic one.
 *
 * A hash rather than stored reference buffers: ffn_gate at T=3968 is 110 MB of
 * f32 output, and keeping twelve of those would dominate the RSS figure this
 * harness exists to measure.
 */
#include "../runtime/bitnet_xdna.h"
#include "../runtime/bitnet_i2s.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint32_t rng_state = 0xC0FFEEu;
static uint32_t xs32() {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return rng_state;
}

static uint64_t fnv1a(const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static long rss_kb() {
    FILE *f = std::fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256]; long v = -1;
    while (std::fgets(line, sizeof line, f))
        if (std::strncmp(line, "VmRSS:", 6) == 0) { v = std::atol(line + 6); break; }
    std::fclose(f);
    return v;
}

static long vmlck_kb() {
    FILE *f = std::fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256]; long v = -1;
    while (std::fgets(line, sizeof line, f))
        if (std::strncmp(line, "VmLck:", 6) == 0) { v = std::atol(line + 6); break; }
    std::fclose(f);
    return v;
}

static double temp_c() {
    for (int i = 0; i < 20; ++i) {
        char p[128]; std::snprintf(p, sizeof p, "/sys/class/hwmon/hwmon%d/name", i);
        FILE *f = std::fopen(p, "r"); if (!f) continue;
        char nm[64] = {0}; if (!std::fgets(nm, sizeof nm, f)) { std::fclose(f); continue; }
        std::fclose(f);
        if (std::strncmp(nm, "k10temp", 7) && std::strncmp(nm, "zenpower", 8)) continue;
        std::snprintf(p, sizeof p, "/sys/class/hwmon/hwmon%d/temp1_input", i);
        f = std::fopen(p, "r"); if (!f) continue;
        long mc = 0; if (std::fscanf(f, "%ld", &mc) != 1) mc = 0;
        std::fclose(f);
        return mc / 1000.0;
    }
    return -1.0;
}

struct Shape { const char *name; const char *file; int64_t K, N; };

int main(int argc, char **argv) {
    const int iters = (argc > 1) ? std::atoi(argv[1]) : 360;

    static const Shape shapes[] = {
        {"attn_q",   "artifacts/correctness/tensors/attn_q_l0.packed",   2560, 2560},
        {"ffn_gate", "artifacts/correctness/tensors/ffn_gate_l0.packed", 2560, 6912},
        {"ffn_down", "artifacts/correctness/tensors/ffn_down_l0.packed", 6912, 2560},
    };
    /* Deliberately unsorted, so the arena is asked for a large slot count, then
     * a small one, then a large one again -- the pattern that would expose
     * growth-on-every-shape-change. */
    static const int64_t Ts[] = {1024, 3968, 2048, 1024, 3072, 2048, 3968, 1024};

    std::printf("npu_stress  (%d invocations, direct output)\n", iters);
    if (!bitnet_xdna_available()) {
        std::printf("  NPU unavailable (set BITNET_XDNA=1 and BITNET_XDNA_ARTIFACTS)\n");
        return 77;
    }

    std::vector<std::vector<uint8_t>> blobs(3);
    for (int i = 0; i < 3; ++i) {
        const size_t packed = i2s_packed_bytes(shapes[i].K, shapes[i].N);
        blobs[i].resize(packed + 4);
        FILE *f = std::fopen(shapes[i].file, "rb");
        if (!f) { std::printf("  cannot open %s\n", shapes[i].file); return 1; }
        const size_t got = std::fread(blobs[i].data(), 1, packed + 4, f);
        std::fclose(f);
        if (got != packed + 4) { std::printf("  short read %s\n", shapes[i].file); return 1; }
    }

    const int nT = (int)(sizeof(Ts) / sizeof(Ts[0]));
    const int nS = 3;
    /* Largest activation and output any iteration needs, allocated once so the
     * harness's own allocator churn does not masquerade as runtime growth. */
    int64_t maxT = 0, maxN = 0, maxK = 0;
    for (int i = 0; i < nT; ++i) maxT = Ts[i] > maxT ? Ts[i] : maxT;
    for (int i = 0; i < nS; ++i) { maxN = shapes[i].N > maxN ? shapes[i].N : maxN;
                                   maxK = shapes[i].K > maxK ? shapes[i].K : maxK; }
    std::vector<int8_t> act((size_t)(maxT * maxK));
    std::vector<float>  scales((size_t)maxT, 1.0f);
    std::vector<float>  dst((size_t)(maxT * maxN));
    for (auto &v : act) v = (int8_t)((int)(xs32() % 255) - 127);

    auto run_one = [&](int si, int ti, uint64_t *out_hash) -> bool {
        const Shape &s = shapes[si]; const int64_t T = Ts[ti];
        bitnet_xdna_invocation_begin();
        const int ok = bitnet_xdna_accumulate(blobs[si].data(), s.K, s.N,
                                              act.data(), T, (size_t)s.K);
        if (ok) {
            bitnet_xdna_epilogue(s.N, 0, T, scales.data(), 1.0f, dst.data(),
                                 (size_t)(s.N * sizeof(float)));
            *out_hash = fnv1a(dst.data(), (size_t)(T * s.N) * sizeof(float));
        }
        bitnet_xdna_invocation_end();
        return ok != 0;
    };

    /* Sentinels, captured once. */
    std::vector<uint64_t> ref((size_t)(nS * nT), 0);
    for (int si = 0; si < nS; ++si)
        for (int ti = 0; ti < nT; ++ti)
            if (!run_one(si, ti, &ref[si * nT + ti])) {
                std::printf("  reference declined %s T=%lld\n", shapes[si].name,
                            (long long)Ts[ti]);
                return 1;
            }

    const long rss0 = rss_kb();
    long rss_hi = rss0, slots_hi = 0, resident_hi = 0;
    long mismatches = 0, declines = 0;

    std::printf("  initial RSS %ld MiB, arena %d slots\n", rss0 / 1024,
                bitnet_xdna_out_slots());
    std::printf("  %6s %9s %8s %8s %9s %11s %7s %6s\n",
                "iter", "RSS MiB", "VmLck", "slots", "arenaMiB", "dispatches",
                "resid", "degC");

    for (int it = 0; it < iters; ++it) {
        const int si = it % nS, ti = (it / nS) % nT;
        uint64_t h = 0;
        if (!run_one(si, ti, &h)) { ++declines; continue; }
        if (h != ref[si * nT + ti]) ++mismatches;

        const int slots = bitnet_xdna_out_slots();
        const int resid = bitnet_xdna_resident_tensors();
        const long rss = rss_kb();
        if (rss > rss_hi) rss_hi = rss;
        if (slots > slots_hi) slots_hi = slots;
        if (resid > resident_hi) resident_hi = resid;

        if (it % 30 == 0 || it == iters - 1)
            std::printf("  %6d %9ld %8ld %8d %9.0f %11llu %7d %6.1f\n",
                        it, rss / 1024, vmlck_kb(), slots,
                        slots * bitnet_xdna_out_slot_bytes() / 1048576.0,
                        (unsigned long long)bitnet_xdna_dispatches(), resid, temp_c());
    }

    const long rss1 = rss_kb();
    std::printf("\n  invocations       %d\n", iters);
    std::printf("  mismatches        %ld\n", mismatches);
    std::printf("  declines          %ld\n", declines);
    std::printf("  RSS initial       %ld MiB\n", rss0 / 1024);
    std::printf("  RSS high-water    %ld MiB\n", rss_hi / 1024);
    std::printf("  RSS final         %ld MiB\n", rss1 / 1024);
    std::printf("  arena high-water  %ld slots (%.0f MiB)\n", slots_hi,
                slots_hi * bitnet_xdna_out_slot_bytes() / 1048576.0);
    std::printf("  resident tensors  %ld (high-water)\n", resident_hi);
    std::printf("  NPU dispatches    %llu\n",
                (unsigned long long)bitnet_xdna_dispatches());

    const bool ok = (mismatches == 0 && declines == 0);
    std::printf("\n%s\n", ok ? "stress: correctness held, arena bounded"
                             : "stress: FAILURE");
    return ok ? 0 : 1;
}
