/* npu_switch_cost -- decompose the in-model per-dispatch cost.
 *
 * Milestone A measured 0.82-1.95 ms of pure kernel time per shape, and the
 * weighted mean across BitNet's 7 linears is 1.46 ms. The integrated runtime
 * measures 2.66 ms. This locates the missing ~1.2 ms by timing the same
 * dispatches (a) one program at a time and (b) cycling the three programs in
 * the order a BitNet layer actually uses them. */
#include "../runtime/xdna_gemm.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <cstdlib>

using clk = std::chrono::steady_clock;

struct Prog {
    std::string name;
    std::unique_ptr<xdna::Program> p;
    std::unique_ptr<xdna::Weights> w;
    std::vector<std::unique_ptr<xdna::Weights>> extra;  // residency ballast
};

static double p50(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char **argv) {
    const std::string dir = argc > 1 ? argv[1] : "artifacts/xclbin";
    struct Spec { const char *name; int64_t K, N; };
    const Spec specs[] = {
        {"K2560_N2560 (q, o)",       2560, 2560},
        {"K2560_N3456 (gate/up 1/2)", 2560, 3456},
        {"K6912_N2560 (down)",       6912, 2560},
    };

    std::vector<Prog> progs;
    for (const auto &s : specs) {
        const std::string stem = dir + "/mm_M512_K" + std::to_string(s.K) +
                                 "_N" + std::to_string(s.N);
        Prog pr;
        pr.name = s.name;
        pr.p = std::make_unique<xdna::Program>(stem + ".xclbin", stem + ".insts.bin",
                                               512, s.K, s.N);
        std::vector<int8_t> b((size_t)(s.K * s.N));
        for (size_t i = 0; i < b.size(); ++i) b[i] = (int8_t)((i % 3) - 1);
        // Match the in-model residency: BitNet-2B has 30 layers, so each shape
        // backs ~30-60 distinct weight tensors, ~1.84 GiB in total. If resident
        // footprint (IOMMU/TLB pressure) is what inflates dispatch cost, it will
        // show up here and nowhere else.
        const int copies = (argc > 2) ? std::atoi(argv[2]) : 1;
        for (int c = 0; c < copies; ++c) pr.extra.push_back(pr.p->upload(b.data()));
        pr.w = pr.p->upload(b.data());
        progs.push_back(std::move(pr));
    }

    std::printf("npu_switch_cost\n\n  (a) each program in isolation, repeated:\n");
    double isolated_mean = 0;
    // The per-layer mix: q, o (x2 on K2560_N2560), gate+up (x4 on K2560_N3456),
    // down (x1 on K6912_N2560).
    const int mix[3] = {2, 4, 1};
    for (size_t i = 0; i < progs.size(); ++i) {
        for (int w = 0; w < 10; ++w) progs[i].p->run_mapped(*progs[i].w);
        std::vector<double> t;
        for (int r = 0; r < 60; ++r) {
            auto t0 = clk::now();
            progs[i].p->run_mapped(*progs[i].w);
            t.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
        }
        const double m = p50(t);
        isolated_mean += m * mix[i];
        std::printf("      %-28s %7.3f ms  (x%d per layer)\n", progs[i].name.c_str(), m, mix[i]);
    }
    isolated_mean /= 7.0;
    std::printf("      %-28s %7.3f ms\n", "weighted mean, no switching", isolated_mean);

    // (b) the real per-layer order: q, o, gate, gate, up, up, down
    const int order[7] = {0, 0, 1, 1, 1, 1, 2};
    for (int w = 0; w < 14; ++w) progs[order[w % 7]].p->run_mapped(*progs[order[w % 7]].w);
    std::vector<double> t;
    for (int r = 0; r < 70 * 4; ++r) {
        const int i = order[r % 7];
        auto t0 = clk::now();
        progs[i].p->run_mapped(*progs[i].w);
        t.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    }
    const double cycled = p50(t);
    std::printf("\n  (b) cycling the 3 programs in BitNet's per-layer order:\n");
    std::printf("      %-28s %7.3f ms\n", "median dispatch", cycled);
    std::printf("\n  => switching penalty: %.3f ms per dispatch (%.0f%% overhead)\n",
                cycled - isolated_mean, 100.0 * (cycled - isolated_mean) / isolated_mean);
    std::printf("     over 210 dispatches/prefill that is %.0f ms\n",
                (cycled - isolated_mean) * 210);
    return 0;
}
