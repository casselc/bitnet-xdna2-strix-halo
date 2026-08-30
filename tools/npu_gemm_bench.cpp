/* npu_gemm_bench -- why does an in-model dispatch cost 2.86 ms when the same
 * kernel standalone costs 1.53 ms?
 *
 * The in-model path differs in one way that standalone benchmarks never do: it
 * cycles through ~150 distinct resident weight buffers rather than reusing one.
 * This isolates that variable, holding everything else fixed. */
#include "../runtime/xdna_gemm.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

using clk = std::chrono::steady_clock;

static double bench(xdna::Program &prog,
                    std::vector<std::unique_ptr<xdna::Weights>> &ws,
                    int iters) {
    std::vector<double> t;
    t.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        prog.run_mapped(*ws[i % ws.size()]);
        t.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int main(int argc, char **argv) {
    const std::string dir = argc > 1 ? argv[1] : "artifacts/xclbin";
    // Defaults match the tuned artifacts; override on the command line.
    const int64_t M = argc > 2 ? atoll(argv[2]) : 1024;
    const int64_t K = argc > 3 ? atoll(argv[3]) : 2560;
    const int64_t N = argc > 4 ? atoll(argv[4]) : 2560;
    const std::string stem = dir + "/mm_M" + std::to_string(M) +
                             "_K" + std::to_string(K) + "_N" + std::to_string(N);

    xdna::Program prog(stem + ".xclbin", stem + ".insts.bin", M, K, N);

    // Deterministic ternary weights; content is irrelevant to timing.
    std::vector<int8_t> b((size_t)(K * N));
    for (size_t i = 0; i < b.size(); ++i) b[i] = (int8_t)((i % 3) - 1);

    std::printf("npu_gemm_bench  M=%lld K=%lld N=%lld\n\n",
                (long long)M, (long long)K, (long long)N);
    std::printf("  %-30s %8s  %6s\n", "distinct weight buffers", "p50", "TOPS");

    for (int count : {1, 2, 4, 8, 16, 30, 60}) {
        std::vector<std::unique_ptr<xdna::Weights>> ws;
        try {
            for (int i = 0; i < count; ++i) ws.push_back(prog.upload(b.data()));
        } catch (const std::exception &e) {
            std::printf("  %-42d upload failed: %s\n", count, e.what());
            break;
        }
        for (int i = 0; i < 10; ++i) prog.run_mapped(*ws[i % ws.size()]);  // warm
        const double p50 = bench(prog, ws, std::max(60, count * 3));
        const double tops = 2.0 * (double)M * (double)K * (double)N / (p50 * 1e-3) / 1e12;
        std::printf("  %-30d %8.3f ms  %6.2f TOPS  (%.0f MiB resident)\n",
                    count, p50, tops, count * (double)(K * N) / 1048576.0);
    }
    return 0;
}
