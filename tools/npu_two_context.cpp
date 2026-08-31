/* What does adding a SECOND production-sized hardware context cost?
 *
 * This is the decisive economic question for putting attention on the NPU. The
 * current design deliberately serves every BitNet shape from ONE program so a
 * prefill performs zero context switches -- artifacts/kernels/context_switching.md
 * measured 3-context cycling at +53% to +210% and notes the penalty SCALES WITH
 * DESIGN SIZE (the same probe over the smaller M=512 designs shows ~0%).
 *
 * An attention xclbin would be a second large context, alternating with the GEMM
 * context roughly twice per layer. So: measure two PRODUCTION-sized designs
 * alternating, against each alone, interleaved.
 */
#include "../runtime/xdna_gemm.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double p50(std::vector<double> v) {
    std::sort(v.begin(), v.end()); return v[v.size()/2];
}

struct P {
    std::string name;
    std::unique_ptr<xdna::Program> prog;
    std::unique_ptr<xdna::Weights> w;
};

int main(int argc, char **argv) {
    const std::string dir = argc > 1 ? argv[1] : "artifacts/xclbin-tuned";
    const int iters = argc > 2 ? std::atoi(argv[2]) : 40;
    struct Spec { const char *name; int64_t M, K, N; };
    const Spec specs[] = {
        {"A: M1024 K2560 N2560 (q/o, the resident GEMM context)", 1024, 2560, 2560},
        {"B: M1024 K6912 N2560 (down, standing in for an attention context)", 1024, 6912, 2560},
    };
    std::vector<P> ps;
    for (const auto &s : specs) {
        const std::string stem = dir + "/mm_M" + std::to_string(s.M) + "_K" +
                                 std::to_string(s.K) + "_N" + std::to_string(s.N);
        P p; p.name = s.name;
        p.prog = std::make_unique<xdna::Program>(stem + ".xclbin", stem + ".insts.bin",
                                                 s.M, s.K, s.N);
        std::vector<int8_t> b((size_t)(s.K * s.N));
        for (size_t i = 0; i < b.size(); ++i) b[i] = (int8_t)((i % 3) - 1);
        p.w = p.prog->upload(b.data());
        ps.push_back(std::move(p));
    }

    std::printf("npu_two_context  (%d iters, interleaved)\n\n", iters);
    // Interleave the "alone" and "alternating" measurements rather than running
    // them in blocks: this machine drifts 10-30% between runs.
    std::vector<double> aloneA, aloneB, altA, altB;
    for (int r = 0; r < iters; ++r) {
        for (int k = 0; k < 3; ++k) ps[0].prog->run_mapped(*ps[0].w);
        { auto t0=clk::now(); ps[0].prog->run_mapped(*ps[0].w);
          aloneA.push_back(std::chrono::duration<double,std::milli>(clk::now()-t0).count()); }
        for (int k = 0; k < 3; ++k) ps[1].prog->run_mapped(*ps[1].w);
        { auto t0=clk::now(); ps[1].prog->run_mapped(*ps[1].w);
          aloneB.push_back(std::chrono::duration<double,std::milli>(clk::now()-t0).count()); }
        // alternating: every dispatch follows a dispatch on the OTHER context
        { auto t0=clk::now(); ps[0].prog->run_mapped(*ps[0].w);
          altA.push_back(std::chrono::duration<double,std::milli>(clk::now()-t0).count()); }
        { auto t0=clk::now(); ps[1].prog->run_mapped(*ps[1].w);
          altB.push_back(std::chrono::duration<double,std::milli>(clk::now()-t0).count()); }
    }
    const double aA=p50(aloneA), aB=p50(aloneB), xA=p50(altA), xB=p50(altB);
    std::printf("  %-58s alone %6.3f ms   alternating %6.3f ms   %+.0f%%\n",
                specs[0].name, aA, xA, (xA/aA-1)*100);
    std::printf("  %-58s alone %6.3f ms   alternating %6.3f ms   %+.0f%%\n",
                specs[1].name, aB, xB, (xB/aB-1)*100);
    const double pen = (xA+xB) - (aA+aB);
    std::printf("\n  switch penalty: %+.3f ms per context-alternating PAIR (%+.0f%%)\n",
                pen, ((xA+xB)/(aA+aB)-1)*100);
    std::printf("  a prefill alternating twice per layer over 30 layers pays %.0f ms\n",
                pen * 30);
    return 0;
}
