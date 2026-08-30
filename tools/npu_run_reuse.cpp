/* npu_run_reuse -- does reusing one xrt::run beat constructing one per dispatch?
 *
 * The current hot path calls kern(3, insts, n, a, b, c), which builds a fresh
 * xrt::run (and with it an ERT command packet, plus argument validation) on
 * every dispatch. XRT also exposes a persistent form: construct the run once,
 * update arguments with set_arg, then start/wait. If the per-call construction
 * is material, this shows it directly. */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <experimental/xrt_xclbin.h>

using clk = std::chrono::steady_clock;
static double p50(std::vector<double> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }

int main(int argc, char **argv) {
    const std::string stem = (argc > 1 ? std::string(argv[1])
                                       : std::string("artifacts/xclbin")) + "/mm_M512_K2560_N2560";
    const int64_t M=512, K=2560, N=2560;

    xrt::device dev(0);
    xrt::xclbin xclbin(stem + ".xclbin");
    dev.register_xclbin(xclbin);
    xrt::hw_context ctx(dev, xclbin.get_uuid());
    std::string kname;
    for (auto &k : xclbin.get_kernels())
        if (k.get_name().rfind("MLIR_AIE",0)==0) { kname=k.get_name(); break; }
    xrt::kernel kern(ctx, kname);

    std::ifstream f(stem + ".insts.bin", std::ios::binary|std::ios::ate);
    const size_t ib = (size_t)f.tellg(); f.seekg(0);
    std::vector<uint32_t> insts(ib/4);
    f.read((char*)insts.data(), ib);

    xrt::bo bo_i(dev, ib, XCL_BO_FLAGS_CACHEABLE, kern.group_id(1));
    std::memcpy(bo_i.map<void*>(), insts.data(), ib);
    bo_i.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::bo bo_a(dev,(size_t)(M*K),  XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
    xrt::bo bo_b(dev,(size_t)(K*N),  XRT_BO_FLAGS_HOST_ONLY, kern.group_id(4));
    xrt::bo bo_c(dev,(size_t)(M*N*4),XRT_BO_FLAGS_HOST_ONLY, kern.group_id(5));
    std::memset(bo_b.map<void*>(), 1, (size_t)(K*N));
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    const int iters = 400;

    // One persistent run, plus a second weight BO and an arena of sub-buffers,
    // so all four variants can be compared under identical conditions.
    xrt::run run(kern);
    run.set_arg(0, 3); run.set_arg(1, bo_i); run.set_arg(2, (uint32_t)ib);
    run.set_arg(3, bo_a); run.set_arg(4, bo_b); run.set_arg(5, bo_c);

    xrt::bo bo_b2(dev,(size_t)(K*N), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(4));
    std::memset(bo_b2.map<void*>(), 1, (size_t)(K*N));
    bo_b2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    const size_t wbytes = (size_t)(K*N);
    xrt::bo arena(dev, wbytes*4, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(4));
    std::memset(arena.map<void*>(), 1, wbytes*4);
    arena.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::vector<xrt::bo> subs;
    for (int i=0;i<4;i++) subs.emplace_back(arena, wbytes, wbytes*i);

    std::vector<double> t[4];
    auto once = [&](int variant) {
        auto t0 = clk::now();
        switch (variant) {
        case 0: { auto r = kern(3, bo_i, (uint32_t)ib, bo_a, bo_b, bo_c); r.wait(); break; }
        case 1: { run.set_arg(4, bo_b);  run.start(); run.wait(); break; }
        case 2: { run.set_arg(4, bo_b2); run.start(); run.wait(); break; }
        case 3: { static int k=0; run.set_arg(4, subs[k++ & 3]); run.start(); run.wait(); break; }
        }
        return std::chrono::duration<double,std::milli>(clk::now()-t0).count();
    };

    for (int i=0;i<40;i++) for (int v=0;v<4;v++) once(v);          // warm all four
    // Interleave round-robin so drift hits every variant equally.
    for (int i=0;i<iters;i++) for (int v=0;v<4;v++) t[v].push_back(once(v));

    const char *names[4] = {
        "(a) fresh xrt::run per dispatch (current)",
        "(b) persistent run, same weight BO",
        "(c) persistent run, set_arg a distinct BO",
        "(d) persistent run, set_arg a sub-buffer",
    };
    std::printf("npu_run_reuse  M=%lld K=%lld N=%lld, %d iters, interleaved\n\n",
                (long long)M,(long long)K,(long long)N,iters);
    for (int v=0;v<4;v++)
        std::printf("  %-46s %7.3f ms\n", names[v], p50(t[v]));
    std::printf("\n  persistent-run saving vs fresh:  %+.3f ms\n", p50(t[0])-p50(t[1]));
    std::printf("  cost of rebinding a distinct BO: %+.3f ms\n", p50(t[2])-p50(t[1]));
    std::printf("  cost of rebinding a sub-buffer:  %+.3f ms\n", p50(t[3])-p50(t[1]));
    return 0;
}
