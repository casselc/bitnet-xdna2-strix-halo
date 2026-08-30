/* npu_sustained -- does dispatch cost degrade under continuous load?
 *
 * Two observations point here:
 *   - repeated runs of the same benchmark binary differ by ~32% (0.771 vs
 *     1.017 ms baseline), which is far more than run-to-run noise within a run
 *   - the in-model dispatch cost (2.66 ms) is ~50% above what the same kernels
 *     cost in a short standalone burst (1.74 ms)
 * A real prefill drives the NPU continuously for hundreds of dispatches, while
 * a micro-benchmark runs in short bursts. If the device clocks down under
 * sustained load, that difference is the explanation and no amount of API
 * tuning will recover it. This dispatches continuously and reports the cost as
 * a function of elapsed time. */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <experimental/xrt_xclbin.h>

using clk = std::chrono::steady_clock;
static double p50(std::vector<double> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }

int main(int argc, char **argv) {
    const std::string stem = (argc>1?std::string(argv[1]):std::string("artifacts/xclbin"))
                             + "/mm_M512_K2560_N2560";
    const double seconds = argc>2 ? atof(argv[2]) : 60.0;
    const int64_t M=512,K=2560,N=2560;

    xrt::device dev(0);
    xrt::xclbin xclbin(stem+".xclbin");
    dev.register_xclbin(xclbin);
    xrt::hw_context ctx(dev, xclbin.get_uuid());
    std::string kn; for (auto &k: xclbin.get_kernels())
        if (k.get_name().rfind("MLIR_AIE",0)==0){ kn=k.get_name(); break; }
    xrt::kernel kern(ctx,kn);

    std::ifstream f(stem+".insts.bin", std::ios::binary|std::ios::ate);
    size_t ib=(size_t)f.tellg(); f.seekg(0);
    std::vector<uint32_t> insts(ib/4); f.read((char*)insts.data(), ib);
    xrt::bo bo_i(dev, ib, XCL_BO_FLAGS_CACHEABLE, kern.group_id(1));
    std::memcpy(bo_i.map<void*>(), insts.data(), ib); bo_i.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    xrt::bo bo_a(dev,(size_t)(M*K),  XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
    xrt::bo bo_b(dev,(size_t)(K*N),  XRT_BO_FLAGS_HOST_ONLY, kern.group_id(4));
    xrt::bo bo_c(dev,(size_t)(M*N*4),XRT_BO_FLAGS_HOST_ONLY, kern.group_id(5));
    std::memset(bo_b.map<void*>(),1,(size_t)(K*N)); bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::printf("npu_sustained  continuous dispatch for %.0f s\n\n", seconds);
    std::printf("  %-10s %-12s %-12s %s\n", "window", "p50 (ms)", "TOPS", "vs first window");

    const double bucket = 5.0;
    const auto start = clk::now();
    std::vector<double> cur;
    double first = 0;
    double elapsed = 0, bucket_end = bucket;
    while (elapsed < seconds) {
        auto t0 = clk::now();
        auto r = kern(3, bo_i, (uint32_t)ib, bo_a, bo_b, bo_c);
        r.wait();
        cur.push_back(std::chrono::duration<double,std::milli>(clk::now()-t0).count());
        elapsed = std::chrono::duration<double>(clk::now()-start).count();
        if (elapsed >= bucket_end) {
            const double m = p50(cur);
            if (first == 0) first = m;
            const double tops = 2.0*M*K*N / (m*1e-3) / 1e12;
            std::printf("  %-10.0f %-12.3f %-12.2f %+.1f%%\n",
                        bucket_end, m, tops, 100.0*(m-first)/first);
            cur.clear();
            bucket_end += bucket;
        }
    }
    return 0;
}
