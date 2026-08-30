/* npu_probe -- what does the NPU cost us before any kernel is involved?
 *
 * The whole offload design hinges on one number: how expensive is a host->NPU
 * ->host round trip. Published XDNA2 figures (0.66 ms round trip, 2.67 ms
 * context switch) were measured on Krackan, not Strix Halo, so we measure here
 * rather than inherit them.
 *
 * This probe covers the parts that need no compiled kernel: device open, BO
 * allocation, and host<->device sync at realistic BitNet sizes. Kernel dispatch
 * is measured separately once mlir-aie has produced an xclbin. */
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

struct Stat { double mean, min, max, p50; };

static Stat summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double sum = 0; for (double x : v) sum += x;
    return { sum / v.size(), v.front(), v.back(), v[v.size() / 2] };
}

static void report(const char *label, const Stat &s, double bytes = 0) {
    std::printf("  %-38s mean %8.3f ms   p50 %8.3f   min %8.3f   max %8.3f",
                label, s.mean, s.p50, s.min, s.max);
    if (bytes > 0) std::printf("   %7.2f GB/s", bytes / (s.p50 * 1e-3) / 1e9);
    std::printf("\n");
}

int main() {
    std::printf("npu_probe\n\n");

    // --- device open -------------------------------------------------------
    // Measured once because it is a per-process cost, not a per-dispatch one.
    auto t0 = clk::now();
    xrt::device dev(0);
    double open_ms = ms_since(t0);
    std::printf("  %-38s %8.3f ms\n", "device open (one-time)", open_ms);
    std::printf("  %-38s %s / %s\n\n", "device",
                dev.get_info<xrt::info::device::name>().c_str(),
                dev.get_info<xrt::info::device::bdf>().c_str());

    // --- BO allocation + sync at real BitNet sizes -------------------------
    // Sizes chosen from the actual model: one 2560x2560 weight tile expanded to
    // int8 (6.55 MB), one 512-token activation block (1.31 MB), and one output
    // accumulator block (5.24 MB).
    struct Case { const char *name; size_t bytes; };
    const std::vector<Case> cases = {
        {"activations 512x2560 int8",     512ull * 2560},
        {"weights 2560x2560 int8",       2560ull * 2560},
        {"output 512x2560 int32",         512ull * 2560 * 4},
        {"weights 2560x6912 int8",       2560ull * 6912},
    };

    const int iters = 50;
    for (const auto &c : cases) {
        std::printf("  --- %s (%.2f MiB)\n", c.name, c.bytes / 1048576.0);

        std::vector<double> alloc, to_dev, from_dev;
        for (int i = 0; i < iters; ++i) {
            auto ta = clk::now();
            xrt::bo bo(dev, c.bytes, xrt::bo::flags::host_only, 0);
            alloc.push_back(ms_since(ta));

            // Touch the mapping so the sync measures real traffic, not a no-op
            // on untouched pages.
            auto *p = bo.map<uint8_t *>();
            std::memset(p, i & 0xff, c.bytes);

            auto tb = clk::now();
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            to_dev.push_back(ms_since(tb));

            auto tc = clk::now();
            bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            from_dev.push_back(ms_since(tc));
        }
        report("bo alloc + map", summarize(alloc));
        report("sync TO_DEVICE",   summarize(to_dev),   (double)c.bytes);
        report("sync FROM_DEVICE", summarize(from_dev), (double)c.bytes);
        std::printf("\n");
    }

    std::printf("note: sync cost here is the host-side cache operation only.\n"
                "      Kernel dispatch latency is measured separately in\n"
                "      npu_dispatch_bench once an xclbin exists.\n");
    return 0;
}
