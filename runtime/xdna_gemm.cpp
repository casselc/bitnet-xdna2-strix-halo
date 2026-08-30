#include "xdna_gemm.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <experimental/xrt_xclbin.h>

namespace xdna {
namespace {

std::atomic<uint64_t> g_dispatches{0};
std::atomic<uint64_t> g_dispatch_ns{0};
/* The dispatch is four distinct costs and an aggregate cannot separate them.
 * sync() on this part is not a DMA and not a no-op: amdxdna's KMQ shim reports
 * is_cache_coherent()==false unconditionally, so sync() is a userspace CLFLUSH
 * loop run by this thread, and its cost depends on how much of the buffer is
 * resident and dirty. In llama.cpp we memcpy megabytes in and out around every
 * dispatch while 15 other threads churn L3; a micro-benchmark that never
 * touches the buffers pays almost nothing. */
std::atomic<uint64_t> g_sync_in_ns{0}, g_submit_ns{0}, g_wait_ns{0}, g_sync_out_ns{0};

/* One xrt::device per process: device open measured 12.3 ms, and XRT does not
 * appreciate repeated opens. */
xrt::device &shared_device() {
    static xrt::device dev(0);
    return dev;
}

std::vector<uint32_t> read_insts(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open insts file: " + path);
    const auto bytes = static_cast<size_t>(f.tellg());
    if (bytes % 4) throw std::runtime_error("insts file is not a whole number of words");
    f.seekg(0);
    std::vector<uint32_t> v(bytes / 4);
    f.read(reinterpret_cast<char *>(v.data()), bytes);
    return v;
}

} // namespace

struct Weights::Impl { xrt::bo bo; };
Weights::Weights(std::unique_ptr<Impl> i) : p_(std::move(i)) {}
Weights::~Weights() = default;

struct Program::Impl {
    int64_t M, K, N;
    xrt::hw_context ctx;
    xrt::kernel     kern;
    xrt::bo         bo_insts, bo_a, bo_c;   // staging buffers shared by all tensors
    size_t          insts_bytes;
};

Program::Program(const std::string &xclbin_path, const std::string &insts_path,
                 int64_t M_tile, int64_t K, int64_t N)
    : p_(std::make_unique<Impl>()) {
    p_->M = M_tile; p_->K = K; p_->N = N;

    auto &dev = shared_device();
    xrt::xclbin xclbin(xclbin_path);
    dev.register_xclbin(xclbin);
    p_->ctx = xrt::hw_context(dev, xclbin.get_uuid());

    std::string kname;
    for (const auto &k : xclbin.get_kernels()) {
        if (k.get_name().rfind("MLIR_AIE", 0) == 0) { kname = k.get_name(); break; }
    }
    if (kname.empty()) throw std::runtime_error("no MLIR_AIE kernel in " + xclbin_path);
    p_->kern = xrt::kernel(p_->ctx, kname);

    const auto insts = read_insts(insts_path);
    p_->insts_bytes = insts.size() * sizeof(uint32_t);
    p_->bo_insts = xrt::bo(dev, p_->insts_bytes, XCL_BO_FLAGS_CACHEABLE, p_->kern.group_id(1));
    std::memcpy(p_->bo_insts.map<void *>(), insts.data(), p_->insts_bytes);
    p_->bo_insts.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Args 3,4,5 are A,B,C; args 0..2 are (opcode, insts, ninsts).
    p_->bo_a = xrt::bo(dev, (size_t)(M_tile * K),     XRT_BO_FLAGS_HOST_ONLY, p_->kern.group_id(3));
    p_->bo_c = xrt::bo(dev, (size_t)(M_tile * N * 4), XRT_BO_FLAGS_HOST_ONLY, p_->kern.group_id(5));
}

Program::~Program() = default;

std::unique_ptr<Weights> Program::upload(const int8_t *b_kn) {
    auto impl = std::make_unique<Weights::Impl>();
    impl->bo = xrt::bo(shared_device(), (size_t)(p_->K * p_->N),
                       XRT_BO_FLAGS_HOST_ONLY, p_->kern.group_id(4));
    std::memcpy(impl->bo.map<void *>(), b_kn, (size_t)(p_->K * p_->N));
    impl->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return std::make_unique<Weights>(std::move(impl));
}

int8_t  *Program::a_map() { return p_->bo_a.map<int8_t *>(); }
int32_t *Program::c_map() { return p_->bo_c.map<int32_t *>(); }

void Program::sync_a() {
    const auto t0 = std::chrono::steady_clock::now();
    p_->bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    g_sync_in_ns.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - t0).count(),
                           std::memory_order_relaxed);
}

void Program::run_mapped(const Weights &w) {
    sync_a();
    run_mapped_presynced(w);
}

void Program::run_mapped_presynced(const Weights &w) {
    using nsec = std::chrono::nanoseconds;
    const auto t1 = std::chrono::steady_clock::now();
    const auto t0 = t1;

    auto run = p_->kern(3, p_->bo_insts, (uint32_t)p_->insts_bytes,
                        p_->bo_a, w.p_->bo, p_->bo_c);
    const auto t2 = std::chrono::steady_clock::now();

    const auto state = run.wait();
    if (state != ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error("NPU dispatch did not complete, ert state " +
                                 std::to_string(static_cast<int>(state)));
    const auto t3 = std::chrono::steady_clock::now();

    p_->bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const auto t4 = std::chrono::steady_clock::now();

    g_submit_ns  .fetch_add((uint64_t)std::chrono::duration_cast<nsec>(t2-t1).count(), std::memory_order_relaxed);
    g_wait_ns    .fetch_add((uint64_t)std::chrono::duration_cast<nsec>(t3-t2).count(), std::memory_order_relaxed);
    g_sync_out_ns.fetch_add((uint64_t)std::chrono::duration_cast<nsec>(t4-t3).count(), std::memory_order_relaxed);

    const auto ns = std::chrono::duration_cast<nsec>(t4 - t0).count();
    g_dispatches.fetch_add(1, std::memory_order_relaxed);
    g_dispatch_ns.fetch_add((uint64_t)ns, std::memory_order_relaxed);
}

void Program::run(const Weights &w, const int8_t *a, int32_t *c) {
    std::memcpy(a_map(), a, (size_t)(p_->M * p_->K));
    run_mapped(w);
    std::memcpy(c, c_map(), (size_t)(p_->M * p_->N * 4));
}

int64_t Program::m_tile() const { return p_->M; }
int64_t Program::k()      const { return p_->K; }
int64_t Program::n()      const { return p_->N; }

uint64_t Program::dispatch_count() { return g_dispatches.load(std::memory_order_relaxed); }
double   Program::dispatch_ms()    { return g_dispatch_ns.load(std::memory_order_relaxed) / 1e6; }
void     Program::reset_counters() { g_dispatches = 0; g_dispatch_ns = 0;
    g_sync_in_ns = 0; g_submit_ns = 0; g_wait_ns = 0; g_sync_out_ns = 0; }

void Program::breakdown_ms(double *sync_in, double *submit, double *wait, double *sync_out) {
    *sync_in  = g_sync_in_ns.load()  / 1e6;
    *submit   = g_submit_ns.load()   / 1e6;
    *wait     = g_wait_ns.load()     / 1e6;
    *sync_out = g_sync_out_ns.load() / 1e6;
}

bool device_available() {
    try { (void)shared_device(); return true; }
    catch (...) { return false; }
}

} // namespace xdna
