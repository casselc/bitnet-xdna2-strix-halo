/* xdna_gemm.h -- resident XDNA2 int8 GEMM for BitNet ternary linears.
 *
 * Design rules, each forced by a measurement in artifacts/kernels/milestone_a.md:
 *
 *  - A Program (xclbin + hw_context + kernel + instruction BO + the shared A/C
 *    staging buffers) is created ONCE PER SHAPE and shared by every weight
 *    tensor of that shape. This is not an optimization: the driver allows only
 *    16 hardware contexts, and BitNet-2B has ~150 offloadable weight tensors.
 *    Creating a context per tensor exhausts them and silently falls back to CPU.
 *  - Weights (the per-tensor part) are uploaded once and stay resident. BO
 *    allocation costs ~0.1 ms/MiB, which would dominate if done per call.
 *  - Per dispatch only the activation goes in and the accumulator comes out.
 *
 * Numerics: the NPU computes a pure signed integer GEMM
 *      C[t,n] = sum_k A[t,k] * B[k,n]      A int8, B int8 (ternary), C int32
 * Because weights are stored as SIGNED ternary {-1,0,+1} rather than BitNet's
 * unsigned {0,1,2} codes, the accumulator is already the true ternary dot
 * product and the CPU's `- act_sums` correction is not applied. That identity
 * (sum (w+1)a == sum wa + sum a) is proven in tests/test_i2s_packing.c.
 *
 * Dispatches are serialized: the NPU is single-tenant and hardware contexts
 * time-slice the whole array, so the shared staging buffers are safe under the
 * caller's lock.
 */
#ifndef XDNA_GEMM_H
#define XDNA_GEMM_H

#include <cstdint>
#include <memory>
#include <string>

namespace xdna {

class Weights;

/* One compiled kernel for a fixed (M_tile, K, N), shared across tensors. */
class Program {
public:
    Program(const std::string &xclbin_path, const std::string &insts_path,
            int64_t M_tile, int64_t K, int64_t N);
    ~Program();

    /* Upload one [K,N] int8 weight matrix; the returned handle stays resident. */
    std::unique_ptr<Weights> upload(const int8_t *b_kn);

    /* Direct access to the mapped staging buffers. Writing activations into
     * a_map() and reading accumulators from c_map() removes one full copy on
     * each side of every dispatch -- at 512x3456 the C buffer alone is 7 MiB,
     * so the saving is larger than the kernel time it surrounds. */
    int8_t  *a_map();
    int32_t *c_map();

    /* Dispatch using whatever is currently in a_map(); result lands in c_map().
     * Syncs the activation buffer to the device first. */
    void run_mapped(const Weights &w);

    /* Split form, for reusing one activation slice across several weight chunks.
     * A wide-N tensor is served by several N-chunks that all consume the SAME
     * activation K-slice; calling run_mapped for each re-flushes an unchanged
     * buffer. sync_a() once, then dispatch each chunk. */
    void sync_a();
    void run_mapped_presynced(const Weights &w);

    /* Asynchronous dispatch, for overlapping the host-side evacuation of one
     * chunk's results with the NPU's execution of the next.
     *
     * The dispatch loop copies each chunk's int32 results out of the mapped
     * output buffer immediately after waiting for it, which happens while the
     * device is idle -- measured at 162 ms per prefill, single-threaded (see
     * artifacts/overlap-de-risk/RESULTS.md section 2). Submitting the next chunk
     * before evacuating the current one hides that copy under device time.
     *
     * Two output buffers alternate so the copy out of slot i cannot race the
     * device writing slot i^1. Every N-chunk of one K-slice consumes the same
     * activations, so the A buffer is stable across the submits being pipelined;
     * a K-slice boundary must drain before restaging A.
     *
     * Contract: at most one dispatch outstanding. submit_async() on a Program
     * with one already pending is a logic error and throws. */
    void submit_async(const Weights &w, int c_slot);
    void wait_pending();          // wait + sync the outstanding slot; no-op if none
    bool has_pending() const;
    int32_t *c_map_slot(int slot);
    static constexpr int kCSlots = 2;

    /* Convenience wrapper that copies through (used by tests). */
    void run(const Weights &w, const int8_t *a, int32_t *c);

    int64_t m_tile() const;
    int64_t k() const;
    int64_t n() const;

    static uint64_t dispatch_count();
    static double   dispatch_ms();
    static void     reset_counters();
    /* Split of the timed region: cache flush in, ioctl submit, fence wait,
     * cache invalidate out. Totals in ms across all dispatches. */
    static void     breakdown_ms(double *sync_in, double *submit,
                                 double *wait, double *sync_out);
    /* Split of the MOST RECENT dispatch only, so a caller can attribute
     * submit/wait/sync-out to the logical tensor shape it just issued. Valid
     * only for the single owner thread that issues dispatches. */
    static void     last_breakdown_ms(double *submit, double *wait, double *sync_out);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

/* Opaque resident weight buffer belonging to a Program. */
class Weights {
public:
    ~Weights();
    struct Impl;
    explicit Weights(std::unique_ptr<Impl> i);
    std::unique_ptr<Impl> p_;
};

bool device_available();

} // namespace xdna
#endif
