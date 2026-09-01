/* bitnet_xdna.h -- C shim letting ggml's I2_S mul_mat offload prefill to XDNA2.
 *
 * Deliberately a C API with no XRT types in the signature, so ggml-cpu.c only
 * needs this header and the patch stays a few lines.
 *
 * Backend selection is runtime, never a model property:
 *   BITNET_XDNA=0 (or unset)  -> CPU only, this shim reports unavailable
 *   BITNET_XDNA=1             -> offload eligible I2_S GEMMs during prefill
 *   BITNET_XDNA_MIN_TOKENS=n  -> below n tokens stay on CPU (default 64);
 *                                dispatch overhead does not repay a small batch
 */
#ifndef BITNET_XDNA_H
#define BITNET_XDNA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Idempotent. Returns 1 if the NPU path is usable, 0 otherwise (missing device,
 * missing artifacts, BITNET_XDNA unset). Never throws into C. */
int bitnet_xdna_available(void);

/* Single-flight lease over one NPU invocation's whole lifetime:
 *   begin() -> accumulate() -> [barrier] -> epilogue() on all threads
 *            -> [barrier] -> end()
 * Taken by the thread that drives the device; released only once every CPU
 * reader of that invocation's output has finished. Required because the epilogue
 * consumes process-global state (output-slot plan, accumulator, shape identity)
 * that a second inference context would otherwise overwrite mid-read. */
void bitnet_xdna_invocation_begin(void);
void bitnet_xdna_invocation_end(void);

/* Single-flight lease counters, for service-level measurement.
 *
 * Enabled by BITNET_XDNA_LEASE_STATS=1; when disabled the lease path costs one
 * relaxed atomic load and reads no clock. Cumulative since process start, so a
 * caller measures a window by differencing. Snapshot is not atomic across
 * fields -- the counters move independently and a torn read only matters at
 * sub-microsecond granularity, which is far below what is being measured. */
struct bitnet_xdna_lease_stats {
    unsigned long long acquisitions;
    unsigned long long immediate;      /* uncontended */
    unsigned long long waited;         /* had to block */
    unsigned long long wait_ns;        /* summed blocked time */
    unsigned long long hold_ns;        /* summed held time */
    unsigned long long wait_max_ns;
    unsigned long long hold_max_ns;
    int                waiters_now;
    int                waiters_max;
};
void bitnet_xdna_lease_snapshot(struct bitnet_xdna_lease_stats *out);
int  bitnet_xdna_lease_stats_enabled(void);

/* Token-dimension (ne11) histogram at the ggml/XDNA boundary.
 *
 * Answers two questions the service benchmarks could not: what token batch
 * sizes actually reach offloadable linear nodes, and why a 1954-token prefill
 * produces ~146 NPU invocations rather than the 210 (7 I2_S matmuls x 30
 * layers) the architecture predicts.
 *
 * Every I2_S 2-D mul_mat node is observed once per graph, from thread 0 only --
 * the offload gate is evaluated by all nth threads, so observing inside
 * worth_it() would overcount by a factor of nth.
 *
 * Buckets are log2(ne11): bucket b holds ne11 in [2^b, 2^(b+1)). */
#define BITNET_XDNA_NE11_BUCKETS 20

struct bitnet_xdna_ne11_stats {
    uint64_t seen[BITNET_XDNA_NE11_BUCKETS];      /* every I2_S 2-D node      */
    uint64_t worth[BITNET_XDNA_NE11_BUCKETS];     /* passed the token gate    */
    uint64_t offloaded[BITNET_XDNA_NE11_BUCKETS]; /* also a supported shape   */
    uint64_t nodes_seen, nodes_worth, nodes_offloaded;
    uint64_t declined_small;  /* ne11 below the token threshold */
    uint64_t declined_shape;  /* big enough, but no plan for (K, N) */
};

/* Observe one node. Call from thread 0 only, with the two gate predicates
 * already evaluated, so short-circuit semantics at the call site are unchanged. */
void bitnet_xdna_observe_node(int64_t n_tokens, int worth, int offloaded);
void bitnet_xdna_ne11_snapshot(struct bitnet_xdna_ne11_stats *out);
int  bitnet_xdna_ne11_stats_enabled(void);

/* Can this (K,N) be served? Shapes are fixed by the AOT-compiled xclbins. */
int bitnet_xdna_supports(int64_t K, int64_t N);

/* Is this token count worth a dispatch? */
int bitnet_xdna_worth_it(int64_t n_tokens);

/* How many of the batch's tokens should the NPU take, so that the NPU and the
 * CPU threads can work on disjoint token ranges at the same time?
 *
 * Exclusive offload leaves 16 cores idle on a barrier for ~76% of a prefill,
 * which is why exclusive offload lost: it replaced 16 working cores with one
 * comparable engine instead of adding to them. Splitting the batch lets both run.
 *
 * Returns a multiple of the NPU's token tile (so no dispatch is padded), or the
 * whole batch when it is too small to divide. 0 means "leave it all to the CPU".
 * The fraction is BITNET_XDNA_SPLIT (default 0.5; 1.0 restores exclusive offload). */
int64_t bitnet_xdna_token_split(int64_t n_tokens);

/* Thread-aware form. Thread 0 is consumed driving the device, so only n_threads-1
 * workers remain for the CPU's share -- a fixed 0.5 split is therefore wrong at
 * both ends. Measured on this machine the NPU is worth ~10 Zen 5 threads on these
 * shapes, so the balance point is
 *
 *     f = R / (R + (n_threads - 1)),   R = 10
 *
 * snapped to a whole NPU tile. That reproduces the measured optimum in all six
 * swept (tiles-available, thread-count) cases, including declining the NPU
 * entirely at 15 threads with only one tile available. Override with
 * BITNET_XDNA_NPU_THREADS. */
int64_t bitnet_xdna_token_split_nt(int64_t n_tokens, int n_threads);

/* Compute dst = ((W_ternary . a_q) ) * (ws / act_scale[t]) for a whole batch.
 *
 *   src0_i2s   packed I2_S weight blob for an [N,K] tensor (scale lives inside)
 *   K, N       tensor dims (ggml ne00, ne01)
 *   a_q        [T,K] int8 activations, row stride a_row_stride bytes
 *   T          token count (ggml ne11)
 *   act_scales [T] f32, act_sums [T] int32 (act_sums unused: our weights are
 *              signed, so no bias correction is needed -- taken for interface
 *              symmetry with the CPU path and to assert agreement in tests)
 *   ws         the per-tensor weight scale
 *   dst        [T,N] f32 output, row stride dst_row_stride bytes
 *
 * Returns 1 on success, 0 if the caller must fall back to CPU. Never partially
 * writes dst on failure. */
int bitnet_xdna_mul_mat(const void *src0_i2s, int64_t K, int64_t N,
                        const int8_t *a_q, int64_t T, size_t a_row_stride,
                        const float *act_scales, const int32_t *act_sums,
                        float ws,
                        float *dst, size_t dst_row_stride);

/* Split form, so the f32 epilogue can use the whole ggml threadpool instead of
 * leaving 15 threads spinning on a barrier while thread 0 converts millions of
 * accumulators by itself.
 *
 *   thread 0:      bitnet_xdna_accumulate(...)   -- all NPU dispatches
 *   ggml_barrier
 *   every thread:  bitnet_xdna_epilogue(...)     -- disjoint row range
 *   ggml_barrier   (before the accumulator buffer is reused)
 *
 * bitnet_xdna_accumulate leaves a [T,N] int32 buffer owned by the shim; it stays
 * valid until the next accumulate call. Returns 1 on success, 0 to fall back. */
int bitnet_xdna_accumulate(const void *src0_i2s, int64_t K, int64_t N,
                           const int8_t *a_q, int64_t T, size_t a_row_stride);

void bitnet_xdna_epilogue(int64_t N, int64_t row_begin, int64_t row_end,
                          const float *act_scales, float ws,
                          float *dst, size_t dst_row_stride);

/* Evidence counters: prove prefill used the NPU and decode did not. */
uint64_t bitnet_xdna_dispatches(void);
double   bitnet_xdna_dispatch_ms(void);
double   bitnet_xdna_repack_ms(void);
/* CPU-side cost of USING the NPU rather than the NPU's own device time:
 * the int32->f32 epilogue that converts the NPU's accumulators. */
double   bitnet_xdna_epilogue_ms(void);
uint64_t bitnet_xdna_resident_bytes(void);
/* Introspection for the long-lived stress harness: the direct-output arena must
 * reach a stable high-water mark rather than growing as differently sized
 * requests arrive. */
int      bitnet_xdna_out_slots(void);
uint64_t bitnet_xdna_out_slot_bytes(void);
int      bitnet_xdna_resident_tensors(void);
void     bitnet_xdna_reset_counters(void);

#ifdef __cplusplus
}
#endif
#endif
