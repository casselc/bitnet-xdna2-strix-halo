/* Per-node execution profiler for the ggml CPU graph.
 *
 * Purpose: break down the CPU-side prefill residue -- the work the current NPU
 * offload path does NOT execute -- into dependency-meaningful categories, and
 * emit a trace rich enough to reconstruct the operation DAG.
 *
 * Design notes:
 *
 *  - Only thread 0 records. In ggml's CPU graph every node is computed by all
 *    threads and followed by ggml_barrier(), so the interval from "before
 *    compute" to "after barrier" on thread 0 is that node's true wall-clock
 *    contribution, and the intervals tile the graph without overlap.
 *
 *  - Enabled purely at runtime via BITNET_PROFILE=<path>. A build with the
 *    profiler compiled in and the variable unset pays two predictable branches
 *    per node and nothing else, so the same binary produces both the profiled
 *    and the reference timings.
 *
 *  - Records are appended to a preallocated arena and written once at exit.
 *    Nothing is formatted or allocated on the hot path.
 *
 *  - NPU attribution is exact rather than inferred: the XDNA dispatch counter
 *    and device-time accumulator are sampled around each node, so a node's NPU
 *    device time is a measured delta, not a guess from the node's name.
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

/* Non-zero when BITNET_PROFILE is set. Cheap; safe to call per node. */
int bnp_enabled(void);

/* Bracket one graph node. `idx` is its index within the current graph. */
void bnp_node_begin(int idx);
void bnp_node_end(int idx, const struct ggml_tensor *node);

/* Bump the graph counter; lets the trace separate micro-batches. */
void bnp_graph_begin(int n_nodes);

#ifdef __cplusplus
}
#endif
