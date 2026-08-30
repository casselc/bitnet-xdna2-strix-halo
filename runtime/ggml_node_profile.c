#include "ggml_node_profile.h"

#include "ggml.h"
#include "ggml-impl.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef GGML_BITNET_XDNA
#include "bitnet_xdna.h"
#endif

#define BNP_MAX_RECORDS (1u << 21)   /* 2M nodes; ~450/graph, so ample */
#define BNP_NAME_LEN    48
#define BNP_NSRC        2

struct bnp_rec {
    uint64_t t0_ns, t1_ns;
    uint64_t npu_ns;        /* NPU device time attributed to this node */
    uint32_t npu_dispatch;  /* NPU dispatches issued during this node */
    int32_t  graph;
    int32_t  idx;
    int32_t  op;
    int32_t  src_type0;
    int64_t  ne[4];
    char     name[BNP_NAME_LEN];
    char     src[BNP_NSRC][BNP_NAME_LEN];
};

static struct bnp_rec *g_rec;
static _Atomic size_t  g_n;
static int             g_on = -1;
static const char     *g_path;
static _Atomic int32_t g_graph;

/* per-node scratch (thread 0 only, so plain statics are fine) */
static uint64_t g_t0, g_npu_ms0;
static uint64_t g_disp0;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void bnp_dump(void);

int bnp_enabled(void) {
    if (g_on >= 0) return g_on;
    g_path = getenv("BITNET_PROFILE");
    if (!g_path || !*g_path) { g_on = 0; return 0; }
    g_rec = (struct bnp_rec *)calloc(BNP_MAX_RECORDS, sizeof(struct bnp_rec));
    if (!g_rec) { fprintf(stderr, "[bnp] arena alloc failed; profiling off\n"); g_on = 0; return 0; }
    atexit(bnp_dump);
    g_on = 1;
    fprintf(stderr, "[bnp] node profiling -> %s\n", g_path);
    return 1;
}

void bnp_graph_begin(int n_nodes) {
    (void)n_nodes;
    if (g_on <= 0) return;
    atomic_fetch_add_explicit(&g_graph, 1, memory_order_relaxed);
}

void bnp_node_begin(int idx) {
    (void)idx;
    if (g_on <= 0) return;
#ifdef GGML_BITNET_XDNA
    g_disp0   = bitnet_xdna_dispatches();
    g_npu_ms0 = (uint64_t)(bitnet_xdna_dispatch_ms() * 1e6);  /* ms -> ns */
#endif
    g_t0 = now_ns();
}

void bnp_node_end(int idx, const struct ggml_tensor *node) {
    if (g_on <= 0 || !node) return;
    const uint64_t t1 = now_ns();
    size_t i = atomic_fetch_add_explicit(&g_n, 1, memory_order_relaxed);
    if (i >= BNP_MAX_RECORDS) return;

    struct bnp_rec *r = &g_rec[i];
    r->t0_ns = g_t0;
    r->t1_ns = t1;
    r->graph = atomic_load_explicit(&g_graph, memory_order_relaxed);
    r->idx   = idx;
    r->op    = (int32_t)node->op;
    for (int d = 0; d < 4; d++) r->ne[d] = node->ne[d];
    snprintf(r->name, BNP_NAME_LEN, "%s", node->name);
    r->src_type0 = node->src[0] ? (int32_t)node->src[0]->type : -1;
    for (int s = 0; s < BNP_NSRC; s++)
        snprintf(r->src[s], BNP_NAME_LEN, "%s", node->src[s] ? node->src[s]->name : "");

#ifdef GGML_BITNET_XDNA
    r->npu_dispatch = (uint32_t)(bitnet_xdna_dispatches() - g_disp0);
    const uint64_t ms1 = (uint64_t)(bitnet_xdna_dispatch_ms() * 1e6);
    r->npu_ns = ms1 > g_npu_ms0 ? ms1 - g_npu_ms0 : 0;
#endif
}

static void json_esc(FILE *f, const char *s) {
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') fputc('\\', f);
        fputc(*s, f);
    }
}

static void bnp_dump(void) {
    if (g_on <= 0 || !g_rec) return;
    size_t n = atomic_load(&g_n);
    if (n > BNP_MAX_RECORDS) n = BNP_MAX_RECORDS;
    FILE *f = fopen(g_path, "w");
    if (!f) { fprintf(stderr, "[bnp] cannot write %s\n", g_path); return; }
    for (size_t i = 0; i < n; i++) {
        const struct bnp_rec *r = &g_rec[i];
        fprintf(f, "{\"graph\":%d,\"idx\":%d,\"op\":\"%s\",\"name\":\"",
                r->graph, r->idx, ggml_op_name((enum ggml_op)r->op));
        json_esc(f, r->name);
        fprintf(f, "\",\"t0_us\":%.3f,\"t1_us\":%.3f,\"dur_us\":%.3f",
                r->t0_ns / 1e3, r->t1_ns / 1e3, (r->t1_ns - r->t0_ns) / 1e3);
        fprintf(f, ",\"npu_us\":%.3f,\"npu_dispatch\":%u",
                r->npu_ns / 1e3, r->npu_dispatch);
        fprintf(f, ",\"src0_type\":%d,\"ne\":[%lld,%lld,%lld,%lld],\"src\":[",
                r->src_type0, (long long)r->ne[0], (long long)r->ne[1],
                (long long)r->ne[2], (long long)r->ne[3]);
        for (int s = 0; s < BNP_NSRC; s++) {
            if (s) fputc(',', f);
            fputc('"', f);
            json_esc(f, r->src[s]);
            fputc('"', f);
        }
        fprintf(f, "]}\n");
    }
    fclose(f);
    fprintf(stderr, "[bnp] wrote %zu node records to %s\n", n, g_path);
}
