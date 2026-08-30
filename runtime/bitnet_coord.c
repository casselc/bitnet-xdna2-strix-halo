#include "bitnet_coord.h"
#include <string.h>
#include <stdio.h>

static const char *g_reason = "";
const char *bitnet_coord_last_mismatch(void) { return g_reason; }

#define FAIL(msg) do { g_reason = (msg); return 0; } while (0)

int bitnet_model_coord_eq(const bitnet_model_coord *a, const bitnet_model_coord *b) {
    if (strncmp(a->gguf_sha256, b->gguf_sha256, sizeof(a->gguf_sha256)) != 0)
        FAIL("checkpoint sha256 differs");
    if (a->tokenizer_hash != b->tokenizer_hash) FAIL("tokenizer differs");
    if (a->n_embd  != b->n_embd)  FAIL("n_embd differs");
    if (a->n_layer != b->n_layer) FAIL("n_layer differs");
    if (a->n_head  != b->n_head)  FAIL("n_head differs");
    if (a->n_head_kv != b->n_head_kv) FAIL("n_head_kv differs");
    if (a->n_vocab != b->n_vocab) FAIL("n_vocab differs");
    if (a->n_ctx_train != b->n_ctx_train) FAIL("training context length differs");
    /* RoPE is compared bitwise on purpose: a theta that differs in the last ulp
     * still yields different position embeddings, so "close enough" is wrong. */
    if (memcmp(&a->rope_theta, &b->rope_theta, sizeof(float)) != 0)
        FAIL("rope_theta differs");
    if (memcmp(&a->rms_norm_eps, &b->rms_norm_eps, sizeof(float)) != 0)
        FAIL("rms_norm_eps differs");
    if (a->ggml_ftype != b->ggml_ftype) FAIL("quantization ftype differs");
    if (a->act_parallel != b->act_parallel)
        FAIL("ACT_PARALLEL differs -- the on-disk I2_S packing layout is not the same");
    g_reason = "";
    return 1;
}

int bitnet_context_coord_eq(const bitnet_context_coord *a, const bitnet_context_coord *b) {
    if (!bitnet_model_coord_eq(&a->model, &b->model)) return 0;
    if (a->n_prefix_tokens != b->n_prefix_tokens) FAIL("prompt prefix length differs");
    if (a->prefix_hash != b->prefix_hash) FAIL("prompt token prefix differs");
    g_reason = "";
    return 1;
}

int bitnet_kv_compatible(const bitnet_kv_coord *kv, const bitnet_context_coord *want) {
    if (!bitnet_context_coord_eq(&kv->ctx, want)) return 0;
    if (kv->n_kv_tokens < want->n_prefix_tokens)
        FAIL("KV holds fewer tokens than the requested prefix");
    if (kv->kv_head_dim <= 0 || kv->n_kv_heads <= 0) FAIL("KV geometry is unset");
    g_reason = "";
    return 1;
}

uint64_t bitnet_model_coord_digest(const bitnet_model_coord *c) {
    /* FNV-1a over the struct's meaningful bytes. Not cryptographic -- this is a
     * logging aid, while the real comparison is field-by-field above. */
    uint64_t h = 1469598103934665603ULL;
    const unsigned char *p = (const unsigned char *)c;
    for (size_t i = 0; i < sizeof(*c); ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}
