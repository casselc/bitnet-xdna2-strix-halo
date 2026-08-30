/* bitnet_coord.h -- identity contract preventing incompatible reuse.
 *
 * The MVP's KV cache never leaves host memory: the offload sits inside ggml's
 * mul_mat, so KV is produced by unmodified llama.cpp in its canonical layout and
 * there is no NPU-side KV to convert. The KVCoordinate below is therefore thin
 * BY CONSTRUCTION, not by omission -- it exists so that a later design which
 * does move KV cannot silently reuse state across incompatible runs.
 *
 * What the identity must cover, per the milestone contract:
 *   - model/checkpoint identity      -> gguf_sha256
 *   - tokenizer identity             -> tokenizer_hash
 *   - RoPE / config identity         -> rope_theta, dims, layer/head counts
 *   - quantization configuration     -> ggml_ftype + the ACT_PARALLEL layout flag
 *   - prompt token prefix            -> ContextCoordinate::prefix_hash
 *
 * The backend is deliberately NOT part of the model identity: :cpu and
 * :hybrid-npu-cpu must produce the same ModelCoordinate, or backend choice would
 * become a model property and the fallback guarantee would be meaningless.
 */
#ifndef BITNET_COORD_H
#define BITNET_COORD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char     gguf_sha256[65];   /* checkpoint identity */
    uint64_t tokenizer_hash;    /* vocab + merges + special ids */
    int32_t  n_embd, n_layer, n_head, n_head_kv, n_vocab, n_ctx_train;
    float    rope_theta, rms_norm_eps;
    int32_t  ggml_ftype;        /* 40 = MOSTLY_I2_S */
    int32_t  act_parallel;      /* selects the on-disk I2_S packing layout */
} bitnet_model_coord;

typedef struct {
    bitnet_model_coord model;
    uint64_t prefix_hash;       /* hash of the prompt token prefix */
    int32_t  n_prefix_tokens;
} bitnet_context_coord;

typedef struct {
    bitnet_context_coord ctx;
    int32_t  n_kv_tokens;
    int32_t  kv_head_dim, n_kv_heads;
    int32_t  kv_type;           /* ggml type of the K/V tensors */
} bitnet_kv_coord;

/* Which backend produced a view. Never part of model identity. */
typedef enum { BITNET_BACKEND_CPU = 0, BITNET_BACKEND_HYBRID_NPU_CPU = 1 } bitnet_backend;

/* 1 if the two coordinates describe the same logical model. */
int bitnet_model_coord_eq(const bitnet_model_coord *a, const bitnet_model_coord *b);
int bitnet_context_coord_eq(const bitnet_context_coord *a, const bitnet_context_coord *b);

/* 1 if `kv` may legitimately be continued under `want`. Requires identical model
 * and context identity and a KV layout the consumer understands. */
int bitnet_kv_compatible(const bitnet_kv_coord *kv, const bitnet_context_coord *want);

/* Stable 64-bit digest, for logging into evidence. */
uint64_t bitnet_model_coord_digest(const bitnet_model_coord *c);

/* Human-readable reason the last *_eq / _compatible call returned 0. */
const char *bitnet_coord_last_mismatch(void);

#ifdef __cplusplus
}
#endif
#endif
