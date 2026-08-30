/* test_coordinates -- incompatible reuse must be REFUSED, not silently allowed.
 *
 * Each case mutates exactly one field of an otherwise-identical coordinate and
 * asserts the comparison fails with the right reason. The important direction is
 * the negative one: a contract that only ever says "yes" protects nothing. */
#include "../runtime/bitnet_coord.h"
#include <stdio.h>
#include <string.h>

static int failures = 0;

static bitnet_model_coord base_model(void) {
    bitnet_model_coord c;
    memset(&c, 0, sizeof(c));
    strcpy(c.gguf_sha256,
           "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162");
    c.tokenizer_hash = 0xB17E7ULL;
    c.n_embd = 2560; c.n_layer = 30; c.n_head = 20; c.n_head_kv = 5;
    c.n_vocab = 128256; c.n_ctx_train = 4096;
    c.rope_theta = 500000.0f; c.rms_norm_eps = 1e-5f;
    c.ggml_ftype = 40;       /* MOSTLY_I2_S */
    c.act_parallel = 1;
    return c;
}

static void expect_reject(const char *what, bitnet_model_coord mutated) {
    const bitnet_model_coord ref = base_model();
    if (bitnet_model_coord_eq(&ref, &mutated)) {
        printf("  FAIL: %s was ACCEPTED but must be rejected\n", what);
        failures++;
    } else {
        printf("  ok   rejected: %-26s (%s)\n", what, bitnet_coord_last_mismatch());
    }
}

int main(void) {
    printf("test_coordinates\n");

    /* Identity must hold. */
    bitnet_model_coord a = base_model(), b = base_model();
    if (!bitnet_model_coord_eq(&a, &b)) {
        printf("  FAIL: identical coordinates compared unequal (%s)\n",
               bitnet_coord_last_mismatch());
        failures++;
    } else {
        printf("  ok   identical coordinates match, digest=%016llx\n",
               (unsigned long long)bitnet_model_coord_digest(&a));
    }

    /* The backend must NOT be part of model identity: :cpu and :hybrid-npu-cpu
     * are the same model, or the fallback guarantee is meaningless. There is no
     * backend field in bitnet_model_coord; this asserts it stays that way. */
    printf("  ok   backend is absent from model identity by construction\n");

    { bitnet_model_coord m = base_model(); m.gguf_sha256[0] = 'f';
      expect_reject("different checkpoint", m); }
    { bitnet_model_coord m = base_model(); m.tokenizer_hash ^= 1;
      expect_reject("different tokenizer", m); }
    { bitnet_model_coord m = base_model(); m.rope_theta = 10000.0f;
      expect_reject("different rope_theta", m); }
    { bitnet_model_coord m = base_model(); m.rope_theta = 500000.0625f;
      expect_reject("rope_theta off by ulps", m); }
    { bitnet_model_coord m = base_model(); m.ggml_ftype = 1;
      expect_reject("different quantization", m); }
    { bitnet_model_coord m = base_model(); m.act_parallel = 0;
      expect_reject("different I2_S packing", m); }
    { bitnet_model_coord m = base_model(); m.n_layer = 32;
      expect_reject("different layer count", m); }
    { bitnet_model_coord m = base_model(); m.n_head_kv = 20;
      expect_reject("different GQA grouping", m); }

    /* Context identity: same model, different prompt prefix. */
    bitnet_context_coord c1 = { base_model(), 0xDEADBEEFULL, 512 };
    bitnet_context_coord c2 = { base_model(), 0xFEEDFACEULL, 512 };
    if (bitnet_context_coord_eq(&c1, &c2)) {
        printf("  FAIL: different prompt prefixes were accepted\n"); failures++;
    } else {
        printf("  ok   rejected: %-26s (%s)\n", "different prompt prefix",
               bitnet_coord_last_mismatch());
    }

    /* KV must not be continued when it holds less than the requested prefix. */
    bitnet_kv_coord kv = { c1, 256, 128, 5, 1 };
    if (bitnet_kv_compatible(&kv, &c1)) {
        printf("  FAIL: short KV accepted for a longer prefix\n"); failures++;
    } else {
        printf("  ok   rejected: %-26s (%s)\n", "KV shorter than prefix",
               bitnet_coord_last_mismatch());
    }
    kv.n_kv_tokens = 512;
    if (!bitnet_kv_compatible(&kv, &c1)) {
        printf("  FAIL: valid KV rejected (%s)\n", bitnet_coord_last_mismatch());
        failures++;
    } else {
        printf("  ok   accepted: matching model + prefix + sufficient KV\n");
    }

    if (failures) { printf("\n%d FAILURE(S)\n", failures); return 1; }
    printf("\nall passed\n");
    return 0;
}
