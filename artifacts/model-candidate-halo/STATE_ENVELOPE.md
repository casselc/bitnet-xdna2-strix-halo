# Warm state size and state-cache semantics, per architecture [MEASURED]

The highest-value comparison in this bakeoff, and it splits into two questions
that have different answers:

1. **How many bytes does one warm domain cost?** Hybrid architectures win, by
   3.2x to 6.3x against the BitNet incumbent.
2. **Can that state be checkpointed and restored correctly?** Only the
   pure-attention incumbent is bit-faithful. Every hybrid candidate diverges,
   and the divergence grows when a slot has previously held a different domain.

Both were measured, not estimated. State size is taken from
`POST /slots/{id}?action=save` — the bytes actually written to disk — not from
RSS, which folds weights, allocator slack and the prompt-cache pool into one
number and cannot separate token-dependent from fixed state.

All candidates run on the **same pinned llama.cpp build** as the incumbent
controller (`9918 / 390c30775`), which already carries `LLM_ARCH_QWEN35`,
`LLM_ARCH_LFM2`, `LLM_ARCH_LFM2MOE` and `LLM_ARCH_NEMOTRON_H`. No separate
worktree was needed, so no result here is confounded by a runtime difference.

---

## 1. Ordinary prefix reuse does not work for ANY hybrid candidate

This is architectural, and the runtime says so itself
(`tools/server/server-context.cpp:3334`):

> forcing full prompt re-processing due to lack of cache data (likely due to SWA
> or hybrid/recurrent memory)

A recurrent state after N tokens is one fixed-size object with no way to rewind
it to position P < N. A turn that shares only the spine therefore cannot drop
the previous turn's delta, so every turn is a full prefill:

| model | warm TTFT p50, `cache_prompt` | reused tokens | vs its own cold TTFT |
|---|---:|---:|---|
| BitNet-b1.58-2B (pure attention) | **197.4 ms** | ~1600 | 7.2x faster |
| LFM2.5-1.2B | 1187.9 ms | **0** | 1.09x |
| Qwen3.5-0.8B | 1713.7 ms | **0** | 1.12x |
| LFM2.5-2.6B | 2750.6 ms | **0** | 1.04x |
| Qwen3.5-2B | 3008.3 ms | **0** | 1.02x |

Taken alone this would end the bakeoff: every hybrid is 6-15x slower than the
incumbent on the warm workload. It is not the whole story.

## 2. Explicit spine checkpoint/restore recovers it

Checkpoint the state **at the spine boundary** and restore it before each delta.
The common prefix then equals the entire restored state, so nothing has to be
removed and the invalid rollback never happens.

| model | restore p50 | TTFT p50 | reused | decision p50 (restore+total) |
|---|---:|---:|---:|---:|
| **LFM2.5-1.2B** | 3.2 ms | **114.8 ms** | 1572 | **164.4 ms** |
| **Qwen3.5-0.8B** | 4.7 ms | 144.0 ms | 1621 | **188.2 ms** |
| BitNet-b1.58-2B *(native reuse)* | — | 199.1 ms | 1601 | **263.2 ms** |
| LFM2.5-2.6B | 4.9 ms | 245.7 ms | 1611 | 352.5 ms |
| Qwen3.5-2B | 4.5 ms | 261.7 ms | 1621 | 350.0 ms |

For Qwen3.5-2B that is **3008 ms -> 262 ms**, an 11.5x recovery. Two candidates
then beat the incumbent's best path per decision: LFM2.5-1.2B by **1.60x** and
Qwen3.5-0.8B by **1.40x**.

The incumbent is quoted on its native server-side reuse because that is its best
path. Forced down the explicit route it is a **regression** — a 117 MiB restore
costs 13.0 ms and the decision lands at 290.5 ms, worse than the 263.2 ms it
already gets for free. Explicit checkpointing is a workaround for architectures
that cannot use prefix reuse, not an improvement in itself.

The operational model differs and the comparison should not hide it. BitNet's
reuse is automatic and server-managed; the hybrids require the client to own a
checkpoint file per domain and restore it explicitly. That is exactly the
"authoritative state is truth, model KV is disposable" architecture this project
already uses — but it is client work that does not exist today.

## 3. State size: geometry predicts it, and measurement confirms

Predicted from the config before measuring, then checked:

| model | attn KV/token | fixed recurrent | predicted @~1750 | **measured** | error |
|---|---:|---:|---:|---:|---:|
| LFM2.5-1.2B | 12.0 KiB | 0.117 MiB | 20.16 MiB | **20.23 MiB** | 0.3% |
| LFM2.5-2.6B | 16.0 KiB | 0.258 MiB | 27.59 MiB | **27.71 MiB** | 0.4% |
| Qwen3.5-0.8B | 12.0 KiB | 18.63 MiB | 39.30 MiB | **39.94 MiB** | 1.6% |
| Qwen3.5-2B | 12.0 KiB | 18.63 MiB | 39.30 MiB | **39.94 MiB** | 1.6% |
| BitNet-b1.58-2B | 75.0 KiB | 0 | 127.08 MiB | **127.18 MiB** | 0.1% |

Measured marginal cost per delta token — 12.02, 16.02, 11.76, 11.76 KiB — lands
on the geometric prediction, so the model is sound and can be trusted for
candidates that were not served locally.

The incumbent is the strongest check on the whole measurement chain: run through
the **identical** harness it returns **133,353,352 B = 127.18 MiB/domain** and a
marginal **75.02 KiB/token**, against the frozen `controller-state-envelope`
values of 127.1 MiB and 75.0 KiB/token. Those numbers were derived on another
branch by a different route, so reproducing them to 0.1% says the serialization
measurement is sound.

**Two findings that only appear once the components are separated:**

- **Qwen3.5-0.8B and Qwen3.5-2B have byte-identical warm state** (40,209,124 B
  spine; 41,883,556 B spine+delta). Both are 24 layers / 6 full-attention / 18
  DeltaNet with `head_dim 256` and 2 KV heads; only `hidden_size` differs, and
  hidden size does not enter the state. **Choosing the smaller Qwen3.5 buys
  speed and training cost, not warm-state capacity.**
- **Qwen3.5 pays an 18.6 MiB floor per domain** for 18 fp32 DeltaNet states
  (16 heads x 128 x 128 x 4 B x 18 layers), independent of sequence length. At a
  1750-token spine that is 47% of its footprint. LFM2.5's short-conv state is
  ~0.1-0.26 MiB — three orders of magnitude smaller — because a conv cache is
  O(kernel x dim), not O(heads x d_k x d_v).

### Warm domains per cache budget

Derived from the measured spine+delta bytes:

| model | MiB/domain | @8 GiB | @16 GiB | @32 GiB | vs BitNet |
|---|---:|---:|---:|---:|---:|
| **LFM2.5-1.2B** | 20.23 | **404** | 809 | **1619** | **6.3x** |
| LFM2.5-2.6B | 27.71 | 295 | 591 | 1182 | 4.6x |
| Qwen3.5-0.8B / 2B | 39.94 | 205 | 410 | 820 | 3.2x |
| BitNet-b1.58-2B | 127.08 | 64 | 128 | 257 | 1.0x |
| Qwen3-0.6B / 1.7B | 189.77 | 43 | 86 | 172 | 0.67x |

The current *controls* Qwen3-0.6B and Qwen3-1.7B are the **worst** candidates
for warm state at 112 KiB/token — worse than the BitNet incumbent — because they
are dense-attention with 8 KV heads across 28 layers. Their small parameter count
does nothing for residency.

## 4. Restore correctness: the incumbent is faithful, the hybrids are not

Latency cannot answer this. A restore that silently recomputes looks identical
in every timing chart, and argmax equality is a blunt instrument — several
candidates emit the same short action regardless of state. So the probe compares
the **full next-token logprob distribution** after `restore(spine) + delta`
against processing `spine + delta` from an empty slot, and varies only what the
slot held beforehand.

`max |Δ logprob|` over the top-20, versus a full recompute:

| model | architecture | clean slot | after a FOREIGN domain | cache_n stable? |
|---|---|---:|---:|---|
| **BitNet-b1.58-2B** | 30 attn, pure | **0.00000** | **0.00000** | yes (1600) |
| Qwen3.5-0.8B | 6 attn + 18 DeltaNet | 0.00000 | **0.10913** | no |
| LFM2.5-1.2B | 6 attn + 10 conv | 0.00000 | **0.18302** | no |
| Qwen3.5-2B | 6 attn + 18 DeltaNet | **0.20806** | **0.41244** | no |
| LFM2.5-2.6B | 8 attn + 22 conv | **0.30381** | **0.46035** | no |
| Nemotron-3-Nano-4B | 4 attn + 21 mamba2 | **0.13394** | inconclusive † | no |

† Nemotron's two polluted arms returned `cache_n = 0` — they fell back to a full
reprocess and never exercised the restore, so their apparent 0.00000 is the
"passes for the wrong reason" case this probe exists to catch and is reported as
inconclusive rather than clean. Its clean-slate 0.13394 is a real measurement.

**The BitNet row is the control that makes the rest interpretable.** It is
bit-exact on all four arms with `cache_n` pinned at 1600, which rules out
batch-shape numerical noise as the explanation and shows the protocol can
observe an exact restore. Every nonzero above is therefore a real defect, not
measurement error.

It is not benign. On LFM2.5-1.2B it **changes the emitted decision**: restoring
the identical checkpoint and asking the identical question returns `HOLD` if the
slot was clean and a different action if the slot had previously held another
domain — and it does **not** recover on a subsequent clean restore:

| prior slot content | cache_n | output |
|---|---:|---|
| none | 1572 | `' 1\n```\n\nOUTPUT\n '` ✅ |
| its own prefix | 1572 | `' 1\n```\n\nOUTPUT\n '` ✅ |
| **a foreign domain** | 1568 | `' 1\nRECOMMENDATION'` ❌ |
| **none, after the above** | 1568 | `' 1\nRECOMMENDATION'` ❌ persists |

The `cache_n` shift is the observable fingerprint: the runtime settles on a
4-token-shorter common prefix and re-evaluates those tokens from a recurrent
state that cannot legally be rolled back to that position.

**Verdict per candidate:**

| model | verdict |
|---|---|
| BitNet-b1.58-2B | state restore **FAITHFUL** |
| Qwen3.5-0.8B, LFM2.5-1.2B | exact on a clean slot; **corrupted by a prior foreign domain** |
| Qwen3.5-2B, LFM2.5-2.6B | **never bit-exact**, even on a clean slot |
| Nemotron-3-Nano-4B | see below — excluded on throughput before this mattered |

This does **not** say the models are unusable. It says that in **this runtime**,
multi-domain warm-state deployment on a hybrid model requires either one slot
per domain with no reuse across domains, or an upstream fix. A single-domain
deployment is unaffected. The mission's label applies:

> **STATE-SPINE MULTI-DOMAIN DEPLOYMENT BLOCKED IN THIS RUNTIME** for every
> hybrid candidate tested, on correctness rather than on speed.

## 5. Nemotron-3-Nano-4B: excluded on CPU decode

Loads and generates correctly on the pinned build, but decodes at **0.8 tok/s**
on CPU while consuming ~1550% CPU — roughly 50-100x slower than every other
candidate (39.5-99.9 tok/s). The Mamba2 CPU path in this build is not optimised.
Its state geometry is otherwise unremarkable for this workload: 16 KiB/token
across only 4 attention layers, but a ~40 MiB fixed Mamba2 floor
(96 heads x 80 x 128 x 21 layers), giving ~67 MiB/domain — better than BitNet,
worse than either hybrid family.

Per the mission's stop condition this is recorded as a blocker rather than
pursued: a warm controller emitting 4 tokens per decision would spend 5 seconds
per decision on decode alone.

## 6. Limitations

- Argmax equivalence is reported alongside logprob divergence because several
  candidates emit the same short action on this synthetic workload; the logprob
  comparison is the load-bearing evidence, and the argmax column should not be
  read as independent confirmation.
- `max |Δ logprob|` is over the top-20 tokens common to both distributions. A
  divergence confined to the tail would be understated.
- One spine length (~1600 tokens) and one delta (~135) were swept for
  correctness. The 4-token `cache_n` shift may be length-dependent.
- The checkpoint-restore path was measured with the state file in page cache.
  A cold-disk restore of a 20-40 MiB file would cost more than the 3-5 ms here.
- Quantisation differs slightly across candidates (Q4_K_M, except Qwen3.5-0.8B
  which is Q4_0 as published by `ggml-org`). llama.cpp's KV cache is f16
  regardless, so state sizes are unaffected; decode throughput is mildly so.
