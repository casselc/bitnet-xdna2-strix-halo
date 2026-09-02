# Exact-spine runtime gate — final synthesis

Branch `exact-spine-runtime-gate`, cut from the exact tip of
`origin/hybrid-state-training-gate` (`efab39fc5784e85923700bff5f71a224385e9df4`).
All prior evidence branches frozen and unmodified. Corrections are new commits.

| document | covers |
|---|---|
| `RUNTIME.md` | Tasks 1-9 — exact-spine contract, restore matrix, LFM2 split defect, alternating domains, checkpoint instrumentation, memory |
| `TRAINING.md` | Tasks 10-16 — gradient equivalence, seq 2048, campaign units, Conv1d cost |
| `measurements/` | raw JSON |
| `patch/` | the experimental fast path, recorded and **not** promoted |

---

## PRIMARY RUNTIME VERDICT

> ## EXACT-SPINE RECURRENT FAST PATH VALIDATED
>
> — for **Qwen3.5-0.8B**, and **without any runtime change**.

Both acceptance conditions met, on the **stock unpatched pinned binary**:

| | |
|---|---|
| reuse | **1621 of 1625** spine tokens; `prompt_n` = 137, the delta only |
| speed | **129.2 ms** vs 1771.1 ms full recompute — **13.7x** |
| correctness | **max \|Δlogprob\| = 0.000000** against full recompute |
| durability | **100 alternating turns**, 4 tagged domains, one slot: 0 mismatches, 0 contamination |
| scale | 8 and 32 domains: TTFT p50 130.8-133.7 ms, no degradation |

**Scope, stated plainly.** The verdict is per-model, not per-architecture:

| model | exact-spine path |
|---|---|
| **Qwen3.5-0.8B** | **VALIDATED** |
| BitNet-b1.58-2B | works (control) — 282.8 ms at a 1600-token spine, bit-exact |
| LFM2.5-1.2B | **UNSAFE on this build** — split evaluation diverges (0.372713) |
| Qwen3.5-2B | **UNSAFE on this build** — diverges (0.271305) and flips top-1 |

## The central correction

**The hybrid reuse blocker was ours, not llama.cpp's.** Previous passes saved the
spine by sending its *text*. Text prefixing is not token prefixing — BPE merges
across the spine/delta seam, so the saved state covered one token *more* than the
query actually shared. That asks for a one-token rollback, which recurrent memory
cannot do, so the server discarded the whole prefix: `cache_n = 0`.

Save at the **token-exact** common prefix instead — feeding token ids, not text —
and the stock server reuses everything and is bit-exact. Qwen3.5-0.8B loses 4
tokens to the seam, LFM2.5-1.2B one.

Consequently the experimental `--exact-spine-fast-path` built for this pass
**is not needed**: measured identical with it on, off, and on the original pinned
binary, because once the boundary is exact the stock path already skips the
rollback search. It is recorded in `patch/` and explicitly not promoted.

## Secondary questions

**What does the public sidecar patch fix?** **Not tested** — see below. Its
premise (hybrid restore cannot reuse) is now known to be avoidable without it, so
the arm was deprioritised rather than run.

**Does `create_checkpoint` mutate live state?** **No — measured directly.**
Fingerprinting the live sequence before and after capture (with a
hash-twice-before stability control) gives byte-identical results on both hybrid
families. This **corrects** the previous branch's inference that capture perturbs
state. Combined with the split-evaluation finding, the divergence is most
consistent with **chunked evaluation**, with checkpoints changing how a prefill
is chunked — stated as what the measurements jointly support, not as a
demonstrated mechanism.

**LFM2.5-1.2B is the surprise.** It diverges under split evaluation *alone* — no
checkpoints, no save, no restore — deterministically by split position across
separate server processes (exact at 400/800/1200, divergent at 1010/1575/1700).
That is an LFM2 chunked-evaluation defect and it lands on our production
boundary.

---

## TRAINING VERDICT

> ## HALO CONTROLLER-SFT THROUGHPUT CHARACTERIZED

At production length (seq 2048), action-only loss, restricted logits, eager, mb1:

| model | input tok/s | examples/s | act-tok/s | peak | **100k decisions** |
|---|---:|---:|---:|---:|---:|
| **LFM2.5-1.2B** | 1474.0 | **0.72** | 1.9 | 12.8 GiB | **38.6 h** |
| Qwen3.5-0.8B | 816.2 | 0.40 | 0.6 | 15.6 GiB | 69.4 h |
| Qwen3.5-2B | 650.8 | 0.32 | 0.5 | 19.9 GiB | 86.8 h |
| Qwen3.5-4B | 256.5 | 0.13 | 0.2 | 46.6 GiB | 213.7 h |

**The units matter.** At seq 2048 only 0.079-0.127% of tokens are supervised, so
"100M input tokens" is about **49,000 decisions**, not a large labelled set. A
100k-decision campaign is 1.6 days on the fastest candidate and nine days at 4B —
roughly 2x what the input-token framing suggests.

Also established:

- **Restricted logits are bit-identical**, now properly: per-element over every
  trainable tensor (228 / 7.27 M and 184 / 11.11 M elements, max\|Δ\| = 0) *and*
  every parameter after a full AdamW step. Saves 24.9% / 9.4% peak memory. The
  previous pass compared only an aggregate norm, which did not support the claim.
- **Reaching the recurrent pathway is cheap**: unfreezing every `Conv1d` alongside
  LoRA costs LFM2.5-1.2B +61,440 params (+0.55%), 4.0% throughput, no measurable
  memory; Qwen3.5-0.8B +442,368 (+6.1%), 14.2%, +3% memory. No quality claim.
- **Coexistence, re-measured under the corrected objective**: controller TTFT p50
  195.6 → 309.8 ms (**+58.4%**), p95 +55.1%, req/s −44.0%, while training is
  unaffected (1470.5 vs 1474.0 tok/s). A third arm returns to 200.3 ms, so this
  is interference and not drift. This **supersedes** the +48% measured on the
  full-sequence objective: the corrected workload costs the controller *more*.
  Wording deliberately: this is consistent with shared-memory contention; no
  hardware counters were collected, so no causal claim is made.

## Correction chain

| claim | origin | status |
|---|---|---|
| "hybrid restore yields no prefix reuse (#28194)" | `hybrid-state-training-gate` | **CAUSE REATTRIBUTED** — a token-exact boundary gives full reuse on the stock binary; the symptom was real, the cause was our text-level spine |
| "restore is bit-exact at ctxcp 0" | same | **TRUE BUT VACUOUS AS EVIDENCED** — measured when `cache_n = 0`, so restored state was never exercised. Now supported: `R vs S = 0.000000` |
| "context-checkpoint capture perturbs hybrid state" | same | **REFUTED** — capture is byte-identical; divergence lies in chunked evaluation |
| "hybrid multi-domain deployment blocked on reuse" | same | **NARROWED** — not blocked for Qwen3.5-0.8B; remains blocked for LFM2.5-1.2B and Qwen3.5-2B on a *different* defect |
| coexistence +48% | same | **SUPERSEDED** — +58.4% under the corrected objective |
| restricted-logit "equivalence" | same | **NOW JUSTIFIED** — per-element and post-step, previously only an aggregate norm |
| "6x server-resident context density" | `model-candidate-halo` lineage | remains **SEQUENCE-STATE DENSITY**; the exact-spine path adds no sidecar, so it carries over unchanged |

## Not done, and why

- **Task 7, the public sidecar patch (`headbouyJB/llama.cpp` `c369f24`)** — not
  fetched or built. Its purpose is to make restore reuse work; this pass showed
  reuse already works without it once the boundary is token-exact, so the arm lost
  its decision value. It would still be the right experiment for the
  *non*-exact-boundary case, which Samizdat does not have.
- **LFM2.5-2.6B and BitNet BF16 controller-SFT** — chained to their downloads;
  see the final commit for whichever completed. The BitNet BF16 repo's `auto_map`
  points at `configuration_bitnet.py` / `modeling_bitnet.py` that are **absent
  from the repo**, so `trust_remote_code` fails; transformers 5.16.1 has a native
  `bitnet` implementation that loads correctly once `auto_map` is removed.
- **The LFM2 split-evaluation mechanism** — reproduced and bounded, not isolated
  to a kernel.
- **Fixed-semantic benchmarking** — still outstanding from the previous branch and
  still required before any model is promoted.

## Recommendation

The runtime question is **resolved for the one candidate that matters most**, and
resolved in the cheapest possible way: a protocol change in our own harness, no
patched llama.cpp, no promoted runtime change. Hardware exploration can stop.

The open items are behavioural, not hardware: whether Qwen3.5-0.8B is *good
enough*, and whether LFM2.5's superior state density and training speed justify
either waiting for an upstream LFM2 fix or accepting full re-prefill at 1176 ms.
Both belong to the off-box evaluation.
