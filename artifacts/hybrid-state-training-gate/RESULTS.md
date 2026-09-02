# Correction / discriminator pass: hybrid state and controller-SFT

Branch `hybrid-state-training-gate`, cut from the exact tip of
`origin/model-candidate-halo` (`c9915023d5f0dd024ba4177798ea9201b1cf99a8`).
That branch and every other evidence branch are frozen and unmodified; the
corrections below are new commits here, and the superseded evidence is preserved
in place.

| document | covers |
|---|---|
| `RESTORE.md` | Tasks 1-4, 13, 14 — boundary, restore matrix, upstream discriminator, equal-memory spine |
| `TRAINING.md` | Tasks 6-11 — controller-SFT objective, logits, micro-batch, SDPA, LoRA coverage |
| `measurements/` | raw JSON behind every number |

## Verdicts

**State restore** — of the four permitted options, none fits cleanly, so the
closest is stated with the precise reason:

> **HYBRID EXACT-BOUNDARY RESTORE VALID — PRIOR BLOCKER WAS TEST-PROTOCOL
> INDUCED**, with the correction that the inducing factor was **server context
> checkpoints**, not an overlong checkpoint. The restore itself is bit-exact.
>
> But the prior branch's *latency* claim is separately **OVERTURNED**: restore
> yields **no prefix reuse at all** on hybrid models (upstream #28194,
> reproduced), and the only settings that produce reuse also produce numerically
> wrong state. So hybrid multi-domain warm deployment remains **BLOCKED on this
> build** — on reuse, not on correctness.

**Training**:

> **HALO CONTROLLER-SFT THROUGHPUT CHARACTERIZED.** The action-only objective is
> implemented and verified (guaranteed supervised tokens, exact restricted-logit
> equivalence), micro-batch and attention backend are swept, and three model
> sizes are measured.

## Answers to the twelve questions

**1. Was the prior restore defect caused by an accidental +1 generated token in
the checkpoint?** **No — that premise was false.** The +1 is BOS.
`/tokenize add_special=false` gives 1575, `add_special=true` gives 1576 with
first id 1, and the saved state is 1576. `n_predict=0` and `n_predict=1` produce
**byte-identical** files on every model. The boundary was always exact.

**2. If exact-boundary restore still fails, is it explained by missing server
prompt/context-checkpoint metadata?** **Partly, and the diagnosis needs
extending.** #28194 reproduces exactly — `SLOT_RESTORE` restores sequence state
and tokens but never `slot.prompt.checkpoints`, `prompt_clear()` never clears
them, and hybrids fall into the `do_reset` full-reprocess branch. But a second,
independent defect was found: **enabling context checkpoints perturbs hybrid
state on a clean single request with no restore at all** (LFM2.5 0.371,
Qwen3.5-2B 0.271, Qwen3.5-0.8B 0.181 max |Δlogprob|; pure-attention control
exactly 0.0). So the earlier branch's "reference" was itself perturbed.

**3. Does a current or patched llama.cpp restore hybrid state faithfully after a
foreign domain held the slot?** With `-ctxcp 0`, **yes — 0.00000 across all
twelve arms**, including clean-after-foreign. The author's patch for #28194 is
not public (no fork, no PR), so it was not built; the minimal discriminator
instead shows that **persisting checkpoints would make restore fast and still
wrong here**, because the checkpoint mechanism itself does not preserve hybrid
state. A correct upstream fix must address `create_checkpoint`'s partial-state
capture, not only its serialisation.

**4. If fixed, what are the real state-spine decision latencies?** On the
correct baseline today, hybrids re-prefill every turn: **LFM2.5-1.2B
1170-1186 ms**, **Qwen3.5-0.8B 1684-1711 ms**, **Qwen3.5-2B ~2970 ms**, against
the incumbent's **206-214 ms**. If reuse were fixed, the measured ceiling is
**118 ms (LFM2.5-1.2B)** and **143 ms (Qwen3.5-0.8B)** at `-ub 128` — recorded
as the size of the prize, not a current capability.

**5. Actual controller-SFT throughput with action-only loss?** At each model's
best micro-batch: **LFM2.5-1.2B 1791/1771 tok/s** (seq 512/1024),
**Qwen3.5-0.8B 1289/1114**, **Qwen3.5-2B 799** at seq 1024. That is **5-13%
faster** than the superseded full-sequence objective.

**6. How much do restricted logits save?** **21.7%** peak memory on Qwen3.5-0.8B
(6229 → 4880 MiB), **6.7%** on LFM2.5-1.2B (4220 → 3939 MiB) — the 3x gap
tracks the 248k vs 65k vocabulary. Loss and gradient are **exactly** unchanged
(0.00e+00 on both).

**7. Which micro-batch maximises throughput?** **1**, everywhere except
Qwen3.5-0.8B at seq 512 where **2** wins by 12%. Forward time is flat across
micro-batch while backward grows, so the memory headroom does not convert into
throughput. Peak memory is 3-4x lower at mb 1-2.

**8. Does SDPA materially improve training?** **No — it is slower on every
arm**, by 3.2% to 12.8%, worst on the model with fewest attention layers.
`eager` remains both the controlled and the practical choice here.

**9. Measured Qwen3.5-4B and LFM2.5-2.6B rates?** **Not measured.** Qwen3.5-4B
weights were still downloading when this pass closed; LFM2.5-2.6B was not
fetched in HF format. Qwen3.5-2B (798.6 tok/s at seq 1024, 11.6 GiB peak) is the
largest measured point, so the 2-4B tier has one real measurement rather than
none, but the 4B corner is still extrapolated.

**10. At equal ~127 MiB/domain, how much more context?** **~6x.**
LFM2.5-1.2B holds **9,587** spine tokens at 112.66 MiB and Qwen3.5-0.8B **9,014**
at 125.11 MiB, against the incumbent's **1,600** at 117.29 MiB. This is the
strongest architectural argument for the hybrid family — and it is unusable
until reuse works, since each decision at that length is a ~9-10 s full prefill.

**11. Which earlier conclusions survive unchanged?**

- Serialized state sizes and the geometry model (predicted within 1.6%).
- Qwen3.5-0.8B and -2B have byte-identical warm state.
- Qwen3-0.6B/1.7B are the worst candidates for warm state (112 KiB/token).
- BitNet restore is faithful — now strengthened: also insensitive to `-ctxcp`,
  and it is the only model that reuses after restore.
- Nemotron-3-Nano-4B excluded on 0.8 tok/s CPU decode.
- The *relative ordering* of training throughput across candidates.
- Checkpoint/resume bit-exactness (both models, 0.00e+00).
- Coexistence: GPU training costs the controller ~48% TTFT.

**12. Which require correction or narrower scope?**

| claim | change |
|---|---|
| hybrid explicit-checkpoint decisions at 164/188 ms, beating the incumbent | **OVERTURNED** — artifact of stale in-slot checkpoints; real figures are 1176/1695 ms |
| "hybrid state restore is corrupted, changes the decision, does not recover" | **NARROWED** — restore is bit-exact; the corruption was induced by `-ctxcp` |
| fidelity divergences of 0.109-0.460 as restore corruption | **RETRACTED as such** — measured against a perturbed reference; they are perturbed-vs-perturbed |
| "404 / 1619 warm domains @8/32 GiB" | **RELABELLED** equivalent checkpoint-storage capacity, not measured prompt-cache residency |
| LoRA training tok/s (718-1704) | **RELABELLED** FULL-SEQUENCE LM THROUGHPUT; controller-SFT is 5-13% faster |
| "seq 512 is the efficient band" at mb1 only | **EXTENDED** — micro-batch matters; mb1-2 optimal, mb4-8 counterproductive |

## Refreshed hardware shortlist

Behavioural quality remains **unresolved** and is not addressed here.

| model | hardware standing after this pass |
|---|---|
| **BitNet-b1.58-2B** (incumbent) | **Only architecture that works end-to-end today.** Faithful restore, real reuse after restore (1600 tokens, 214 ms), insensitive to `-ctxcp`. Costs 127 MiB/domain and holds only ~1,600 tokens at that budget. |
| **LFM2.5-1.2B** | **Best hybrid on every hardware axis, gated on one upstream bug.** Smallest state (20.2 MiB/domain), ~6x more context at equal budget, fastest training (1791/1771 tok/s) in the least memory. Currently 1176 ms/decision because reuse is broken. |
| **Qwen3.5-0.8B** | Same gate, slightly behind: 39.9 MiB/domain, 5.6x context, 1289/1114 tok/s, Apache-2.0. Its 18.6 MiB fixed DeltaNet floor is why it holds less context than LFM2.5 despite equal per-token cost. |
| Qwen3.5-2B | Dominated on hardware by its 0.8B sibling — identical state, ~1.8x the decision latency, 0.72x the training throughput. Retained only because quality may separate them. |
| Nemotron-3-Nano-4B | Excluded — 0.8 tok/s CPU decode. |

**The decisive question is no longer which model, it is whether the runtime is
fixed.** Every hybrid advantage measured here — 3-6x state density, 6x resident
context, faster training — is real and none of it is usable for multi-domain
warm serving until hybrid prefix reuse is both available and correct.

## Limitations and next steps

**Fixed-token vs fixed-semantic (Task 15).** Everything here is **fixed-token**:
each model receives ~1600 spine + ~135 delta *tokens*, which normalises for
hardware cost and lets architectures be compared. It is **not** the production
question, which is fixed-*semantic*: the same serialized state object, whose
token count then differs by tokenizer. Once a stable Training ABI projection
exists, **the fixed-semantic benchmark must be run before any model is
promoted** — a model with a denser tokenizer could win on real payloads while
losing here, or the reverse.

Also outstanding:

- **Qwen3.5-4B and LFM2.5-2.6B** controller-SFT (Task 10) — not measured.
- **Coexistence under the corrected objective** (Task 16) — not re-run; the
  prior +48% figure was measured under the full-sequence objective at mb1, and
  the corrected objective at mb1-2 has a similar memory and utilisation profile,
  but that is an assumption, not a measurement.
- **A multi-domain 8/32-domain residency test** (Task 5) — deliberately not run,
  because it is meaningless while hybrid reuse is broken.
- **The mechanism behind `-ctxcp` perturbation** — the effect is measured; the
  cause inside `create_checkpoint`'s partial-state capture is inferred from
  source reading and would need instrumentation inside the recurrent memory to
  confirm.
- **Teacher-tier models** — deferred by instruction (Task 17).
