# Controller model candidates: hardware suitability on Strix Halo

**This document does not name a controller.** Hardware measurement cannot select
one, and the frozen behavioural/semantic eval suite that could does not exist
yet. What follows is a hardware screen: which candidates this box can serve
warm, hold in RAM, restore correctly, and train — and which it cannot.

Branch: `model-candidate-halo`, created from
`origin/controller-state-envelope` @ `60230b59adb30c8bf97f1419cd804d5371037d40`.
`origin/halo-training-smoke` @ `f4c27323f819e7d62290ac88870cbb7ae42d7f0a` was
**not merged**; `tools/train_scaling.py` and `tools/train_resume.py` are new work
that supersedes its `tools/train_smoke.py`, cited in place.

Companion documents:

| document | covers |
|---|---|
| `REGRESSION.md` | Task 0 — system ROCm 7.1 disturbed nothing |
| `TRAINING_ENV.md` | Task 1 — reproducible PyTorch/HSA invocation |
| `STATE_ENVELOPE.md` | Tasks 4, 5, 6 — state size, warm latency, restore correctness |
| `TRAINING.md` | Tasks 7, 8 — LoRA scaling and real checkpoint resume |
| `candidates.csv` | Task 2 — the manifest, with revisions and licences |
| `state_geometry.csv` | derived per-architecture state geometry |
| `measurements/` | raw JSON behind every number here |

---

## The headline

**The two things a controller must do well on this box are in tension, and no
candidate wins both.**

- Hybrid architectures hold **3.2x to 6.3x more warm domains** in the same RAM,
  and two of them make a decision **faster** than the incumbent.
- But **only the pure-attention incumbent restores its state correctly under
  domain switching.** The two smallest hybrids (Qwen3.5-0.8B, LFM2.5-1.2B) are
  bit-exact on a *clean* slot and then diverge once that slot has held a
  different domain; the larger ones (Qwen3.5-2B, LFM2.5-2.6B, Nemotron-3-Nano-4B)
  are never bit-exact even on a clean slot. On LFM2.5-1.2B the divergence
  **changes the emitted decision**, and does not recover.

So the residency win is real and large, and it is currently unbankable for
multi-domain deployment without either a runtime fix or one slot per domain —
which is precisely the configuration that a high-domain-count deployment cannot
afford.

## Hardware classification

| model | role | classification | why |
|---|---|---|---|
| **LFM2.5-1.2B-Instruct** | candidate | **GOOD HALO FIT** | fastest decision (164 ms), smallest state (20.2 MiB), fastest training; restore corrupted by a prior foreign domain |
| **Qwen3.5-0.8B** | candidate | **GOOD HALO FIT** | 188 ms decision, exact resume, bit-exact restore on a clean slot, apache-2.0; 18.6 MiB fixed state floor |
| **BitNet-b1.58-2B-4T** | incumbent | **WORKABLE / reference** | the ONLY faithful state restore, and automatic server-side reuse; costs 127 MiB/domain |
| Qwen3.5-2B | candidate | **WORKABLE** | byte-identical state to 0.8B but 1.86x slower per decision — dominated on hardware alone |
| LFM2.5-2.6B | candidate | **WORKABLE** | 4.6x residency, but never bit-exact on restore and 2.1x slower than its 1.2B sibling |
| Qwen3-0.6B / 1.7B | current controls | **POOR HALO FIT** | 112 KiB/token — *worse* warm state than the incumbent; small parameter count buys nothing |
| Nemotron-3-Nano-4B | candidate | **RUNTIME SUPPORT BLOCKED** | 0.8 tok/s CPU decode (50-100x slower than every peer) |
| Muse-Glimmer-30B | teacher | **RUNTIME SUPPORT BLOCKED** | `muse_glimmer` absent from the pinned llama.cpp build |
| Qwen3.8-27B | teacher | **NOT MEASURED** | declares `qwen3_5`, so likely supported; 16.35 GiB download did not complete |
| Nemotron-3.5-Lightning-30B | teacher | **NOT MEASURED** | CUDA-first; no GGUF found at probe time |

## Warm state-spine scorecard

Workload: ~1600-token stable spine, ~135-token changing delta, 4-token
deterministic output. Every model's spine is calibrated **in tokens** with its
own tokenizer, so a large vocabulary is not handed a shorter prompt. All served
on the same pinned llama.cpp (`9918 / 390c30775`), 4 threads, `-tb 16`,
`b/ub 4096`.

| model | cold TTFT | best warm TTFT | **decision p50** | CPU decode | state/domain | @8 GiB | @32 GiB | restore |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **LFM2.5-1.2B** | 1301 ms | 114.8 ms † | **164.4 ms** | 85.6 t/s | **20.23 MiB** | **404** | **1619** | ⚠ corrupted by foreign domain |
| **Qwen3.5-0.8B** | 1917 ms | 144.0 ms † | **188.2 ms** | **99.9 t/s** | 39.94 MiB | 205 | 820 | ⚠ corrupted by foreign domain |
| BitNet-b1.58-2B | 1675 ms | **199.1 ms** | 263.2 ms | 60.5 t/s | 127.18 MiB | 64 | 257 | ✅ **bit-exact** |
| Qwen3.5-2B | 3083 ms | 261.7 ms † | 350.0 ms | 48.6 t/s | 39.94 MiB | 205 | 820 | ✗ never exact |
| LFM2.5-2.6B | 2858 ms | 245.7 ms † | 352.5 ms | 39.5 t/s | 27.71 MiB | 295 | 1182 | ✗ never exact |
| Nemotron-3-Nano-4B | — | — | — | **0.8 t/s** | 108.0 MiB ‡ | 75 | 303 | ✗ not exact (0.134) |

† via **explicit spine checkpoint/restore**. Ordinary `cache_prompt` prefix reuse
does not work for any hybrid — the runtime forces a full re-prefill — which
costs them 1188-3008 ms per warm turn instead. The checkpoint path is what makes
them competitive at all; it is client work that does not exist today.
‡ Nemotron's serialized state is sized by the configured slot context, not by
tokens used, so it does not shrink with a shorter spine.

**Decision latency includes the restore** (3.2-4.9 ms) for the hybrids. The
incumbent is quoted on its native server-side reuse, which is its best path;
forced down the explicit route it regresses to 290.5 ms.

Every row above comes from a single run of one harness per model, so the columns
are internally consistent. The incumbent's separately-measured regression check
(`bitnet_ref.json`, run under `service_ctl.sh` with `--cache-ram 8192`) gives
197.4 ms warm TTFT and 67.5 tok/s decode — the small differences from the row
above are configuration, not drift.

## Training scorecard

Fixed ~4096 tokens per optimizer update, BF16, LoRA r=16, matched target
coverage (~0.94-0.96% trainable in both).

| model | tok/s @512 | @1024 | @2048 | peak GPU @1024 | @2048 | trainable | resume |
|---|---:|---:|---:|---:|---:|---:|---|
| **LFM2.5-1.2B** | **1703.8** | **1637.1** | **1347.7** | **7.0 GiB** | **13.9 GiB** | 11.1 M | ✅ **bit-exact** |
| Qwen3.5-0.8B | 1090.2 | 1014.3 | 717.7 | 11.7 GiB | 22.2 GiB | 7.3 M | ✅ **bit-exact** |

LFM2.5-1.2B trains 1.56-1.88x faster than Qwen3.5-0.8B while being a 1.6x larger
model, and needs 1.6x less memory — 6 of 16 blocks carry attention, the rest are
linear-cost short-conv. Throughput peaks at **seq 512** for both. Power is flat
at 95-112 W; nothing is power-limited.

Checkpoint resume: **both** candidates reproduce their continuous run to
**0.00e+00** across all 8 steps after the process is destroyed — no
fresh-optimizer spike. Qwen3.5-0.8B's checkpoint is 83.48 MiB (adapter 27.78 +
optimizer 55.69), LFM2.5-1.2B's is 127.31 MiB (42.40 + 84.91); both hold the
same 2:1 optimizer-to-adapter ratio that an adapter-only save throws away.

## Coexistence: training costs the controller ~48%, and the controller costs training nothing

Measured twice, at seq 1024, against the reference controller configuration:

| | controller TTFT p50 | p95 | controller req/s | training tok/s | package W |
|---|---:|---:|---:|---:|---:|
| controller alone | 199.2 / 200.2 ms | 203.2 / 204.9 ms | 3.769 | — | 109-110 |
| both | 294.9 / 292.0 ms | 303.7 / 303.0 ms | 2.513 | 1022.8 | 120 |
| training alone | — | — | — | 1014.3 | — |
| **delta** | **+48.0% / +45.9%** | **+49.5% / +47.9%** | **−33.3%** | **+0.8%** | +9% |

**There is a material controller penalty and it reproduces** — this is *not* the
"no material penalty resolved" case. The interference is one-directional:
training does not notice the controller, the controller loses a third of its
throughput. `cache_n` is identical (1615) in both arms, so this is contention,
not a cache artifact.

This **supersedes by scope** the `halo-training-smoke` figure of +0.7%: that
branch trained a ~20-example smoke batch of short sequences, where the GPU was
busy only in bursts. At 98-99% sustained GPU utilisation the cost appears. The
frozen record is not rewritten; the scope of its claim is narrowed.

## Concurrency: a second domain is not worth it, for either candidate

Two points only, each client owning its own domain and its own slot — sharing a
domain would measure deduplication rather than concurrency.

| model | c | decisions/s | decision p50 | p95 |
|---|---:|---:|---:|---:|
| LFM2.5-1.2B | 1 | 5.920 | 169.5 ms | 169.9 ms |
| LFM2.5-1.2B | 2 | 6.739 | 290.0 ms | 302.6 ms |
| | | **1.138x** | **1.711x** | **1.781x** |
| Qwen3.5-0.8B | 1 | 5.205 | 192.9 ms | 195.7 ms |
| Qwen3.5-0.8B | 2 | 5.590 | 352.8 ms | 366.5 ms |
| | | **1.074x** | **1.829x** | **1.872x** |

Doubling concurrency buys **7-14% throughput for 71-83% more latency** on both.
This is the same "concurrency ~= 1" shape `service-cotenancy` measured for the
incumbent, reproduced on two different architectures — so it is a property of a
CPU-bound prefill on this part, not of any model. **Run one domain at a time and
scale by adding time, not threads.**

## Low-bit path: LFM2.5 QAD-Q4 costs nothing on this hardware

LiquidAI publishes a **quantisation-aware distilled** `QAD-Q4_0` GGUF alongside
the ordinary `Q4_0`. That is directly relevant here because QAD is an
independent low-bit distillation route, an alternative to the ternary/BitDistill
idea rather than a competitor to it. The hardware question is whether adopting
it costs anything.

LFM2.5-1.2B, same workload, same harness:

| form | state bytes | restore p50 | TTFT p50 | decision p50 |
|---|---:|---:|---:|---:|
| `Q4_0` | 19,555,404 | 4.27 ms | 113.60 ms | 148.59 ms |
| `QAD-Q4_0` | 19,555,404 | 4.48 ms | 113.49 ms | 149.11 ms |
| delta | **0 bytes** | +0.2 ms | **−0.1%** | **+0.3%** |

**Byte-identical state and indistinguishable latency.** Both are `Q4_0` with the
same tensor shapes, so this is the expected result and it is the useful one:
**QAD is free on this box.** Whatever it buys is behavioural, measurable only by
the off-box eval, and there is no hardware reason not to prefer it.

The LFM2.5-**2.6B** `Q4_0`/`QAD-Q4_0` pair was not downloaded, so this rests on
the 1.2B pair alone.

## Answers to the specific questions

1. **Does Qwen3.5-0.8B or 2B materially reduce warm-state RAM vs BitNet?**
   Yes — 39.94 vs 127.18 MiB/domain, a **3.2x** reduction, 205 vs 64 warm
   domains at 8 GiB. But 18.6 MiB of that is a fixed fp32 DeltaNet floor
   (47% of the footprint at a 1750-token spine), so the advantage shrinks for
   shorter spines and grows for longer ones.
2. **Do their DeltaNet states survive save/restore correctly?** **No, not
   reliably.** Qwen3.5-0.8B is bit-exact on a clean slot but diverges
   (|Δlogprob| 0.109) once the slot has held another domain; Qwen3.5-2B is never
   exact (0.208 clean, 0.412 polluted). The BitNet control is 0.00000 on every
   arm, which is what proves these are real defects and not measurement noise.
3. **Does that translate to more warm domains?** Yes arithmetically — 3.2x for
   Qwen3.5, 6.3x for LFM2.5-1.2B — but **not safely**, because the defect in (2)
   is triggered precisely by reusing a slot across domains, which is what a
   high-domain-count deployment does.
4. **Which is fastest on the real workload?** LFM2.5-1.2B at **164.4 ms** per
   decision, then Qwen3.5-0.8B at 188.2, then the incumbent at 263.2 — but only
   via explicit checkpointing. On the deployed `cache_prompt` path the incumbent
   wins by 6-15x.
5. **Does LFM2.5 beat the Qwen candidates on CPU?** On latency yes (164 vs
   188 ms) and on state decisively (20.2 vs 39.9 MiB). On raw decode Qwen3.5-0.8B
   is faster (99.9 vs 85.6 tok/s); LFM2.5 wins the decision because its prefill
   is cheaper, which is what this workload is made of.
6. **Is published LFM QAD-Q4 useful here?** It is **free** — byte-identical
   state and latency within 0.3% of ordinary `Q4_0` on LFM2.5-1.2B. So there is
   no hardware argument against adopting it, and its value is entirely a
   behavioural question for the off-box eval. Measured on the 1.2B pair only.
7. **Does Nemotron's Mamba-heavy structure give cheap state or decode?**
   Neither, here. 108 MiB serialized (sized by slot context, not tokens) and
   0.8 tok/s CPU decode. Its 4-attention-layer geometry *should* be cheap;
   this runtime does not deliver it. Its state restore is also not bit-exact on
   a clean slot (|Δlogprob| 0.134).
8. **Realistic LoRA throughput?** 718-1704 tok/s across seq 512-2048 for
   0.8-1.2B models at ~4096 tokens/update — not the 2819 tok/s the 20-example
   smoke batch suggested.
9. **How large a model trains comfortably?** Memory is not the limit (7-22 GiB
   of ~97.6 GiB at seq 2048); **throughput is**. ~2-4B is the practical ceiling
   for iterative work; beyond that a 100M-token campaign runs into days.
   Extrapolated from two measured points.
10. **Muse Glimmer / Qwen3.8-27B as a local architect?** Metadata recon only.
    Muse-Glimmer-30B (55.5 GiB BF16, official GGUF exists) needs a **newer
    llama.cpp than the pinned build** — `muse_glimmer` is absent from its arch
    table. Qwen3.8-27B declares `model_type: qwen3_5`, which the pinned build
    *does* carry, making it the cheaper experiment; its 16.35 GiB download
    stalled and was abandoned rather than left to consume bandwidth for hours.
11. **Does ROCm 7.1 alter any XDNA or Vulkan reference result?** **No.** See
    `REGRESSION.md`: perplexity bit-identical, all XDNA shapes bit-exact, 1926
    dispatches, controller within −2.3%, Vulkan 12.38 vs 11.76 tok/s. XRT and
    Mesa are unchanged, and ROCm ships no NPU component.
12. **What should the off-box team test first?** See the handoff below.

## Three candidates for behavioural evaluation

Ranked by hardware fitness only. **The incumbent must remain in the eval as the
control** — it is the only architecture whose state restore is provably
faithful, so any hybrid must beat it by enough to justify that loss.

### 1. LFM2.5-1.2B-Instruct — the best hardware profile measured

- **Why it survived:** fastest decision (164.4 ms, 1.60x the incumbent),
  smallest state (20.23 MiB, **6.3x** more warm domains), fastest training
  (1637 tok/s @1024, 1.6x Qwen3.5-0.8B) in the least memory (7.0 GiB). It won
  every hardware axis it was measured on.
- **Low-bit path:** LiquidAI publishes an official **QAD-Q4_0** GGUF — a
  quantisation-aware distillation, an independent alternative to the
  ternary/BitDistill route rather than a competitor to it. That is a real
  advantage: a published low-bit form exists today.
- **Unknown:** everything behavioural. Also its licence is the LFM Open License
  v1.0, not Apache — **legal review is required before it is a default**. And
  its restore defect is the one that demonstrably changed a decision.

### 2. Qwen3.5-0.8B — the cleanest hybrid, and Apache-2.0

- **Why it survived:** 188.2 ms per decision, 99.9 tok/s decode (fastest
  measured), 3.2x residency, bit-exact restore on a clean slot, and the model
  that carried the bit-exact checkpoint-resume result.
- **Low-bit path:** **unknown.** BitDistill compatibility with gated DeltaNet is
  not established and is explicitly off-box work. Do not assume the BitNet
  recipe transfers.
- **Unknown:** whether a 0.8B model is behaviourally sufficient at all. Note its
  18.6 MiB fixed state floor makes it *worse* than LFM2.5 for short spines and
  relatively better for long ones.

### 3. Qwen3.5-2B — because hardware cannot separate it from its 0.8B sibling

- **Why it survived:** it has **byte-identical warm state** to Qwen3.5-0.8B
  (40,209,124 B), so the residency argument for the small one applies unchanged
  to the large one. The only hardware cost of choosing 2B is latency: 350.0 vs
  188.2 ms per decision, and 48.6 vs 99.9 tok/s decode.
- **Low-bit path:** same as (2) — unknown, off-box.
- **Unknown:** **exactly the question this bakeoff cannot answer.** If 2B is
  materially smarter than 0.8B, its 1.86x latency is likely worth paying, since
  it costs nothing in RAM. If it is not, 0.8B dominates it outright. This is the
  single highest-value comparison to hand off.

### What to test first, off-box

1. **0.8B vs 2B Qwen3.5 on identical controller tasks** — the residency cost is
   identical, so this is a pure quality-per-millisecond question.
2. **Whether any of them can emit a constrained action reliably** — on this
   synthetic workload several candidates emitted the same token regardless of
   state, which made argmax comparison nearly useless as a probe. Real tasks
   must discriminate.
3. **LFM2.5-1.2B against Qwen3.5-0.8B at matched latency budget**, since they
   are 164 vs 188 ms and the choice is otherwise a licence question.

## What was not done

- **Task 10 at the 2.6B size** — the 1.2B `Q4_0`/`QAD-Q4_0` pair was compared;
  the 2.6B pair did not download.
- **Task 7 at the 2-4B tier** — BF16 checkpoints did not finish downloading.

- **Teacher-tier inference smoke** — recon only; no local 27-30B model was run.
- Nemotron's state-semantics probes, abandoned after the decode measurement made
  it non-viable as a warm controller.
