# Controller-SFT at production length [MEASURED]

Builds on the corrected harness from `hybrid-state-training-gate`. The
methodology is not rebuilt; only the remaining gaps are closed.

## 1. Restricted logits: the equivalence claim, properly justified

The previous pass compared a scalar loss and one **aggregate gradient norm** and
called the paths "equivalent". Two different gradient vectors can share a norm,
so that evidence did not support the claim. Checked per element, and then
end-to-end through an optimizer step from identical adapter and optimizer state:

| model | trainable tensors / elements | grad max\|Δ\| | grad rel L2 | params after one AdamW step |
|---|---|---:|---:|---|
| Qwen3.5-0.8B | 228 / 7,274,496 | **0.00e+00** | **0.00e+00** | **identical** |
| LFM2.5-1.2B | 184 / 11,108,352 | **0.00e+00** | **0.00e+00** | **identical** |

Bit-identical, not merely within tolerance — on every trainable element and on
every parameter after a full optimizer step. Peak memory on the same protocol:
Qwen3.5-0.8B **11,039.8 → 8,293.3 MiB (−24.9%)**, LFM2.5-1.2B 6,709.8 → 6,076.3
(−9.4%). The 2.6x gap tracks the 248k vs 65k vocabulary.

## 2. Production length: seq 2048, and the units that matter

Action-only loss, restricted logits, eager, micro-batch 1, ~4096 input
tokens/update. This is the real serving shape (~1600 spine + ~135 delta + a tiny
action).

| model | input tok/s | **examples/s** | action-tok/s | step | peak GPU |
|---|---:|---:|---:|---:|---:|
| **LFM2.5-1.2B** | **1474.0** | **0.72** | **1.9** | 2.779 s | 12.8 GiB |
| Qwen3.5-0.8B | 816.2 | 0.40 | 0.6 | 5.018 s | 15.6 GiB |
| Qwen3.5-2B | 650.8 | 0.32 | 0.5 | 6.294 s | 19.9 GiB |
| Qwen3.5-4B | 256.5 | 0.13 | 0.2 | 15.967 s | **46.6 GiB** |

### Why the units change the conclusion

Reporting only input tokens/s makes this look healthier than it is. At seq 2048
each example carries **1.6-2.6 supervised tokens out of 2048** — a supervised
fraction of 0.079-0.127%. So "100M input tokens" is only about **49,000
decisions**, not a large labelled set.

Both framings, for planning:

| model | 100M input tokens | **100k controller decisions** |
|---|---:|---:|
| LFM2.5-1.2B | 18.8 h | **38.6 h** |
| Qwen3.5-0.8B | 34.0 h | 69.4 h |
| Qwen3.5-2B | 42.7 h | 86.8 h |
| Qwen3.5-4B | 108.3 h | **213.7 h (8.9 days)** |

**A 100k-decision campaign at production length is 1.6 days on the fastest
candidate and nine days at 4B.** That is the number the off-box data pipeline
should plan against, and it is roughly 2x the input-token framing.

Sequence length costs throughput roughly proportionally, so shorter training
sequences buy decisions cheaply — at seq 1024 LFM2.5-1.2B does 1.66 examples/s
against 0.72 at 2048. Whether a shorter training context is acceptable is a
behavioural question, not a hardware one.

Memory finally starts to bind at the top: Qwen3.5-4B at seq 2048 peaks at
**46.6 GiB of ~97.6**, so micro-batch 2 would not fit and the 4B corner is
constrained by both throughput and memory.

## 3. Reaching the recurrent pathway is cheap (Task 15)

Ordinary LoRA leaves every `nn.Conv1d` frozen — the short-conv in LFM2 and the
DeltaNet convolution in Qwen3.5. Unfreezing those parameters alongside LoRA, no
custom adapter:

| model | trainable, LoRA | trainable, +Conv1d | added | tok/s | peak |
|---|---:|---:|---:|---|---:|
| LFM2.5-1.2B | 11,108,352 (0.940%) | 11,169,792 (0.945%) | **+61,440 (+0.55%)** | 1771.1 → 1699.4 (**−4.0%**) | 6159.0 → 6159.4 MiB |
| Qwen3.5-0.8B | 7,274,496 (0.958%) | 7,716,864 (1.016%) | +442,368 (+6.1%) | 1114.4 → 955.6 (−14.2%) | 8345.4 → 8596.0 MiB |

**For LFM2.5-1.2B the recurrent pathway is essentially free to adapt** — 61k
extra parameters, no measurable memory cost, 4% throughput. For Qwen3.5-0.8B it
costs more but is still modest.

This does not claim it *helps*; no quality was measured. It tells the off-box
team that including the recurrent path is a cheap option rather than a
significant footprint change.

## 4. The two remaining family points (Tasks 13 and 14)

Both completed. **BitNet BF16 attaches to standard PEFT with no obstacle** —
the answer to Task 14 is the positive one:

| | BitNet-b1.58-2B BF16 | LFM2.5-2.6B |
|---|---|---|
| adapted modules | `q,k,v,o_proj`, `gate,up,down_proj` | `q,k,v_proj`, `in_proj`, `out_proj`, `w1,w2,w3` |
| `Conv1d` present | **0** (pure attention) | 22 (not adapted) |
| trainable | 21,626,880 / 2.43 B (0.888%) | 24,461,312 / 2.72 B (0.899%) |
| seq 1024 | 855.8 tok/s, 0.84 ex/s, 18.8 GiB | 889.8 tok/s, 0.87 ex/s, 12.6 GiB |
| seq 2048 | 671.2 tok/s, 0.33 ex/s, 35.8 GiB | 784.5 tok/s, 0.38 ex/s, 22.7 GiB |

**BitNet standard-LoRA support is NOT blocked.** The BF16 master exposes ordinary
`nn.Linear` projections under conventional names, so `peft` attaches cleanly and
no custom trainer is required. The only obstacle was packaging: the repo's
`config.json` carries an `auto_map` pointing at `configuration_bitnet.py` /
`modeling_bitnet.py` that are **absent from the repo**, so `trust_remote_code=True`
fails. transformers 5.16.1 has a native `bitnet` implementation that loads
correctly once `auto_map` is removed — a one-line fix, recorded so the next
person does not lose time to it.

### Complete seq-2048 table

| model | params | input tok/s | examples/s | act-tok/s | peak | 100k decisions |
|---|---:|---:|---:|---:|---:|---:|
| **LFM2.5-1.2B** | 1.18 B | **1474.0** | **0.72** | 1.9 | **12.8 GiB** | **38.6 h** |
| LFM2.5-2.6B | 2.72 B | 784.5 | 0.38 | 0.8 | 22.7 GiB | 73.1 h |
| Qwen3.5-0.8B | 0.76 B | 816.2 | 0.40 | 0.6 | 15.6 GiB | 69.4 h |
| BitNet-b1.58-2B | 2.43 B | 671.2 | 0.33 | 0.5 | 35.8 GiB | 84.2 h |
| Qwen3.5-2B | 1.89 B | 650.8 | 0.32 | 0.5 | 19.9 GiB | 86.8 h |
| Qwen3.5-4B | 4.23 B | 256.5 | 0.13 | 0.2 | 46.6 GiB | 213.7 h |

Two things the larger LFM2.5 point settles:

- **LFM2.5's training lead is not a small-model artifact.** At 2.72 B it still
  beats Qwen3.5-**2B** (1.89 B) on throughput — 784.5 against 650.8 tok/s — while
  being 1.4x the parameters. The family scales as its architecture predicts.
- **Memory separates the incumbent sharply.** BitNet needs **35.8 GiB** at seq
  2048 against LFM2.5-2.6B's 22.7 GiB at a *larger* parameter count, because all
  30 of its layers carry full attention.

## 5. What is not covered

- **Micro-batch sweeps for the two new points** — only mb1 was run for
  LFM2.5-2.6B and BitNet, by instruction. The mb1 optimum found elsewhere makes
  that unlikely to be the wrong choice, but it is not measured for these two.
- **Quality of any of it.** Every number here is throughput, memory or
  numerical agreement. Nothing in this pass says any model decides well.
