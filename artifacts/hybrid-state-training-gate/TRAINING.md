# Controller-SFT throughput, corrected [MEASURED]

Correction pass over `model-candidate-halo`'s `TRAINING.md`. That branch is
frozen; its numbers are not deleted, they are **relabelled**.

## What was actually being measured before

`tools/train_scaling.py`'s generator grew the state text until the state ALONE
already reached `seq_len + 1`, then appended `"ACTION: <verb>"`, then truncated
to `seq_len + 1`. The action was therefore **always past the truncation
boundary and never present in the training sequence**. Loss was then applied to
every position via `labels=`.

So those runs measured **full-sequence causal-LM throughput over synthetic state
text**. That is a real and useful number — it is close to the cost of the
forward/backward itself — but it is not the controller objective, which is a
long context and a 1-4 token supervised decision. The earlier numbers are
retained under the name **FULL-SEQUENCE LM THROUGHPUT**.

`tools/train_controller_sft.py` implements the intended objective:

```
[ context: stable spine + dynamic state ]  [ "\nACTION:" ]  [ 1-4 action tokens ]
```

with the **context trimmed to make room for the action** rather than the action
being pushed out, `-100` on every context position, and a per-example assertion
that each sequence carries between 1 and `--max-action-tokens` supervised
tokens. A violation aborts rather than silently training on nothing.

Verified corpus shape:

| model | seq | supervised tokens | supervised fraction |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 512 | 1.6 | 0.314% |
| Qwen3.5-0.8B | 1024 | 1.6 | 0.157% |
| LFM2.5-1.2B | 512 | 2.6 | 0.510% |
| LFM2.5-1.2B | 1024 | 2.6 | 0.255% |

## Restricted logit materialisation is exact, and worth it

Because only the tail is supervised, the loss is computed in the harness rather
than by passing `labels=`, which allows `logits_to_keep`. Both candidate
families expose it (`Qwen3_5ForCausalLM.forward` and `Lfm2ForCausalLM.forward`
in transformers 5.16.1).

**Correctness control first**, before any timing was reported:

| model | Δ loss | Δ grad-norm (relative) | equivalent |
|---|---:|---:|---|
| Qwen3.5-0.8B | **0.00e+00** | **0.00e+00** | yes |
| LFM2.5-1.2B | **0.00e+00** | **0.00e+00** | yes |

Exactly equal, not merely within tolerance. The harness refuses the restricted
path and falls back to full logits if this check fails.

Memory on the single-example check:

| model | full logits | restricted (`logits_to_keep=8`) | saved |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 6229.0 MiB | 4880.3 MiB | **−21.7%** |
| LFM2.5-1.2B | 4220.2 MiB | 3939.2 MiB | −6.7% |

Qwen3.5 benefits ~3x more, which is what a 248,320-token vocabulary against
LFM2's 65,536 predicts: the `[B, T, V]` logits tensor is the dominant term and
restricting `T` to 8 removes almost all of it.

## Micro-batch sweep at a fixed ~4096 tokens/update

**seq 512**

| model | mb1 × ga8 | mb2 × ga4 | mb4 × ga2 | mb8 × ga1 |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | 1151.5 | **1289.1** | 1233.5 | 1112.4 |
| peak MiB | 4,936 | 8,241 | 14,697 | 27,674 |
| LFM2.5-1.2B | **1791.1** | 1732.3 | 1723.3 | 1704.9 |
| peak MiB | 4,024 | 5,582 | 8,698 | 14,888 |

**seq 1024**

| model | mb1 × ga4 | mb2 × ga2 | mb4 × ga1 |
|---|---:|---:|---:|
| Qwen3.5-0.8B | **1114.4** | 1006.1 | 907.1 |
| peak MiB | 8,345 | 14,921 | 28,126 |
| LFM2.5-1.2B | **1771.1** | 1470.0 | 1465.8 |
| peak MiB | 6,159 | 9,852 | 17,196 |
| Qwen3.5-2B | 798.6 | — | — |
| peak MiB | 11,556 | | |

**The throughput-maximising micro-batch is 1 in every case except Qwen3.5-0.8B
at seq 512, where 2 wins by 12%.** This is the opposite of the usual
expectation, and the reason is visible in the split timings: forward time is
almost flat across micro-batch (Qwen3.5-0.8B at seq 1024: 1.103 / 1.096 / 1.099 s)
while **backward time grows steadily** (2.550 / 2.954 / 3.395 s). Larger
micro-batches buy nothing on the forward and cost real time on the backward, so
the memory headroom that motivated Task 8 does not convert into throughput here.

Practical reading: **keep micro-batch at 1-2 and make up the budget with
gradient accumulation.** That also keeps peak memory 3-4x lower, which matters
for the coexistence result rather than for training itself.

## Corrected objective vs the superseded one

At micro-batch 1, the same models, same token budget:

| model | seq | FULL-SEQUENCE LM (superseded) | CONTROLLER-SFT (this pass) | change |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | 512 | 1090.2 | 1151.5 | +5.6% |
| Qwen3.5-0.8B | 1024 | 1014.3 | 1114.4 | +9.9% |
| LFM2.5-1.2B | 512 | 1703.8 | 1791.1 | +5.1% |
| LFM2.5-1.2B | 1024 | 1637.1 | 1771.1 | +8.2% |
| Qwen3.5-2B | 1024 | 704.2 | 798.6 | +13.4% |

The corrected objective is **5-13% faster**, from restricted logits and from not
computing a full-vocabulary loss at every position. The relative ordering of the
candidates is unchanged, so the earlier branch's *comparative* training
conclusion survives even though its objective was wrong; the absolute numbers
move.

Best measured controller-SFT throughput, at each model's optimal micro-batch:

| model | seq 512 | seq 1024 |
|---|---:|---:|
| **LFM2.5-1.2B** | **1791.1** tok/s | **1771.1** tok/s |
| Qwen3.5-0.8B | 1289.1 | 1114.4 |
| Qwen3.5-2B | 800.4 | 798.6 |
| Qwen3.5-4B | 426.1 | 368.5 |

## The 2-4B tier, measured rather than extrapolated (Task 10)

`model-candidate-halo` extrapolated a ~2-4B local ceiling from two points.
Qwen3.5-4B is now measured, which makes the whole Qwen3.5 family a real curve at
micro-batch 1:

| model | params | seq 512 | peak | seq 1024 | peak | trainable |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 0.760 B | 1151.5 | 4.9 GiB | 1114.4 | 8.3 GiB | 7.27 M (0.958%) |
| Qwen3.5-2B | 1.894 B | 800.4 | 7.6 GiB | 798.6 | 11.6 GiB | 12.09 M (0.638%) |
| Qwen3.5-4B | 4.230 B | 426.1 | 17.6 GiB | 368.5 | 26.8 GiB | 23.79 M (0.563%) |

**The three points do not lie on a single power law**, so the earlier
extrapolation should not be trusted at face value. A fit through the endpoints
gives `params^-0.645`, which reproduces 0.8B and 4B exactly by construction but
misses the 2B midpoint by **−22.5%** — Qwen3.5-2B is materially faster than a
smooth curve predicts. The honest statement is therefore the measured table, plus
a *rough* bound: at 7-8B the same fit suggests ~255 tok/s, and given the midpoint
error that figure could easily be 20-30% off in either direction.

What the measurement does settle, in campaign terms for 100M tokens at seq 1024:

| model | tok/s | 100M-token campaign |
|---|---:|---:|
| LFM2.5-1.2B | 1771.1 | **15.7 h** |
| Qwen3.5-0.8B | 1114.4 | 24.9 h |
| Qwen3.5-2B | 798.6 | 34.8 h |
| Qwen3.5-4B | 368.5 | **75.4 h** (3.1 days) |
| ~7-8B (extrapolated, weak) | ~255 | ~109 h (4.6 days) |

**Memory still never binds** — the heaviest arm measured, Qwen3.5-4B at seq 1024,
peaks at 26.8 GiB of ~97.6 available. The ceiling is entirely throughput. So the
practical reading tightens: **~2B is the comfortable ceiling for iterative work**
(a campaign inside a working day and a half), **4B is viable for occasional
runs** at three days, and the earlier branch's "~2-4B" was the right range but
its 4B end is slower than its extrapolation implied.

LFM2.5-2.6B was not fetched in HF format, so the LFM2.5 family still has only its
1.2B point; whether it scales as favourably as it leads at 1.2B is untested.

## LoRA coverage — stated, not implied (Task 11)

Neither family is fully adapted, and the report does not claim otherwise.

| model | adapted modules | not adapted | `Conv1d` modules present |
|---|---|---|---|
| Qwen3.5-0.8B / 2B | `q,k,v,o_proj`, `gate,up,down_proj`, `out_proj` | `conv1d` | **18** |
| LFM2.5-1.2B | `q,k,v_proj`, `in_proj`, `out_proj`, `w1,w2,w3` | `conv` | 10 |

peft's `Linear` adapter does not apply to `nn.Conv1d`, so the short-conv /
DeltaNet convolution in every hybrid block is frozen. For LFM2.5-1.2B that is
**10 of 16 blocks** carrying an unadapted convolution; for Qwen3.5, 18 of 24.
Attention and MLP projections inside those blocks *are* adapted, so the blocks
are not untouched — but **this is not architecture-wide adaptation**, and no
claim here should be read as evidence that LoRA reaches the recurrent path. A
conv-capable adapter was not written in this pass, by instruction.

## Attention backend: SDPA is slower here (Task 9)

The superseded pass used `attn_implementation="eager"` for controlled fairness.
That arm is kept. The practical alternative on this stack is PyTorch SDPA; no
FlashAttention build work was done, by instruction.

| model | seq | EAGER CONTROLLED | SDPA PRACTICAL | change |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B (mb2) | 512 | **1291.3** | 1250.3 | −3.2% |
| Qwen3.5-0.8B (mb2) | 1024 | **1005.0** | 958.4 | −4.6% |
| LFM2.5-1.2B (mb1) | 512 | **1796.6** | 1652.0 | −8.0% |
| LFM2.5-1.2B (mb1) | 1024 | **1768.4** | 1542.7 | −12.8% |

**SDPA is slower on every arm**, and worst on the model with the fewest
attention layers. That is consistent with SDPA's dispatch overhead not being
repaid when only 6 of 16 (LFM2.5) or 6 of 24 (Qwen3.5) blocks are attention at
all, and with these sequence lengths being short enough that the kernel has
little to win. It is a measurement on this ROCm/gfx1151 stack, not a general
claim about SDPA.

**The practical-training estimate therefore uses `eager`**, which is also the
controlled arm — so for this workload the two coincide and no separate
"practical" number is needed.
