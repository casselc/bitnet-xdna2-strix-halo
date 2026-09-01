# Realistic LoRA training scaling on the Radeon 8060S [MEASURED]

`halo-training-smoke` proved the stack runs and said plainly that its 2819 tok/s
at Qwen3-0.6B and 1480 tok/s at Qwen3-1.7B came from a deliberately tiny ~20
example batch. This measures what a campaign would actually see: sequence length
swept 256 -> 2048 at a **fixed ~4096 tokens per optimizer update**, so the arms
are comparable, with >= 2 warmup and 10 timed updates each.

Environment: torch 2.10.0+rocm7.0, HIP 7.0.51831, system ROCm 7.1, gfx1151,
BF16, LoRA r=16 alpha=32, AdamW, microbatch 1 with gradient accumulation making
up the budget. Every run launched through `tools/halo_rocm_env.sh exec`.

## The LoRA target list is part of the measurement, not a detail

A first LFM2.5 run adapted only 5 modules for **3,244,032** trainable params
(0.28%) while Qwen3.5 got 8 modules and **7,274,496** (0.96%). The cause was
that LFM2's MLP is named `w1/w2/w3` (SwiGLU) rather than
`gate_proj/up_proj/down_proj`, so it was silently skipped. Comparing those two
runs would have been comparing two different jobs, and would have flattered
LFM2.5 by ~1.4x.

Both are reported below at **matched coverage** — attention + MLP + the
hybrid-block projections in each architecture:

| model | LoRA targets | trainable | % of model |
|---|---|---:|---:|
| Qwen3.5-0.8B | `q,k,v,o_proj`, `gate,up,down_proj`, `out_proj` | 7,274,496 | 0.958% |
| LFM2.5-1.2B | `q,k,v_proj`, `in_proj`, `out_proj`, `w1,w2,w3` | 11,108,352 | 0.940% |

1-D convolutions are **not** adapted in either (`conv1d` for Qwen3.5, `conv` for
LFM2) — peft's Linear adapter does not apply to them. This is recorded rather
than hidden: on LFM2.5 those are 10 of 16 blocks, so a real adaptation study
would need a conv-capable adapter, and no claim here should be read as "LoRA
covers this architecture".

## Scaling

**Qwen3.5-0.8B** (759,667,520 params)

| seq | microbatch x accum | tok/update | tok/s | step | peak GPU | package W |
|---:|---|---:|---:|---:|---:|---:|
| 256 | 1 x 16 | 4096 | 892.1 | 4.591 s | 4,102 MiB | 94.7 |
| 512 | 1 x 8 | 4096 | **1090.2** | 3.757 s | 6,623 MiB | 102.7 |
| 1024 | 1 x 4 | 4096 | 1014.3 | 4.038 s | 11,722 MiB | — |
| 2048 | 1 x 2 | 4096 | 717.7 | 5.707 s | 22,227 MiB | 102.3 |

**LFM2.5-1.2B** (1,181,448,960 params — 1.6x larger)

| seq | microbatch x accum | tok/update | tok/s | step | peak GPU | package W |
|---:|---|---:|---:|---:|---:|---:|
| 256 | 1 x 16 | 4096 | 1388.8 | 2.949 s | 3,400 MiB | 94.8 |
| 512 | 1 x 8 | 4096 | **1703.8** | 2.404 s | 4,450 MiB | 105.5 |
| 1024 | 1 x 4 | 4096 | 1637.1 | 2.502 s | 7,009 MiB | 110.4 |
| 2048 | 1 x 2 | 4096 | 1347.7 | 3.039 s | 13,856 MiB | 108.5 |

### What the curves say

- **Throughput peaks at seq 512 for both**, then falls: 1090 -> 718 tok/s for
  Qwen3.5 (-34%) and 1704 -> 1348 for LFM2.5 (-21%) going 512 -> 2048. Short
  sequences lose to per-update overhead (16 accumulation micro-steps), long ones
  to quadratic attention and activation pressure. **seq 512-1024 is the
  efficient band on this part.**
- **LFM2.5-1.2B trains 1.56-1.88x faster than Qwen3.5-0.8B at every length**
  while being a 1.6x larger model with 1.5x more trainable parameters. Only 6 of
  its 16 blocks carry attention; the rest are short-conv, which is linear in
  sequence length. Its advantage *widens* with length exactly as that predicts.
- **Memory is the binding constraint, not throughput.** Qwen3.5-0.8B needs
  **22.2 GiB** at seq 2048 — 3.2x LFM2.5's 13.9 GiB at the same length, for a
  smaller model. Two causes compound: a 248,320-token vocabulary (vs 65,536)
  makes the logits tensor ~3.8x larger, and 6 of 24 layers hold full attention
  against LFM2's 6 of 16 with a much smaller head dim.
- **Power is flat at 95-112 W** across every arm. Nothing here is
  power-limited, and no arm approached the part's envelope.

### How large a model trains comfortably

Extrapolating from measured peak memory against ~97.6 GiB of usable unified
memory, at seq 1024 and this LoRA configuration:

| model | measured peak @1024 | headroom |
|---|---:|---|
| LFM2.5-1.2B | 7.0 GiB | ~13x |
| Qwen3.5-0.8B | 11.7 GiB | ~8x |

Memory is not the limit at this scale — **throughput is**. At ~1000-1700 tok/s a
1B-class LoRA campaign over 100M tokens is 16-28 hours. A 7-8B model would land
near 200-400 tok/s by parameter scaling, making the same campaign a week. The
practical local training ceiling on this box is therefore **~2-4B for iterative
work**, with larger models viable only for short adaptation runs. This is an
extrapolation from two measured points and is labelled as such.

## Checkpoint resume: fixed, and verified against a continuous run

The smoke test saved adapter weights only and exposed that AdamW moments were
lost. The harness now checkpoints **adapter + optimizer + scheduler + step +
RNG**, and the test is a three-way comparison in **separate processes**, because
`save` and `resume` inside one process share allocator and RNG state and can
pass while a real restart fails.

Qwen3.5-0.8B, seq 512, 4096 tokens/update, 12 updates then checkpoint, process
destroyed, 8 more updates:

| step | continuous | resumed | difference |
|---:|---:|---:|---:|
| 12 | 2.21485 | 2.21485 | 0.00e+00 |
| 13 | 2.18317 | 2.18317 | 0.00e+00 |
| 14 | 2.12541 | 2.12541 | 0.00e+00 |
| 15 | 2.09906 | 2.09906 | 0.00e+00 |
| 16 | 2.07027 | 2.07027 | 0.00e+00 |
| 17 | 2.02552 | 2.02552 | 0.00e+00 |
| 18 | 2.00521 | 2.00521 | 0.00e+00 |
| 19 | 2.00505 | 2.00505 | 0.00e+00 |

**Bit-exact across a process boundary. No fresh-optimizer spike** — the failure
this test exists to detect. 228 optimizer state entries restored.

| | bytes | note |
|---|---:|---|
| adapter | 27.78 MiB | the deployment artifact; still exported separately |
| optimizer + scheduler | 55.69 MiB | 2x the adapter — AdamW's two moments |
| **total checkpoint** | **83.48 MiB** | write 1.279 s, read 0.164 s |

The optimizer state being twice the adapter is the whole point: an adapter-only
save discards two-thirds of what a resume needs.

**LFM2.5-1.2B reproduces the result**, so this is a property of the harness and
not of one architecture: max |continuous − resumed| = **0.00e+00** over 8 steps
(3.96054 → 3.26641, identical in both runs). Its checkpoint is 127.31 MiB
(adapter 42.40 + optimizer/scheduler 84.91), write 0.801 s, read 0.153 s —
larger than Qwen3.5-0.8B's because it carries 11.1 M trainable parameters
against 7.3 M, at the same 2:1 optimizer-to-adapter ratio.

## Not measured, and why

- **Qwen3.5-2B, LFM2.5-2.6B, Nemotron-3-Nano-4B**: their BF16 checkpoints did
  not finish downloading within this session (Qwen3.5-2B stalled at 12 MiB of
  ~4.2 GiB while sharing bandwidth). The harness is architecture-agnostic and
  resolves targets by inspection, so these are a re-run rather than new work.
  The 2-4B tier of Task 7 is therefore **not covered**.
- **Hyperparameters and quality**: out of scope by instruction. The loss curves
  above show gradients flow; they are not evidence any of this learns well.
- The `attn_implementation` is `eager` for all arms, so no arm benefits from a
  fused attention kernel the others lack.

---

# GPU training coexistence with the warm controller [MEASURED]

The controller runs on CPU and LoRA training runs on the iGPU, but on this part
they share one LPDDR5X memory system, one power budget and one thermal envelope.
"Different devices" is not an argument that they do not interfere.

Arms, in order, so the baseline is taken on a machine in the state the loaded
arm starts from: controller alone, then the identical turns while Qwen3.5-0.8B
trains at **seq 1024**. Controller configuration is the reference
`t4 / tb16 / b4096 / ub4096 / c40960`, `--cache-ram 8192`. Run twice.

| | TTFT p50 | TTFT p95 | total p50 | req/s | package W |
|---|---:|---:|---:|---:|---:|
| **run 1** alone | 199.2 ms | 203.2 ms | 260.8 ms | 3.769 | 110.3 |
| **run 1** training | 294.9 ms | 303.7 ms | 394.9 ms | 2.513 | 120.0 |
| | **+48.0%** | **+49.5%** | **+51.5%** | **−33.3%** | +8.8% |
| **run 2** alone | 200.2 ms | 204.9 ms | 263.6 ms | — | 109.0 |
| **run 2** training | 292.0 ms | 303.0 ms | 396.9 ms | — | 119.9 |
| | **+45.9%** | **+47.9%** | **+50.6%** | — | +10.0% |

**There IS a material controller penalty, and it reproduces.** This is not the
"no material penalty resolved" case the mission described for indistinguishable
results — the two runs agree to within 2 points on every axis and the effect is
~10x larger than the run-to-run spread.

## The interference is one-directional

| | alone | under the other tenant |
|---|---:|---:|
| training throughput @ seq 1024 | 1014.3 tok/s | **1022.8 tok/s** |
| controller TTFT p50 | 199.2 ms | **294.9 ms** |

**Training does not notice the controller. The controller loses a third of its
throughput to training.** A plausible mechanism, consistent with everything
measured: the training job is compute-bound on the iGPU while a BitNet CPU
prefill is memory-bandwidth-bound, so a saturated shared memory bus costs the
CPU tenant and not the GPU one. That mechanism is *consistent with* the data,
not demonstrated by it — separating bandwidth from power or thermal causes would
need counters this pass did not collect.

`cache_n` is identical at 1615 in both arms, so the degradation is contention
and not a prompt-cache artifact.

## This corrects a prior branch, and the reason is the load

`halo-training-smoke` recorded **+0.7%** controller TTFT under GPU training.
That is not contradicted so much as superseded by scope: it trained Qwen3-0.6B
on a fixed ~20-example batch of short sequences, which the same branch describes
as a plumbing smoke test. The GPU there was busy in brief bursts. Here the GPU
sits at 98-99% utilisation for minutes at seq 1024 with 11.7 GiB resident, and
the cost appears.

**The frozen record on that branch is not rewritten.** The correction is that
"local GPU training coexists at +0.7%" holds for a smoke-sized job and **does
not** hold for a realistic training campaign, where the figure is ~+48%.

Operationally: a training campaign and a latency-sensitive warm controller
should not share this box without admission control — which is the same
conclusion `service-cotenancy` reached for a different tenant pair.
