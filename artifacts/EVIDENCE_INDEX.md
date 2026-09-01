# Evidence index

Every result in this project lives on a **frozen evidence branch**, not on `main`.
Nothing is merged; each branch is the provenance for its own numbers, and later
branches correct earlier ones by adding a new section rather than rewriting history.

This index is current as of the `controller-state-envelope` pass (2026-09-01).

## Branches, in the order the work happened

| branch | tip | what it establishes | primary artifact |
|---|---|---|---|
| `next-pass-results` | `fb4493e` | first profiling pass: the NPU is idle 68% of prefill | `artifacts/next-pass/` |
| `overlap-de-risk` | `3dff59b` | cross-micro-batch pipelining worth only 1.02–1.05x | `artifacts/overlap/` |
| `direct-output-arena` | `9f6008a` | async dispatch spike; concurrent contexts segfault -> the lease | `artifacts/direct-output/` |
| `direct-output-closeout` | `ed97cfc` | direct mapped-output epilogue promoted to default; R recalibrated | `artifacts/direct-output/` |
| `gemm-tile-resweep` | `f19fff4` | tile `64x128x64` is 1.038x faster — measured, **not** promoted | `artifacts/gemm-tile/RESULTS.md` |
| `attention-feasibility` | `0328d7f` | XDNA2 attention **REJECTED**: stock aie2p kernel 2x slower than CPU | `artifacts/attention-feasibility/` |
| `attention-geometry-gate` | `f44ae02` | d=128 geometry gate | `artifacts/attention-feasibility/GEOMETRY_GATE.md` |
| `attention-fused-core` | `3015271` | fused QK/softmax/PV core; includes a retracted claim, struck in place | `artifacts/attention-feasibility/FUSED_CORE.md` |
| `attention-final-gate` | `f2a3de3` | attention **CLOSED**, with scoped verdict and reopening conditions | `artifacts/attention-feasibility/FINAL_GATE.md` |
| `runtime-v1-promotion` | `712b7c6` | the promoted XDNA runtime, frozen as reference v1 | `docs/RUNTIME_STATUS.md` |
| `gpu-cotenancy` | `fbd8bf0` | tri-device co-tenancy validated; NPU footprint costs the GPU nothing | `artifacts/gpu-cotenancy/RESULTS.md` |
| `service-cotenancy` | `9295df0` | warm persistent service under concurrency; **admission control needed** | `artifacts/service-cotenancy/RESULTS.md` |
| `controller-cache-batching` | `2ca2e51` | `cache_prompt` was the dominant confound: 71x TTFT, 86x concurrency | `artifacts/controller-cache-batching/RESULTS.md` |
| `controller-state-scheduler` | `6d225c9` | state spine, forks, memoization, prompt-cache residency | `artifacts/controller-state-scheduler/RESULTS.md` |
| `service-batching-gate` | `856357c` | `-ub` gated batch formation; `-tb` was never set and cost 38% TTFT | `artifacts/service-batching-gate/RESULTS.md` |
| **`controller-state-envelope`** | current | **the real multi-domain warm-state envelope**, cache-RAM scaling, thrash, warm open loop, candidate-shape probe | `artifacts/controller-state-envelope/RESULTS.md` |
| **`halo-training-smoke`** | `f4c2732` | **local ROCm/PyTorch LoRA training on gfx1151**, and its coexistence cost | `artifacts/halo-training-smoke/RESULTS.md` |
| **`model-candidate-halo`** | current | **controller-model hardware bakeoff**: warm state size by serialization, hybrid state-restore correctness, realistic LoRA scaling, real checkpoint resume, coexistence cost | `artifacts/model-candidate-halo/RESULTS.md` |

`model-candidate-halo` shows `current` rather than a SHA because this copy of the
index lives in that branch and cannot name its own commit. Run `git rev-parse
origin/model-candidate-halo` for its tip. (`controller-state-envelope`, which
this branch was cut from, is `60230b5`.)

`main` (`885df0c`) carries none of this work; nothing has been merged into it.

## Corrections chain — read these together

A claim published on an earlier branch and corrected on a later one is **never**
rewritten in place. The chain:

| corrected claim | published on | corrected on |
|---|---|---|
| "controller concurrency ~= 1" — true, but configuration-scoped | `service-cotenancy` §3 | `service-batching-gate` §1, §5 |
| lease held "essentially the whole prefill", "~210 invocations, 14.87 ms" | `service-cotenancy` §3 | `service-batching-gate` §1 (measured: 97.9–148 acq/req, 5.5–11.2 ms, 38% of prefill) |
| state-spine "TTFT p50 147 ms / total p50 122 ms" (impossible) | `controller-state-scheduler` §6 | `controller-state-envelope` §0 (24 of 50 turns were silent HTTP 400s) |
| "rebase cadence 10 vs 25 indistinguishable" | `controller-state-scheduler` §6 | `controller-state-envelope` §0 — same conclusion, valid data |
| "exceeding cache capacity collapses the hit rate to 0%" | `controller-state-scheduler` §10 | `controller-state-envelope` §8 (true only for cyclic access; random keeps its median) |
| "HALO TRAINING STACK BLOCKED" | `halo-training-smoke` (first half) | same file, CORRECTION section — system ROCm 7.1 resolved it |
| GPU training costs the controller "+0.7% TTFT" | `halo-training-smoke` | `model-candidate-halo` — **+48%** at a realistic sustained load; the original figure holds only for the ~20-example smoke batch it was measured on |
| "2819 tok/s at Qwen3-0.6B" as training throughput | `halo-training-smoke` (self-qualified) | `model-candidate-halo` — 718–1090 tok/s at seq 256–2048 with a fixed token budget |
| `artifacts/invocations.md` NPU examples (`artifacts/xclbin`, `-p 512`) | pre-`runtime-v1` | `model-candidate-halo` `REGRESSION.md` — needs `artifacts/xclbin-tuned` and `-ub 2048` under the promoted runtime |

## Current headline numbers

| question | answer | source |
|---|---|---|
| warm controller TTFT p50 | **231 ms** (cold 1242–1276 ms) | `controller-state-envelope` §3 |
| warm domains resident | **58** @ 8 GiB, **251** @ 32 GiB, 127.1 MiB each | §6 |
| sustainable request rate | **~2.5 req/s** open loop (capacity 5.1) | §5 |
| NPU engagement, warm steady state | **0.00%** (cold miss: 56.33%) | §9 |
| does a bigger cache hurt the GPU? | **no** — 11.76 tok/s at both 8 and 32 GiB | §7 |
| does model geometry help the NPU? | **no** — 1.7B ≈ 2B; smaller is worse; N ≤ 4096 is a hard limit | Appendix |
| cheapest warm state per domain | **20.2 MiB** (LFM2.5-1.2B) vs 127.2 MiB (BitNet) — 6.3x more domains | `model-candidate-halo` |
| do hybrid models restore state correctly? | **no** — only the pure-attention incumbent is bit-exact | `model-candidate-halo` |
| realistic local LoRA throughput | **718–1704 tok/s** at seq 256–2048, ~4096 tok/update | `model-candidate-halo` |
| does GPU training disturb the controller? | **yes, +48% TTFT** under sustained load | `model-candidate-halo` |
| local training on this box | **ready** — 2819 tok/s @ 0.6B, 1480 @ 1.7B | `halo-training-smoke` |
| training's cost to the controller | **GPU +0.7% / CPU +74%** TTFT | `halo-training-smoke` T5 |

## Reproducing

Tools live beside the results on each branch. The service work needs
`tools/service_ctl.sh` (which now warms XDNA explicitly before any measurement) and
`tools/multi_domain.py`; the training work needs `.venv-train` and
`tools/train_smoke.py`. See each `RESULTS.md` for the exact invocation.
