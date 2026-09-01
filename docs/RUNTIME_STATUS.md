# Runtime status

What is settled, what was measured and rejected, and what is still open. Read
this before proposing work on this repository.

Promoted coordinate: `artifacts/runtime-v1/COORDINATE.md`.

---

## The architecture

| device | role |
|---|---|
| **XDNA2 NPU** | BitNet **linear prefill** offload |
| **Zen 5** | attention, decode, orchestration, verification |
| **Radeon 8060S** | next system-level worker target — not yet enabled |

---

## Supported / promoted

- **XDNA2 linear prefill offload.** One program serves every BitNet linear
  shape, so a prefill performs **zero hardware-context switches**. The wide FFN
  is N-chunked (6912 padded to 7680) and the deep down-projection K-chunked.
- **Direct mapped output.** Persistent per-(token tile, N chunk) XRT output
  buffers; the ggml epilogue reads NPU memory directly. `stage_out` is **0.0 ms
  over 0.00 GB**, by counter.
- **Direct deep-K reduction.** K chunks are reduced in int32 before conversion;
  the host `part` buffer, its zeroing and its copy are all skipped.
  `partacc` and `partcopy` are **0.0 ms**.
- **R = 25 cost model** for the CPU/NPU token split (10 on the `g_acc` path).
- **Partial-token-tile handling.** `ceil` with the tile count clamped to the
  token count, so a trailing micro-batch is not floored to zero NPU tiles.
- **Single-flight NPU invocation contract.** `bitnet_xdna_invocation_begin/end`.
  Concurrent contexts without it **segfault** — reproduced deterministically by
  `tests/test_xdna_concurrent`.
- **CPU attention and CPU decode.**

### Operating requirement

**The micro-batch must be >= 1024 tokens (`kMTile`) or the NPU never runs.**
`llama-bench` and `llama-perplexity` default to `-ub 512`, below that threshold,
so every micro-batch is declined and the run silently falls back to CPU. Pass
`-ub 2048`.

---

## Measured and rejected / superseded

Each of these was measured, not assumed. The evidence branches are listed so a
later reader can check rather than retry.

| rejected | why | evidence |
|---|---|---|
| exclusive NPU prefill | the CPU is not idle during offload; the split is worth more than either extreme | `next-pass-results` |
| default `-ub 512` for XDNA | below `kMTile`, so the NPU is never invoked at all | this document, section 4 |
| multiple large hardware contexts for GEMM shapes | contexts hold all 8 columns and cannot be co-resident; 3-context cycling cost **+53% to +210%** | `artifacts/kernels/context_switching.md` |
| cross-microbatch pipeline | CPU already 92.5–95.7% utilised and attention is the critical path; modelled gain only 1.02–1.05x | `overlap-de-risk` |
| async evacuation (`BITNET_XDNA_ASYNC`) | its purpose was overlapping evacuation, and direct output removes the evacuation | `direct-output-arena` |
| straightforward AMD/IRON attention port | stock d=64 operator is **1.95x–2.87x slower** than Zen 5, burdened, before a measured **130 ms/prefill** GEMM<->MHA context-switch tax | `attention-feasibility`, `attention-final-gate` |
| d=128 geometry rescue for that MHA | d=128 does not fit L1 at the native 64x64 block; the forced block reduction makes **QK** the new bottleneck. Best modelled path remains **~1.69x slower** than Zen 5 at 4K | `attention-geometry-gate` |
| fused single-core attention | recovers the real 50.5% core-utilisation waste (2.24x measured) but still lands at **+2.7%** at 4K, under the 15% bar | `attention-fused-core` |

**On XDNA2 attention specifically:** the measured stock/port path is closed. This
is **not** a claim that XDNA2 attention is impossible. A novel kernel would be a
distinct research project requiring materially different blocking, pipeline
structure and co-residency, and the branches above quantify what it would have
to beat.

---

## Deferred / open

- ~~**Real Radeon 8060S co-tenancy** — the next question.~~ **ANSWERED** on
  `gpu-cotenancy` and again on `controller-state-envelope` §7: a GPU worker
  coexists fine. The NPU's resident footprint costs the GPU nothing, and a 4x
  larger controller prompt cache (8 -> 32 GiB) leaves GPU decode identical at
  11.76 tok/s. Local GPU *training* also coexists, at +0.7% controller TTFT
  (`halo-training-smoke`). See `artifacts/EVIDENCE_INDEX.md`.
- **Packed ternary residency.** The runtime expands packed weights to ~2.0 GiB
  of resident int8. Only justified if co-tenancy shows it matters.
- **Hardware-aware controller student / BitDistill.** Not started. The backend
  constraints are now measured, and two of them bear directly on it:
  a warm controller engages the NPU **0% of the time** (work sits ~120 tokens,
  far below `kMTile = 1024`), and **model geometry is not a lever for NPU
  throughput** — a 2048/6144/2048 candidate matches the current 2560/6912/2560
  at 9.23 vs 9.25 TOPS, a 1024-wide one is worse, and `N <= 4096` is a hard
  single-kernel DMA-stride limit for any of them
  (`controller-state-envelope` §9 and Appendix). So the student's size should be
  chosen for quality, memory and CPU decode speed, not for XDNA2.
- **GEMM tile `64x128x64`** — measured 1.038x faster with 8 KiB less L1 and 6x
  lower variance, but **not promoted**: under 1% of prefill on its own. Use it
  if anything rebuilds the xclbins anyway (`gemm-tile-resweep`).
- **NPU decode** — only if a future model or representation changes the case.
- **Local training on this box** — now **READY**, not deferred: system ROCm 7.1
  plus `torch 2.10.0+rocm7.0` trains LoRA on gfx1151 at 2819 tok/s (0.6B) and
  1480 tok/s (1.7B). Requires shadowing the wheel's bundled
  `libhsa-runtime64.so`, which a `pip --force-reinstall` would undo
  (`halo-training-smoke`).
- **Admission control** — measured but deliberately not implemented. Two rules
  are needed, not one: bound the resident working set by
  `cache_ram / (state_tokens x 75 KiB)`, and bound arrivals near 50% of
  closed-loop capacity (`controller-state-envelope` §5, §8).

---

## 4. Promotion verification

Re-run on this machine at promotion time, from the promoted coordinate.

| check | result |
|---|---|
| `make check` | green: patch reproduces and applies to a pristine pinned tree, CPU tests, 12/12 shape cases bit-exact, concurrency lease holds |
| CPU-only NPU dispatches | **0** |
| `stage_out` / `partacc` / `partcopy` | **0.000 ms** on all three shapes |
| direct-output arena | 6 slots x 10.0 MiB = **60.0 MiB** |
| resident int8 weights | **2006.2 MiB**, 147 tensors |
| pp2048 t8, 5 interleaved reps | **865.7 t/s** (recorded 862.3) — **1.004x** |
| CPU-only pp2048 t8 | 637.5 t/s (recorded 629.6) — 1.013x |
| speedup over CPU-only | **1.358x** |
| perplexity, NPU engaged | **312.7569 +/- 14.92981**, identical CPU vs XDNA, 1320 dispatches |

### Two things the promotion run caught

**A ten-hour-stale binary.** `refs/BitNet/build-xdna` predated the closeout
runtime: it lacked `BITNET_XDNA_SHAPE_CSV` entirely, reported
`default == BITNET_XDNA_DIRECT_OUT=0`, and measured pp2048 at 837 t/s. Only
after rebuilding did the coordinate reproduce (865.7). Build freshness is not
optional on this repository — check that the binary is newer than `runtime/`.

**The recorded perplexity check never exercised the NPU.** The invocation in
`artifacts/invocations.md` uses the default `-ub 512`, so the XDNA arm ran with
`dispatches=0` and the claimed equivalence was vacuous for the offload path. The
check now passes `-c 2048 -b 2048 -ub 2048`, which produces 1320 dispatches and
still gives identical perplexity. This does not invalidate the earlier
correctness evidence — `tests/test_xdna_shapes` proves the offload path
bit-exact against a scalar oracle on all twelve shape cases — but the perplexity
line alone did not, and now does.
