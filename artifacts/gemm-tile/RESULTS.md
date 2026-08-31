# Re-sweeping the production GEMM tile

| | |
|---|---|
| branch base | `3015271` (`attention-fused-core`) |
| branch | `gemm-tile-resweep` |
| coordinate | M=1024, K=2560, N=2560, i8 -> i32, 8 columns, `b_col_maj=1` |
| build env | `C_FIFO_DEPTH=1`, `TB_MAX_N_ROWS=2` (the shipping settings) |
| incumbent | **128 x 64 x 64** |

---

## 1. Why re-sweep

`npu/sweep_tiling.sh` chose the tile our kernels ship with. It filters on

```bash
l1=$(( m*k + k*n + m*n*4 ))     # "the design double-buffers so budget ~half"
(( l1 > 32768 )) && continue
```

Tuning change #2 (`artifacts/e2e/tuned_results.md`) then set the C ObjectFifo
depth from 2 to 1 — C is written once per tile and drained, so it never needed
double buffering — making the real constraint `2mk + 2kn + 4mn <= ~62 KB`. **The
sweep was never re-run against that.**

The tell: the shipping tile `128x64x64` is itself **rejected by the old filter**
(45056 > 32768). It was found by hand-patching the one parameter the change was
known to unlock (`m=128`), not by searching the space the change opened. Of 96
tiles legal for the deployed shape under the current budget, only **33** were
reachable by the old sweep.

## 2. Sweep [MEASURED]

30 configurations nearest the incumbent, each correctness-gated by the design's
own PASS check. **29/30 PASS, 0 JIT-cache artifact collisions.** Raw:
`resweep.csv`.

| tile | TOPS | vs incumbent | |
|---|---:|---:|---|
| 64x128x80 | 13.817 | 1.136x | never evaluated |
| 64x128x64 | 13.712 | 1.127x | |
| 128x128x32 | 12.502 | 1.028x | never evaluated |
| **128x64x64** | **12.163** | **1.000x** | **incumbent — rank 4 of 29** |
| 64x128x32 | 10.430 | 0.858x | |
| 64x64x64 | 10.314 | 0.848x | |

**The lever is `k`, not `n`**, and the two mechanisms that hid the leaders
differ:

- `64x128x80` was **rejected outright** by the old filter (38912 > 32768).
- `64x128x64` **passed** the filter and was attempted — but under C depth 2 it
  needed 65536 B and could not build. Tried, and unbuildable before the change.

`milestone_a.md` had already recorded *"deep K prefers `k=128`"* for the
6912x2560 shape. It was never carried back to the unified 2560 kernel.

## 3. Interleaved confirmation [MEASURED] — and it shrinks the win

The incumbent measured **13.14 TOPS** standalone and **12.163** inside the
sweep: an 8% spread on the *same* configuration. A 1.14x gap cannot be promoted
on one-shot numbers, so the top three were re-measured **alternating within each
round**, 5 rounds x 20 iterations. Raw: `tile_ab.csv`.

| tile | L1 | median µs | sd | TOPS | sweep said | **interleaved** |
|---|---:|---:|---:|---:|---:|---:|
| **128x64x64** | 57344 | 1082.7 | 14.3 | 12.397 | 1.000x | **1.000x** |
| 64x128x80 | 57344 | 1044.2 | 4.9 | 12.854 | 1.136x | **1.037x** |
| **64x128x64** | **49152** | **1043.4** | **2.4** | **12.863** | 1.127x | **1.038x** |
| 128x128x32 | 57344 | 1179.0 | 28.4 | 11.384 | 1.028x | **0.918x** |

**The 1.136x collapses to 1.038x, and `128x128x32`'s apparent 1.028x gain is
actually a 0.918x regression.** One-shot sweep numbers on this machine carry
~10% of drift; the ranking survived, the magnitude did not.

## 4. What this is worth

`64x128x64` is genuinely and reproducibly better than the shipping tile:

- **1.038x** throughput, interleaved, correctness-gated
- **49152 B of L1 against 57344** — 8 KB more headroom
- **sd 2.4 µs against 14.3** — six times more stable run to run
- legal at every production M (512/1024/2048: all divisible by `m*4 = 256`),
  K=2560 % 128 = 0, N=2560 % (64*8) = 0

End to end it is **negligible**. NPU device time is ~24.5% of prefill, so
1.038x on the kernel is `0.245 x (1 - 1/1.038)` = **~0.9% of prefill**.

### Not promoted

Promoting means rebuilding every shipping xclbin and re-running the full
correctness and perplexity suite for **under 1% of prefill**. That trade is not
worth taking on its own. **If any future work rebuilds those artifacts anyway,
use `64x128x64`** — it is strictly better on all four axes above and there is no
reason to keep the current tile.

## 5. Two priors refuted

No number was predicted going in, because this session had already produced two
bf16 results that reverse sign in int8:

1. **`n` is the strongest lever, bigger is better** (bf16, row-major B: n 64→128
   gave 1.40–1.59x). In int8 with col-major B, large `n` is catastrophic —
   **n=320 gives 0.35–0.63 TOPS and n=160 gives 2.3–4.4**, against 12–14 at
   n=64/80. Measured on the first five points, which is why the sweep was
   reordered from largest-`n`-first to nearest-the-incumbent.
2. **`b_col_maj` costs 1.408x** (bf16, measured this session with `b_col_maj`
   the sole differing config field) — but was measured at **+35% faster** for
   our int8 kernel (`tuned_results.md` #1).

**The bf16 geometry surface is not a usable prior for the int8 path.** It was
right that the tile mattered and wrong about which dimension and which
direction. It earned its keep only as a reason to look.
