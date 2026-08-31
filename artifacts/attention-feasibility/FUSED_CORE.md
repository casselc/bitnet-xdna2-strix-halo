# The fused attention core

Testing the one lever the geometry gate's own measurements pointed at and did not
pursue: AMD's kernel wastes half the cores it occupies.

| | |
|---|---|
| branch base | `f44ae0215d0503cbc4dbb947341c6d023fa646af` (`attention-geometry-gate`) |
| branch | `attention-fused-core` |
| evidence source | `amd/IRON` @ `d9e4ec5fab71d34365befd8127f86c5a676a6ae1` |
| toolchain | `mlir_aie 1.4.2`, `llvm_aie 21.0.0.2026080301+c9c5ecb7`, Peano |

---

## 0. The observation

`GEOMETRY_GATE.md` measured the three stage times to identify the limiting
stage. Read as a **utilisation** statement they say something else:

| core | service | duty cycle |
|---|---:|---:|
| QK (row 2) | 3.938 µs | **36.9%** |
| softmax (row 3) | 9.412 µs | **88.3%** |
| PV (row 4) | 2.797 µs | **26.2%** |

A spatial pipeline runs at its **slowest** stage but pays for **all three
cores**: 31.98 µs of core-time to do 16.147 µs of work. **Compute efficiency
50.5%.** That is not an arithmetic problem — it is a 3.4:1 stage imbalance
mapped onto a topology that cannot absorb it, and it is most of the "5% of
device capability" this project has quoted since the feasibility pass without
ever decomposing it.

The projected fix: one core does all three stages for its own q block, and all
32 cores are used instead of the 24 an 8x3 pipeline occupies. Projected
**2.64x**. Every number below replaces a projection with a measurement.

---

## 1. Program memory: fits [MEASURED]

The gating feasibility question, run before any kernel work. AIE2P cores have
16 KiB of program memory and this project has already seen it overflow.

| | .text |
|---|---:|
| stock QK core | 1696 B |
| stock softmax core | 5328 B |
| stock PV core | 3744 B |
| *sum* | *10768 B* |
| **fused core, built** | **8656 B** |
| program memory | 16384 B |
| **headroom** | **7728 B (+47.2%)** |

It fits, and **better than the sum** — shared runtime and init code is emitted
once instead of three times. Verified as the whole footprint: the ELF has one
LOAD segment (R E, `0x21d0`) and `.text` is its only allocated section.

## 2. A second constraint, found by building [MEASURED]

```
tile (0, 3) requires 3 input/1 output DMA channels, but only 2 input/2 output available
```

A compute tile has **2 input DMA channels**. Q, K and V is three. The stock
design has 3-input cores only because its inter-stage fifos connect *vertically
adjacent* tiles and map to shared local memory, costing no channel; a fused core
has no neighbour to share with. Resolved naturally rather than worked around: K
and V for one kv block are always consumed together, so they share one stream and
are acquired two at a time — 2 in (Q, KV) / 1 out (O).

## 3. The constraint that actually bites: the free relayout [MEASURED]

The stock design's inter-stage fifos carry **layout transforms**:

- `memA` carries `a_dims`, converting the QK matmul's **tile-major** output into
  the **row-major** layout `partial_softmax` indexes with `A[i*B_kv + j]`
- `memP` carries `q_dims`, converting P back into the tile-major operand layout
  the PV matmul needs

**The stock pipeline performs two relayouts per pair for free, in the memtile
DMA.** A fused core has no inter-stage DMA and must pay for them on-core. The
"sum of stage times" projection did not account for this.

This was found the way it should be — by a wrong answer, not by reading. The
first fused build returned rel-L2 1.95 with 97% of elements out of tolerance.

## 4. Measured fusion cost [MEASURED]

One core, `B_q = B_kv = d = 64`, sweeping kv-block count and fitting
`t(n_kv) = intercept + slope * n_kv`. Raw: `fused_core.csv`, `fused_core.json`.

| | |
|---|---:|
| naive sum of stock stage times | 16.147 µs/pair |
| **measured fused core** | **18.845 µs/pair** (**1.167x** — fusing is not free) |
| + inter-stage relayout traffic | **19.065 µs/pair** (+1.2%, **lower bound**) |
| per-q-block cost | 93.687 µs (stock pipeline model fitted 68.38) |
| stock, core-time per pair | 31.981 µs (3 cores x 10.660) |

**The core result holds: fusing is worth 1.677x per core, and 2.237x at 32 fused
cores against 24 pipelined.** Real, and materially less than the 2.64x
projected, for three measured reasons: a single core cannot overlap what three
cores overlapped (1.167x), it must do the relayouts the DMA did for free
(≥1.012x), and its per-q-block cost is higher (93.7 vs 68.4 µs).

**The +1.2% relayout figure is a lower bound and should be read as one.** It is
`passThroughLine` over the block twice — the read+write *traffic* of a relayout
with none of the shuffle. A real transform costs more.

### What this measurement is, and is not [IMPORTANT]

**The fused kernel does not produce correct output.** rel-L2 ≈ 1.9, ~97% of
elements outside AMD's tolerance, because the inter-stage relayout of section 3
is absent. The timing is a valid measurement of *work performed* — the kernels
have no data-dependent control flow (the masks are held uniform by construction,
and `aie::exp2` is constant-time), and the output is finite, so no denormal or
NaN path is being taken — but **this is not a working attention kernel and no
claim here depends on it being one.**

An earlier version *did* silently skip its work: setting `q_block_idx` and
`S_eff` both to `1<<20` made `valid_q_rows` go negative, which takes
`partial_softmax`'s "fully padded block contributes nothing" path. It ran 1.7x
*faster* than the model predicts and returned NaN. The probe now refuses to fit
a slope through non-finite output, because timing a kernel that skipped its work
is worse than no measurement.

---

## 5. The economic gate [DERIVED]

d=128 stage times from the geometry probe, scaled by the **measured** fusion
penalty, over 32 cores. Switch tax and CPU oracle from the frozen prior
branches. Raw: `fused_gate.json`.

| T | config | C_qblock | prefill | + tax | CPU | vs CPU | gain |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2048 | d64 64x64 | 1x | 490 | 620 | 602 | 1.03x | −2.9% |
| 2048 | **d128 64x32** | 1x | 338 | 467 | 602 | **0.78x** | **+22.4%** |
| 2048 | d128 64x32 | 2x | 394 | 524 | 602 | 0.87x | +13.0% |
| **3968** | **d128 64x32** | **1x** | **1221** | **1351** | **1388** | **0.97x** | **+2.7%** |
| 3968 | d128 64x32 | 2x | 1334 | 1463 | 1388 | 1.05x | −5.4% |
| 3968 | d128 32x64 | 1x | 1436 | 1565 | 1388 | 1.13x | −12.8% |
| 3968 | d64 64x64 | 1x | 1712 | 1842 | 1388 | 1.33x | −32.7% |

### **Classification: B. POSSIBLE BUT MARGINAL**

At 4K — the length that matters and the one the gate is set at — the best case is
**+2.7%**, and that is with the *generous* `C_qblock` assumption. The physically
motivated one (the O tile doubles at d=128) gives **−5.4%**. Fusion alone at
d=64 is **−32.7%**: it is not enough by itself, and the win only appears when
stacked with the d=128 geometry the previous branch closed on its own.

The 2K column looks better (+22.4%) but 2K is not where attention hurts: it is
29.8% of prefill there against 51.0% at 4K.

---

## 6. Verdict

### **FUSED CORE CLOSED — MARGINAL**

The threshold set before measuring was **15% fully burdened attention-path gain
at 4K**, below which integration complexity and the accurate-softmax work would
erase it. The measured best case is **+2.7%**, on a kernel that does not yet
produce correct output and whose relayout cost is recorded as a lower bound.

**The observation was right and the payoff is not there.** The 50.5% compute
efficiency is real, fusing genuinely recovers most of it (2.237x measured), and
that is the largest single structural gain measured anywhere in this project's
attention work. It still lands inside the band the brief said not to build.

What is now known that was not before, and would be the starting point if anyone
reopens this:

1. **Program memory is not a constraint** — 8656 of 16384 B, with all three
   stages linked in.
2. **A fused core is DMA-channel-constrained to 2 inputs**, forcing K and V onto
   one stream.
3. **The stock design's inter-core DMA does two per-pair layout transforms for
   free.** Any fused design pays for them. This is the non-obvious cost and it
   is why "one core does the same work" understates the problem.
4. **Fusing costs 1.167x the naive sum of stage times**, measured.
5. The remaining gap at 4K is small enough that a *correct* fused kernel with a
   cheap on-core relayout, or a design that avoids the relayout by having the QK
   matmul emit row-major C directly, could plausibly cross 15%. That is a kernel
   research project, not a port, and it is not proposed as the next default.

This does not change the standing partition: **XDNA2** for BitNet linear
prefill, **Zen 5** for attention, decode and orchestration, **Radeon** for a
future larger local worker.
