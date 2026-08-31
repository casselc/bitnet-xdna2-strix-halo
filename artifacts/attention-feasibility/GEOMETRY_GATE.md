# The d=128 geometry gate

Closing the one uncertainty the final gate left open.

| | |
|---|---|
| branch base | `f2a3de392268df43cb34428e28e243d35d86a786` (`attention-final-gate`) |
| branch | `attention-geometry-gate` |
| evidence source | `amd/IRON` @ `d9e4ec5fab71d34365befd8127f86c5a676a6ae1` |
| toolchain | `mlir_aie 1.4.2`, `llvm_aie 21.0.0.2026080301+c9c5ecb7`, Peano |
| runtime reference | `ed97cfcac564be9f85db415faf076695b871e008` (`direct-output-closeout`, frozen) |

| tag | meaning |
|---|---|
| **[MEASURED]** | observed on this machine |
| **[DERIVED]** | arithmetic over measured quantities |
| **[MODEL]** | stage model, validated on held-out sequence lengths |
| **[NOT DONE]** | with the reason |

---

## 0. What the previous gate's bound did and did not establish

`FINAL_GATE.md` rejected the stock operator using an intentionally favourable
**work-proportional** falsification: pretend d=128 halves the entire d=64 kernel,
and it still loses. That framing stands, with one qualification this branch
exists to make precise.

**It is a favourable work-proportional bound, not a formal lower bound
independent of execution efficiency.** It holds if changing d=64 → d=128 leaves
efficiency-per-unit-of-QK/PV-work unchanged. It does not hold if d=128 maps the
arithmetic onto the AIE datapath materially better. That was the unmeasured
possibility, and it is what is measured here.

**The residual requirement is 1.58x, not 3.16x.** At ~4K:

| | |
|---|---|
| measured d=64 proxy | 3980.7 ms |
| burdened NPU kernel budget | 1258.3 ms |
| total stock-kernel deficit | **3.16x** |
| grant the absurd whole-kernel halving | 1990 ms |
| **residual geometry/efficiency gap still to close** | **1990 / 1258 ≈ 1.58x** |

3.16x is the *stock kernel's* deficit. 1.58x is what geometry would still have to
find **after** being handed the maximum possible softmax benefit. Only the second
is the hypothesis under test here.

---

## 1. The primitives, pinned [MEASURED / read from source]

Identified in the pinned tree, not inferred.

| | QK^T | softmax / rescale | PV |
|---|---|---|---|
| source | `aie_kernels/aie2p/mm.cc` | `aie_kernels/aie2p/softmax.cc` | `mm.cc`, row-major |
| MHA entry | `matmul_bf16_bf16_wrapper` | `partial_softmax` → `partial_softmax_bf16` | `matmul_PV` → `matmul_bf16_bf16_rowmaj` |
| primitive | `matmul_vectorized_2x2_mmul<bf16,bf16,r=8,s=8,t=8>` | 3-pass, `SM_VEC_LEN` vectors, `aie::exp2` LUT | same, `b_row_maj=true` |
| tile | `DIM_M=64 DIM_K=64 DIM_N=64`, `B_COL_MAJ` | rows of `B_kv=64` | `DIM_M=64 DIM_K=64 DIM_N=64` |
| in / acc / out | bf16 / `accauto` / bf16 | bf16, bf16 `m`,`l` statistics | bf16 / `accauto` / bf16 |
| flags | `-Dbf16_bf16_ONLY -DROUND_CONV_EVEN -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` | — | same, without `-DB_COL_MAJ` |
| cores | 8 (one per pipeline, row 2) | 8 (row 3) | 8 (row 4) |

The probe is **IRON's own `GEMM` operator**, which compiles that same `mm.cc`
with that same flag set and exposes `tile_m/tile_k/tile_n` and `b_col_maj`. That
is precisely where `d` enters the two matmuls:

```
QK^T : [B_q x d] @ [d x B_kv]     -> d is the CONTRACTION dim (tile_k)
PV   : [B_q x B_kv] @ [B_kv x d]  -> d is the OUTPUT width    (tile_n)
```

so this is a re-parameterisation of AMD's kernel, not a re-implementation.

**Control.** Every variant runs an identical `(M,K,N) = 2048^3`. Useful MACs
(8.590 GMAC) and total bytes moved are therefore identical across variants and
only the tile geometry differs, so efficiency is MACs/second and the ratio needs
no normalisation argument.

**Artifact identity.** IRON builds artifact names only from `repr=True` fields,
and `GEMM` declares `emulate_bf16_mmul_with_bfp16`, `round_conv_even`,
`prio_accuracy` and `dtype_in/out` as `repr=False` even though all four change
generated code — the same collision class that produced a bogus pipes=4 result
earlier in this project. Each variant therefore gets its own build directory
keyed on a SHA-256 of **every** field, cleared before use. Instruction-stream and
xclbin digests are recorded separately; the instruction streams reproduced
byte-identically across runs while the containers did not, confirming again that
the `.bin` digest is the executable identity. Raw: `geometry_qk_pv_c8.csv`,
`geometry_qk_pv_c1.csv`, `geometry_softmax.csv`, `geometry_summary.json`.

---

## 2. The first result is a constraint, not a ratio [MEASURED]

**d=128 at MHA's native 64x64 block does not build at all.** Peano's allocator:

```
A_L2L1 buff_0/1 : 16384 each     B_L2L1 buff_0/1 : 16384 each
C_L1L2 buff_0/1 :  8192 each     stack 3328   rtp 8
-> 85256 B required, 65536 B available
'aie.tile' op allocated buffers exceeded available memory
```

A closed-form L1 model (`l1_bytes` in the probe) reproduces **85256 exactly**,
and was checked against the allocator's own printed map before being used.
Applying it to **MHA's own buffer types** — `q_ty=(B_q,d)`, `k_ty=(d,B_kv)`,
`qk_ty=(B_q,B_kv)`, `of_depth=2` on every ObjectFifo, all read from `design.py`
— gives the same number for AMD's kernel:

| core | d=64 | d=128 |
|---|---:|---:|
| MHA QK core: Q 2x + K 2x + S 2x | 52488 B — fits | **85256 B — over** |
| MHA PV core: P 2x + V 2x + O 2x | 52488 B — fits | **85256 B — over** |

So this is tile arithmetic, not an artifact of the probe. **Every escape route
costs something:**

| escape route at native 64x64 | L1 | |
|---|---:|---|
| single-buffer Q only | 68872 B | still over |
| single-buffer K only | 68872 B | still over |
| single-buffer **both** inputs | 52488 B | fits, but surrenders *all* input compute/DMA overlap |
| shrink `B_q` 64 → 32 | 60680 B | fits |
| shrink `B_kv` 64 → 32 | 60680 B | fits |

A d=128 port is therefore **forced** to give something up. This is a first-order
finding that no amount of geometry efficiency removes, and it was not visible
from the work-proportional bound.

---

## 3. QK and PV geometry, both escape routes [MEASURED]

Measuring only one block-reduction route would make the answer an artifact of my
choice, so both are measured, each with **its own d=64 control at the same block
shape** — which is what separates "the effect of d" from "the effect of a
smaller block".

**32 cores (`cols=8`), the regime MHA runs in (24 of 32 cores):**

| stage | route | geometry (d128/d64, same block) | block cost (d64, vs 64x64) | **NET vs d=64 64x64** |
|---|---|---:|---:|---:|
| QK | A: `B_q` 64→32 | 1.249 | 0.587 | **0.733** |
| QK | B: `B_kv` 64→32 | 1.020 | 0.722 | **0.737** |
| PV | A: `B_q` 64→32 | 1.388 | 0.656 | **0.911** |
| PV | B: `B_kv` 64→32 | 1.588 | 0.884 | **1.404** |

**d=128 geometry genuinely does help the primitive** — 1.02x to 1.59x at a
matched block, exactly the mechanism the hypothesis proposed. But the forced
block reduction costs 0.59x to 0.88x, and **for QK the block cost is the larger
effect in both routes**. PV comes out ahead; QK does not.

### The ratios are not a pure per-core compute property [MEASURED]

The core-count control (`cols=1`, 4 cores) is reported because it disagrees, and
the disagreement is informative rather than noise:

| | 32 cores | 4 cores |
|---|---:|---:|
| QK route A, NET | 0.733 | 1.338 |
| QK route B, NET | 0.737 | 0.662 |
| PV route A, NET | 0.911 | 1.538 |
| PV route B, NET | 1.404 | 1.420 |

The moving part is the **block-reduction penalty**, not the geometry gain: for
QK route A it is 0.587 at 32 cores and 0.924 at 4. Halving `B_q` doubles the
number of C tiles and so the output DMA descriptors, which costs much more when
the array is already near its data-movement limit. In other words the block
penalty is largely a **data-movement** penalty and it bites harder the busier the
array. MHA runs 24 of 32 cores, so the 32-core column is the representative one,
and it is the one used below. Both regimes agree on the two robust conclusions:
**QK route B is worse at d=128, PV route B is better.**

---

## 4. Softmax, 40 heads vs 20 [MEASURED]

Measured on the same `softmax.cc` that `mha.cc` `#include`s, at `cols=64` = MHA's
key block. Raw: `geometry_softmax.csv`.

| rows | elements | kernel ms | ns/row | Melem/s |
|---:|---:|---:|---:|---:|
| 1024 | 65536 | 0.2309 | 225.52 | 283.8 |
| 4096 | 262144 | 0.6718 | 164.01 | 390.2 |
| 16384 | 1048576 | 2.4789 | 151.30 | 423.0 |
| 65536 | 4194304 | 9.6377 | 147.06 | 435.2 |

Linear fit: **t(ms) = 0.0812 + 145.84 ns/row**. The fixed term is a per-dispatch
cost of 0.081 ms; at MHA's scale (hundreds of thousands of rows per layer) it is
negligible, so softmax **is** linear in row count and the 20-head / 40-head ratio
is **0.500**, not held above 0.5 by any fixed overhead. This is the one place
where d=128 gets exactly the benefit the work argument assumed.

**Proxy caveat, stated because it runs against the verdict.** The standalone
operator calls `softmax_bf16` → `softmax_simple_bf16`; MHA calls
`partial_softmax_bf16` → `partial_softmax_alias_bf16`. Same three passes, same
`SM_VEC_LEN` vectorisation, differing only in the online max/sum bookkeeping and
the scale multiply — so the standalone slightly **understates** MHA's softmax
stage. Understating softmax understates how much halving it can win, which is
*anti*-conservative for a negative verdict. Section 5 removes the concern by
calibrating against the real fused kernel instead.

---

## 5. Reconstructing the d=64 pipeline [MODEL, validated on holdout]

`d128_time = 0.5 * d64_time` is retired here. Structure read from `design.py`:
8 pipelines, each a spatial chain QK core → softmax core → PV core, `B_q=B_kv=64`,
causal so q block *i* consumes kv blocks 0..*i*.

```
t_layer = pairs_per_pipeline * service + q_blocks_per_pipeline * C_qblock
```

`service` is the steady-state cost of one (q block, kv block) pair — a spatial
pipeline runs at its slowest stage. `C_qblock` is work done **once per q block**
rather than per pair: streaming Q in, `init_scale_buffer`, `rescale_O` over the
whole O tile, and writing O out.

A pure steady-state model (no `C_qblock`) underpredicts by **26% at 4K and 63% at
512** — and the error growing as S shrinks is the signature of exactly that
missing per-q-block term, since the ratio of q blocks to pairs is highest at
small S. That failure is what motivated the second term; it is not a free
parameter added to improve a fit.

**Fitted on the two largest lengths only, the two smallest held out:**

| S | pairs/pipe | qblk/pipe | measured | predicted | error | role |
|---:|---:|---:|---:|---:|---:|---|
| 512 | 180 | 40 | 4.611 ms | 4.654 ms | **+0.93%** | **HOLDOUT** |
| 1024 | 680 | 80 | 12.670 ms | 12.719 ms | **+0.39%** | **HOLDOUT** |
| 2048 | 2640 | 160 | 39.084 ms | 39.084 ms | 0.00% | fit |
| 4096 | 10400 | 320 | 132.749 ms | 132.749 ms | 0.00% | fit |

Worst holdout error **0.93%**, against the required 10%. The model is usable.

### Which stage sets the rate — measured, not inferred [MEASURED + MODEL]

Standalone stage times per 64x64 block at d=64:

| stage | per block |
|---|---:|
| QK | 3.938 µs |
| **softmax** | **9.412 µs** |
| PV | 2.797 µs |

The fitted `service` is **10.660 µs** — **1.133x** the standalone softmax stage,
but **2.71x** QK and **3.81x** PV. No combination of QK and PV accounts for it,
and the 13% excess over softmax is the right size and sign for
`partial_softmax_alias_bf16`'s extra online bookkeeping over
`softmax_simple_bf16`. Two independent lines of evidence therefore agree:

> **The stock d=64 pipeline is SOFTMAX-limited.**

This is the case the previous brief called CASE C, and it is why d=128 was worth
measuring rather than assuming. `C_qblock` fits at **68.38 µs**.

*(One alternative reading, stated for completeness: the 10.660 µs could instead
be `max(stage)` plus imperfect pipeline overlap, since the serial sum is
16.15 µs. Either reading leaves softmax dominant, and the calibration factor is
applied uniformly below so the forecast does not depend on which is right.)*

---

## 6. The d=128 forecast [MODEL]

d=128 BitNet is 20 Q heads / 5 KV heads: **half the softmax rows, identical QK/PV
arithmetic**. The block must shrink (section 2), so both routes are forecast.
`C_qblock` is reported as a band — the O tile it rescales doubles at d=128, so
`2x` is the physically motivated case and `1x` is a generous bound.

| S | block | t_QK | t_softmax | t_PV | limiting | d=64 measured | d=128 predicted | ratio |
|---:|---|---:|---:|---:|---|---:|---:|---:|
| 2048 | 64x32 | 5.34 | 4.71 | 1.99 | **QK** | 39.08 ms | 21.45 – 26.92 ms | 0.55 – 0.69 |
| 4096 | 32x64 | 5.37 | 4.71 | 3.07 | **QK** | 132.75 ms | 85.15 – 107.03 ms | 0.64 – 0.81 |
| 4096 | 64x32 | 5.34 | 4.71 | 1.99 | **QK** | 132.75 ms | **73.88 – 84.82 ms** | **0.56 – 0.64** |

**d=128 is a real improvement — roughly 1.8x over the stock d=64 kernel — and it
arrives exactly by the hypothesised mechanism.** Halving the softmax rows drops
that stage from 9.41 µs to 4.71 µs per pair.

**And that is precisely why it is not enough.** The pipeline **rebalances**:
softmax stops being the bottleneck and **QK becomes the limiting stage at
5.34 µs**. Section 3 measured QK as the one stage that does *not* improve at
d=128 once the forced block reduction is paid (NET 0.733–0.737). The two results
compose: d=128 spends its entire softmax win and lands on the stage geometry
cannot help.

---

## 7. The economic gate [DERIVED]

Switch tax and CPU oracle from the frozen prior branch; budgets recomputed by
`tools/geometry_gate.py` from the measured files. Raw: `geometry_gate.json`.

| T | CPU attn | switch tax | budget | stock d=64 | **d=128 best** | d=128 worst | best + tax | vs CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 50.9 | 130 | −78.7 | 138.3 | 73.7 | 197.0 | 203.3 | **3.99x** |
| 1024 | 165.2 | 130 | 35.6 | 380.1 | 205.5 | 452.3 | 335.1 | **2.03x** |
| 2048 | 602.0 | 130 | 472.4 | 1172.5 | 643.4 | 1138.2 | 773.0 | **1.28x** |
| **3968** | **1388.0** | **130** | **1258.3** | **3982.5** | **2216.3** | 3210.8 | **2346.0** | **1.69x** |

At 4K the best case — most favourable block, most generous `C_qblock` — is
**1.69x slower than the CPU** fully burdened, a **−69% attention-path "gain"**.
Classification **A. CLEAR NEGATIVE** at every context length.

**What would have to be true.** At d=128 the limiting stage is QK, so the
residual is a statement about one primitive: per-pair service would have to fall
from the modelled **6.05 µs to 2.98 µs**, a further **2.03x** on the stage that
section 3 measured as *not* improving with d=128.

---

## 8. Hard geometry-efficiency check [MEASURED]

Reported per stage, not averaged — averaging unrelated stage speedups into one
global number would be meaningless.

| quantity | measured | sanity threshold ~1.5x |
|---|---|---|
| **QK efficiency, d128/d64**, matched block | 1.249 (route A), 1.020 (route B) | below |
| **QK NET**, after forced block reduction | **0.733 / 0.737** | far below — a *regression* |
| **PV efficiency, d128/d64**, matched block | 1.388 (route A), 1.588 (route B) | at/above |
| **PV NET**, after forced block reduction | 0.911 / **1.404** | route B clears it |
| **softmax work, 20 heads / 40 heads** | **0.500** exactly | as assumed |

PV alone clears the sanity threshold. It does not matter, because **PV is the
fastest of the three stages at d=64 (2.797 µs) and stays the fastest at d=128
(1.99 µs)**. Improving the stage that was never the bottleneck moves nothing. The
stage that becomes the bottleneck, QK, **regresses to 0.73x**.

That is the whole finding in one line: *the geometry hypothesis was right about
the mechanism and wrong about the outcome, because the win lands on the wrong
stage.*

---

## 9. Numerical status

Unchanged from the prior branch, and deliberately so. Everything was timed with
AMD's stock numerics: bf16 online-softmax statistics, the hardware `exp2` LUT,
bf16-rounded `log2e`. The softmax probe's rel-L2 of 0.025–0.046 (against 0.012
for the matmuls) is that LUT error showing up directly, consistent with AMD's own
documented 6.1–49.1% figure.

The accurate path — f32 statistics, `exp2f_vec`, exact `log2e` — was **[NOT
DONE]**, correctly: all three **add** cost, and they would be added to QK's
neighbour on the critical path. Task 7 of the brief forbids it while the
economic case is negative, and it is negative.

---

## 10. Verdict

### **ATTENTION GEOMETRY CLOSED — NEGATIVE**

| question | answer |
|---|---|
| does d=128 fit MHA's native block? | **No** — 85256 B against 64 KiB L1, confirmed against MHA's own buffer types |
| does d=128 improve the primitives? | **Yes, partially** — QK 1.02–1.25x, PV 1.39–1.59x at matched block |
| does it survive the forced block reduction? | **QK no (0.73x), PV yes (1.40x)** |
| does softmax halve? | **Yes, exactly 0.500** |
| what limits d=64? | **softmax**, 9.41 µs vs QK 3.94 / PV 2.80, confirmed by a holdout-validated model |
| what limits d=128? | **QK** — the pipeline rebalances onto the stage that regressed |
| d=128 vs stock d=64 | **~1.8x better** |
| d=128 vs the CPU at 4K, burdened | **1.69x slower**, best case |
| residual requirement | a further **2.03x** on QK |

**The straightforward IRON path and the remaining d=128 geometry hypothesis are
now both closed for this controller.**

**This does not prove no novel XDNA2 attention kernel can ever win.** A novel
kernel would be a new research project requiring a materially different pipeline
and utilisation model — and this pass sharpens what it would have to do rather
than merely asserting one could exist:

1. It must not be a three-stage spatial pipeline over 24 of 32 cores whose
   limiting stage is a scalar-ish softmax. Fixing softmax alone just relocates
   the bottleneck to QK, which is measured here.
2. It must handle `d=128` without paying the block-reduction penalty — either
   more L1 per core, a different data layout, or a fundamentally different
   blocking. The 64 KiB L1 wall at 85256 B is the hard constraint.
3. It must remove or amortise the 130 ms/prefill context switch, which means
   living in the same xclbin as the GEMM.

None of those is proposed as the next default project. The supported reference
partition stands: **XDNA2** for BitNet linear prefill, **Zen 5** for attention,
decode and orchestration, **Radeon** for a future larger local worker.
