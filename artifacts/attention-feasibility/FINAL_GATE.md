# Attention on XDNA2: the final gate

Closing the uncertainty the stock d=64 experiment left open, with three
measurements and one falsification test.

| | |
|---|---|
| base | `0328d7f6e1ec95e53d322637d020a86d4018a24e` (`attention-feasibility`) |
| runtime reference | `ed97cfcac564be9f85db415faf076695b871e008` (`direct-output-closeout`, frozen) |
| branch | `attention-final-gate` |
| stock operator | `amd/IRON` @ `d9e4ec5fab71d34365befd8127f86c5a676a6ae1` |
| toolchain | `mlir_aie 1.4.2`, `llvm_aie 21.0.0.2026080301+c9c5ecb7`, Peano |

| tag | meaning |
|---|---|
| **[MEASURED]** | observed on this machine |
| **[DERIVED]** | arithmetic over measured quantities, reproducible by `tools/attention_gate.py` |
| **[BOUND]** | a deliberately impossible best case, used to falsify |
| **[NOT DONE]** | with the reason |

---

## 0. Scope corrections carried forward

`RESULTS.md` on the base commit stands as published. Four things in it need to be
read more narrowly than they were written, and this section is the correction.
Nothing there is deleted or rewritten.

**The d=64 proxy is FLOP-equivalent, not softmax-equivalent.** 40 Q heads /
10 KV heads at d=64 performs exactly the same QK^T and PV arithmetic as BitNet's
20 / 5 at d=128, with the same GQA ratio. It does **not** perform the same
softmax work: at a given query position the proxy normalises **40** head-wise
rows where real BitNet normalises **20**, each over the same number of keys. The
proxy therefore does **2x** BitNet's softmax work and **1x** its QK/PV work. If
softmax sets the throughput, real d=128 geometry could behave better than the
proxy. Section 4 is what settles that, and it settles it without needing to
identify the stage.

**"Even perfect scaling to all 32 cores gives 0.73 TFLOPS" does not prove d=128
cannot rebalance the stages.** It is an argument about core count only. It says
nothing about how work redistributes across a three-stage pipeline when the
head geometry changes. Treat it as one datapoint about occupancy, not as a
disproof of d=128.

**The reject is scoped to the existing AMD operator and a straightforward port of
it.** It is not, and was not, a claim that no XDNA2 attention kernel can win.

**The "~4x required improvement" figure should not be repeated.** It was not
derived from a burdened budget. Freshly derived below (section 3), the
break-even requirement is **2.50x at 2K and 3.16x at 4K** — smaller than 4x, and
the case fails anyway. The measured deficit does not need exaggerating.

---

## 1. The d=64 proxy at 4K [MEASURED]

The measurement the earlier pass was missing. Same validated FLOP-equivalent
configuration, 8 pipelines, 20 timed reps after 3 warmups, `--fresh` build per
case. Raw: `npu_mha_4k.csv`. Tool: `tools/attention_npu_probe.py`.

### 3968 is not free: the design pads it to 4096

`iron/operators/mha/design.py:136` pads the sequence to a multiple of
`B_q * pipelines` = 512 and **executes the padded length**. A request for
S=3968 at 8 pipelines runs 4096 tokens of work. Measured side by side, they cost
the same to within 0.43%:

| requested S | executed S | kernel ms/layer |
|---:|---:|---:|
| 3968 | **4096** | 132.184 |
| 4096 | 4096 | 132.749 |

So the operator cannot express BitNet's 3968-token context more cheaply than
4096, and the 3968 row below is credited with only 3968 tokens' worth of work
while paying for 4096 — a **6.6%** handicap that is *in the NPU's favour* in
every ratio that follows, since it is charged to the CPU's smaller workload.
4096 and 3968 are not the same thing and are not reported as such.

### Results

`kernel` is XRT dispatch..wait as the runtime itself measures it. `staging` is
host->device sync of Q/K/V plus device->host of O, measured separately. `burden`
is what an integration would actually pay per layer.

| S | S exec | kernel ms | staging ms | **burden ms** | **x30 layers** | rel L2 | outside AMD tol |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 512 | 4.611 | 0.108 | 4.729 | **141.9 ms** | 0.01423 | 0.000% |
| 1024 | 1024 | 12.670 | 0.106 | 12.783 | **383.5 ms** | 0.01382 | 0.000% |
| 2048 | 2048 | 39.084 | 0.219 | 39.319 | **1179.6 ms** | 0.01304 | 0.000% |
| **3968** | **4096** | **132.184** | **0.466** | **132.689** | **3980.7 ms** | **0.01316** | **0.000%** |
| 4096 | 4096 | 132.749 | 0.467 | 133.239 | 3997.2 ms | 0.01318 | 0.000% |

The whole sweep was run **twice**, each time from a cleared build directory, and
the two runs agree to within **0.3%** at every length (e.g. 3968: 132.157 then
132.184 ms). The table is the second run; the first is in this branch's history.

Staging is negligible — 0.1 to 0.5 ms per layer — which confirms the earlier
finding that this workload is not data-movement-bound. Correctness holds at every
length including 4K: **0.000% of elements outside AMD's own `rel_tol=4e-2 /
abs_tol=1.5e-1` gate**. This remains a correct kernel that is slow.

### Against the CPU oracle

CPU figures are the in-situ per-node measurements from `cpu_oracle.csv`,
15 threads, warmup prefill dropped. Same work, wall time compared directly.

| T | CPU attention | NPU d=64 proxy | | NPU rate | CPU rate |
|---:|---:|---:|---|---:|---:|
| 512 | 50.9 ms | 141.9 ms | NPU **2.79x slower** | 0.28 TFLOPS | 0.79 TFLOPS |
| 1024 | 165.2 ms | 383.5 ms | NPU **2.32x slower** | 0.42 TFLOPS | 0.97 TFLOPS |
| 2048 | 602.0 ms | 1179.6 ms | NPU **1.96x slower** | 0.55 TFLOPS | 1.07 TFLOPS |
| **3968** | **1388.0 ms** | **3980.7 ms** | NPU **2.87x slower** | **0.61 TFLOPS** | **1.74 TFLOPS** |

**The gap widens at 4K, which is the length that matters most.** Both engines
speed up with context, but not equally: from 2K to 4K the CPU's rate rises 1.63x
(1.07 -> 1.74 TFLOPS) while the NPU's rises only 1.11x (0.55 -> 0.61). The NPU
scales almost exactly with the arithmetic and gains almost nothing from the
longer sequence; the CPU gains substantially, because at 4K one layer's KV is
10.2 MiB and sits in the 64 MiB L3.

This reverses the hope the earlier trend invited. Reading 2.71x -> 2.31x -> 1.95x
across 512/1024/2048, one could reasonably have extrapolated that the NPU closes
the gap at longer context. **It does not. It reopens it.** That extrapolation is
the specific thing this measurement existed to test, and the answer is no.

---

## 2. The real GEMM <-> MHA context-switch cost [MEASURED]

The earlier ~153 ms/prefill figure came from a **surrogate**: two production-sized
GEMM designs alternating, because no MHA xclbin existed. One now does, so this
replaces the surrogate with the actual pair. Tool:
`tools/npu_gemm_mha_switch.py`. Raw: `gemm_mha_switch_s2048.json`,
`gemm_mha_switch_s3968.json`.

Both contexts are held open simultaneously, which is the condition being
measured: XDNA2 hardware contexts hold all 8 columns and cannot be co-resident,
so every alternation reconfigures the array. "Alone" and "alternating" samples
are **interleaved within each cycle** rather than run in blocks, because this
machine drifts 10-30% between runs; "alone" is sampled after three same-context
dispatches so the context is certainly resident. 40 cycles, medians.

### Artifact identity

Recorded before timing, because this project has twice been fooled by loading the
wrong xclbin — once via `argv[1]` picking up the M512 designs, once via IRON's
`repr=False` name collision on `num_of_pipelines`.

| role | artifact | bytes | SHA-256 |
|---|---|---:|---|
| GEMM | `artifacts/xclbin-tuned/mm_M1024_K2560_N2560.xclbin` | 109022 | `363319183d5a442eeaa9fd2b2c96f3e9df080ff1fe91a1f65119f96c310cb1e7` |
| GEMM | `artifacts/xclbin-tuned/mm_M1024_K2560_N2560.insts.bin` | 5520 | `0cdea8e2932e04affe846293a8d3c30285a33fa5fc29ea1b74c66f7ac07abc24` |
| MHA | `MHA_h40_s2048_d64_kv10_npu2.xclbin` | 133246 | see `gemm_mha_switch_s2048.json` |
| MHA | `MHA_h40_s3968_d64_kv10_npu2.xclbin` | 133246 | see `gemm_mha_switch_s3968.json` |

The GEMM xclbin is byte-for-byte the one the frozen `direct-output-closeout`
runtime loads. The MHA xclbin is built fresh by this branch into a cleared build
directory.

**One reproducibility caveat, found by building the same geometry twice.** The
MHA **xclbin container is not byte-reproducible**: two builds of
`MHA_h40_s2048_d64_kv10_npu2` from cleared directories produced different
SHA-256 digests. The **instruction stream is** — the `.bin` hashed
`eb867698602b4e1b9f77b0cdde47bd3d47bc6abc9ba8276ba795928d481b2339` in both
builds, and the 3968 one hashed `580e20e4...` in both. So the *program* the array
executes is reproducible and the container's packaging is not, which is why the
`.bin` digest is the identity to trust. The two builds also timed identically
(39.077 vs 39.060 ms), which is the behavioural confirmation.

### Result

| S | context | alone | alternating | penalty |
|---:|---|---:|---:|---:|
| 2048 | GEMM M1024 K2560 N2560 | 0.984 ms | 3.461 ms | **+252%** |
| 2048 | MHA | 39.060 ms | 40.903 ms | +5% |
| 2048 | **pair** | 40.044 ms | 44.364 ms | **+11%, +4.320 ms** |
| 3968 | GEMM M1024 K2560 N2560 | 0.984 ms | 3.492 ms | **+255%** |
| 3968 | MHA | 132.186 ms | 134.002 ms | +1% |
| 3968 | **pair** | 133.170 ms | 137.495 ms | **+3%, +4.324 ms** |

**+4.32 ms per GEMM<->MHA alternating pair, and it is flat across sequence
length** (4.320 vs 4.324 ms, a 0.1% difference over a 2x change in MHA context
size). The cost is dominated by reloading the *GEMM* context — a 1 ms dispatch
that becomes 3.5 ms — not the larger MHA one, whose reload is amortised over
39-132 ms of work.

The surrogate over-estimated: **4.32 ms measured vs 5.10 ms surrogate**, so the
real tax is **15% cheaper** than the number the earlier pass used. Correcting it
moves the budget in the NPU's favour, and the conclusion still does not change.

### Counting assumption, stated explicitly

Per BitNet layer the graph runs `q,k,v` GEMMs -> attention -> `o` GEMM ->
`gate,up` GEMMs -> `down` GEMM. Offloading attention introduces exactly
**two transitions per layer** (GEMM->MHA and MHA->GEMM), which is **one measured
alternating pair per layer**:

```
switch_tax(prefill) = 4.32 ms/pair x 1 pair/layer x 30 layers = 130 ms
```

If an implementation needed two pairs per layer the tax doubles to 259 ms. The
gate below uses the cheaper, more favourable assumption of one.

---

## 3. Revised economic budgets [DERIVED]

```
budget_NPU(T) = CPU_attention(T) - switch_tax(T)
```

Computed by `tools/attention_gate.py` from the measured files; raw output in
`attention_gate.json`.

| T | CPU attention | switch tax | **budget for an NPU kernel** | measured d=64 proxy | over budget by |
|---:|---:|---:|---:|---:|---:|
| 512 | 50.9 ms | 129.6 ms | **-78.7 ms** | 141.9 ms | *no budget exists* |
| 1024 | 165.2 ms | 129.6 ms | **35.6 ms** | 383.5 ms | 10.8x |
| **2048** | **602.0 ms** | **129.6 ms** | **472.4 ms** | **1179.6 ms** | **2.50x** |
| **3968** | **1388.0 ms** | **129.7 ms** | **1258.3 ms** | **3980.7 ms** | **3.16x** |

At T=512 the switch tax **alone** exceeds the entire CPU attention time, so the
NPU cannot win there at any kernel speed whatsoever.

**Break-even requires 2.50x at 2K and 3.16x at 4K** over a kernel AMD ships and
tests on this silicon — and that is break-even, not a margin worth integrating
for. These supersede the "~4x" figure quoted earlier, which was not derived from
a burdened budget.

---

## 4. The impossible-halving falsification [BOUND]

The generous test the softmax caveat demands. **Pretend that moving d=64 -> d=128
halves the entire kernel.**

This deliberately overstates d=128's possible benefit. The proxy is
FLOP-equivalent, so QK^T and PV arithmetic do not shrink **at all**; only the
softmax row count halves (40 -> 20 per query position). In a spatial pipeline
total time is at least the limiting stage's time, and under d=128 every stage's
work is at least half its d=64 value — QK and PV unchanged, softmax halved —
so the limiting stage's time is at least halved, and

```
d128_time >= 0.5 x d64_proxy_time
```

is a genuine lower bound on the kernel, holding **whatever the limiting stage
turns out to be**.

| T | 0.5 x measured proxy | + switch tax | CPU attention | |
|---:|---:|---:|---:|---|
| 512 | 70.9 ms | 200.5 ms | 50.9 ms | **FAILS 3.94x** |
| 1024 | 191.7 ms | 321.4 ms | 165.2 ms | **FAILS 1.95x** |
| **2048** | **589.8 ms** | **719.4 ms** | **602.0 ms** | **FAILS 1.20x** |
| **3968** | **1990.3 ms** | **2120.1 ms** | **1388.0 ms** | **FAILS 1.53x** |

**No context survives, and 4K — the length with the most to gain — fails worst of
the two long ones.** An impossible d=128 kernel that halves work the FLOP
analysis says cannot halve is still 1.53x slower than Zen 5 at 4K.

This is **CASE A** of the decision logic: close negative, do not port d=128.

### Why this also disposes of the stage question

Task 5 (identify QK vs softmax vs PV as the limiting stage) is gated on the
optimistic bound leaving an economic possibility. It does not, so the
discriminator was **[NOT DONE]** — and it could not have changed the outcome:

The entire reason to identify the stage was the possibility that softmax
dominates, since softmax is the only stage where d=128 does less work than the
proxy. The bound above already grants the *maximum* benefit that a fully
softmax-limited kernel could produce — halving the whole kernel — and it still
fails by 1.53x at 4K. A QK- or PV-limited kernel would do **worse**, since those
stages' work is identical between the proxy and d=128. Every possible stage
attribution is therefore already covered, and every one loses.

**The one thing this does not cover**, stated plainly: the bound is an argument
about *work*, not about *efficiency per FLOP*. A d=128 kernel with wider output
tiles could in principle execute the same QK/PV arithmetic more efficiently than
the d=64 one does. But it would have to be **more than 3.16x more efficient at
4K on identical arithmetic**, against a kernel AMD wrote and tests on this
silicon. That is not a port; that is the new-kernel research listed under
reopening conditions below.

---

## 5. Pipeline bottleneck

**UNRESOLVED — deliberately, per the Task 5 gate.**

What *is* established, from the base commit's measurement (pipeline-count
scaling on cleared builds: 8/4/2 pipelines -> 12.670 / 22.404 / 41.469 ms at
S=1024, near-linear): the kernel is **compute/occupancy-bound, not
data-movement-bound**. Section 1's staging measurement independently confirms it
from the other side — 0.1 to 0.5 ms of host staging against 4.6 to 132 ms of
kernel.

Which of QK, softmax or PV sets that compute bound was not determined. Section 4
explains why determining it cannot change the verdict. The smallest additional
discriminator, if the question is ever reopened, is recorded in section 7.

---

## 6. Numerical status

**Adequate as a performance probe; not production-quality BitNet attention.**

The stock kernel was measured **as AMD ships it**, with its loose numerics
intact: bf16 online-softmax statistics `m`/`l`, the hardware `exp2` LUT (6.1% to
49.1% relative error over softmax's range, per AMD's own `exp2f_vec.cc`), and a
bf16-rounded `log2e` (1.4453125 vs 1.4426950, a +0.18% systematic temperature
shift).

Measured against a numpy f32 golden it holds **rel-L2 ~0.013 and 0.000% of
elements outside AMD's `rel_tol=4e-2 / abs_tol=1.5e-1` gate at every length
including 4K**. That is good enough to trust the *timing*; it is not good enough
to put in front of a model.

The accurate path — f32 statistics, `exp2f_vec` (8.9e-5 error, already installed
in our venv), exact `log2e` — was **[NOT DONE]**, correctly. All three fixes
**add** cost, and the economic case is negative before paying for any of them.
Implementing them could only widen the deficit.

---

## 7. Verdict

### **ATTENTION CLOSED — STOCK/PORT PATH REJECTED**

| question | answer |
|---|---|
| correct at 4K? | **Yes** — 0.000% outside AMD's tolerance, rel-L2 0.01316 |
| faster at 4K? | **No** — 3980.7 ms vs the CPU's 1388.0 ms, **2.87x slower**, burdened |
| does the gap close with context? | **No** — it widens from 1.96x at 2K to 2.87x at 4K |
| real switch tax? | **130 ms/prefill**, measured on the actual pair (surrogate said 153) |
| budget at 4K? | 1258.3 ms; the kernel needs **3.16x** just to break even |
| survives halving the whole kernel? | **No** — fails by 1.53x at 4K, at every length |
| could the stage bottleneck rescue it? | **No** — the halving bound already grants the best case |
| data movement? | **Not the constraint** — 0.1-0.5 ms staging per layer |
| numerics? | Adequate to time; the fixes that make it trustworthy add cost |

**This closes straightforward use or porting of the existing IRON MHA for this
controller. It does not prove that no novel XDNA2 attention kernel can ever
outperform the CPU.**

What would be required to reopen the question — all of it new research, none of
it a port:

1. **A kernel with materially different pipeline structure and utilisation.**
   AMD's uses 24 of 32 cores across three rows (QK -> softmax -> PV) and reaches
   0.61 TFLOPS against ~11 TOPS for the same device on int8 GEMM — about 5% of
   demonstrated capability. The headroom is real; nothing in reach exploits it.
2. **A same-context, co-resident design** placing attention in the *same* xclbin
   as the GEMM, removing the 130 ms tax outright. AMD's MHA already uses 24 of 32
   cores, so this is a redesign of both kernels, not a merge.
3. **Some new measured hardware capability** — an f32-accurate `exp2` datapath, a
   larger core program memory, or co-resident hardware contexts — that changes
   the constants this gate is built on.

If the question is reopened, the **smallest useful next discriminator** is the
one deliberately skipped here: stage-local timing inside the three-stage pipeline
(AIE event profiling per worker, or a diagnostic variant replacing one stage with
a passthrough) to establish whether QK, softmax or PV sets the rate. It is cheap,
and it is the first thing a new-kernel effort would need. It is **not** worth
running to defend a d=128 port, because section 4 shows no stage attribution can
make that port pay.

**The cheap gate did its job twice.** The base commit's d=64 measurement avoided a
head_dim port. This pass's 4K measurement avoided the specific error of
extrapolating 2.71x -> 2.31x -> 1.95x into a win at long context — the trend
reverses at 4K, and only measurement could have shown that.
