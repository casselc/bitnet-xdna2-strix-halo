# Direct-output runtime: closeout and frozen reference

Hardens the direct-output runtime into a candidate reference: proves (or fixes) a
concurrency contract, stress-tests it, removes the remaining deep-K output copy,
and re-validates the cost model at long context.

**Base:** `9f6008aa1e46e9d0d54746cbc1e975ecc5cd9526` (`direct-output-arena`, frozen
and not modified by this pass).
**Evidence-producing source:** `a2b0885ed54bee214d67cc2636753d7fbd4b01fb` -- the
last commit on this branch that changes `runtime/` or `patches/`. Everything
after it is artifacts and prose. The live branch tip comes from `git rev-parse`
and is deliberately not embedded here.

Untouched: `main` `885df0ca`, `next-pass-results` `fb4493e9`, `overlap-de-risk`
`3dff59bb`, `direct-output-arena` `9f6008aa`.

| tag | meaning |
|---|---|
| **[MEASURED]** | observed on this machine; raw data in this directory |
| **[DERIVED]** | arithmetic over measured quantities |
| **[DEFERRED]** | not done, with the reason |

---

## 1. Concurrency contract [MEASURED]

**The runtime was not safe for two inference contexts in one process. It is now,
by explicit serialization.**

`accumulate()` publishes process-global state -- `g_direct` (which output slots
hold this tensor's results), `g_acc`, `g_cur_shape` -- and returns. The scaling
epilogue reads that state afterwards, on every ggml worker, holding no lock.
Within one graph the ggml barrier orders the two phases; across two contexts it
orders nothing, so context B overwrites `g_direct` and reuses the same output
slots while context A's workers are still reading them.

`tests/test_xdna_concurrent.cpp` reproduces that schedule directly. Each mode
runs in its own exec'd child so a crash is reported rather than taking the
harness down.

| mode | result |
|---|---|
| unleased (raw API) | **CHILD KILLED BY SIGNAL 11 (SIGSEGV)** |
| leased | **300 invocations, 0 mismatches, 0 declined** |

**Contract: Option A, single-flight.** `bitnet_xdna_invocation_begin/end` bracket
the whole `accumulate -> barrier -> epilogue -> barrier` lifetime, taken by the
thread driving the device and released only after every CPU reader has finished.
Option B (invocation-owned output handles) is architecturally cleaner but is not
needed for a resident-controller runtime. **CPU-only contexts never enter this
path and are unaffected.**

**A second bug, found only by running it.** The first version reused `g_mu` for
the lease. `bitnet_xdna_available()` also locks `g_mu` on its first call, and
every worker calls it on the hot path -- so a worker still inside that call
blocked on the lease while thread 0 waited at the barrier for exactly that
worker. Hard hang at pp2048, reproducible. Availability/config resolution now has
its own `g_init_mu`.

**Cost: none measurable.** One uncontended mutex acquisition per mul_mat node:
t=8 844.7/848.2/835.8 tok/s (pre-lease reference 836.6); t=15 1277.3/1287.7/1284.0
(reference 1274.1).

---

## 2. Long-lived stress [MEASURED]

`tools/npu_stress.cpp`, 360 invocations over three shapes x eight token counts,
deliberately unsorted (1024, 3968, 2048, 1024, 3072, 2048, 3968, 1024) so the
arena is asked for a large slot count, then a small, then large again. Every
iteration carries a correctness sentinel (FNV hash vs a single-threaded
reference).

| | |
|---|---|
| invocations / mismatches / declines | **360 / 0 / 0** |
| RSS initial / high-water / final | **512 / 512 / 512 MiB** |
| arena high-water | **12 slots, 120 MiB** = 4 token tiles x 3 N chunks |
| resident weight tensors | 3 (stable) |
| NPU dispatches | 2016 |
| package temperature | 41.2 -> 38.2 degC |

**The arena is bounded by construction** -- slot count is
`ceil(T/kMTile) x n_chunks x k_chunks`, maximised by the largest request -- and
it reached that bound during reference capture, never growing afterwards.

Confirmed through the real llama.cpp path: 75 prefills across pp1024/2048/3968 in
one process, RSS plateaus at **3944 MiB** and stays flat (that figure includes
the model, KV cache, ggml work buffers and the 1843 MiB of int8 NPU weights).

---

## 3. Deep-K direct reduction [MEASURED]

`ffn_down` (`k_chunks == 3`) now gives each K chunk its own persistent slot; the
CPU workers do the int32 reduction, the scale and the store in one parallel pass.
No `part`, no `g_acc`. **The reduction stays in int32 before conversion**, exactly
as the host path did -- summing in float would change results.

A first measurement showed `partacc` at 0 but `partcopy` still 12-33 ms: the
`part -> g_acc` copy, the `part` allocation and its per-tile zeroing all still ran
unconditionally on `k_chunks > 1`, copying a buffer nothing had written. Guarding
them improved the result:

| config | before guard | after guard |
|---|---:|---:|
| pp2048 t4 | 1.010x | **1.024x** |
| pp2048 t6 | 1.009x | **1.025x** |
| pp2048 t8 | 1.012x | **1.031x** |
| pp2048 t15 | 1.023x | 1.010x |
| pp3968 t8 | 1.003x | 1.008x |
| pp3968 t15 | 1.008x | 1.015x |

`partacc` and `partcopy` are both **0.0 ms** by counter. The pp2048 t8 gain (3.1%)
exceeds the 2.6% ceiling quoted for `partacc + partcopy` because that ceiling did
not count the `part` zeroing -- roughly 600 MB of memset per prefill for
`ffn_down`, which the guard also removes.

**Decision: PROMOTE.** Bit-exact at every T, under both flags and in combination;
concurrency contract and arena bound hold with it on; never negative.

---

## 4. Cost model at long context [MEASURED]

The holdout found **a defect, not a bad constant** -- and two errors in my own
method had to be corrected first. Both are recorded because each produced a
plausible wrong answer.

1. The first sweep tested tiles `{0, 1}` only, believing a pp3968/`-ub 2048`
   micro-batch is 1984 tokens. It is 2048, so the space is `{0, 1, 2}`. `auto`
   then appeared 18% *faster than every exhaustive option* -- impossible for a
   model that must pick one of them, and the tell that the option space was
   wrong.
2. Before spotting that, I assumed thermal bias, since the tool ran `auto` first
   in every cell and a pp3968/t=3 run takes ~1 minute. Rotating the option order
   did **not** close the gap, which is what ruled drift out. The rotation is kept.

**The defect.** `token_split_nt` computed `max_tiles = n_tokens / kMTile`,
flooring. The trailing micro-batch at pp3968 is 1920 tokens, so the partial tile
was unassignable and the NPU could take at most 1024 of those 1920 however
strongly the model favoured it -- the remainder falling to the CPU workers that
can least absorb it. The forced path already clamped to `n_tokens`; auto did not
agree with it.

| threads | best | auto (before) | regret | auto (after) | regret |
|---:|---:|---:|---:|---:|---:|
| 3 | 345.7 | 294.8 | **1.173x** | 345.7 | **1.000x** |
| 5 | 527.9 | 491.1 | 1.075x | 529.3 | 0.995x |
| 7 | 601.0 | 590.8 | 1.017x | 602.6 | 0.998x |
| 9 | 684.4 | 692.4 | 0.988x | 693.8 | 0.992x |
| 12 | 869.9 | 870.5 | 0.999x | 871.6 | 0.997x |
| 15 | 1036.6 | 1028.4 | 1.008x | 1028.0 | 1.003x |
| | | **mean 1.043 / worst 1.173** | | **mean 0.997 / worst 1.003** | |

Fix: `ceil` with `t` clamped to `n_tokens`. Whole-multiple micro-batches are
unaffected, so the 2K/3K calibration cannot move -- verified: pp2048 t8 861.6
(ref 865.1), t15 1318.1 (ref 1278.5); pp3072 t8 760.8, t15 1074.5.

**R = 25 is kept, and no scheduler complexity was added.**

---

## 5. Energy [MEASURED]

Package RAPL only (the `core` subdomain is unusable on this SoC), alternating
arms, 5 reps, pp2048 `-ub 2048`.

| threads | arm | tok/s | avg W | mJ/token | vs arena |
|---:|---|---:|---:|---:|---:|
| 6 | arena (no kreduce) | 731.5 | 70.9 | 96.59 | — |
| 6 | closeout | 751.6 | 71.1 | 94.65 | **0.980x** |
| 8 | arena (no kreduce) | 838.2 | 74.9 | 89.37 | — |
| 8 | closeout | 862.3 | 76.1 | 87.94 | **0.984x** |

---

## 6. Correctness [MEASURED]

Bit-exact across every flag combination, including the multi-token-tile lifetime
paths:

```
                      direct  kreduce      result
default                  1       1      all shapes bit-exact
                         1       0      all shapes bit-exact
                         0       0      all shapes bit-exact
concurrent, leased       1       1      300 invocations, 0 mismatches
```

Perplexity identical in every mode: **307.5806 +/- 27.85495** for CPU-only,
direct+kreduce, direct-only and the `g_acc` reference.

---

## 7. Reproduction from a clean worktree [MEASURED]

A fresh `git worktree` plus **pristine pinned trees**, not the working build:

```
BitNet @ 0b341e5, llama.cpp @ 390c30775   (git archive HEAD -> pristine)
patch applied to pristine pinned tree
configured OK ("XDNA2 offload enabled")
BUILD OK from pristine pinned trees + checked-in patch
```

| config | tok/s | dispatches | stage_out |
|---|---:|---:|---:|
| CPU-only | 629.6 | **0** | n/a |
| default (direct + kreduce) | **866.0** | 2568 | **0.0 ms** |
| `BITNET_XDNA_DIRECT_OUT=0` | 838.7 | 1284 | 598.9 ms |

matching the working build (CPU-only ~640, default ~862). Full suite green in the
worktree: patch reproduction, CPU tests, all twelve shape cases, concurrency
contract.

**External inputs the worktree exposed** (all deliberately untracked, pinned in
`artifacts/coordinates.edn`): `refs/` (pinned upstream checkouts), `models/`,
`.venv/` (used to regenerate the `.packed` test fixtures from the GGUF), and
`.localdeps/` (a local `uuid/uuid.h` for the XRT headers). A reproduction needs
those four present.

---

## 8. Frozen reference

### Final defaults

| setting | value | override |
|---|---|---|
| `BITNET_XDNA_DIRECT_OUT` | **on** | `=0` restores the `g_acc` path |
| `BITNET_XDNA_DIRECT_KREDUCE` | **on** | `=0` restores host `part` accumulation |
| `BITNET_XDNA_ASYNC` | **off** | superseded; see below |
| cost-model `R` | **25** with direct output (10 with `g_acc`) | `BITNET_XDNA_NPU_THREADS` |
| concurrency | single-flight invocation lease, always on | — |

### Superseded

**`BITNET_XDNA_ASYNC`** -- its entire purpose was overlapping evacuation with the
next dispatch, and direct output removes the evacuation. Never better than direct
output alone beyond noise. Retained behind its flag, default off, **not
recommended**; delete when the `g_acc` path is retired.

**The `g_acc` path is deliberately kept** as the reference implementation and as
the fallback, per the brief: deleting it now would remove the oracle that makes
the direct paths verifiable.

### Known limitations

- Concurrency is **serialized, not parallel**. Two contexts using the NPU take
  turns for the whole invocation lifetime. Correct and predictable, but it caps
  aggregate NPU throughput across contexts at one invocation at a time.
- The arena costs **up to 120 MiB** (12 slots) at `-ub 3968`; larger micro-batches
  would grow it proportionally (`ceil(T/1024) x n_chunks x k_chunks` slots).
- The epilogue is measurably dearer reading a freshly-synced device buffer than a
  host copy (623 -> 714 thread-ms at t15) -- far smaller than the staging removed,
  but it is a real cost and it grows with output size.
- `stage_in` (~32 ms/prefill) is untouched; activations must still reach the
  device buffer.
- GPU co-tenancy remains **[DEFERRED]**: no ROCm, Vulkan, torch or GPU-runnable
  model on this machine, and installing one is out of scope.

### Reproduction

```bash
bash tools/scan_artifacts.sh
make check                      # patch reproduction + CPU + NPU + concurrency
export BITNET_XDNA_ARTIFACTS=$PWD/artifacts/xclbin-tuned
BITNET_XDNA=1 build/npu_stress 360        # arena bound + correctness sentinel
python3 tools/kreduce_ab.py                # deep-K A/B
python3 tools/cost_model_4k.py --reps 3    # long-context holdout
bash    tools/energy_closeout.sh           # package energy/token
```

---

## 9. Next

Attention is the largest remaining opportunity and the dependency-constrained
critical path: **35.7% of hybrid prefill at 2K, 49.0% at ~4K**, growing O(T^2).
The next pass is a **bounded standalone feasibility experiment** -- a
flash-attention-shaped aie2p kernel measured against the CPU implementation for
BitNet's real head and KV geometry, fully burdened with data movement -- on a
separate branch, with no llama.cpp integration unless it clearly passes the gate.
