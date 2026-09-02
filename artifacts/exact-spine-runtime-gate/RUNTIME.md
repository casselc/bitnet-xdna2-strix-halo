# Exact-spine reuse: no runtime change was needed [MEASURED]

Branch cut from `origin/hybrid-state-training-gate`
(`efab39fc5784e85923700bff5f71a224385e9df4`). All prior evidence branches frozen.

## Headline

> **The hybrid reuse blocker was never in llama.cpp. It was a one-token BPE seam
> in our own protocol.** Saving the spine state at the token-exact common prefix
> makes the **stock, unpatched, pinned** server reuse the whole spine and produce
> results **bit-identical** to a full recompute.

Qwen3.5-0.8B, original pinned binary, no patch, `-ctxcp 0`:

| arm | cache_n | prompt_n | TTFT | max \|Δlogprob\| |
|---|---:|---:|---:|---:|
| A full recompute | 0 | 1758 | 1771.1 ms | — |
| **R restore + delta** | **1621** | **137** | **129.2 ms** | **0.000000** |

**13.7x faster, bit-exact.** The previous branch's "hybrid restore yields no
reuse (#28194)" was a true observation of a *symptom* whose cause was ours.

## 1. What was actually wrong

The previous pass saved the spine by sending its **text** and compared token
counts. Text prefixing is not token prefixing — BPE merges across the seam:

```
LFM2.5-1.2B:   spine alone      = 1576 tokens
               spine + delta    = 1710 tokens
               common prefix    = 1575        <-- one token short
               at index 1575: spine has id 708, spine+delta has id 509
```

So the saved state covered 1576 tokens while only 1575 were a genuine prefix of
the query. That asks the server to **roll back one token**, which is exactly the
operation recurrent memory cannot do. It searches for a context checkpoint,
finds none, and discards the entire prefix — `cache_n = 0`, full re-prefill.

One token of tokeniser behaviour produced every downstream conclusion about
hybrid reuse being blocked.

**The fix is in the protocol, not the runtime:** compute the longest token
prefix common to the spine *and* the deltas that will follow, and save the state
at exactly that many **tokens**, feeding token ids rather than text.
Qwen3.5-0.8B loses 4 tokens to the seam (1625 → 1621), LFM2.5-1.2B loses 1.

## 2. Full matrix at the token-exact boundary

Stock behaviour, `-ctxcp 0`, token-ids end to end. Arm **S** is a control that
splits evaluation *without* save/restore, separating a chunked-evaluation
difference from a persistence one:

| model | R cache_n | R TTFT | S vs A | **R vs S** | R vs A | verdict |
|---|---:|---:|---:|---:|---:|---|
| BitNet-b1.58-2B | 1600 | 282.8 ms | 0.000000 | **0.000000** | **0.000000** | **REUSE + EXACT** |
| **Qwen3.5-0.8B** | 1621 | **127.8 ms** | 0.000000 | **0.000000** | **0.000000** | **REUSE + EXACT** |
| LFM2.5-1.2B | 1575 | 126.7 ms | 0.372713 | **0.000000** | 0.372713 | REUSE, NUMERICALLY WRONG |
| Qwen3.5-2B | — | — | 0.271305 | — | 0.271305 (top-1 **differs**) | REUSE, NUMERICALLY WRONG |

**`R vs S = 0.000000` on every model tested.** Save/restore is exact — it
reproduces the live split-evaluation state bit for bit, on hybrids included.
This retires any remaining suspicion of the persistence path.

Where a model fails, it fails in **split evaluation itself**, with no save or
restore involved.

### The previous branch's "restore is bit-exact" was true but vacuous

That result was measured when `cache_n = 0` — the restored state was never used,
so it only confirmed that a full recompute is a full recompute. The present
`R vs S` comparison is the first test that actually exercises restored state.
The conclusion happens to survive, but the earlier evidence did not support it.

## 3. LFM2.5-1.2B: chunked evaluation is not reproducible

Fresh server per arm, so no slot history is involved. Split the identical
1710-token prompt at different points and compare to a single-batch evaluation:

| split at | cache_n | max \|Δ\| | |
|---:|---:|---:|---|
| 400 | 400 | 0.000000 | exact |
| 800 | 800 | 0.000000 | exact |
| 1010 | 1010 | 0.324110 | **diverges** |
| 1200 | 1200 | 0.000000 | exact |
| 1575 | 1575 | 0.372713 | **diverges** |
| 1700 | 1700 | 0.273756 | **diverges** |

Deterministic and reproducible across separate server processes: some split
positions are exact and others are not. This is an **LFM2 chunked-evaluation
defect**, not a state-persistence one, and it lands on our production spine
boundary. BitNet and Qwen3.5-0.8B are exact at every split tested.

No mechanism is claimed. Identifying which kernel loses the short-conv window
across a batch boundary needs instrumentation this pass did not add.

## 4. 100 alternating domains through one slot

Task 5, Qwen3.5-0.8B, 4 domains with unguessable 64-bit tags, one physical slot,
restore before every turn, full-recompute verification every 10th turn on a
second slot:

| | |
|---|---|
| turns | 100 |
| reuse | 1621-1623 of ~1625 every turn |
| TTFT | **p50 131.6 ms**, p95 137.4 ms |
| restore | ~6 ms |
| verified turns | 10, **all 0.000000** |
| numerical mismatches | **0** |
| foreign-tag contamination | **0** |

> **STABLE ACROSS ALTERNATING DOMAINS.** No drift, no contamination, no
> degradation from turn 0 to turn 99.

## 5. The experimental patch was built, and is not needed

An experimental server-only fast path was implemented in an isolated worktree
(`--exact-spine-fast-path`, default off) to skip the rollback search when the
cached sequence is entirely a prefix of the incoming prompt. It builds and the
flag is exposed.

**It changes nothing measurable**, because once the boundary is token-exact the
stock path already takes the same branch: with `n_past == prompt.n_tokens()`
there is no rollback to perform, `pos_min < pos_min_thold`, and the checkpoint
search is skipped anyway. Measured identical with the flag on, with it off, and
on the original pinned binary.

It is kept in the branch as documentation of what was tried and as a guard that
makes the invariant explicit, but **no runtime change is required and none
should be promoted**. The frozen BitNet runtime is untouched.

## 6. Where this leaves each candidate

| model | exact-spine fast path |
|---|---|
| **Qwen3.5-0.8B** | **VALIDATED** — 13.7x, bit-exact, stable over 100 alternating turns |
| BitNet-b1.58-2B | works (control), 282.8 ms at a 1600-token spine |
| LFM2.5-1.2B | **unsafe on this build** — split evaluation diverges at our boundary |
| Qwen3.5-2B | **unsafe on this build** — diverges and flips top-1 |

The size pattern within each family (smaller exact, larger divergent) mirrors
the `-ctxcp` sensitivity measured on the previous branch. Whether that is one
defect or two is not established here.

## Reproduce

```bash
tools/exact_spine_probe.py --bin <llama-server> --model <gguf> \
    --label X --out X.json          # arms A / S / R at a token-exact boundary
tools/exact_spine_alternating.py --bin <llama-server> --model <gguf> \
    --domains 4 --turns 100 --verify-every 10 --label X --out X.json
```


---

# 7. Does `create_checkpoint` mutate live state? Measured: no [MEASURED]

Task 8. The previous branch inferred that checkpoint *capture* perturbs hybrid
state, from downstream logit differences, and labelled the mechanism "consistent
with" rather than demonstrated. It is now measured directly and **the inference
was wrong**.

`create_checkpoint` was instrumented in the isolated worktree to fingerprint the
LIVE sequence state (FNV-1a over `llama_state_seq_get_data`) immediately before
and immediately after the capture, with no token decoded in between. It hashes
**twice** before the capture as a control: if those two disagree, the
observation itself is unstable and nothing after it could be trusted.

Run with `-ctxcp 32 -cms 0 -ub 512` so mid-prompt checkpoints are genuinely
created:

| model | before | before again | after | observation stable | capture mutated |
|---|---|---|---|---|---|
| LFM2.5-1.2B | `f03df646…` / 13,522,072 B | `f03df646…` | `f03df646…` | yes | **no** |
| LFM2.5-1.2B | `fd32b4df…` / 19,819,672 B | `fd32b4df…` | `fd32b4df…` | yes | **no** |
| Qwen3.5-0.8B | `19a8392a…` / 33,556,276 B | `19a8392a…` | `19a8392a…` | yes | **no** |
| Qwen3.5-0.8B | `7a82b39c…` / 39,857,972 B | `7a82b39c…` | `7a82b39c…` | yes | **no** |

> **CHECKPOINT CAPTURE DOES NOT MUTATE LIVE STATE.** Byte-identical before and
> after, on both hybrid families, with the observation verified stable.

So the `-ctxcp` sensitivity recorded on the previous branch has a different
cause. Taken together with §3 — where LFM2.5 diverges under **split evaluation
alone, with no checkpoints and no save/restore** — the coherent reading is that
the divergence belongs to **chunked evaluation**, and enabling checkpoints
changes how the prefill is chunked rather than corrupting anything at capture
time. That is stated as the reading the two measurements jointly support, not as
a demonstrated mechanism; isolating which kernel loses reproducibility across a
batch boundary was out of scope.

(BitNet created no checkpoints in this run and therefore contributes no rows.)

# 8. Memory cost of the working reuse path (Task 9)

For the **exact-spine** path, which is the one that works:

| | |
|---|---|
| per domain | **one state file, 38.31 MiB** (Qwen3.5-0.8B at a ~1625-token spine) |
| sidecar | none |
| checkpoint blobs | none |
| metadata | none beyond the file |
| 32 domains, measured | **1.2 GB** on disk |

Nothing is added beyond the sequence state itself, so the sequence-state density
measured on the previous branch carries over to this path unchanged. A generic
context-checkpoint approach would add one or more additional state blobs per
domain on top of this; that arm was not built (see below), so its cost is not
quantified here.
