# Hybrid state restore: what is actually wrong [MEASURED]

Correction pass over `model-candidate-halo`
(`c9915023d5f0dd024ba4177798ea9201b1cf99a8`). That branch is frozen and not
rewritten; this document supersedes two of its claims, in **opposite
directions**, and both corrections come from the same root cause.

## Verdict

> **HYBRID RESTORE IS NUMERICALLY FAITHFUL — THE PRIOR "CORRUPTION" WAS INDUCED
> BY SERVER CONTEXT CHECKPOINTS. BUT RESTORE YIELDS NO PREFIX REUSE ON HYBRID
> MODELS, SO THE PRIOR LATENCY WIN WAS ALSO AN ARTIFACT.**

Two independent defects were conflated:

1. **Context-checkpoint creation perturbs hybrid state** — even on a clean,
   single, first request with no restore involved at all.
2. **Restore does not carry the server bookkeeping hybrid reuse needs**, so a
   restored hybrid sequence is re-prefilled from scratch (upstream #28194).

The pure-attention incumbent is affected by neither.

---

## 0. The premise that motivated this pass was false

The brief asked to rebuild the checkpoint at an exact prefix boundary, on the
reading that `completion(prefix, n_predict=1)` had stored `prefix + 1 generated
token` — evidenced by "spine ~1575, checkpoint ~1576".

That +1 is **BOS**, not a generated token:

```
/tokenize add_special=false -> 1575 tokens
/tokenize add_special=true  -> 1576 tokens, first id = 1   (BOS)
saved state, n_predict=0    -> 1576 tokens
saved state, n_predict=1    -> 1576 tokens, byte-identical file
```

`n_predict=0` and `n_predict=1` produce **byte-identical** saved state on every
model tested. The earlier note compared a bare `/tokenize` count (no BOS)
against the server's count (with BOS). **The checkpoint boundary was already
exact**, so no rollback was ever induced by an overlong checkpoint, and CASE A
of the interpretation gate does not apply.

`tools/restore_matrix.py` now asserts this rather than assuming it: it reports
both token counts and fails the boundary check if the saved count does not equal
the with-BOS count. Qwen3.5 adds no BOS (1622 == 1622) and also passes.

## 1. Enabling context checkpoints changes hybrid output on a clean request

The decisive control. Fresh server process, **one** request, greedy with fixed
seed, no restore, no slot reuse, no prior traffic — run twice, differing only in
`-ctxcp`:

| model | architecture | `-ctxcp 32` (default) | `-ctxcp 0` | max \|Δlogprob\| | verdict |
|---|---|---|---|---:|---|
| **BitNet-b1.58-2B** | 30 attn, pure | `' 1\ns002 81 '` | `' 1\ns002 81 '` | **0.0** | **INSENSITIVE** |
| LFM2.5-1.2B | 6 attn + 10 conv | `' 1\n```\n\nOUTPUT\n '` | `' 1\nRECOMMENDATION'` | **0.371** | SENSITIVE |
| Qwen3.5-2B | 6 attn + 18 DeltaNet | same text | same text | **0.271** | SENSITIVE |
| Qwen3.5-0.8B | 6 attn + 18 DeltaNet | same text | same text | **0.181** | SENSITIVE |

`cache_n = 0` and `prompt_n` identical in every cell — these are all full
recomputes of the same prompt. The only difference is whether the server is
allowed to create context checkpoints.

The pure-attention control being **bit-identical across separate processes**
rules out process-to-process nondeterminism as the explanation, and establishes
that "no difference" is measurable as exactly 0.0 on this machine.

**Consequence for the previous branch:** its fidelity probes measured every
candidate against a full-recompute reference taken from a `-ctxcp 32` server.
That reference was itself perturbed. The divergences reported there
(0.109–0.460) were **differences between two differently-perturbed states**, not
between a correct state and a corrupted one. Those numbers should not be read as
restore corruption.

Mechanism, from the pinned build's source: `create_checkpoint()`
(`server-context.cpp:2347`) calls `update_tgt(..., LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)`
against a hybrid memory. A partial state capture is well-defined for an
attention KV cache and evidently is not for a recurrent cell. Confirming the
precise fault needs instrumentation inside the recurrent memory that this pass
did not add, so the mechanism is **consistent with** the data rather than
demonstrated by it. The *effect* is measured.

## 2. With checkpoints disabled, restore is bit-exact — and useless for hybrids

Full Task 2 matrix at `-ctxcp 0`, against a full-recompute reference, top-100
logprobs, `A` = own domain and `B` = a foreign domain:

| model | arm | restore | cache_n | TTFT | max \|Δ\| | top-1 | decision |
|---|---|---:|---:|---:|---:|---|---|
| **BitNet-b1.58-2B** | B clean | 13.3 ms | **1600** | **214.2 ms** | 0.00000 | same | same |
| | C foreign | 16.1 ms | **1600** | 214.7 ms | 0.00000 | same | same |
| | D clean-after-foreign | 13.4 ms | **1600** | 213.9 ms | 0.00000 | same | same |
| LFM2.5-1.2B | B clean | 3.9 ms | **0** | 1176.0 ms | 0.00000 | same | same |
| | C foreign | 4.3 ms | **0** | 1170.0 ms | 0.00000 | same | same |
| | D clean-after-foreign | 3.9 ms | **0** | 1171.9 ms | 0.00000 | same | same |
| Qwen3.5-0.8B | B clean | 5.7 ms | **0** | 1695.5 ms | 0.00000 | same | same |
| | C foreign | 7.2 ms | **0** | 1711.0 ms | 0.00000 | same | same |
| | D clean-after-foreign | 6.1 ms | **0** | 1702.4 ms | 0.00000 | same | same |
| Qwen3.5-2B | B clean | 5.5 ms | **0** | 2970.2 ms | 0.00000 | same | same |
| | C foreign | 6.0 ms | **0** | 2987.5 ms | 0.00000 | same | same |
| | D clean-after-foreign | 5.5 ms | **0** | 2987.6 ms | 0.00000 | same | same |

Two things, and they must not be conflated:

- **Model sequence state: restored correctly.** `max |Δlogprob| = 0.00000` in
  all twelve arms, including after a foreign domain occupied the slot. There is
  no state corruption attributable to restore, on any architecture tested.
- **Server-side reuse: absent for every hybrid.** `cache_n = 0` means the
  restored state buys nothing — each turn re-prefills the whole 1710–1758 token
  prompt. The incumbent, by contrast, reuses all 1600 tokens and answers in
  214 ms.

This is **CASE B** of the interpretation gate, and it matches upstream
`ggml-org/llama.cpp#28194`: the sequence state and token list are serialized,
`server_slot::prompt.checkpoints` are not, and hybrid reuse requires them.
Confirmed in the pinned build's source:

- `SLOT_RESTORE` (`server-context.cpp:2585`) calls `llama_state_seq_load_file`
  and replaces `slot->prompt.tokens`. It never touches `slot->prompt.checkpoints`.
- `prompt_clear()` (`:255`), used by `SLOT_ERASE`, clears `prompt.tokens` and
  also leaves `checkpoints` untouched.
- The reuse path (`:3302`) searches `slot.prompt.checkpoints` and, finding
  nothing, takes the `do_reset` branch whose own log line reads *"forcing full
  prompt re-processing due to lack of cache data (likely due to SWA or
  hybrid/recurrent memory)"*.

Because checkpoints are never cleared **or** restored, a slot also carries the
previous occupant's checkpoints. That is what produced the earlier
history-dependent behaviour: with `-ctxcp 32` the polluted arms found a stale
checkpoint, reused 1568 tokens, and answered in ~116 ms with a **different
decision**; the clean arm found none and re-prefilled. Both the speed and the
divergence came from the same stale object.

## 3. The two corrections to `model-candidate-halo`

| claim on that branch | status | corrected statement |
|---|---|---|
| "Explicit spine checkpoint/restore recovers hybrid latency: LFM2.5-1.2B **164.4 ms**, Qwen3.5-0.8B **188.2 ms** per decision, beating the incumbent's 263.2" | **OVERTURNED** | Those timings came from stale in-slot context checkpoints, not from the restore. With a clean baseline, a restored hybrid re-prefills: **LFM2.5-1.2B 1176 ms**, **Qwen3.5-0.8B 1695 ms**, **Qwen3.5-2B 2970 ms** TTFT. The incumbent restores usefully at **214 ms**. |
| "Hybrid state restore is corrupted; on LFM2.5-1.2B it changes the emitted decision, and does not recover" | **NARROWED** | Restore itself is bit-exact (0.00000, all arms). The corruption was induced by server context checkpoints, which both perturb hybrid state on a clean request and leak across slot reuse. With `-ctxcp 0` the divergence disappears entirely. |
| "3.2–6.3x more warm domains per GiB" | **STANDS, relabelled** | The serialized state sizes are unchanged and independent of this. But see `STORAGE_DENSITY` below — they are checkpoint-file sizes, not measured server residency. |
| BitNet restore is faithful | **STANDS, strengthened** | Now also shown to be insensitive to `-ctxcp`, and to give real reuse (1600 tokens, 214 ms) after restore. |

## 4. Storage density is not prompt-cache residency

Per Task 14, the per-domain figures derived from checkpoint-file size are
renamed. They are **equivalent checkpoint-storage capacity**, i.e. how many
serialized domains fit in a byte budget:

| model | MiB/domain | domains per 8 GiB | per 32 GiB |
|---|---:|---:|---:|
| LFM2.5-1.2B | 20.23 | 404 | 1619 |
| LFM2.5-2.6B | 27.71 | 295 | 1182 |
| Qwen3.5-0.8B / 2B | 39.94 | 205 | 820 |
| BitNet-b1.58-2B | 127.18 | 64 | 257 |

These are **not** measured `--cache-ram` resident-domain counts, and for the
hybrids they currently describe storage for checkpoints that the server cannot
reuse. Only the incumbent's figure has been demonstrated end-to-end through a
working residency path (`controller-state-envelope` §6).

## 5. What this does and does not settle

- Hybrid **multi-domain state-spine deployment remains blocked in this runtime**,
  but on **reuse**, not on correctness. The state is right; the server will not
  use it.
- The blocker is upstream and identified, with an open issue and a reporter's
  patch. Task 4 addresses whether a current/patched build fixes it.
- `-ctxcp 0` is the correct baseline for any future hybrid measurement on this
  build, and any earlier hybrid number taken at the default is suspect.

## Reproduce

```bash
tools/ctxcp_sensitivity.py --model <gguf> --label <name> --out <json>
# fresh server, one request, -ctxcp 32 vs 0

tools/restore_matrix.py --port <p> --label <name> \
    --save-dir <dir> --out <json>
# boundary assertion + clean / foreign / clean-after-foreign restore arms
```

---

# 6. Reuse and correctness are mutually exclusive for hybrids here [MEASURED]

Task 4 asked whether persisting `slot.prompt.checkpoints` — the upstream fix for
#28194 — would make hybrid restore both correct and fast. The author's patch is
**not publicly available** (no llama.cpp fork on their account, no PR
referencing the issue, and they state they are not opening one), so rather than
reimplement ~750 lines this pass asks the question the patch would answer using
only stock server behaviour.

## The minimal discriminator

A checkpoint only needs persisting if it works in the first place. So drop
save/restore and exercise the same code path in-process:

    turn 1:  A_prefix + delta_1      (populates the slot)
    turn 2:  A_prefix + delta_2      (shares the 1575-token prefix)

Turn 2 is the situation a restored slot would be in if checkpoints had survived.

| model | turn-2 cache_n @ `-ctxcp 0` | @ `-ctxcp 32` |
|---|---:|---:|
| BitNet-b1.58-2B | 1614 | 1614 |
| LFM2.5-1.2B | **0** | **0** at `-ub 4096` |
| Qwen3.5-0.8B | **0** | **0** at `-ub 4096` |

The incumbent reuses identically with checkpoints on or off — **checkpoints are
not what gates its reuse.** The hybrids reuse nothing at the project's standard
`-ub 4096`, with or without checkpoints.

## `-ub` is the lever, and it is not free

Hybrid reuse is gated by `pos_min >= pos_min_thold` (`server-context.cpp:3252`),
where `pos_min` is the earliest position the memory can still represent. For
recurrent memory that is bounded by the last micro-batch boundary, so a smaller
`-ub` leaves a nearer roll-back point. It works — and it is wrong.

**LFM2.5-1.2B**, two turns, versus a `-ctxcp 0` full-recompute reference:

| `-ub` | `-ctxcp 0` cache_n / max\|Δ\| | `-ctxcp 32` cache_n / TTFT / max\|Δ\| |
|---:|---|---|
| 4096 | 0 / **0.00000** | 0 / 1186 ms / 0.30757 |
| 1024 | 0 / **0.00000** | 683 / 738 ms / 0.29952 |
| 512 | 0 / **0.00000** | 1195 / 385 ms / 0.29491 |
| 256 | 0 / **0.00000** | 1451 / **207 ms** / 0.33977 |
| 128 | 0 / 0.24348 | 1579 / **118 ms** / **0.48868** |

**Qwen3.5-0.8B**, same protocol:

| `-ub` | `-ctxcp 0` cache_n / max\|Δ\| | `-ctxcp 32` cache_n / TTFT / max\|Δ\| |
|---:|---|---|
| 4096 | 0 / **0.00000** | 0 / 1711 ms / 0.11125 |
| 1024 | 0 / **0.00000** | 730 / 1008 ms / 0.16998 |
| 512 | 0 / **0.00000** | 1242 / 516 ms / 0.12822 |
| 256 | 0 / **0.00000** | 1498 / 259 ms / 0.15820 |
| 128 | 0 / **0.00000** | 1626 / **143 ms** / 0.15424 |

Read the two columns together:

- **`-ctxcp 0`: correct and useless.** Exactly 0.00000 at every micro-batch from
  4096 down to 256, and zero reuse in all of them. The self-consistency across
  four micro-batch sizes is what establishes this column as the trustworthy
  reference. (`-ub 128` on LFM2.5 diverges at 0.243 even here, so that setting is
  independently excluded on numerical grounds.)
- **`-ctxcp 32`: fast and wrong.** Reuse climbs to 97-100% of the spine and TTFT
  falls to 118-143 ms — genuinely competitive with the incumbent's ~207 ms — but
  **every single arm is numerically wrong, including the `-ub 4096` arm that
  reuses nothing.** That last cell is the tell: the divergence is not caused by
  reuse, it is caused by checkpointing being enabled at all, and reuse then adds
  to it (LFM2.5: 0.308 with no reuse, 0.489 at 97% reuse).

## Verdict for Task 4

> **HYBRID PREFIX REUSE AND NUMERICAL CORRECTNESS ARE MUTUALLY EXCLUSIVE IN THIS
> RUNTIME.** Reuse requires context checkpoints; context checkpoints perturb
> hybrid state. There is no `-ub` / `-ctxcp` combination that delivers both.

This **narrows the upstream issue's implied remedy**, and the distinction
matters to anyone picking up that patch:

- #28194 is correct that restore does not persist `prompt.checkpoints`, and that
  hybrid reuse needs them. Reproduced here exactly (`cache_n = 0` after restore,
  full reuse on a pure-attention model).
- But persisting them would make hybrid restore **fast and still wrong on this
  build**, because the checkpoint mechanism itself does not preserve hybrid state
  faithfully. A correct fix has to address `create_checkpoint`'s partial-state
  capture for recurrent memory, not only its serialisation.

The pure-attention incumbent is unaffected throughout: it reuses without
checkpoints, and its restore is bit-exact.

## What this means for the model bakeoff

For any hybrid candidate on the pinned build, the honest warm-decision numbers
are the **correct** ones — `-ctxcp 0`, no reuse, a full re-prefill every turn:

| model | correct warm TTFT | incumbent |
|---|---:|---:|
| LFM2.5-1.2B | 1170-1186 ms | **206-214 ms** |
| Qwen3.5-0.8B | 1684-1711 ms | |
| Qwen3.5-2B | 2970-2988 ms | |

The 118-143 ms figures are reachable only by accepting a state the model did not
compute. They are recorded here as the size of the prize if the runtime is
fixed, **not** as a current capability.

---

# 7. Equal-memory long spine (Task 13) [MEASURED]

If a hybrid's per-token state is 3-6x smaller, that gain can be spent on more
domains **or** on a longer spine per domain. At the incumbent's ~127 MiB/domain
budget:

| model | longest spine within 127.18 MiB | state | vs incumbent | cold decision at that length |
|---|---:|---:|---:|---:|
| BitNet-b1.58-2B | **1,600** tok | 117.29 MiB | 1.0x | 1,680.6 ms |
| Qwen3.5-0.8B | **9,014** tok | 125.11 MiB | **5.6x** | 10,239.4 ms |
| **LFM2.5-1.2B** | **9,587** tok | 112.66 MiB | **6.0x** | 8,938.0 ms |

Sweep detail (LFM2.5-1.2B): 1,575 tok / 18.65 MiB, 3,188 / 37.58, 6,414 / 75.43,
9,587 / 112.66, 11,009 / 129.35 (first over budget). The incumbent crosses the
budget between 1,600 tok (117.29 MiB) and 3,224 tok (236.26 MiB).

**The capacity answer is real and large: at the same per-domain RAM, a hybrid
controller can hold roughly 6x more resident context.** That is the strongest
architectural argument for the hybrid family in this whole bakeoff.

**The latency answer is currently prohibitive, and that is the reuse blocker
again, not the architecture.** Because hybrid prefix reuse is either absent or
wrong (§2, §6), every decision at a 9.6K-token spine is a full prefill costing
~8.9 s. The capacity is usable only once warm reuse works; until then a longer
spine makes each decision proportionally more expensive rather than free.

No semantic claim is made — this is capacity and latency only.
