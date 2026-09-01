# The real multi-domain warm-state controller envelope

| | |
|---|---|
| branch base | `controller-state-scheduler` @ `6d225c9734a1771ccd573ecb8ee5b27031862f9e` |
| branch | `controller-state-envelope` |
| controller | BitNet-b1.58-2B-4T I2_S, promoted XDNA runtime, `llama-server` |

Fetched coordinates at the start of this pass:

```
main                        885df0ca793c23aed8e090a90fb19d8f591e75f4
controller-state-scheduler  6d225c9734a1771ccd573ecb8ee5b27031862f9e
service-batching-gate       856357c68b14486448426b08af91ca87fb1ff084
controller-cache-batching   2ca2e51916f1aa3fddfae94ef4db4ceb02d02296
runtime-v1-promotion        712b7c6d7f1fb018fdc727b8b3e254d33d8865dd
gpu-cotenancy               fbd8bf00108480c68919752fbb521fafd786d47d
service-cotenancy           9295df0a6167eaa43c983c13f702fce1033e4b1f
```

All evidence branches are frozen; nothing here rewrites them. `service-batching-gate`
was **not** merged — the streaming-TTFT client, the warmup step and the launcher
parameters were ported as isolated edits, because merging a parallel evidence history
to obtain scripts would confuse provenance.

| tag | meaning |
|---|---|
| **[MEASURED]** | observed on this machine |
| **[DERIVED]** | arithmetic over measured quantities |
| **[CORRECTION]** | a prior published claim found wrong here |
| **[DEFERRED]** | not done, with the reason |

### Standing environment confound [MEASURED]

A third-party `lemonade` `llama-server` (uid `lemonade`, not owned by this work) is
resident throughout at **8.7 GiB RSS**, idle (CPU ticks static across sampling). It was
deliberately **not** terminated — the brief restricts termination to owned processes.
It reduces available RAM and is therefore relevant to the cache-RAM ceiling in §6.
Swap was already fully consumed (7/7 GiB) at start but **static** (`si`/`so` = 0), i.e.
stale anonymous pages, not thrash. ZFS ARC was 3.9 GiB against a 121 GiB `c_max`.

---

## 0. The impossible TTFT/total relationship, resolved [CORRECTION]

`controller-state-scheduler` §6 published:

| regime | TTFT p50 | total p50 |
|---|---:|---:|
| C rebase/10 | 146.73 ms | **121.86 ms** |
| C rebase/25 | 145.21 ms | **130.49 ms** |

For a single request `total < TTFT` is impossible. It is not a clock problem, a
mislabelled field, or a streaming bug.

### What actually happened [MEASURED]

Auditing all 200 raw records in `state_spine.csv`:

```
rows where ttft_ms > total_ms:  0 / 200
per regime: n=50, ttft_ms present on 26, total_ms present on 50
```

**Every individual record was self-consistent. The two statistics were computed over
different populations.** `summarize()` takes `[r[key] for r in rows if r.get(key) is
not None]` per key, so a row missing `ttft_ms` still contributed to `total_ms`.

Why 24 rows lacked TTFT:

| regime | rows WITH ttft | their median total | rows WITHOUT ttft | their median total |
|---|---:|---:|---:|---:|
| A rebuild | 26 | 3076.80 ms | 24 | **2.26 ms** |
| B spine reuse | 26 | 191.18 ms | 24 | **1.88 ms** |
| C rebase/10 | 26 | 192.72 ms | 24 | **1.87 ms** |
| C rebase/25 | 26 | 188.99 ms | 24 | **1.96 ms** |

A ~2 ms "request" is an HTTP error return. The missing turns are **27–50 in every
regime**, and the mechanism is exact:

```
turn 26: supplied_n = 2544        slot context = min(n_ctx/n_parallel, 4096) = 2560
turn 27: 2544 + 18 = 2562  >  2560   ->  HTTP 400, ~2 ms
```

The spine grew 18 tokens per turn. **At turn 27 it outgrew the per-slot context, and
every remaining request in every regime returned HTTP 400.** Those failures carried no
server timings (so no TTFT) but their client wall was still recorded as `total_ms`,
dragging the total percentiles below the TTFT percentiles.

**So the published 50-turn state-spine experiment was actually a 26-turn experiment
with 24 silent failures.** The harness never recorded `err`, so nothing surfaced it.

### What this invalidates, and what survives

- **INVALID:** every `total_ms` statistic in `controller-state-scheduler` §6, and the
  derived claim that rebase cadence 10 vs 25 is "indistinguishable" (that comparison
  rested on the contaminated totals), and "~122–130 ms total".
- **Survives, correctly scoped:** the TTFT figures were computed over the 26 *valid*
  turns only, so they were honest numbers about turns 1–26 — but they were reported as
  50-turn results.

### Fixes applied [MEASURED]

1. `Req.row()` now **returns no timing at all for a failed request**. A failure cannot
   be counted as a fast completion.
2. `turn_row()` carries `err`.
3. `assert_timing_sane()` enforces `request_start <= first_token <= request_end` per
   record, returns only usable rows, and reports exclusions. Summaries take the **same**
   population for every statistic.
4. `preflight_length()` tokenizes the worst case (the final turn) against the server's
   real slot context **before** the run and raises immediately, instead of collecting
   400s for an hour.
5. TTFT is now the **measured** first-token arrival from a streaming client
   (`ttft_source = client_stream`), not `prompt_ms + predicted_ms/predicted_n`.
6. The real slot ceiling is recorded: `/props` reports **4096**, i.e.
   `min(n_ctx/n_parallel, max_position_embeddings)` — a larger `-c` does *not* buy a
   longer single sequence, because BitNet-2B caps at 4096.

### Corrected re-run [MEASURED]

50 turns, `t4/tb16 b4096 ub4096 np8 c40960`, output 4 tokens, streaming TTFT:

```
preflight: final-turn prompt is 2973 tokens, slot context 4096 -- fits
```

| regime | turns usable | excluded | eval p50 | reused p50 | TTFT p50 | TTFT p95 | total p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A rebuild (no reuse) | **50** | 0 | 2544 | 0 | 2276.76 | 2676.45 | 2348.51 |
| B spine reuse | **50** | 0 | 28 | 2516 | 97.05 | 110.66 | 167.75 |
| C rebase/10 | **50** | 0 | 28 | 2516 | 93.13 | 109.66 | 168.50 |
| C rebase/25 | **50** | 0 | 28 | 2516 | 95.41 | 108.75 | 167.22 |

**`total > TTFT` in every row, 200/200 records valid, 0 exclusions, all TTFT
client-measured.** The invariant now holds mechanically.

Spine reuse still removes ~98.9% of per-turn prefill (2544 -> 28 evaluated) and is
worth **23.5x on TTFT** (2276.76 -> 97.05 ms) — the qualitative conclusion of the
previous branch is unchanged and is now measured over the full 50 turns. Ephemeral
forks still pass all four invariants (`query_forks.csv`).

Rebase cadence 10 vs 25 remains indistinguishable, but now on *valid* totals
(168.50 vs 167.22 ms) rather than on error-contaminated ones. **The conclusion was
right for the wrong reason; it is now right for the right one.**

## 1. Reconciling the state-spine config with the best service config [MEASURED]

`service-batching-gate` found `t4/tb16/b4096/ub4096/np8` better for the *cold*
controller. Tested on the *state-spine* workload: 8 domains, 1600-token stable prefix,
135-token delta, 4-token output, 2 interleaved rounds of 20 requests per config.

Both arms use `-c 40960`, because the original `-c 20480` with `-np 8` gives a
2560-token slot and **cannot run this workload at all** — that is the defect in §0.

| round | A `t8 b2048 ub2048` | B `t4 tb16 b4096 ub4096` |
|---|---:|---:|
| 0 | 292.7 ms | 217.5 ms |
| 1 (order reversed) | 294.1 ms | 218.1 ms |

| config | n | TTFT p50 | p95 | sd |
|---|---:|---:|---:|---:|
| A state-scheduler | 40 | 293.0 | 308.2 | 133.7 |
| B service-batching | 40 | **217.9** | **226.2** | **94.6** |

**B is 25.66% faster on median TTFT**, with lower p95 and lower dispersion, and the
per-round values are tight (292.7/294.1 vs 217.5/218.1) — these are separable, unlike
the t4-vs-t8 comparison on `service-batching-gate`. Evaluated and reused tokens are
identical across arms (72.3 / 1660.8), so this is a pure execution-speed difference.

**Adopted as the common baseline for everything below**, which also preserves 4 CPU
threads of headroom for the GPU worker and verifier.

## 2. The production-shaped multi-domain workload [MEASURED]

`tools/multi_domain.py`. Each domain is an independent controller with:

- **stable prefix, byte-identical across every turn** (so reuse is real, not
  deduplication): objective, controller contract, policy version, action schema,
  WorkGraph summary, authoritative state spine, and its own tag — **1600 tokens**
- **volatile suffix, different every turn**: state version, changed cells, new event,
  resource delta, verification delta — calibrated against the server's own tokenizer
  to **39 / 135 / 265 tokens** (never a chars/token estimate)
- **output: 4 tokens**, deterministic sampling

Tags are **64-bit random hex** derived by SHA-256, deliberately not sequential:
`controller-cache-batching` produced a false contamination signal because sequential
tags were guessable and the model simply predicted a neighbour. Only unguessable tags
distinguish "saw foreign state" from "guessed".

### Two harness bugs found and fixed before publishing anything [CORRECTION]

Both were the same class — a measured pass silently re-sending prompts an earlier pass
had already warmed — and both would have inflated the results substantially.

1. **Warm/measure turn collision.** With D domains and D requests, `turn = 1 + i//D`
   is 1 for every request, so a measured pass starting at turn 1 re-sent exactly what
   the warm pass sent. Observed as `eval = 1` at 32 domains for *all three* delta
   sizes. Fixed with `turn0`: warm at turn 0, measure from turn 1.
   Preserved as `multi_domain_DUPLICATE_TURNS.csv`.
2. **Turn collision across cells.** Each concurrency level restarted at turn 1, so
   `c>=2` re-sent what `c=1` had just sent — `eval` fell to 40.7 at 32 domains and to
   **1** at 64. Fixed with a turn cursor that advances across cells.
   Preserved as `concurrency_DUPLICATE_TURNS.csv`.

After both fixes `eval` sits at the real delta size (~120 tokens for the 135-token
arm) in every cell, which is the signature of a genuine warm-spine + fresh-delta
request.

## 3. Multi-domain matrix [MEASURED]

`t4/tb16 b4096 ub4096 np8 c40960`, default `--cache-ram 8192`, output 4 tokens,
>= 2 fresh turns per domain.

| domains | delta tok | TTFT p50 | TTFT p95 | total p50 | eval | reused | req/s | contam | RSS MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39 | **64.6** | 78.0 | 130.8 | 25.1 | 1614.9 | 7.560 | 0 | 6752 |
| 1 | 135 | 202.0 | 212.3 | 264.7 | 120.5 | 1614.9 | 3.790 | 0 | 6752 |
| 1 | 265 | 349.0 | 357.2 | 414.3 | 251.2 | 1614.9 | 2.409 | 0 | 6753 |
| 8 | 39 | 86.5 | 101.8 | 148.9 | 25.0 | 1613.5 | 6.611 | 0 | 11449 |
| 8 | 135 | 223.5 | 231.3 | 286.2 | 119.9 | 1613.5 | 3.473 | 0 | 14703 |
| 8 | 265 | 378.6 | 387.5 | 443.7 | 250.6 | 1613.5 | 2.257 | 0 | 14748 |
| 32 | 39 | 96.6 | 104.4 | 157.2 | 24.5 | 1615.0 | 6.283 | 0 | 14459 |
| 32 | 135 | 231.7 | 243.8 | 293.9 | 119.6 | 1615.0 | 3.382 | 0 | 14786 |
| 32 | 265 | 383.6 | 392.8 | 448.4 | 226.6 | 1638.8 | 2.367 | 0 | 14690 |
| 64 | 39 | 96.6 | 104.4 | 158.9 | 25.0 | 1614.6 | 6.232 | 0 | 14679 |
| 64 | 135 | 232.7 | 243.4 | 294.2 | 120.1 | 1614.6 | 3.381 | 0 | 14764 |
| **64** | **265** | **1368.1** | 1420.6 | 1434.1 | **1808.0** | **57.5** | 0.696 | 0 | 14693 |
| **96** | 39 | **1182.6** | 1205.9 | 1246.5 | **1582.0** | **57.4** | 0.801 | 0 | 14797 |
| **96** | 135 | **1264.7** | 1316.8 | 1327.0 | **1677.1** | **57.4** | 0.751 | 0 | 14762 |
| **96** | 265 | **1363.1** | 1404.0 | 1430.6 | **1807.9** | **57.4** | 0.698 | 0 | 14691 |

### Domain count is free until the cache runs out, then it is a cliff [MEASURED]

Up to the capacity limit, **adding domains costs almost nothing**: at delta 135, TTFT
p50 moves 202.0 -> 223.5 -> 231.7 -> 232.7 ms going 1 -> 8 -> 32 -> 64 domains. The
whole cost is the delta size, not how many domains are resident.

Past capacity, `reused` collapses from ~1615 to **57.4** and every request becomes a
near-full prefill. 57 tokens is the preamble text common to all domains — the only
thing still shared once each domain's own spine has been evicted.

### The capacity model predicts the knee, stated before the sweep [DERIVED + MEASURED]

KV per token = 5 KV heads x 128 head_dim x 30 layers x 2 (K,V) x 2 B = **75.0 KiB**.

| delta | total tok | MiB/domain | predicted capacity @8 GiB | observed |
|---|---:|---:|---:|---|
| 39 | 1639 | 120.0 | **68.2** | 64 warm, 96 miss ✓ |
| 135 | 1735 | 127.1 | **64.5** | 64 warm (at the edge), 96 miss ✓ |
| 265 | 1865 | 136.6 | **60.0** | 32 warm, **64 miss** ✓ |

All three brackets are consistent with the prediction, including the one that matters
most: at delta 265 the model says 60 and 64 domains **did** miss.

**Zero foreign-domain tags in any cell (`contam 0` in all 15).** State version and
fork invariants hold (§0, `query_forks.csv`).

## 4. Closed-loop concurrency over DISTINCT warm domains [MEASURED]

Not identical requests. Each request targets a different warm domain with its own
fresh delta. Delta 135, output 4 tokens.

| domains | c | req/s | TTFT p50 | TTFT p95 | total p95 | eval | contam |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 3.489 | 220.8 | 261.1 | 324.3 | 120.0 | 0 |
| 8 | 2 | 4.004 | 416.1 | 448.0 | 574.7 | 120.2 | 0 |
| 8 | 4 | 4.525 | 793.6 | 821.5 | 976.7 | 120.3 | 0 |
| 8 | 8 | 5.450 | 1342.9 | 1359.6 | 1488.2 | 120.2 | 0 |
| 32 | 1 | 3.389 | 230.6 | 242.7 | 309.6 | 120.1 | 0 |
| 32 | 2 | 3.874 | 432.1 | 500.2 | 600.5 | 120.0 | 0 |
| 32 | 4 | 4.403 | 822.3 | 858.8 | 1006.9 | 120.2 | 0 |
| 32 | 8 | 5.098 | 1427.0 | 1511.4 | 1642.3 | 120.5 | 0 |
| 64 | 1 | 3.396 | 231.5 | 239.8 | 306.3 | 120.1 | 0 |
| 64 | 2 | 4.045 | 419.8 | 449.0 | 520.6 | 119.9 | 0 |
| 64 | 4 | 4.734 | 766.9 | 801.2 | 888.0 | 120.2 | 0 |
| 64 | 8 | 5.075 | 1447.0 | 1514.3 | 1645.8 | 120.6 | 0 |

**The warm envelope is ~5x the cold one.** A warm state-spine controller sustains
**3.4–3.5 req/s at c=1 with 231 ms TTFT**, against 0.665 req/s and 1470 ms measured
cold on `service-batching-gate` — 5.1x throughput at 6.4x lower latency, from state
residency alone.

**Concurrency still buys little and costs a lot**, and the shape is unchanged by
warmth: c1 -> c8 is +50% to +56% throughput for **+508% to +525% TTFT**. c=2 is the
knee (+15% throughput for +81% TTFT).

**Domain count is irrelevant to the concurrency curve while everything is warm** —
8, 32 and 64 domains give 3.49/3.39/3.40 req/s at c=1 and 5.45/5.10/5.08 at c=8. What
matters is whether the working set fits, not how many domains there are.
