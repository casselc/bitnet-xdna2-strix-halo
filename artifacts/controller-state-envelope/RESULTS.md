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
