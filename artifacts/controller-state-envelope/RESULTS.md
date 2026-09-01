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

## 5. Warm open-loop characterization [MEASURED]

`tools/warm_open_loop.py`. 32 warm domains (well inside the 58-domain capacity at
8 GiB), 1600-token prefix, 135-token delta, 4-token output. Arrivals are scheduled
*before* the run using the construction corrected on `service-batching-gate` —
conditioned on N arrivals in [0, T] a Poisson process places them as N uniform order
statistics, so fixing `N = round(rate x T)` makes the offered rate exact instead of
seed-dependent. Latency is charged from the **scheduled** arrival.

Closed-loop reference capacity: 5.0 req/s (32 domains, c=8, §4).

| offered | frac | completed | n | cache hit | TTFT p50 | TTFT p95 | total p95 | in-flight max | err |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.250 | 25% | 1.255 | 300 | 100% | **285.1** | 666.0 | 1186.9 | 9 | 0 |
| 2.500 | 50% | 2.505 | 600 | 100% | **409.2** | 1291.7 | 2583.8 | 15 | 0 |
| 3.750 | 75% | 3.744 | 900 | 100% | 1482.4 | 4081.6 | 5333.4 | 28 | 0 |
| 4.500 | 90% | **3.974** | 1080 | 100% | **17026.6** | 31198.0 | 32443.2 | **137** | 0 |

**Cache hit rate is 100% at every rate** — residency is not the limit here, execution
is. Zero errors and zero cross-domain contamination across 2880 requests.

**Instability sits between 75% and 90% of closed-loop capacity.** At 90% completed
(3.974/s) falls below offered (4.500/s), in-flight climbs to 137, and TTFT p50 reaches
17.0 s — a 74x inflation over the unloaded 231 ms. The closed-loop figure of 5.0 req/s
is a throughput ceiling, not a service level.

**Service budget:** ~1.25 req/s holds p50 at 285 ms and p95 at 666 ms; ~2.5 req/s holds
p50 at 409 ms and p95 at 1.3 s. Beyond ~3.75 req/s the median has already degraded 6.4x
even though throughput still looks healthy — throughput is a lagging indicator here.

## 6. Cache-RAM scaling [MEASURED]

`tools/cache_ram_scale.py`. Capacity predicted **before** each run from the KV
geometry (75.0 KiB/token x 1735 tokens = 127.1 MiB/domain), then probed
newest-to-oldest around the prediction, stopping at the first run of misses. My GPU
worker was stopped for this sweep to free unified memory; the third-party `lemonade`
server (8.7 GiB) could not be and remains a standing tax.

| `--cache-ram` | predicted | **observed** | ratio | warm TTFT p50 | cold TTFT p50 | ctrl RSS | mem avail | swap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 MiB | 64.5 | **58** | 0.90 | 233.1 ms | 1241.7 ms | 14762 MiB | 48529 MiB | 8191 (static) |
| 16384 MiB | 128.9 | **122** | 0.95 | 232.4 ms | 1270.6 ms | 22910 MiB | 40442 MiB | 8191 (static) |
| 32768 MiB | 257.9 | **251** | 0.97 | 231.6 ms | 1248.8 ms | 39330 MiB | 23963 MiB | 8191 (static) |

**The knob works, and it is linear.** Quadrupling the budget quadruples capacity
(58 -> 251, 4.33x for 4x the RAM). RSS tracks the budget almost exactly
(+8148 MiB for +8192 MiB, then +16420 for +16384).

**Warm TTFT is completely flat across a 4x cache: 233.1 / 232.4 / 231.6 ms.** A larger
prompt cache costs nothing at lookup time — the LRU list is walked with
`get_common_prefix` per entry, but at these sizes that is invisible against a 231 ms
request. So capacity is purchasable with RAM at no latency penalty.

**The prediction is conservative and gets better with size** (0.90 -> 0.95 -> 0.97).
The shortfall is fixed overhead — slot KV, the model, allocator slack — amortized over
more domains as the budget grows.

Swap did not move (8191 MiB throughout, and it was already static at session start),
so none of these numbers are swap-contaminated.

## 7. GPU co-tenancy at large prompt-cache budget [MEASURED]

`tools/gpu_cache_residency.py`. This is why the cache-RAM question belongs on this
machine: the Radeon 8060S is an iGPU on unified memory, so the host-side prompt cache
and the worker's GTT come out of the same 122 GiB. The **fill** phase and the
**resident steady-state** phase are measured separately — filling hundreds of domains
is itself a heavy prefill workload, and judging GPU decode during it would measure the
fill.

| cache | domains | phase | GPU | ctrl TTFT p50 | ctrl TTFT p95 | GPU decode tok/s | W | GTT MiB | mem avail | swap |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8 GiB | 58 | fill | idle | — | — | — | 114.5 | 59805 | 31621 | 8191 |
| 8 GiB | 58 | steady | idle | 230.6 | 243.1 | — | 96.0 | 59805 | 31604 | 8191 |
| 8 GiB | 58 | steady | decode | 329.4 | 365.6 | **11.76** | 118.7 | 59814 | 31060 | 8191 |
| 32 GiB | 251 | fill | idle | — | — | — | 115.5 | 59814 | 6214 | 8191 |
| 32 GiB | 251 | steady | idle | 232.7 | 246.7 | — | 95.9 | 59814 | 6104 | 8191 |
| 32 GiB | 251 | steady | decode | 334.9 | 353.4 | **11.76** | 118.9 | 59814 | 6634 | 8191 |

**Buying 4.3x more warm controller domains costs the GPU worker exactly nothing.**
Decode throughput is **11.76 tok/s at both 8 GiB and 32 GiB** — identical to three
significant figures. Controller TTFT under GPU decode is likewise unchanged
(329.4 vs 334.9 ms, +1.7%), as is power (118.7 vs 118.9 W) and GTT (59814 MiB, flat).

**So this is a straightforward service-capacity knob**, with one real limit that is
*memory headroom, not performance*: at 32 GiB the machine has only **6.2 GiB
available** with the worker resident and the 8.7 GiB third-party `lemonade` server
still holding memory. Swap never moved (8191 MiB, static). 32 GiB is safe here but is
close to the ceiling on this box as currently populated.

**Cold-start cost of large residency [MEASURED]:** filling is **0.74–0.75 domains/s
regardless of budget**, so 58 domains take 78 s and 251 take 337 s. Residency is cheap
to hold and slow to build — which argues for filling in the background and for
persisting/rebuilding on a schedule rather than on demand.

GPU decode costs the controller +43% TTFT (230.6 -> 329.4 ms), consistent with
`service-batching-gate`'s finding that decode is the harsher GPU phase.

## 8. Thrash characterization [MEASURED — random degrades, cyclic collapses]

`tools/cache_thrash.py`. `--cache-ram 8192`, measured capacity 58 domains, 120
requests per pattern.

| regime | working set | pattern | hit rate | TTFT p50 | TTFT p95 | req/s |
|---|---:|---|---:|---:|---:|---:|
| below capacity (0.75x) | 44 | random | **100.0%** | 233.5 | 237.4 | 3.398 |
| below capacity (0.75x) | 44 | cyclic | **100.0%** | 234.5 | 242.3 | 3.376 |
| above capacity (1.25x) | 72 | random | **80.0%** | **233.8** | 1272.7 | 1.989 |
| above capacity (1.25x) | 72 | cyclic | **29.2%** | **1246.9** | 1297.4 | 0.977 |

This **refines** the previous branch's finding rather than repeating it.
`controller-state-scheduler` reported that exceeding capacity collapses the hit rate to
zero — but that was measured with a cyclic scan, which is the pattern LRU is worst at
by construction.

**Under realistic random access, a 24%-oversized working set still gets 80% hits and
its median latency is unchanged (233.8 vs 233.5 ms).** Only the tail suffers
(p95 237 -> 1273 ms) and throughput halves. Under adversarial cyclic access at the same
working set the hit rate falls to 29.2% and the **median** goes to 1246.9 ms — 5.3x
worse than random at identical pressure.

**Below capacity, access pattern does not matter at all** (100% both, 3.40 vs 3.38
req/s).

### The rule an admission controller would need [DERIVED]

Not implemented tonight — this is the input for that work:

1. **Keep the resident working set at or below measured capacity.** Capacity is
   `cache_ram_MiB / (state_tokens x 75.0 KiB / 1024)`, which predicted all three
   budgets to within 3–10%, conservatively.
2. **Below capacity, admit freely** — pattern and domain count are irrelevant
   (§3, §4, and both rows above).
3. **Above capacity, the access pattern decides the severity.** Randomised or
   LRU-friendly ordering keeps the median intact and costs throughput; round-robin
   over the whole set destroys the median. So a scheduler that must exceed capacity
   should avoid sweeping domains in a fixed cycle.
4. **Rate, not just residency, needs a cap**: even at 100% hit rate, offered load above
   ~75% of closed-loop capacity degrades the median 6.4x and above ~90% is unstable
   (§5).

## 9. Steady-state NPU engagement [MEASURED]

Windowed `ne11` histogram deltas around a warm cell and a cold cell.

| window | evaluated tokens | TTFT p50 | ne11 buckets | NPU-eligible (ne11 >= `kMTile` 1024) |
|---|---:|---:|---|---:|
| warm steady state, 24 requests | 120.0 | 199.7 ms | `b1: 61632, b64: 79488` | **0 / 141120 = 0.00%** |
| cold, 6 fresh domains | 1677.8 | 1276.0 ms | `b1: 15408, b1024: 19872` | 19872 / 35280 = **56.33%** |

**A warm state-spine controller runs entirely on the CPU.** Every offloadable node sees
`ne11` around 120 — the delta size — which is far below the 1024-token threshold, so
the XDNA path correctly declines on every request. The NPU engages only on cold fills
and cache misses, where 56% of nodes qualify.

This is the same tension recorded on `controller-state-scheduler`, now measured on the
production-shaped workload: **the configuration that makes the controller fast is the
one that leaves the NPU idle.** The offload is not broken and was not disabled — it is
correctly declining work that is too small to be worth a dispatch.

(The counter is incremented inside `worth_it()`, which the ggml gate evaluates on all
`nth` threads, so absolute counts overcount by roughly `nth`. Bucket *ratios*, which
are what is used here, are unaffected.)

---

# REAL STEADY-STATE CONTROLLER BUDGET

Measured on this machine, `t4 tb16 b4096 ub4096 np8 -c 40960`, BitNet-b1.58-2B-4T I2_S.

| quantity | value |
|---|---|
| representative stable-prefix tokens | **1600** |
| representative changing-delta tokens | **135** (39 and 265 also measured) |
| output tokens | **4** |
| active warm domains | **58** at `--cache-ram 8192`; **251** at 32768 |
| cache RAM per domain | **127.1 MiB** (= state_tokens x 75.0 KiB) |
| warm TTFT p50 / p95, c=1, idle | **231 ms / 240 ms** |
| warm total p50 / p95, c=1 | **294 ms / 306 ms** |
| warm TTFT under GPU decode | **329–335 ms** |
| cold / cache-miss TTFT p50 | **1242–1276 ms** |
| closed-loop capacity (c=8) | **5.1 req/s** |
| sustainable open-loop req/s | **~2.5 req/s** (p50 409 ms, p95 1292 ms); ~1.25 req/s for p95 666 ms |
| unstable above | **~3.75–4.5 req/s** (completed < offered at 90%) |
| NPU hit fraction, warm steady state | **0.00%** |
| NPU hit fraction, cold miss | **56.33%** |
| GPU interference (worker decoding) | controller +43% TTFT; **GPU decode unaffected by cache size (11.76 tok/s at both 8 and 32 GiB)** |
| verifier interference | not re-measured this pass; `service-cotenancy` measured 1230 ops/s, p95 1.196 ms unaffected [DEFERRED] |
| GPU-training interference | **+0.7% TTFT, −1.8% throughput** — see `halo-training-smoke` |

## Verdict

### VALID WITH RESIDENCY-BASED ADMISSION

The multi-domain state-spine service works and is **~5x the cold envelope**
(3.4 req/s at 231 ms vs 0.665 req/s at 1470 ms). Domain count is nearly free while the
working set fits: 1 -> 64 domains moves TTFT p50 only 202 -> 233 ms. Residency scales
linearly and cheaply with `--cache-ram` (58 / 122 / 251 domains at 8 / 16 / 32 GiB) at
**zero** warm-latency cost and **zero** measured cost to the GPU worker.

It is not unconditionally validated, and it is not thrash-limited either — both of the
stronger verdicts overstate the evidence:

- **Not "MULTI-DOMAIN STATE-SPINE SERVICE VALIDATED"**, because two limits bind and
  must be respected: the working set must stay within a capacity that is *predictable*
  (`cache_ram / (tokens x 75 KiB)`, accurate to 3–10%, conservative), and offered load
  must stay near ~50% of closed-loop capacity — at 90% the service is unstable with a
  100% cache hit rate, so rate admission is needed *independently* of residency.
- **Not "PROMPT-CACHE THRASH IS THE BINDING LIMIT"**, because at 1.25x capacity with
  realistic random access the median is *unchanged* (233.8 vs 233.5 ms) and 80% of
  requests still hit. Thrash is a real cliff only under adversarial cyclic access, and
  it is avoidable by construction. In the open-loop runs the cache hit rate was 100%
  at every rate that mattered — execution, not residency, set the limit.

**Admission needs two rules, not one:** bound the resident working set by the predicted
capacity, and bound the arrival rate near half of closed-loop capacity. Neither alone
is sufficient.

*(The appendix below is a separately-scoped hardware probe run after this verdict, on
request. It concerns candidate model geometry, not the service, and does not bear on
the verdict above — which remains the single service verdict for this pass.)*

---

# Appendix — P4 candidate-shape feasibility probe [MEASURED]

Run after the primary tasks, on request. `tools/shape_probe.py`. Bounded strictly: it
asks only whether a differently-sized controller's linear layers would run better or
worse on XDNA2 than BitNet-2B's. **No new backend work was required** — the stock IRON
`whole_array` example already takes `-M/-K/-N` and tile sizes, so arbitrary INT8 shapes
are evaluable with existing tooling. (The *runtime's* `plan_for` is restricted to
`K,N ∈ {2560, 6912}`, but that is a dispatch constraint, not a build one.)

Tile `64x64x64`, 8 columns, int8 -> int32, legality enforced up front
(`N % (n*cols)`, `K % k`, `M % m`, `M/(m*rows)` even, L1 <= 32 KiB).

| geometry | role | K | N | M=512 | M=1024 | M=2048 | M=4096 |
|---|---|---:|---:|---:|---:|---:|---:|
| current-2B (2560/6912/2560) | attn q,o | 2560 | 2560 | 8.39 | 6.63 | 9.02 | 8.97 |
| current-2B | ffn up/gate | 2560 | 6912 | **FAIL** | **FAIL** | **FAIL** | **FAIL** |
| current-2B | ffn down | 6912 | 2560 | 9.13 | 9.43 | 9.34 | 9.65 |
| small (1024/3072/1024) | attn q,o | 1024 | 1024 | 4.76 | 6.34 | 7.42 | 7.87 |
| small | ffn up/gate | 1024 | 3072 | 6.38 | 7.53 | 7.78 | 8.02 |
| small | ffn down | 3072 | 1024 | 7.79 | 8.92 | 9.77 | 9.61 |
| cand-1.7B (2048/6144/2048) | attn q,o | 2048 | 2048 | 7.59 | 8.44 | 8.75 | 8.95 |
| cand-1.7B | ffn up/gate | 2048 | 6144 | **FAIL** | **FAIL** | **FAIL** | **FAIL** |
| cand-1.7B | ffn down | 6144 | 2048 | 8.95 | 9.26 | 9.55 | 9.69 |

(TOPS = 2·M·K·N / device time, i.e. int8 MACs counted as two ops.)

| geometry | mean TOPS | best | at M>=2048 | buildable |
|---|---:|---:|---:|---:|
| current-2B (2560/6912/2560) | 8.82 | 9.65 | **9.25** | 8/12 |
| small (1024/3072/1024) | 7.68 | 9.77 | 8.41 | **12/12** |
| cand-1.7B (2048/6144/2048) | 8.90 | 9.69 | **9.23** | 8/12 |

### 1. No candidate geometry is materially better [MEASURED]

At the sizes that matter (`M >= 2048`), the 1.7B candidate and the current 2B geometry
are **indistinguishable — 9.23 vs 9.25 TOPS**. The small 1024-wide geometry is
**worse**, not better: 8.41 TOPS, and only 4.76 at `M=512`, because a 1024x1024 matmul
is too small to fill 32 cores. **Shrinking the model does not buy NPU efficiency.**

### 2. `N <= 4096` is a hard single-kernel limit, and it is not model-specific [MEASURED]

Both `N=6912` and `N=6144` fail to build, identically, at every M and every legal tile.
The error is the DMA descriptor stride:

```
N=6912: static_strides = [1769472, 384, 6912, 1]
N=6144: static_strides = [1572864, 512, 6144, 1]
aiecc: edge 'npu_dma_lowered.mlir' failed
```

The outer stride is `256 x N` against the `aie.dma_bd` range `[1, 1048576]`, so
**any N above 4096 is unbuildable as one kernel** regardless of geometry. This is a
property of the hardware/toolchain, not of BitNet — the current runtime already works
around it by serving `N=6912` as chunks. **A 1.7B candidate with a 6144-wide FFN would
inherit exactly the same constraint and exactly the same workaround.**

### 3. Efficiency is set by M, not by model width [MEASURED]

Every geometry improves monotonically with M (small: 4.76 -> 7.87; cand-1.7B:
7.59 -> 8.95), and all three converge to ~9–9.7 TOPS once M >= 2048. The device wants
large token batches far more than it wants a particular hidden size.

### 4. An anomaly, reported rather than smoothed [MEASURED]

`current-2B attn_qo` at `M=1024` reads **6.63 TOPS**, below both its M=512 (8.39) and
M=2048 (9.02) neighbours. A standalone run earlier in the session measured 8.93 TOPS
for the same shape, so this was re-measured three times: **6.64 / 7.16 / 6.67** — the
dip is reproducible and the single earlier 8.93 reading is the outlier.

The dip lands exactly on `kMTile = 1024`, the runtime's production coordinate, which is
suggestive — but the runtime uses tile `128x64x64` and this probe used `64x64x64`, so
the result does **not** transfer to the shipped kernel and no claim is made about it.
It is recorded as a lead for a future tiling pass, not a finding.

### Conclusion

**Changing the controller's hidden geometry is not a lever for NPU throughput.** A
1.7B-shaped model performs the same as the current 2B one, a smaller one performs
worse, and the FFN-width constraint is identical across all of them. Combined with §9
— a warm controller engages the NPU **0%** of the time — geometry choice for the
controller specialist should be driven by quality, memory and CPU decode speed, not by
XDNA2 characteristics. [DEFERRED: no new AOT artifacts were built; a different geometry
would need them, and that is the follow-on work this probe exists to scope.]
