# Prompt-cache and batching semantics of the pinned server

Read from the **pinned tree**, not from current upstream documentation.

| | |
|---|---|
| branch base | `service-cotenancy` @ `9295df0a6167eaa43c983c13f702fce1033e4b1f` |
| llama.cpp | `390c307752ab78fd8189f359d6954c9ba1be74af` (build 9918) |
| server source | `refs/BitNet/3rdparty/llama.cpp/tools/server/` |

## What `cache_prompt` does here [MEASURED — source]

`server-context.cpp:3166`:

```cpp
if (slot.task->params.cache_prompt) {
    // reuse any previously computed tokens that are common with the new prompt
    n_past = slot.prompt.tokens.get_common_prefix(input_tokens);
```

- **Common-prefix matching is implemented** and is the primary mechanism.
- The cache is **slot-local**: `slot.prompt.tokens`, the token sequence that slot
  last evaluated. There is no global/shared prompt cache in this revision.
- Reuse is gated on `slot.can_split()`; a task needing embeddings that cannot be
  split skips the cache path entirely.
- `n_past` becomes the number of tokens **not** re-evaluated.

## Slot selection [MEASURED — source]

`server-context.cpp:1599-1636`. Default `slot_prompt_similarity = 0.1`
(`common/common.h:671`), i.e. **on by default**:

```cpp
const float sim_cur = float(tokens.get_common_prefix(task.tokens)) / task.tokens.size();
if (sim_cur > sim_best && sim_cur > slot_prompt_similarity) { ... }
```

A new request is routed to the slot with the **longest common prefix**, provided
that prefix covers more than 10% of the request. So repeated or shared-prefix
requests are steered back to the slot that already holds their KV — the
mechanism and the routing cooperate.

## Cache reuse beyond the common prefix

`--cache-reuse N` enables reusing *chunks* after the divergence point by shifting
their KV into new positions. Requires `llama_memory_can_shift()` and no mtmd.
**Not enabled in the previous pass.**

## Other relevant flags in this build

| flag | meaning |
|---|---|
| `-np, --parallel N` | number of server **slots** (each with its own KV and its own prompt cache) |
| `-cb / --cont-batching` | continuous batching |
| `--cache-reuse N` | min chunk size for KV-shift reuse past the common prefix |
| `-sps, --slot-prompt-similarity` | LCP similarity threshold for slot routing (default 0.1) |
| `--kv-unified` | unified KV across sequences |
| `--slot-save-path` | slot KV save/restore to disk |

## The instrument

`server-task.cpp:242` emits `cache_n` inside `timings`, set at
`server-context.cpp:507` from `n_prompt_tokens_cache`. So every response reports
**how many prompt tokens were reused rather than evaluated**, which makes the
question "did caching actually engage?" directly measurable rather than inferred
from timing.

## Consequence for the previous pass

`tools/service_bench.py` sent `cache_prompt=False` on every controller, worker
and chained request. Every mechanism above was therefore disabled by
construction, and the reported 2366 ms TTFT and concurrency-1 saturation
characterise **full prefill on every request**. That is a real operating point —
it is what a cold, all-distinct workload costs — but it is not the steady-state
controller workload, and the previous pass's numbers must be read with that
scope.
