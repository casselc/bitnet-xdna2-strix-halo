# Milestone C — end-to-end: NPU-assisted prefill + CPU decode

Model `BitNet-b1.58-2B-4T` (`ggml-model-i2_s.gguf`, sha256 `4221b252...`), 16 CPU
threads, `-ngl 0` (GPU uninvolved). Same binary, same weights, same tokenizer for
both rows; the only difference is the `BITNET_XDNA` environment variable.

## The hybrid split is real and verifiable

`llama-bench` isolates the two phases, and the dispatch counter settles it:

```
prefill only  (-p 512 -n 0)  : dispatches=410     (2 runs x 205)
decode  only  (-p 0  -n 128) : dispatches=0
```

Decode never touches the NPU, and the prompt is not recomputed on CPU: the hook
returns from `ggml_compute_forward_mul_mat` once the NPU path succeeds, so the
CPU SIMD kernel does not run for those tensors.

## Output equivalence

Same 512-token controller-style prompt, greedy (`--temp 0 --seed 42`), 32 tokens:

```
CPU-only : " node_40: status=ready deps=[40] cost=8 region=r3"
Hybrid   : " node_40: status=ready deps=[40] cost=8 region=r3"
```
Character-identical. This is expected rather than lucky: the NPU computes the
same integer accumulator bit-exactly (see `artifacts/correctness/`), and the only
floating-point difference is the order of the epilogue multiply, which is a
single multiply per element and so is also exact.

## Timing

**These numbers replace an earlier, incorrect set.** The first run showed
CPU-only pp512 at 878 t/s, ~30% below the same model in the reference build. The
cause was mine: `bitnet_xdna_available()` took a mutex, and it is called from
every I2_S `mul_mat` on every thread, so the instrumentation was taxing the
CPU-only baseline and flattering the hybrid. Replacing it with a relaxed atomic
restored CPU-only to 1255 t/s. Both rows below come from the fixed build.

| prompt | CPU-only t/s | hybrid t/s | CPU-only prefill | hybrid prefill | speedup |
|---:|---:|---:|---:|---:|---:|
| 128 | 864.0 | 213.1 | 148 ms | 601 ms | **0.25x** |
| 512 | 1255.1 | 642.4 | 408 ms | 797 ms | **0.51x** |
| 2048 | 1019.2 | 571.0 | 2009 ms | 3587 ms | **0.56x** |
| 3968 | 799.1 | 484.3 | 4966 ms | 8193 ms | **0.61x** |
| tg32 (decode) | 79.76 | 80.22 | — | — | 1.01x |

**The NPU-assisted path is 1.6x to 4x slower than CPU-only at every prompt
length.** Decode is unchanged (1.01x), which is the intended behaviour and
confirms the offload is confined to prefill.

The trend is the one Milestone A predicted — the deficit narrows as prompts grow,
because per-dispatch overhead amortizes — but it does not cross over within the
model's 4096-token context.

## Where the time goes

Per-dispatch cost, decomposed by direct measurement (`tools/npu_switch_cost.cpp`):

| component | cost |
|---|---|
| kernel only, weighted over BitNet's 7 linears | **1.52 ms** |
| + cycling the 3 xclbins in per-layer order | +0.18–0.22 ms |
| + ~1.8 GiB of resident weight buffers | +0.21 ms (at 60 buffers/program) |
| **standalone total** | **~1.9 ms** |
| **measured inside llama.cpp** | **2.66 ms** |

So roughly 0.75 ms per dispatch is attributable to running inside the real
inference loop rather than a benchmark harness. Thread count does not explain it
(dispatch cost is flat at 2.84–2.87 ms from 2 to 16 threads), nor does the number
of resident weight buffers on its own.

At 205 dispatches per 512-token prefill, 2.66 ms each is **545 ms of device
time** — already more than the CPU's entire 583 ms prefill, before the CPU's
remaining work (norms, RoPE, attention, softmax, `attn_k`/`attn_v`, the f16
tied lm_head) is counted.

## Costs the kernel-only benchmark would have hidden

The brief warns against hiding an expensive conversion behind a kernel-only
number. Measured explicitly:

| one-time cost | value |
|---|---|
| ternary -> int8 repack + upload of all offloaded weights | **3.2–3.5 s** |
| resident int8 weight footprint | **1843.1 MiB** (4x the 461 MiB of I2_S linears) |
| device open | 12.3 ms |

The repack is one-time per process, but at 3.4 s it dwarfs any single prefill and
would need amortizing across a long-lived resident controller to be acceptable.
