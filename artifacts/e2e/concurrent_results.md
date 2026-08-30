# Round 3: concurrent CPU+NPU execution — the hybrid finally wins

## The change

Offload was **exclusive**: thread 0 drove the NPU while 15 CPU cores idled on a
barrier for ~76% of the prefill. The NPU was already the faster engine for this
arithmetic (777 ms against the CPU's 2009 ms for a whole 2048-token prefill), but
replacing a 1.0x engine with a 0.6x-of-total one is still a loss.

Now the token batch is split: **thread 0 drives the NPU over tokens `[0, t_npu)`
while threads 1..nth-1 compute `[t_npu, ne11)` on the CPU**, simultaneously. One
barrier joins them, then every thread shares the f32 epilogue.

`BITNET_XDNA_SPLIT` sets the NPU's fraction (default 0.5; 1.0 restores exclusive
offload). `t_npu` is rounded to a whole NPU tile so no dispatch is padded.

## A second, larger effect: the micro-batch

The split appeared to do nothing at first — every ratio gave 626-636 t/s. The
reason is that **`ne11` is llama.cpp's micro-batch, not the prompt length**, and
it defaults to 512. With a 1024-token NPU tile that meant two things at once:

- the batch could never exceed one tile, so the split never engaged, and
- every batch was zero-padded 512 -> 1024, so the NPU did **2x the useful work**.

Raising `-ub` fixes both:

| `-ub` | hybrid pp2048 | dispatches |
|---|---|---|
| 512 (default) | 616 t/s | 2568 |
| 1024 | 849 t/s | 1284 |
| 2048 | **1113 t/s** | 642 |

## Result

5 reps per point, backends run alternately to cancel drift, **both at `-ub 2048`**
so the comparison is matched:

| prompt | CPU-only | hybrid | ratio |
|---:|---:|---:|---:|
| 512 | 1247.1 | 1249.2 | **1.00x** (falls back to CPU by policy) |
| 2048 | 1038.5 | 1165.3 | **1.12x** |
| 3968 | 832.7 | 896.9 | **1.08x** |
| tg32 (decode) | 80.5 | 80.8 | 1.00x (NPU unused by design) |

**The hybrid beats CPU-only at 2048 and 3968 tokens** — the range a resident
controller actually operates in. Correctness is unchanged: perplexity is
`307.5806 +/- 27.85495` at every split ratio tested (0.25 / 0.5 / 1.0) and for
CPU-only, and the kernel remains bit-exact.

## Why 512 falls back rather than losing

A 512-token batch cannot fill the 1024-token tile, so the NPU would do a full
tile's work for half the useful output. Measured: 728 t/s offloaded vs 1241 t/s
on CPU. The minimum batch for offload is therefore one full tile, which turns a
0.59x regression into a clean 1.00x. Fixing this properly needs a second,
smaller compiled program — `m=128` (the tiling worth 13.2 TOPS) is illegal below
M=1024, so a 512 tile would have to drop to `m=64` and ~9.0 TOPS.

## The split ratio is coarse

`t_npu` is rounded to whole 1024-token tiles, so at `ne11=2048` the only choices
are 0, 1024 and 2048. Ratios 0.25-0.625 all resolve to 1024 and measure the same
within noise (1122-1193 t/s); 0.75 and 1.0 both resolve to 2048 and collapse back
to exclusive offload (837-867 t/s). The apparent "optimum at 0.5" is really
"one tile each", and finer control needs a smaller tile.

## Two findings from an adversarial review pass

### The CPU baseline is the *stronger* of the fork's two kernels, not the weaker

`-DGGML_LLAMAFILE=OFF` is required for our hook to be reachable, and it disables
`tinyBLAS_I2S_AVX` (`ggml/src/ggml-cpu/llamafile/sgemm.cpp:1357`) — a real
register-tiled AVX2 I2_S kernel with the epilogue fused inline. That looked like a
serious confound: we might have been benchmarking against a deliberately weakened
CPU.

Measured, both at `-ub 2048`, 4 reps, alternated:

| prompt | llamafile **OFF** (our baseline) | llamafile **ON** |
|---:|---:|---:|
| 512 | 900.2 | 841.9 |
| 2048 | **1033.2** | 710.1 |
| 3968 | 826.8 | 739.7 |

`tinyBLAS_I2S_AVX` is **slower** than the `ggml_gemm_i2_i8_s` path for this model,
by up to 1.45x. So the baseline we have been comparing against is the faster of the
two CPU kernels, and the "vs CPU" ratios are if anything conservative. Worth stating
plainly, because the opposite was a reasonable thing to suspect.

### Only 147 of 150 offloadable tensors are resident — and that is correct

Instrumenting every rejection shows **zero failures** and **147 resident tensors**,
not the expected 150. The pattern is exactly `2x30 + 3x29`: every layer's `attn_q`
and `attn_output` are offloaded, but one layer's `ffn_gate`/`ffn_up`/`ffn_down` are
not.

The cause is a llama.cpp optimization, not a bug. `src/models/bitnet.cpp:114`:

```c
if (il == n_layer - 1 && inp_out_ids) {
    cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
    inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
}
```

At the last layer, after attention and *before* the FFN, the token set is reduced to
just the output tokens — 1 for benchmarking. So that FFN legitimately runs with
`ne11 = 1`, falls below the offload threshold, and correctly stays on the CPU.

This matters for how the correctness evidence should be read: "perplexity identical
while N matmuls ran on the NPU" is only as strong as the coverage behind it, and the
coverage is now measured and explained rather than assumed. The runtime also now
logs any tensor it declines to offload, with the reason, so a genuine silent
fallback cannot hide.
