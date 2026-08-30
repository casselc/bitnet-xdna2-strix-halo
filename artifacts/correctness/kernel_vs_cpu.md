# Correctness: XDNA2 kernel vs CPU reference

Weights are the real `blk.0.attn_q.weight` extracted from the shipped GGUF
(sha256 4221b252...), not synthetic data. The op is pure integer arithmetic,
so the bar is bit-exactness, not a tolerance.

```
test_xdna_gemm  [M=512 K=2560 N=2560]  weights: artifacts/correctness/tensors/attn_q_l0.packed
  cpu reference: 360.3 ms (scalar, single-threaded oracle)
  npu: 21 dispatches, 1.531 ms mean round trip

  ok  all 1310720 int32 accumulators BIT-EXACT vs CPU reference
```

## Packing / semantics tests
```
test_i2s_packing
  ok   code mapping: 0->-1, 1->0, 2->+1
  ok   interleave: byte b holds weights {b,b+32,b+64,b+96}, MSB-first
  ok   roundtrip K=128   (1 blocks)
  ok   roundtrip K=2560  (20 blocks)
  ok   roundtrip K=6912  (54 blocks)
  ok   offset identity: acc_u - act_sum == true ternary dot
  ok   activation quant: per-token absmax int8, scale=127/amax, sum consistent
  ok   activation quant: all-zero row clamped by the 1e-5 floor

all passed
```

## Real-GGUF layout validation
```
test_i2s_realdata  artifacts/correctness/tensors/attn_q_l0.packed  [K=2560 N=2560]
  per-tensor scale: 1.2188547850
  code 0 (-1):    1649645   25.17%
  code 1 ( 0):    3251715   49.62%
  code 2 (+1):    1652240   25.21%
  code 3 (--):          0    0.00%   <- must be exactly 0
  +/-1 skew: 0.0008 (symmetric ternarization expects ~0)

  ok  real GGUF weights decode cleanly under our layout
```

## End-to-end numerical equivalence (the strongest evidence here)

Text diffing through `llama-cli` proved unreliable: its animated progress spinner
is interleaved with generated tokens and shifts the terminal wrap points, so the
two runs differ in whitespace while the tokens agree. Perplexity is deterministic,
numeric, and has no UI noise, so it is the better instrument.

`llama-perplexity`, 4 chunks of n_ctx=512 (2048 tokens of real model execution),
same binary, same weights, only `BITNET_XDNA` differs:

```
--- BITNET_XDNA=0   (CPU only)
[1]364.9199,[2]312.3577,[3]288.2628,[4]307.5806,
Final estimate: PPL = 307.5806 +/- 27.85495

--- BITNET_XDNA=1   (NPU-assisted prefill)
[1]364.9199,[2]312.3577,[3]288.2628,[4]307.5806,
Final estimate: PPL = 307.5806 +/- 27.85495
[bitnet-xdna] dispatches=830  dispatch_total=2271.5 ms  mean=2.737 ms
```

**Identical to every printed digit, per chunk and overall, while 830 matmuls ran
on the NPU.** Wall clock differed as expected (3.24 s CPU vs 7.59 s hybrid).

This is stronger than a token-stream diff: perplexity is a sum over per-token log
probabilities across the whole vocabulary, so any divergence in any offloaded
matmul would perturb it.

## Fallback is real, not just declared

NPU enabled but artifacts unreachable — the path must degrade silently and
correctly rather than fail or diverge:

```
BITNET_XDNA=1 BITNET_XDNA_ARTIFACTS=/nonexistent  -> PPL = 312.3577 +/- 40.90633, dispatches=0, 1.744 s
BITNET_XDNA=0                                     -> PPL = 312.3577 +/- 40.90633,               1.717 s
```

Identical perplexity, zero dispatches, and no measurable cost to the CPU path.
NPU availability is not part of model semantics.
