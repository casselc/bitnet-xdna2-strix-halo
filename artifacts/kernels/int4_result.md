# int8 x int4 on aie2p: works, bit-exact, halves weight DMA — but is NOT faster

Settles the question left open in `int4_investigation.md`, on hardware.

## What was built

`npu/kernels/mm_i8_i4.cc` — a mixed-precision AIE kernel using
`aie::mmul<4,16,16,int8,int4>`. mlir-aie's stock `mm.cc` cannot express this: it
hardcodes `aie::mmul<r,s,t,T_in,T_in,accauto>` with the *same* type for both
operands, and IRON's `kernels.mm()` exposes no int4 combination. The file also
carries a matched int8xint8 path with an identical loop nest, so the comparison
below differs in exactly one thing.

`npu/experiments/int4_gemm.py` runs it end to end and checks against numpy.

## Correctness

```
int8 x int4 GEMM, M=K=N=64, mmul 4x16x16
  B bytes on the wire: 2048 (int8 would be 4096)
  CORRECT: all 4096 int32 accumulators bit-exact vs numpy
```
Random ternary weights, verified at K = 16, 32 and 64 (one, two and four mmul
K-blocks). **Half the weight bytes cross the DMA, and the 4-bit expansion is done
by the load-store unit** — the disassembly shows `vldb.unpack y0, unpacksign1,
[p5, ...]`, one per MAC, with no `interleave_unzip` emulation sequence.

## Throughput: no gain

Both kernels run REPEAT full 64x64x64 GEMMs per dispatch, and two REPEAT values
solve out the fixed dispatch cost (`T = dispatch + REPEAT * c`):

| | REPEAT=64 | REPEAT=512 | per-GEMM | dispatch |
|---|---|---|---|---|
| int8 x int8 | 0.259 ms | 0.967 ms | **1.580 us** | 0.158 ms |
| int8 x int4 | 0.257 ms | 0.941 ms | **1.527 us** | 0.159 ms |

**Compute-only ratio: 1.035x.**

A 64^3 GEMM is 262,144 MACs. At 512 MACs per `mac_8x8_8x8` that is 512 mmul
instructions; at 1024 per `mac_4x16_16x16` it is 256. **The int4 kernel issues
half as many instructions in the same time, so the wide instruction takes twice as
long. The MAC rate is identical.**

This also rules out the kernel being load-bound: int4 does half the loads too, and
still shows no gain.

## Conclusion

| claim | verdict |
|---|---|
| int4 halves weight DMA (1843 -> 922 MiB) | **confirmed** |
| ternary fits int4 exactly, bit-exact results | **confirmed on hardware** |
| the 4-bit unpack is free (load-store unit) | **confirmed in disassembly** |
| int4 is native on aie2p, not the aie2ps emulation | **confirmed** |
| **int4 delivers 2x MAC throughput** | **FALSE — measured 1.035x** |

So the earlier reasoning was right about the mechanism and wrong about the
payoff: `mac_4x16_16x16` really is a single native instruction doing 1024 MACs,
and it really does issue at half the rate of the 512-MAC one. **Roughly speaking,
AMD's "emulated on top of int8 x int8" description is the right intuition for
performance even where it is the wrong description of the instruction encoding.**

int4 remains worth doing — halving weight traffic is what makes NPU decode
arithmetically possible at all (28.5 -> 113.8 tok/s ceiling, see
`concurrent_results.md`) — but it should be budgeted as a **bandwidth**
optimization, not a compute one.

## Caveat

The matched kernels reach only ~115-131 GOPS on one core against a ~1.84 TOPS
single-core peak (~7%). They are deliberately simple, not tuned. If both are bound
by something common — loop overhead, pipeline stalls, insufficient unrolling —
that could mask a real difference. The 2x instruction-count argument above is
independent of tuning and points the same way, but a properly pipelined pair would
make the result airtight.
