# Is int8 x int4 worth 2x on this part? Partly settled.

This entry was wrong twice. Recording the evidence so it stops flip-flopping.

## What the headers say

`aie::mmul<4,16,16,int8,int4>` on **`__AIE_ARCH__ == 21` (aie2p — our part)**
compiles to a **single native intrinsic**:

```cpp
this->data = ::mac_4x16_16x16_conf(a, a_sign, b, b_sign, this->data, ...);
```
4x16x16 = **1024 MACs per instruction**, against `mac_8x8_8x8`'s 512 for
int8xint8. The B operand is also twice as wide in bits (256 x int4 = 1024b vs
64 x int8 = 512b).

The "emulated on top of int8 x int8" language in AMD's intrinsics guide — which an
intermediate review cited, and which I wrongly applied to this part — describes
**`__AIE_ARCH__ == 22` (aie2ps)**, whose branch in the *same header* does:

```cpp
auto [a_left, a_right] = interleave_unzip<TypeA, 64>::run(a, a, 8);
this->data = ::mac_4x8_8x16_conf(a_left,  ..., b.extract<128>(0).unpack_sign(...), ...);
this->data = ::mac_4x8_8x16_conf(a_right, ..., b.extract<128>(1).unpack_sign(...), ...);
```
Two MACs plus a shuffle — that is what emulation looks like, and it is not our
architecture. Source: `mlir_aie/include/aie_api/detail/aie2p/mmul_8_4.hpp`, `:23`
(arch 21) vs `:46` (arch 22).

## What compiling it actually shows

Both kernels build for `aie2p-none-unknown-elf` with Peano
(`-D__AIE_API_AIE_ADF_HPP__` is required — it pre-trips aie_api's own guard so
stock upstream headers do not pull in Vitis-only `<adf.h>`).

Disassembling the int4 kernel shows something better than expected:

```
vldb.unpack  y0, unpacksign1, [p1, #0x0];  mov crunpacksize, ...
```

**The 4-bit unpack happens in the load-store unit, on the load path.** It costs no
MAC slots and no vector-ALU slots. That directly answers the "does unpack_sign eat
the 2x" worry: on aie2p there is no software unpack at all.

The int8 kernel's MAC is `vmul dm1, x0, x2, r1` (x-register operands); the int4
kernel's are `vmul dm0, x6, y0` / `vmac dm3, dm0, x4, y1` — using the **y**
register file for the wider B operand.

## What is NOT settled

**Issue latency.** A 1024-MAC instruction that issues every two cycles has the same
throughput as a 512-MAC instruction every one cycle. Static disassembly cannot tell
me which, and my microbenchmark was flawed for this purpose: the operands are
loop-invariant, so the compiler is free to restructure the loop, and the pipelined
loop body is not cleanly visible in the object file.

So the honest position:

| claim | status |
|---|---|
| int4 halves weight DMA (1843 -> 922 MiB) | **certain** |
| ternary fits int4 exactly, no accuracy cost | **certain** |
| the unpack is free (hardware, on the load path) | **confirmed from disassembly** |
| int4 is native on aie2p, not emulated | **confirmed from headers + disassembly** |
| int4 delivers 2x MAC *throughput* | **plausible, unproven** |

## The experiment that would settle it

A timed on-hardware run with **non-loop-invariant operands** — stream A and B tiles
from L1 so the compiler cannot hoist, issue an equal number of mmul instructions
for each dtype, and compare wall time. Equal time means a real 2x (int4 does twice
the MACs per instruction); 2x the time means the wide instruction is half-rate and
only the DMA saving is real.

`npu/experiments/int4_vs_int8_mac.py` has the scaffolding; it needs the operand
streaming and an IRON design to dispatch it. IRON's `kernels.mm()` exposes no int4
combination (`_MM_COMBOS` in `python/aie/iron/kernels/linalg.py`), so this requires
a custom `ExternalFunction` — which is why it was not a one-liner.
