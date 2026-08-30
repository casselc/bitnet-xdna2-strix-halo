# The dominant cost is hardware-context switching, not kernel throughput

## How it was found

In-model dispatch was ~2.4x slower than the same kernel standalone. Hypotheses
tested and **eliminated**, each by direct measurement:

| hypothesis | test | result |
|---|---|---|
| `xrt::run` construction per dispatch | persistent run + `set_arg` vs `kern(...)`, interleaved | **0.000-0.007 ms.** No effect |
| BO rebinding cost | distinct BO vs sub-buffer vs same, interleaved | **~0.02 ms.** No effect |
| Sustained-load throttling | 90 s continuous dispatch | **Flat at 0.763 ms.** No droop |
| Cache flush cost (`sync()` is a userspace CLFLUSH loop) | four-way split of the timed region | **sync_in 1% + sync_out 3%** |
| Submission overhead | same | **submit 0%** |
| CPU memory-bandwidth contention | 1 / 2 / 4 / 8 / 16 ggml threads | **Flat** (5.61 → 5.55 TOPS) |
| Number of resident weight buffers | 1 → 60 buffers, 375 MiB | **Flat** (10.8 → 11.7 TOPS) |

The four-way split localized it precisely: **95% of dispatch time is inside
`run.wait()`** — the device is genuinely busy that long.

## The cause

Cycling three `xrt::hw_context`s (one per compiled shape) in BitNet's per-layer
order, measured interleaved against the same kernels run in isolation:

| shape | alone | cycled | penalty | TOPS alone → cycled |
|---|---|---|---|---|
| K2560 N2560 | 1.159 ms | 3.592 ms | **+210%** | 11.58 → 3.74 |
| K2560 N6912 | 3.556 ms | 5.926 ms | **+67%** | 10.19 → 6.12 |
| K6912 N2560 | 4.756 ms | 7.253 ms | **+53%** | 7.62 → 5.00 |

Those cycled figures match what the integrated runtime reports per shape
(2.42 / 4.78 / 7.32 ms), so this accounts for essentially the whole gap.

**This is 10-20x worse than the 0.22 ms measured earlier with the smaller M=512
designs** — the penalty scales with design size, consistent with the firmware
reconfiguring an 8-column partition on each switch. Every context holds all 8
columns, so the three cannot be co-resident and the array is reprogrammed
between consecutive dispatches.

## Consequence

Kernel tuning raised throughput 9.3 → 13.2 TOPS (+42%), but context switching
then gives back 50-210% of it. **Optimizing the kernel further is close to
pointless while three contexts are in play.**

The fix is structural: use **one** program for everything, decomposing the other
shapes onto it (N-chunking for the wide FFN, K-chunking with accumulation for
the deep down-projection), so a prefill performs zero context switches. The
extra dispatches are far cheaper than the switches they remove:

```
per layer, 3 contexts (measured):  3.59 + 3.59 + 5.93 + 5.93 + 7.25 = 26.3 ms
per layer, 1 context  (projected): 11 dispatches x 1.159            = 12.7 ms
```
