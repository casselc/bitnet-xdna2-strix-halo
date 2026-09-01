# Post-ROCm-7.1 regression check [MEASURED]

The system ROCm 7.1 installation is deliberate machine state, not an accident to
be rolled back. This pass establishes whether it disturbed any frozen execution
path before the machine is used as the new baseline for a model bakeoff.

Branch point: `model-candidate-halo` was created from
`origin/controller-state-envelope` at `60230b59adb30c8bf97f1419cd804d5371037d40`.
`origin/halo-training-smoke` (`f4c27323f819e7d62290ac88870cbb7ae42d7f0a`) was
**not** merged; only tooling is ported, with provenance.

## Verdict: PASS — no frozen result moved

| check | reference | measured now | verdict |
|---|---|---|---|
| `make check` (full suite) | passes | passes, exit 0 | **PASS** |
| XDNA GEMM bit-exactness | all shapes bit-exact | all bit-exact, T=1024/1536/2048 | **PASS** |
| concurrent-context lease | holds; unleased path SIGSEGVs | holds, 300 invocations, 0 mismatch | **PASS** |
| perplexity, CPU vs NPU | `307.5806 +/- 27.85495` | `307.5806 +/- 27.85495`, both modes | **PASS** (identical) |
| NPU dispatch engagement | > 0 on a qualifying prefill | **1926** dispatches, 3 shapes | **PASS** |
| warm controller TTFT p50 | 202.0 ms (1 domain, 135 delta) | **197.4 ms** (−2.3%) | **PASS** |
| warm controller TTFT p95 | 212.3 ms | **203.1 ms** (−4.3%) | **PASS** |
| warm controller total p50 | 264.7 ms | **257.5 ms** (−2.7%) | **PASS** |
| GPU (Vulkan) worker decode | 11.76 tok/s (envelope §7) | **12.38 tok/s** | **PASS** |

The controller numbers are reproduced with the reference configuration
`t4 / tb16 / b4096 / ub4096 / np8 / c40960`, `--cache-ram 8192`, 4-token output,
on the `~1600`-token spine + `~135`-token delta workload. Calibration landed on
exactly 1600 spine tokens and 135 delta tokens.

## Two things that look like regressions and are not

Both were reproduced, diagnosed and closed. They are recorded because a later
reader running the documented commands will hit them.

**1. `artifacts/invocations.md` is stale for the promoted runtime.** It documents
`BITNET_XDNA_ARTIFACTS=artifacts/xclbin` and `llama-bench -p 512`. Under the
runtime promoted on `runtime-v1-promotion`, that combination now fails twice
over: `artifacts/xclbin` holds only `M512` xclbins while the runtime requests
`mm_M1024_*`, and `-p 512` leaves the micro-batch below `kMTile = 1024`. The
suite itself uses `artifacts/xclbin-tuned` (see `Makefile:80`), and
`docs/RUNTIME_STATUS.md` already states the `-ub` requirement. Correct
invocation:

```bash
BITNET_XDNA=1 BITNET_XDNA_STATS=1 \
BITNET_XDNA_ARTIFACTS=$PWD/artifacts/xclbin-tuned \
  refs/BitNet/build-xdna/bin/llama-bench -m $M -p 2048 -n 0 -t 16 -ngl 0 -ub 2048
```

With `-ub 512` the run reports `dispatches=0` — correct behaviour, not a fault.
This document does not rewrite `invocations.md`; the evidence branches are frozen.

**2. A 338 ms warm TTFT that was configuration, not ROCm.** A first run measured
TTFT p50 338 ms against the referenced ~202 ms. The cause was entirely local: it
used `t6`, `b/ub 2048` and **no `-tb`**. `service-batching-gate` had already
established that an unset `-tb` alone costs 38% TTFT. Re-run under the reference
configuration it lands at 197.4 ms. This is an independent re-confirmation of
that branch's finding, not a new result.

## Session prerequisite that silently degrades results

`/dev/kfd` and `/dev/dri/renderD128` are `root:render`. A login session created
before the user's `render` membership does not carry the group, and the failure
is silent in both directions: `rocminfo` reports no agents, and RADV falls back
to a CPU rasteriser so a "GPU" arm measures the CPU. `tools/service_ctl.sh`
already wraps the worker in `sg render -c` for exactly this reason;
`tools/halo_rocm_env.sh exec` now does the same for training.

## Exact package coordinates at the time of this check

```
libhsa-runtime64-1      7.1.0+dfsg-0ubuntu9      libamdhip64-7      7.1.0-0ubuntu2
libhsa-runtime-dev      7.1.0+dfsg-0ubuntu9      libamdhip64-dev    7.1.0-0ubuntu2
libhsakmt1              7.1.0+dfsg-0ubuntu9      libhiprtc7         7.1.0-0ubuntu2
libamd-comgr3           7.1.1+dfsg-0ubuntu1      hipcc              7.1.1+dfsg-0ubuntu1
rocm-device-libs-21     7.1.1+dfsg-0ubuntu1      librocblas5        7.1.0-1ubuntu4
rocminfo                7.1.1-0ubuntu1           libhipblas3        7.1.0-0ubuntu5
rocm-smi                7.1.1-0ubuntu1           librocsolver0      7.1.0-0ubuntu2
```

Unchanged from the frozen `artifacts/environment.md`, and therefore **not** a
candidate explanation for any movement:

```
XRT                     1:2.25.0-4~resolute1     kernel             7.0.0-29-generic
libxrt-npu2             1:2.25.0-4~resolute1     NPU firmware       1.1.2.65
mesa-vulkan-drivers     26.0.3-1ubuntu1          libvulkan1         1.4.341.0-1
```

ROCm ships no NPU component: the XDNA path runs through XRT/`amdxdna`, which is
untouched. The Vulkan path runs through Mesa RADV, also untouched. The one path
ROCm genuinely owns is PyTorch/HIP training, and that is the subject of
`TRAINING_ENV.md`.

## Reproduce

```bash
make check
BITNET_XDNA=0 refs/BitNet/build-xdna/bin/llama-perplexity \
    -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    -f artifacts/correctness/ppl_input.txt -t 16 -ngl 0 --chunks 4
# then the -ub 2048 llama-bench line above, then:
CTRL_TB=16 CTRL_B=4096 CTRL_UB=4096 CTRL_CTX=40960 CACHE_RAM=8192 \
    tools/service_ctl.sh start-ctrl 4
tools/model_bakeoff.py --port 8081 --label bitnet-ref --skip-state \
    --out artifacts/model-candidate-halo/bitnet_ref.json
```
