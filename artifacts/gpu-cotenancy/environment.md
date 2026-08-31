# GPU co-tenancy — machine inventory

Taken **before** installing anything, per the "discovery first" rule. Sanitized:
no hostname, no raw kernel cmdline, no storage pool names, no credentials.

Branch base: `origin/runtime-v1-promotion` @ `712b7c6`.

## Device

| item | value |
|---|---|
| GPU | `[1002:1586]` rev c1 — Strix Halo iGPU (Radeon 8060S) at `c6:00.0` |
| class | Display controller `[0380]` |
| kernel driver in use | **`amdgpu`** |
| render node | `/dev/dri/renderD128` (card0 present) |
| VRAM (carve-out) | **0.5 GiB** |
| **GTT (system memory the GPU can use)** | **97.7 GiB** |
| NPU | `amdxdna` loaded alongside, `[1022:17f0]` at `c7:00.1` |
| kernel | `7.0.0-29-generic` |

The GPU is an **iGPU sharing the same LPDDR5X** as the CPU and NPU. VRAM is a
0.5 GiB carve-out and real capacity comes from GTT, so this machine cannot
separate "GPU memory" from "system memory" — which is precisely why the
co-tenancy question is worth measuring rather than assuming.

## Graphics / compute stack, as found

| component | state |
|---|---|
| **RADV (Mesa AMD Vulkan)** | **present** — `radeon_icd.json`, `mesa-vulkan-drivers 26.0.3` |
| Vulkan loader | `libvulkan1 1.4.341.0` |
| `libdrm-amdgpu1` | 2.4.131 |
| **ROCm / HIP / HSA** | **absent** — no `/opt/rocm`, no `rocminfo`, `hipcc` or `rocm-smi` |
| `ggml-vulkan` in the pinned llama.cpp fork | **present** (`ggml/src/ggml-vulkan/`) |

## Backend decision

**Vulkan (RADV) is the worker path.** It is already installed, it is the
least invasive option, and the pinned fork already carries the backend.

**ROCm is deliberately not installed.** It is a large system-wide change, it is
not packaged for this distribution release, and the working
XRT + mlir-aie + Peano environment is the most valuable thing on this machine.
The brief's rule — stop rather than improvise a destructive change — applies.
This costs the planned ROCm-vs-Vulkan cross-check; the substitute oracle is
**CPU**, which this repository already trusts bit-exactly.

## Missing build prerequisites

`GGML_VULKAN=ON` needs a shader compiler and Vulkan headers, neither present:

| package | candidate | XRT/AIE deps |
|---|---|---|
| `glslc` | 2026.1-1 | **0** |
| `glslang-tools` | 16.2.0-2 | **0** |
| `libvulkan-dev` | 1.4.341.0-1 | **0** |
| `vulkan-tools` | 1.4.341.0+dfsg1-1 | **0** |

`apt-cache depends` shows **none** of them pulls in anything matching
`xrt`, `aie` or `xilinx`. They are additive graphics development packages and
cannot disturb the NPU toolchain.

## What was actually changed

| change | scope |
|---|---|
| `glslc`, `glslang-tools`, `libvulkan-dev`, `vulkan-tools` installed | system, additive; **0** XRT/AIE dependencies |
| user added to the `render` group | needed to open `/dev/dri/renderD128`, which is `root:render` |
| user added to the `lemonade` group | read access to the model cache |
| SPIRV-Headers built into `.localdeps/usr` | **project-local**, not system |
| `refs/BitNet/build-vulkan/` | **new build dir**; `build-xdna/` untouched |

The XDNA toolchain was not modified. `GGML_VULKAN=ON` additionally needs
`SPIRV-Headers`; rather than install it system-wide, it is built into the
repository's existing `.localdeps/usr` prefix and passed via `CMAKE_PREFIX_PATH`
plus an explicit `-I` (its cmake config exports the include dir, but the
`ggml-vulkan` target does not consume it, so `spirv/unified1/spirv.hpp` is not
found without the flag).

Group membership does not reach an already-running session, so commands use
`sg render -c '...'`. `sg` **replaces** rather than extends the group set, so
nested `sg lemonade -c 'sg render -c ...'` yields only `render` — the worker
model was therefore copied once into `models/worker/` and only `render` is
needed at run time.

## Backend verification [MEASURED]

```
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = Radeon 8060S Graphics (RADV STRIX_HALO) (radv) | uma: 1 |
             fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 65536 |
             int dot: 1 | matrix cores: KHR_coopmat
```

| | |
|---|---|
| model | Qwen3.6-27B `UD-Q4_K_XL`, 16.67 GiB, 27.32 B params |
| pp512 | **243.90 t/s** |
| tg32 | **12.21 t/s** |

`uma: 1` — the driver itself reports unified memory, confirming the GPU and the
NPU draw on the same physical LPDDR5X.

## Models

| role | path | size |
|---|---|---|
| controller | `models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf` | 1.11 GiB |
| WORKER-REPRESENTATIVE | `models/worker/Qwen3.6-27B-UD-Q4_K_XL.gguf` | 16.68 GiB |
| WORKER-SMALL | **none available locally** | — |

No small dense instruct model exists on this machine; the readable cache holds
only 27–35 B models plus BitNet embedding models (not generative). The 27 B
model loads and benchmarks fast enough to iterate on, so it serves as the
primary worker. Weights are **not** committed.

## Two traps hit, both recorded

**`llama-cli` in this fork ignores `-no-cnv`.** With stdin not a TTY it enters
interactive mode and emits `> ` forever — one run produced an **880 MB** log and
looked exactly like a hang at 99% CPU. Benchmarks use `llama-bench`, and any
`llama-cli` invocation must redirect `< /dev/null`.

**First Vulkan run compiles the whole shader set.** Several minutes at ~99% CPU
with no output before the model even loads, cached afterwards in
`~/.cache/mesa_shader_cache`. Cold-start timings must exclude it.

## Correctness oracle [MEASURED]

ROCm being unavailable, the independent path is **CPU** — the backend this
repository already trusts bit-exactly. Same model, same input, same chunking;
only the backend differs.

| backend | perplexity | s/pass |
|---|---|---|
| GPU, Vulkan `-ngl 99` | **19.6300 +/- 0.68692** | 13.55 |
| CPU, `-ngl 0 -t 16` | **19.4907 +/- 0.67850** | 50.78 |

**0.71% apart, each well inside the other's confidence interval.** That is the
expected size for fp16 versus fp32 accumulation on a Q4 model across two
backends, and it is not evidence of corruption. The GPU is **3.75x** the CPU on
prompt processing at this size.

Perplexity rather than generated text, deliberately: fluent output does not
prove a backend is numerically sound, and this is a number that can be compared.
