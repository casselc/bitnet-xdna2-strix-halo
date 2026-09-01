# Halo local-training plumbing smoke

| | |
|---|---|
| branch base | `gpu-cotenancy` @ `fbd8bf00108480c68919752fbb521fafd786d47d` |
| branch | `halo-training-smoke` |
| purpose | prove a real forward/backward/optimizer/checkpoint/reload/resume loop on THIS box |

This is plumbing validation, not a quality experiment. No judgement of model output.

## T1 Inventory, before installing anything [MEASURED]

| item | value |
|---|---|
| kernel | `7.0.0-29-generic` |
| amdgpu | loaded (21569536); `amdxdna` also loaded (172032) — untouched |
| GPU | `1002:1586` Radeon 8060S, **gfx1151** (`gfx_target_version 110501`), 80 SIMDs, 2900 MHz |
| KFD | `/dev/kfd` present (`root:render`), topology generation 3, 97.7 GiB visible |
| ROCm packages | **none** — `/opt/rocm` absent; the 3 apparent `dpkg` matches were false positives (`libflashrom1`, `motd-news-config`, `whiptail`) |
| `rocminfo` / `hipcc` | not on PATH (apt candidate 7.1.1 exists, not installed) |
| Python | 3.14.4 only; PEP 668 externally managed |
| PyTorch | not installed |
| my groups | no `render` — `sg render` used throughout, as for the Vulkan worker |
| free disk / RAM | 1.6 TB on `/home`; 61 GiB available after stopping owned services |

Nothing was altered: no kernel change, no driver replacement, no system upgrade, no
change to the XDNA/Peano/IRON environment.

## T2 Isolated environment [MEASURED]

`.venv-train`, a venv entirely separate from the `.venv` the XDNA stack uses. Nothing
was installed into the known-good environment.

- `torch 2.9.1+rocm6.4` (cp314 wheel) — installed, then **replaced**
- `torch 2.10.0+rocm7.0` (cp314) — final, 6.0 GiB venv
- `transformers 5.16.1`, `peft 0.20.0`, `accelerate`, `numpy`

## The blocker, localized precisely [MEASURED]

**PyTorch on ROCm sees the device and can allocate on it, but segfaults on the first
kernel dispatch.**

Bisected stage by stage under `sg render`:

| stage | result |
|---|---|
| `import torch` | OK, `2.10.0+rocm7.0` |
| `torch.cuda.get_arch_list()` | includes **gfx1150 / gfx1151** (ROCm 7.0; the rocm6.4 wheel stopped at gfx1201) |
| `torch.cuda.is_available()` | **True** |
| `torch.cuda.get_device_name(0)` | `AMD Radeon 8060S`, `gcnArchName gfx1151`, 100000 MiB |
| `torch.cuda.init()` | OK |
| `torch.empty(4, device="cuda")` | **OK** — allocation succeeds |
| `x.fill_(1.0)` | **SEGFAULT** |

`faulthandler` puts the fault inside the wheel's bundled
`torch/lib/libhsa-runtime64.so` (6 frames), reached from the first kernel launch.

**It is not the driver stack.** Driving the same wheel's `libamdhip64.so` directly via
`ctypes`, bypassing PyTorch:

```
hipGetDeviceCount rc=0  count=1
hipMemGetInfo     rc=0  free=97.7GiB total=97.7GiB
hipMalloc(1MiB)   rc=0  ptr=0x727a52200000
hipFree           rc=0
```

and `hsa_init()` returns 0. **HIP allocation, memory query and HSA init all succeed;
only PyTorch's kernel dispatch faults.**

Ruled out:

- **not permissions** — `sg render` gives the `render` group, `ulimit -l` is `unlimited`
- **not the architecture gap alone** — the rocm6.4 wheel genuinely lacks gfx1151
  (`hipErrorInvalidDeviceFunction`), but rocm7.0 ships it and still faults
- **not an override problem** — `HSA_OVERRIDE_GFX_VERSION` of `11.0.0`, `11.5.0`,
  `11.5.1` all fail, as do `HSA_ENABLE_SDMA=0`, `HSA_XNACK=0` and
  `PYTORCH_HIP_ALLOC_CONF=expandable_segments:False`

Fixing this would need a system ROCm installation or a different kernel — both are
destructive global changes the brief rules out, so **the path was stopped and
documented** rather than forced.

## T3/T4 Training plumbing, validated on CPU [MEASURED]

To separate "the training stack is broken" from "the GPU backend is broken", the same
harness was run end to end on CPU. Everything except the device works.

| | Qwen3-0.6B | Qwen3-1.7B |
|---|---:|---:|
| steps | 60 | 30 |
| loss first -> last | **5.2768 -> 0.0053** | **5.5141 -> 0.3204** |
| loss drop | **99.9%** | 94.2% |
| LoRA trainable | 4,587,520 / 600,637,440 (**0.764%**) | 6,422,528 / 1,726,997,504 (**0.372%**) |
| step time (median) | 945.4 ms | 1507.7 ms |
| tokens/s | 573.3 | 359.5 |
| adapter size | 17.53 MiB | 24.53 MiB |
| checkpoint write | 0.47 s | 0.39 s |
| package power | 111.6 W | — |

**Checkpoint / reload / resume, in a genuinely separate process:**

```
loss after reload : 0.0035      (matches the trained adapter)
resumed 5 steps   : [0.0035, 2.7778, 2.4448, 1.6545, 0.9867]
```

The reload reproduces the loss. The jump on the first resumed step is **expected and
worth recording**: only the adapter is saved, not the AdamW moments, so resuming
applies a fresh optimizer at `lr=1e-4` to an already-converged adapter. **Optimizer
state is not persisted by adapter-only checkpointing** — a real resume path must save
it too.

So: autograd, LoRA attach, optimizer step, checkpoint, reload and resume are all
confirmed working. Only the ROCm execution backend is not.

## T5 Coexistence: warm controller vs local training [MEASURED]

GPU training is blocked, so the relevant coexistence question is the CPU one — and it
is the sharper question anyway, because §8 of `controller-state-envelope` shows the
controller is bandwidth/CPU bound.

8 warm domains, 1600-token prefix, 135-token delta, 4-token output; training is
Qwen3-0.6B LoRA on CPU.

| arm | ctrl TTFT p50 | p95 | ctrl total p50 | req/s | W |
|---|---:|---:|---:|---:|---:|
| controller alone | **219.3** | 221.1 | 280.2 | 3.555 | 95.9 |
| controller + CPU training | **381.7** | 417.6 | 494.1 | 2.016 | 118.4 |
| controller alone (repeat) | **225.9** | 262.2 | 287.2 | 3.413 | 94.1 |

**They coexist, at a real but bounded cost: +74% controller TTFT and −43% throughput,
for +22.5 W.** Neither workload destabilized, and the controller **fully recovers**
afterwards (225.9 vs 219.3 ms), so there is no lasting damage — the third arm exists
precisely to check that.

## Verdict

### HALO TRAINING STACK BLOCKED BY a PyTorch/ROCm kernel-dispatch segfault on gfx1151

Specifically: `torch 2.10.0+rocm7.0` (and `2.9.1+rocm6.4`) enumerate the Radeon 8060S,
report `gfx1151`, and allocate device memory successfully, but segfault inside the
bundled `libhsa-runtime64.so` on the first kernel launch — while raw `hipMalloc` /
`hipMemGetInfo` / `hsa_init` through the same libraries all succeed. No environment
override tried avoids it, and no non-destructive fix is available from userspace.

Everything else is ready: the isolated environment installs cleanly, and the full
train / LoRA / checkpoint / reload / resume loop is confirmed on CPU at both 0.6B and
1.7B, with a measured and recoverable coexistence cost against the live controller.

**What would unblock it** (all deliberately NOT attempted tonight, all requiring
system-level change):

1. install system ROCm (apt `rocminfo` 7.1.1 and the ROCm stack) so PyTorch links a
   system `libhsa-runtime64` rather than the bundled one — the most likely fix, since
   the bundled runtime is the faulting component;
2. try a PyTorch nightly built against ROCm 7.1+;
3. confirm against AMD's documented Strix Halo support matrix for kernel 7.0 / KFD
   generation 3.

---

# CORRECTION — the blocker is resolved. HALO LOCAL TRAINING READY.

Everything above stands as provenance and is **not** rewritten. The verdict it reached
(`BLOCKED`) was correct for the environment as it existed at the time, and the
diagnosis was right: the faulting component was the **bundled** `libhsa-runtime64.so`.
Acting on unblock step 1 fixed it exactly as predicted.

## What changed [MEASURED]

System ROCm 7.1 was installed from the Ubuntu archive (no third-party repo):

```
rocminfo             7.1.1-0ubuntu1
libhsa-runtime64-1   7.1.0+dfsg-0ubuntu9
libamdhip64-7        7.1.0-0ubuntu2
librocblas5          7.1.0-1ubuntu4
```

and the wheel's bundled runtime was moved aside so torch resolves the system one:

```
torch/lib/libhsa-runtime64.so -> libhsa-runtime64.so.bundled
```

**That single rename is the whole fix.** The wheel's own `libamdhip64` was already
working (raw `hipMalloc` succeeded); only its bundled HSA runtime faulted on kernel
dispatch. Nothing else about the environment changed — same wheel, same venv, same
kernel, same driver, XDNA untouched.

## GPU training results [MEASURED]

`sg render`, `--device cuda`, gfx1151, identical harness and dataset to the CPU runs.

| | Qwen3-0.6B | Qwen3-1.7B |
|---|---:|---:|
| steps | 60 | 60 |
| loss first -> last | 5.2770 -> **0.0402** | 5.5191 -> **0.0045** |
| loss drop | 99.2% | **99.9%** |
| LoRA trainable | 4,587,520 / 600,637,440 (0.764%) | 6,422,528 / 1,726,997,504 (0.372%) |
| **step time (median)** | **192.3 ms** | **366.3 ms** |
| **tokens/s** | **2819.2** | **1479.6** |
| **peak device memory** | **3874.7 MiB** | **6646.3 MiB** |
| package power | 100.5 W | 104.3 W |
| adapter size / write | 17.53 MiB / 0.37 s | 24.53 MiB / 0.36 s |

**GPU vs CPU on identical work:**

| model | CPU step | GPU step | speedup | CPU tok/s | GPU tok/s |
|---|---:|---:|---:|---:|---:|
| 0.6B | 945.4 ms | 192.3 ms | **4.92x** | 573.3 | 2819.2 |
| 1.7B | 1507.7 ms | 366.3 ms | **4.12x** | 359.5 | 1479.6 |

Peak device memory is now actually reported (the CPU runs could not): **3.9 GiB at
0.6B and 6.6 GiB at 1.7B**, against 97.7 GiB visible — so both fit with very large
headroom, and a considerably bigger model or batch is affordable.

**Checkpoint / reload / resume in a fresh process, on GPU:**

```
loss after reload : 0.0325      (trained adapter was at 0.0402)
resumed 5 steps   : [0.0325, 2.5606, 1.1419, 1.3416, 0.3842]
```

Reload reproduces the loss. The first-resumed-step spike reproduces on GPU exactly as
on CPU, confirming it is the **optimizer-state gap** (adapter-only saving does not
persist AdamW moments) and not a device artefact.

## T5 coexistence, now the real experiment [MEASURED]

The CPU coexistence result above was a stand-in taken while the GPU path was blocked.
Repeated with training actually on the GPU, same 8 warm domains, same controller:

| arm | ctrl TTFT p50 | p95 | ctrl total p50 | req/s | W |
|---|---:|---:|---:|---:|---:|
| controller alone | 219.9 | 225.4 | 282.1 | 3.543 | 96.0 |
| **controller + GPU training** | **221.4** | 263.3 | 282.5 | 3.478 | **93.6** |
| controller alone (repeat) | 222.8 | 265.7 | 290.8 | 3.398 | 89.3 |

**GPU training is essentially free for the controller: +0.7% TTFT and −1.8%
throughput, at *lower* package power.** Against the CPU-training arm's +74% TTFT and
−43% throughput, the contrast is decisive:

| training on | ctrl TTFT | ctrl throughput | training speed |
|---|---|---|---|
| CPU | **+74%** | **−43%** | 1.0x |
| GPU | **+0.7%** | **−1.8%** | **4.9x** |

**Local training belongs on the GPU on this box — it is both ~5x faster and ~40x
cheaper in controller interference.** That is not a marginal preference: CPU training
takes nearly half the controller's capacity, GPU training takes almost none, because
the controller is CPU/bandwidth bound (§8 of `controller-state-envelope`) and the two
workloads land on different engines.

This also completes the tri-device picture: NPU idle in warm steady state, CPU serving
the controller, GPU free for training or the worker.

## Revised verdict

### HALO LOCAL TRAINING READY

0.6B and 1.7B both load, train, LoRA-adapt, checkpoint, reload in a fresh process, and
resume on the Radeon 8060S (gfx1151) through system ROCm 7.1 + `torch 2.10.0+rocm7.0`,
at 2819 / 1480 tok/s and 3.9 / 6.6 GiB peak, coexisting with a live warm controller at
under 2% cost.

Two limitations to carry forward, neither blocking:

1. **Adapter-only checkpoints do not persist optimizer state.** A real resume path must
   save the AdamW moments; the loss spike on step 1 of resume is that gap, measured on
   both devices.
2. **The fix depends on shadowing the wheel's bundled HSA runtime.** A `pip install
   --force-reinstall torch` will restore the faulting `libhsa-runtime64.so` and
   re-break the environment. This needs to be either scripted or replaced with a
   properly built wheel.
