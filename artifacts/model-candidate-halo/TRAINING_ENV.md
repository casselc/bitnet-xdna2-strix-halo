# Reproducible PyTorch/ROCm invocation on gfx1151 [MEASURED]

The training venv on this box worked for a reason that no longer survives a
reinstall: the wheel's bundled `torch/lib/libhsa-runtime64.so` had been renamed
aside by hand, so torch resolved the system ROCm 7.1 runtime instead. Any
`pip install --force-reinstall torch`, any rebuilt venv, any dependency
resolution that happens to reinstall torch silently restores the bad file, and
the resulting failure does not look like a library-resolution problem.

## The failure, reproduced from scratch

A **fresh** venv with `torch==2.10.0+rocm7.0` (i.e. exactly what a reinstall
produces) fails as follows:

| step | bundled HSA (fresh venv) | system HSA (LD_PRELOAD) |
|---|---|---|
| import torch | ok | ok |
| device enumeration | ok — reports `AMD Radeon 8060S` | ok — reports `Radeon 8060S Graphics` |
| allocation | ok | ok |
| **first kernel launch** | **SIGSEGV (exit 139)** | ok |
| matmul | — | ok |
| backward | — | ok |
| optimizer step | — | ok |

Enumeration and allocation both succeeding is what makes this expensive to
diagnose: everything reports a healthy GPU right up to the first real dispatch.
The device *name string* is a reliable fingerprint of which runtime is live —
`AMD Radeon 8060S` is the bundled runtime, `Radeon 8060S Graphics` is the system
one.

## Why LD_LIBRARY_PATH cannot fix it — measured, not assumed

torch's HIP libraries carry **`DT_RPATH`** (`$ORIGIN`), not `DT_RUNPATH`:

```
$ readelf -d torch/lib/libtorch_hip.so | grep -E 'RPATH|RUNPATH'
 0x000000000000000f (RPATH)   Library rpath: [$ORIGIN]
```

`DT_RPATH` is searched **before** `LD_LIBRARY_PATH`, so the bundled copy wins
regardless. Tested directly: with `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu`
the device still reports as `AMD Radeon 8060S` and still dies at the first
kernel. Option 2 of the three candidate fixes is therefore **ruled out by
measurement**, not skipped.

## The fix: LD_PRELOAD (least invasive of the three that were tried)

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1
```

This passes the full acceptance test on an untouched, freshly installed wheel.
It requires no write to the venv, so it survives reinstalls rather than being
undone by them.

### Honest limitation

LD_PRELOAD does **not** prevent the bundled library from being mapped. HIP's
`DT_NEEDED` is the unversioned `libhsa-runtime64.so`, while *both* files carry
`SONAME libhsa-runtime64.so.1`, so the preloaded object does not satisfy the
dependency by name. Both end up in the process image:

```
LOADED HSA: ['.../torch/lib/libhsa-runtime64.so',
             '/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1.18.0']
```

The preloaded one wins symbol resolution, which is sufficient — verified through
a real forward/backward/optimizer step, not just an import. But two runtimes are
mapped, so if a future wheel misbehaves under it, the `shadow` fallback below
guarantees a single mapping.

## `tools/halo_rocm_env.sh`

```bash
tools/halo_rocm_env.sh check            # verify; exit 1 if unusable
tools/halo_rocm_env.sh exec CMD [ARGS]  # verify, then run with the fix applied
tools/halo_rocm_env.sh shadow VENV      # fallback: rename the bundled file aside
eval "$(tools/halo_rocm_env.sh export)" # emit env for an interactive shell
```

`check` verifies the system HSA runtime and its soversion, records the ROCm
package coordinates, verifies `/dev/kfd` is reachable **from this session**,
verifies `gfx1151`, and detects the known-bad bundled file. Against the
deliberately broken probe venv it reports:

```
  system HSA     /usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1.18.0
  libhsa pkg     7.1.0+dfsg-0ubuntu9
  libamdhip pkg  7.1.0-0ubuntu2
  /dev/kfd       PRESENT BUT NOT ACCESSIBLE (need the owning group in THIS session)
  bundled HSA    PRESENT -- known-bad on this box; LD_PRELOAD required
```

`exec` additionally re-execs once under the group that owns `/dev/kfd` when the
current session lacks it. That is a second silent-failure mode on this box: a
login session predating the `render` membership cannot open `/dev/kfd`, and
RADV/Vulkan quietly falls back to a CPU rasteriser rather than erroring, so a
"GPU" measurement silently becomes a CPU measurement.

## Acceptance test

`import -> enumerate -> allocate -> first kernel -> matmul -> backward ->
optimizer step`, plus a readback of `/proc/self/maps` to record which HSA
runtime actually loaded. Against a **fresh** venv:

```
$ tools/halo_rocm_env.sh exec .venv-probe/bin/python hsa_accept.py
STEP 0 import ... STEP 6 optimizer step
  ok delta 0.4853633940219879
ALL STEPS PASS
```

The same script without the wrapper exits 139 at STEP 3. That difference is the
whole test.

## What was NOT done

No further change to system ROCm, no kernel or driver change, no second ROCm
installation. The fix is confined to how a process is launched.
