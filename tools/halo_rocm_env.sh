#!/usr/bin/env bash
# Reproducible PyTorch-on-ROCm invocation for this box (gfx1151 / Strix Halo).
#
# THE PROBLEM
#
# The pytorch ROCm wheel ships its own HSA runtime at
# torch/lib/libhsa-runtime64.so. On this machine that bundled runtime
# enumerates the GPU and allocates memory successfully and then SIGSEGVs on the
# first kernel launch. The system ROCm 7.1 runtime works. Anything that
# reinstalls the wheel -- `pip install --force-reinstall torch`, a rebuilt venv,
# a `pip install` that happens to upgrade torch -- silently restores the bad
# file, and the failure looks like a GPU/driver problem rather than a library
# resolution problem.
#
# WHY LD_PRELOAD AND NOT LD_LIBRARY_PATH
#
# torch's HIP libraries carry DT_RPATH ($ORIGIN), not DT_RUNPATH. DT_RPATH is
# searched BEFORE LD_LIBRARY_PATH, so LD_LIBRARY_PATH cannot displace the
# bundled copy -- measured, not assumed: with LD_LIBRARY_PATH set the device
# still reports as "AMD Radeon 8060S" (the bundled runtime's string) and still
# dies at the first kernel. With LD_PRELOAD it reports "Radeon 8060S Graphics"
# and the full forward/backward/optimizer path passes.
#
# HONEST LIMITATION
#
# LD_PRELOAD does not prevent the bundled library from being mapped -- HIP's
# DT_NEEDED is the unversioned "libhsa-runtime64.so" while both files carry
# SONAME "libhsa-runtime64.so.1", so the preloaded object does not satisfy the
# dependency by name. Both end up in the process image and the preloaded one
# wins symbol resolution. That is sufficient (verified through a real LoRA
# training step), but if a future wheel breaks under it, `shadow` below is the
# heavier fallback that guarantees only one runtime is mapped.
#
# USAGE
#
#   tools/halo_rocm_env.sh check            # verify environment, exit 1 if unusable
#   tools/halo_rocm_env.sh exec CMD [ARGS]  # verify, then run CMD with the fix
#   tools/halo_rocm_env.sh shadow VENV      # fallback: rename the bundled file aside
#   eval "$(tools/halo_rocm_env.sh export)" # emit the env for an interactive shell
#
# `exec` is the supported entry point. It re-execs under the `render` group when
# the current session lacks it, which is its own silent-failure mode on this box:
# a login session predating the render-group membership cannot open /dev/kfd,
# and RADV/Vulkan quietly falls back to a CPU rasteriser instead of erroring.

set -uo pipefail

EXPECT_GFX="${HALO_EXPECT_GFX:-gfx1151}"
EXPECT_HSA_MAJOR="${HALO_EXPECT_HSA_MAJOR:-1}"
SYS_HSA_CANDIDATES=(
    /usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1
    /opt/rocm/lib/libhsa-runtime64.so.1
)

note() { printf '  %-14s %s\n' "$1" "$2"; }
fail() { printf 'halo_rocm_env: FAIL: %s\n' "$1" >&2; exit 1; }

find_sys_hsa() {
    for p in "${SYS_HSA_CANDIDATES[@]}"; do
        [ -e "$p" ] && { echo "$p"; return 0; }
    done
    return 1
}

# The venv whose torch we are about to use. Derived from the python on PATH so
# that `exec` and `check` agree about which install is being validated.
torch_lib_dir() {
    local py="${HALO_PYTHON:-python3}"
    command -v "$py" >/dev/null 2>&1 || return 1
    "$py" - <<'PY' 2>/dev/null
import os, sys
try:
    import torch
except Exception:
    sys.exit(1)
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
}

check_env() {
    local rc=0
    echo "halo_rocm_env: checking"

    # 1. system HSA runtime
    local sys_hsa
    if ! sys_hsa=$(find_sys_hsa); then
        note "system HSA" "MISSING"
        echo "    install the distro ROCm runtime (libhsa-runtime64-1)" >&2
        rc=1
    else
        local real; real=$(readlink -f "$sys_hsa")
        note "system HSA" "$real"
        case "$(basename "$real")" in
            libhsa-runtime64.so."$EXPECT_HSA_MAJOR".*) ;;
            *) note "system HSA" "WARNING: unexpected soversion (want .$EXPECT_HSA_MAJOR.x)";;
        esac
    fi

    # 2. package coordinates, recorded so a later reader can tell whether the
    #    machine drifted rather than the code.
    if command -v dpkg >/dev/null 2>&1; then
        local hsa_pkg hip_pkg
        hsa_pkg=$(dpkg-query -W -f='${Version}' libhsa-runtime64-1 2>/dev/null || echo "?")
        hip_pkg=$(dpkg-query -W -f='${Version}' libamdhip64-7 2>/dev/null || echo "?")
        note "libhsa pkg" "$hsa_pkg"
        note "libamdhip pkg" "$hip_pkg"
    fi

    # 3. device node reachability. Being in the group per /etc/group is not the
    #    same as having it in this session's credentials.
    if [ -e /dev/kfd ]; then
        if [ -r /dev/kfd ] && [ -w /dev/kfd ]; then
            note "/dev/kfd" "readable+writable"
        else
            note "/dev/kfd" "PRESENT BUT NOT ACCESSIBLE (need the owning group in THIS session)"
            note "" "re-run via: $0 exec <cmd>   (it applies sg automatically)"
            rc=1
        fi
    else
        note "/dev/kfd" "MISSING -- amdgpu/kfd not loaded"; rc=1
    fi

    # 4. the actual target architecture
    if command -v rocminfo >/dev/null 2>&1; then
        local gfx
        gfx=$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | sort -u | tr '\n' ' ')
        if [ -n "$gfx" ]; then
            note "rocminfo gfx" "$gfx"
            echo "$gfx" | grep -qw "$EXPECT_GFX" || { note "gfx" "WARNING: $EXPECT_GFX not found"; }
        else
            note "rocminfo gfx" "no gfx reported (device not accessible from this session?)"
            rc=1
        fi
    else
        note "rocminfo" "not installed (skipping arch check)"
    fi

    # 5. THE KNOWN-BAD CASE: a bundled HSA runtime inside the torch wheel.
    local tl
    if tl=$(torch_lib_dir) && [ -n "$tl" ]; then
        note "torch lib" "$tl"
        if [ -e "$tl/libhsa-runtime64.so" ]; then
            note "bundled HSA" "PRESENT -- known-bad on this box; LD_PRELOAD required"
            BUNDLED_PRESENT=1
        else
            note "bundled HSA" "absent (already shadowed)"
            BUNDLED_PRESENT=0
        fi
    else
        note "torch lib" "torch not importable with ${HALO_PYTHON:-python3} (skipping wheel check)"
        BUNDLED_PRESENT=0
    fi

    return $rc
}

emit_export() {
    local sys_hsa; sys_hsa=$(find_sys_hsa) || fail "no system libhsa-runtime64.so.1"
    printf 'export LD_PRELOAD="%s${LD_PRELOAD:+:$LD_PRELOAD}"\n' "$sys_hsa"
}

case "${1:-check}" in
  check)
    check_env || exit 1
    echo "halo_rocm_env: OK"
    ;;

  export)
    emit_export
    ;;

  exec)
    shift
    [ $# -gt 0 ] || fail "exec needs a command"
    sys_hsa=$(find_sys_hsa) || fail "no system libhsa-runtime64.so.1"

    # If this session cannot open /dev/kfd but the owning group would grant it,
    # re-exec once under that group rather than failing with a confusing
    # kernel-launch crash much later.
    if [ -e /dev/kfd ] && { [ ! -r /dev/kfd ] || [ ! -w /dev/kfd ]; }; then
        grp=$(stat -c '%G' /dev/kfd 2>/dev/null || echo "")
        if [ -n "$grp" ] && [ "${HALO_REEXEC:-0}" != "1" ] && id -nG "$(id -un)" >/dev/null 2>&1 \
           && getent group "$grp" 2>/dev/null | grep -qw "$(id -un)"; then
            export HALO_REEXEC=1
            exec sg "$grp" -c "$(printf '%q ' "$0" exec "$@")"
        fi
    fi

    export LD_PRELOAD="$sys_hsa${LD_PRELOAD:+:$LD_PRELOAD}"
    exec "$@"
    ;;

  shadow)
    # Fallback only. Guarantees a single mapped runtime by renaming the wheel's
    # copy aside. Reversible: the .bundled file is kept, never deleted.
    venv="${2:-}"
    [ -n "$venv" ] || fail "shadow needs a venv path"
    found=0
    while IFS= read -r f; do
        found=1
        if [ -e "$f" ]; then
            mv -v "$f" "$f.bundled"
        fi
    done < <(find "$venv" -path '*/torch/lib/libhsa-runtime64.so' 2>/dev/null)
    [ "$found" = 1 ] || echo "shadow: nothing to do (no bundled libhsa-runtime64.so under $venv)"
    ;;

  *)
    echo "usage: $0 {check|export|exec CMD...|shadow VENV}" >&2
    exit 2
    ;;
esac
