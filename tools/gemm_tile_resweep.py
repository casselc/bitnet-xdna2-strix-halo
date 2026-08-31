#!/usr/bin/env python3
"""Re-sweep the production BitNet GEMM's per-core tile, after the L1 budget changed.

npu/sweep_tiling.sh chose the tile our kernels ship with. It filters candidates on

    l1 = m*k + k*n + m*n*4          # "the design double-buffers so budget ~half"
    (( l1 > 32768 )) && continue

Tuning change #2 (artifacts/e2e/tuned_results.md) then set the C ObjectFifo depth
from 2 to 1 -- C is written once per tile and drained, so it never needed double
buffering -- which made the real constraint

    2*m*k + 2*k*n + m*n*4 <= ~62 KB

The sweep was never re-run against that. The tell is that the SHIPPING tile,
128x64x64, is itself rejected by the old filter (45056 > 32768): it was found by
hand-patching the one parameter the change was known to unlock, not by searching
the space the change opened. Two thirds of the legal space has never been
evaluated.

Independent motivation from artifacts/attention-feasibility/geometry_qk_pv_c8.csv:
on identical arithmetic, per-core tile geometry spans 2.50-8.42 TFLOPS on the
same kernel source -- a 3.4x range -- and n is the strongest single lever there.
The old sweep's n list stops at 80; the deployed N-chunk of 2560 also admits
n = 160 and n = 320.

NO RESULT IS PREDICTED. Those geometry numbers are bf16 with row-major B, and
this path is int8 with col-major B, where one effect has already been measured
to REVERSE sign: b_col_maj costs 1.408x in bf16 but was measured at +35% FASTER
for our int8 kernel (tuned_results.md #1). This re-runs the search on the real
dtype and layout; it does not transfer a number.

Every configuration is correctness-gated by the design's own PASS/FAIL check, and
its JIT cache key is recorded so a collision cannot masquerade as a result.
"""
import argparse, csv, hashlib, json, os, re, shutil, statistics as st, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CACHE = Path.home() / ".npu" / "cache"

# Production coordinate: one program serves every BitNet shape by N-chunking the
# wide FFN and K-chunking the deep down-projection, so the tile is tuned for it.
PROD = dict(M=1024, K=2560, N=2560, cols=8, b_col_maj=1,
            dtype_in="i8", dtype_out="i32")
INCUMBENT = (128, 64, 64)
ENV = dict(C_FIFO_DEPTH="1", TB_MAX_N_ROWS="2")   # the shipping build's settings

OLD_SWEEP_M = {32, 64, 128}
OLD_SWEEP_K = {32, 64, 128, 256}
OLD_SWEEP_N = {16, 32, 64, 80}
L1_BUDGET = 62208


def old_filter(m, k, n):
    return m * k + k * n + m * n * 4 <= 32768


def l1_now(m, k, n):
    return 2 * m * k + 2 * k * n + m * n * 4


def reachable_by_old_sweep(m, k, n):
    return (m in OLD_SWEEP_M and k in OLD_SWEEP_K and n in OLD_SWEEP_N
            and old_filter(m, k, n))


def legal(m, k, n, p=PROD, rows=4):
    """Constraints the design itself enforces, plus the mmul micro-tile shape."""
    return (p["N"] % (n * p["cols"]) == 0 and p["K"] % k == 0
            and p["M"] % (m * rows) == 0
            and m % 16 == 0 and k % 8 == 0 and n % 16 == 0)


def candidates():
    out = []
    for m in (16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 256):
        for k in (16, 32, 64, 128, 256, 320, 512):
            for n in (16, 20, 32, 40, 64, 80, 160, 320):
                if not legal(m, k, n):
                    continue
                if l1_now(m, k, n) > L1_BUDGET:
                    continue
                out.append((m, k, n))
    # Ordering matters because each configuration costs ~6 minutes to build and
    # run, so a full 96-point sweep is ~10 hours. The first five points measured
    # (largest n first, on the prior that n was the strongest lever in bf16)
    # settled the ordering question empirically instead:
    #
    #     32x16x320  0.545 TOPS      64x32x160  4.424 TOPS
    #     16x32x320  0.631 TOPS      64x16x160  2.309 TOPS
    #     16x16x320  0.354 TOPS      incumbent 128x64x64: 13.14 TOPS
    #
    # Large n is CATASTROPHIC for int8 with col-major B -- the opposite of the
    # bf16 row-major result that motivated looking. That is the second effect in
    # this investigation to reverse sign between the two paths (b_col_maj was
    # the first), so the bf16 geometry surface is not a usable prior here at all.
    #
    # Order by distance from the incumbent instead, in log space over all three
    # tile dimensions, so the plausible neighbourhood is measured first and the
    # already-refuted extremes last.
    import math
    im, ik, iN = INCUMBENT
    def dist(c):
        return (abs(math.log2(c[0] / im)) + abs(math.log2(c[1] / ik))
                + abs(math.log2(c[2] / iN)))
    return sorted(out, key=dist)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def newest_cache_entry(before):
    now = {d.name for d in CACHE.iterdir()} if CACHE.exists() else set()
    fresh = now - before
    if not fresh:
        return None, None
    d = CACHE / sorted(fresh)[0]
    x = next(iter(sorted(d.glob("*.xclbin"))), None)
    return d.name, (sha256(x) if x else None)


NPU_RE = re.compile(r"NPU time.*?:\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)")
E2E_RE = re.compile(r"End-to-end.*?:\s*([\d.]+)")
GF_RE = re.compile(r"NPU GFLOPS\s*:\s*([\d.]+)")


def run_one(m, k, n, warm, iters, timeout=1200):
    before = {d.name for d in CACHE.iterdir()} if CACHE.exists() else set()
    env = {**os.environ, **ENV}
    cmd = [str(REPO / ".venv" / "bin" / "python"),
           str(REPO / "npu" / "whole_array_tuned.py"),
           "-M", str(PROD["M"]), "-K", str(PROD["K"]), "-N", str(PROD["N"]),
           "-m", str(m), "-k", str(k), "-n", str(n),
           "--n-aie-cols", str(PROD["cols"]),
           "--b-col-maj", str(PROD["b_col_maj"]),
           "--dtype_in", PROD["dtype_in"], "--dtype_out", PROD["dtype_out"],
           "-w", str(warm), "-i", str(iters)]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                           timeout=timeout, env=env)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        out = "TIMEOUT"
    wall = time.time() - t0
    key, xsha = newest_cache_entry(before)

    rec = dict(m=m, k=k, n=n, l1_bytes=l1_now(m, k, n),
               new_to_sweep=int(not reachable_by_old_sweep(m, k, n)),
               wall_s=round(wall, 1), cache_key=key, xclbin_sha256=xsha)
    if "TIMEOUT" in out:
        rec.update(status="TIMEOUT"); return rec
    mm = NPU_RE.search(out)
    gf = GF_RE.search(out)
    if mm and gf and "PASS" in out:
        rec.update(status="PASS", npu_us=float(mm.group(1)),
                   npu_us_min=float(mm.group(2)), npu_us_max=float(mm.group(3)),
                   e2e_us=float(E2E_RE.search(out).group(1)) if E2E_RE.search(out) else None,
                   gops=float(gf.group(1)), tops=round(float(gf.group(1)) / 1000, 3))
    elif mm and gf:
        rec.update(status="NUMERIC_FAIL", npu_us=float(mm.group(1)),
                   gops=float(gf.group(1)), tops=round(float(gf.group(1)) / 1000, 3))
    elif "exceeds the [1:1048576] range" in out:
        rec.update(status="DMA_STRIDE")
    elif "exceeded available memory" in out:
        rec.update(status="L1_OVERFLOW")
    else:
        m2 = re.search(r"error: (.{0,110})", out)
        rec.update(status="BUILD_FAIL", detail=(m2.group(1) if m2 else out[-110:]).strip())
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", default="",
                    help="comma-separated mxkxn already measured, to merge in")
    ap.add_argument("--out", default="artifacts/gemm-tile/resweep.csv")
    a = ap.parse_args()

    cands = candidates()
    done = {tuple(int(v) for v in t.split("x"))
            for t in a.skip.split(",") if t.strip()}
    if done:
        cands = [c for c in cands if c not in done]
    if a.limit:
        cands = cands[:a.limit]
    new_n = sum(1 for c in cands if not reachable_by_old_sweep(*c))
    print(f"production coordinate: M={PROD['M']} K={PROD['K']} N={PROD['N']} "
          f"{PROD['dtype_in']}->{PROD['dtype_out']} cols={PROD['cols']} "
          f"b_col_maj={PROD['b_col_maj']}")
    print(f"build env: {ENV}   incumbent tile: {INCUMBENT}")
    print(f"{len(cands)} candidates fit the CURRENT L1 budget; "
          f"{new_n} were unreachable by the old sweep\n")
    print(f"  {'m':>4}{'k':>5}{'n':>5}{'L1':>8}{'new':>5}{'status':>14}"
          f"{'us':>10}{'TOPS':>8}")

    rows = []
    for (m, k, n) in cands:
        r = run_one(m, k, n, a.warm, a.iters)
        rows.append(r)
        print(f"  {m:>4}{k:>5}{n:>5}{r['l1_bytes']:>8}{r['new_to_sweep']:>5}"
              f"{r['status']:>14}"
              f"{r.get('npu_us', float('nan')):>10.1f}"
              f"{r.get('tops', float('nan')):>8.3f}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    keys, seen = [], set()
    for r in rows:
        for kk in r:
            if kk not in seen:
                seen.add(kk); keys.append(kk)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({kk: r.get(kk, "") for kk in keys})

    ok = [r for r in rows if r["status"] == "PASS"]
    # A repeated xclbin hash across distinct configs would mean the JIT cache
    # served a stale artifact -- the exact failure mode this project has hit.
    shas = [r["xclbin_sha256"] for r in ok if r["xclbin_sha256"]]
    dup = len(shas) - len(set(shas))
    print(f"\n  {len(ok)}/{len(rows)} PASS   "
          f"artifact collisions: {dup} {'(CLEAN)' if dup == 0 else '(SUSPECT!)'}")
    if ok:
        ok.sort(key=lambda r: -r["tops"])
        inc = next((r for r in ok if (r["m"], r["k"], r["n"]) == INCUMBENT), None)
        print(f"\n  top 10 by throughput:")
        for r in ok[:10]:
            tag = "  <-- INCUMBENT" if (r["m"], r["k"], r["n"]) == INCUMBENT else \
                  ("  (new)" if r["new_to_sweep"] else "")
            gain = f"{r['tops']/inc['tops']:.3f}x" if inc else "-"
            print(f"    {r['m']:>4}x{r['k']:<4}x{r['n']:<4} {r['tops']:>7.3f} TOPS"
                  f"  {gain:>8}{tag}")
        if inc:
            print(f"\n  incumbent {INCUMBENT}: {inc['tops']:.3f} TOPS "
                  f"(rank {ok.index(inc)+1} of {len(ok)})")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
