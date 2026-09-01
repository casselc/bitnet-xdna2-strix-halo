#!/usr/bin/env python3
"""P4 -- candidate-geometry INT8 GEMM feasibility probe.

Bounded on purpose. This asks ONE question: if the controller specialist were a
different size, would its linear layers run better or worse on XDNA2 than the
current BitNet-2B geometry? It does NOT integrate any new model, build a
transformer backend, or design kernels.

It is possible only because the existing tooling is already generic: the stock
IRON `whole_array` example takes -M/-K/-N and tile sizes directly, so arbitrary
INT8 shapes are evaluable with no new backend code. (The RUNTIME's plan_for is
restricted to K,N in {2560, 6912}, but that is a dispatch constraint, not a
build constraint -- a different geometry would need new AOT artifacts, which is
exactly the follow-on work this probe is meant to inform.)

Legality constraints, enforced rather than discovered by failure:
  N % (n * cols) == 0      design requirement
  K % k == 0
  M % m == 0  and  M/(m*rows) even     transfer-block row count
  m*k + k*n + m*n*4 <= 32768           L1 budget (64 KB/core, double buffered)
"""
import argparse, csv, json, re, subprocess, sys, time

ROWS = 4          # compute rows per column on aie2p
COLS = 8
L1_BUDGET = 32768

GEOMETRIES = {
    "current-2B  (2560/6912/2560)":  [("attn_qo", 2560, 2560),
                                      ("ffn_up",  2560, 6912),
                                      ("ffn_down", 6912, 2560)],
    "small       (1024/3072/1024)":  [("attn_qo", 1024, 1024),
                                      ("ffn_up",  1024, 3072),
                                      ("ffn_down", 3072, 1024)],
    "cand-1.7B   (2048/6144/2048)":  [("attn_qo", 2048, 2048),
                                      ("ffn_up",  2048, 6144),
                                      ("ffn_down", 6144, 2048)],
}


def legal(M, K, N, m, k, n, cols=COLS):
    if N % (n * cols):
        return "N % (n*cols)"
    if K % k:
        return "K % k"
    if M % m:
        return "M % m"
    q = M // (m * ROWS)
    if q < 1 or q % 2:
        return "M/(m*rows) not even"
    if m * k + k * n + m * n * 4 > L1_BUDGET:
        return "L1"
    return None


def pick_tile(M, K, N):
    """First legal tile from a small preferred set; None if the shape needs padding."""
    for (m, k, n) in ((64, 64, 64), (64, 64, 48), (64, 64, 32),
                      (32, 64, 64), (64, 32, 64), (32, 64, 32)):
        if legal(M, K, N, m, k, n) is None:
            return (m, k, n)
    return None


def pad_N(N, n=64, cols=COLS):
    """Smallest N' >= N divisible by n*cols -- what the shape would cost padded."""
    step = n * cols
    return ((N + step - 1) // step) * step


def run(M, K, N, m, k, n, timeout=1200):
    t0 = time.time()
    r = subprocess.run([".venv/bin/python", "npu/ref/whole_array.py",
                        "-M", str(M), "-K", str(K), "-N", str(N),
                        "-m", str(m), "-k", str(k), "-n", str(n),
                        "--dtype_in", "i8", "--dtype_out", "i32",
                        "--n-aie-cols", str(COLS)],
                       capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr
    us = re.search(r"NPU time\s+\(avg/min/max us\):\s*([0-9.]+)", out)
    gf = re.search(r"NPU GFLOPS\s*:\s*([0-9.]+)", out)
    ok = "PASS" in out
    return (float(us.group(1)) if us else None,
            float(gf.group(1)) if gf else None,
            ok, round(time.time() - t0, 1), out[-200:] if not ok else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="512,1024,2048,4096")
    ap.add_argument("--out", default="artifacts/controller-state-envelope/shape_probe.csv")
    a = ap.parse_args()
    Ms = [int(x) for x in a.ms.split(",")]
    rows = []
    print(f"{'geometry':<30}{'role':<10}{'K':>6}{'N':>6}{'M':>6}"
          f"{'tile':>14}{'us':>10}{'TOPS':>8}{'padN':>7}  res")
    print("-" * 105)
    for geo, shapes in GEOMETRIES.items():
        for role, K, N in shapes:
            for M in Ms:
                tile = pick_tile(M, K, N)
                padded = pad_N(N)
                if tile is None:
                    rows.append(dict(geometry=geo, role=role, K=K, N=N, M=M,
                                     tile=None, us=None, tops=None,
                                     natural_fits=0, padded_N=padded,
                                     pad_pct=round(100*(padded-N)/N, 2),
                                     result="NO_LEGAL_TILE"))
                    print(f"{geo:<30}{role:<10}{K:>6}{N:>6}{M:>6}"
                          f"{'-':>14}{'-':>10}{'-':>8}{padded:>7}  NO_LEGAL_TILE", flush=True)
                    continue
                m, k, n = tile
                us, gf, ok, wall, err = run(M, K, N, m, k, n)
                tops = round(gf / 1000, 2) if gf else None
                rows.append(dict(geometry=geo, role=role, K=K, N=N, M=M,
                                 tile=f"{m}x{k}x{n}", us=us, tops=tops,
                                 natural_fits=int(n == 64), padded_N=padded,
                                 pad_pct=round(100*(padded-N)/N, 2),
                                 build_run_s=wall,
                                 result="PASS" if ok else "FAIL", err=err))
                print(f"{geo:<30}{role:<10}{K:>6}{N:>6}{M:>6}"
                      f"{f'{m}x{k}x{n}':>14}{(us or 0):>10.1f}{(tops or 0):>8.2f}"
                      f"{padded:>7}  {'PASS' if ok else 'FAIL'}", flush=True)
                with open(a.out, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=sorted({k2 for r in rows for k2 in r}))
                    w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
