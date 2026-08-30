#!/usr/bin/env python3
"""Task 1: where does C-output time actually go, per logical shape?

The aggregate stage_out counter is not sufficient, and in particular it lives
inside the `k_chunks == 1` branch of the evacuation, so it never measured
ffn_down's deep-K partial-accumulation path at all. This separates:

  wait        blocked on the NPU fence
  sync_out    XRT sync of the C buffer from device
  stage_in    activations -> mapped A buffer      (thread 0, serial)
  stage_out   mapped C -> g_acc, k_chunks == 1    (thread 0, serial)
  partacc     int32 partial accumulation, k_chunks > 1 (thread 0, serial)
  partcopy    part -> g_acc, k_chunks > 1         (thread 0, serial)
  epilogue    int32 -> f32 into dst               (summed over all threads)

llama-bench runs one warmup prefill in addition to the -r timed reps, and the
warmup dispatches like a timed rep, so all cumulative counters are divided by
(reps + 1) to give per-prefill figures.
"""
import argparse, csv, os, re, subprocess
from pathlib import Path

BIN   = "refs/BitNet/build-xdna3/bin/llama-bench"
MODEL = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
ART   = os.path.abspath("artifacts/xclbin-tuned")
SHAPE_NAME = {(2560,2560): "attn_q + attn_out", (2560,6912): "ffn_gate + ffn_up",
              (6912,2560): "ffn_down"}

def run(prompt, ub, threads, tiles, reps, csv_path, direct=False, async_on=False):
    env = dict(os.environ, BITNET_XDNA="1", BITNET_XDNA_ARTIFACTS=ART,
               BITNET_XDNA_STATS="1", BITNET_XDNA_SHAPE_CSV=csv_path,
               BITNET_XDNA_DIRECT_OUT="1" if direct else "0",
               BITNET_XDNA_ASYNC="1" if async_on else "0")
    if tiles is not None:
        env["BITNET_XDNA_TILES"] = str(tiles)
    p = subprocess.run([BIN, "-m", MODEL, "-p", str(prompt), "-n", "0", "-t", str(threads),
                        "-ngl", "0", "-r", str(reps), "-ub", str(ub)],
                       capture_output=True, text=True, env=env, timeout=3600)
    o = p.stdout + p.stderr
    m = re.search(rf"pp{prompt} \|\s*([0-9.]+)", o)
    return float(m.group(1)) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=15)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="artifacts/direct-output/output_cost_by_shape.csv")
    a = ap.parse_args()

    configs = [("pp2048 ub2048 auto",     2048, 2048, None),
               ("pp2048 ub2048 all-NPU",  2048, 2048, 2),
               ("pp3968 ub2048 auto",     3968, 2048, None)]
    rows = []
    for label, prompt, ub, tiles in configs:
        tmp = f"/tmp/shape_{prompt}_{tiles}.csv"
        tok = run(prompt, ub, a.threads, tiles, a.reps, tmp)
        n_pref = a.reps + 1                      # + warmup prefill
        print(f"\n== {label}, {a.threads}T: {tok:.1f} tok/s  "
              f"(counters / {n_pref} prefills) ==")
        print(f"  {'shape':<20}{'n':>5}{'wait':>8}{'sync_out':>9}{'stage_in':>9}"
              f"{'stage_out':>10}{'partacc':>9}{'partcopy':>9}{'epi wall':>9}")
        tot = {}
        for r in csv.DictReader(open(tmp)):
            K, N = int(r["K"]), int(r["N"])
            g = lambda k: float(r[k]) / n_pref
            epi_wall = g("epi_thread_ms") / a.threads
            vals = dict(wait=g("wait_ms"), sync_out=g("sync_out_ms"),
                        stage_in=g("stage_in_ms"), stage_out=g("stage_out_ms"),
                        partacc=g("partacc_ms"), partcopy=g("partcopy_ms"),
                        epi_wall=epi_wall)
            for k, v in vals.items():
                tot[k] = tot.get(k, 0.0) + v
            print(f"  {SHAPE_NAME.get((K,N), f'{K}x{N}'):<20}{int(r['dispatches'])//n_pref:>5}"
                  f"{vals['wait']:>8.1f}{vals['sync_out']:>9.1f}{vals['stage_in']:>9.1f}"
                  f"{vals['stage_out']:>10.1f}{vals['partacc']:>9.1f}"
                  f"{vals['partcopy']:>9.1f}{epi_wall:>9.1f}")
            rows.append(dict(config=label, prompt=prompt, ub=ub, threads=a.threads,
                             tiles=("auto" if tiles is None else tiles), tok_s=tok,
                             K=K, N=N, shape=SHAPE_NAME.get((K,N), f"{K}x{N}"),
                             dispatches=int(r["dispatches"])//n_pref,
                             dispatch_ms=round(g("dispatch_ms"),2),
                             wait_ms=round(vals["wait"],2),
                             sync_out_ms=round(vals["sync_out"],2),
                             stage_in_ms=round(vals["stage_in"],2),
                             stage_in_mb=round(g("stage_in_mb"),1),
                             stage_out_ms=round(vals["stage_out"],2),
                             stage_out_mb=round(g("stage_out_mb"),1),
                             partacc_ms=round(vals["partacc"],2),
                             partacc_mb=round(g("partacc_mb"),1),
                             partcopy_ms=round(vals["partcopy"],2),
                             partcopy_mb=round(g("partcopy_mb"),1),
                             epi_wall_ms=round(epi_wall,2),
                             epi_thread_ms=round(g("epi_thread_ms"),1)))
        ser = tot["stage_in"] + tot["stage_out"] + tot["partacc"] + tot["partcopy"]
        wall = prompt / tok * 1000
        print(f"  {'-'*88}")
        print(f"  totals              {'':>5}{tot['wait']:>8.1f}{tot['sync_out']:>9.1f}"
              f"{tot['stage_in']:>9.1f}{tot['stage_out']:>10.1f}{tot['partacc']:>9.1f}"
              f"{tot['partcopy']:>9.1f}{tot['epi_wall']:>9.1f}")
        print(f"  wall/prefill {wall:.0f} ms;  thread-0 serial output work "
              f"{ser:.1f} ms ({ser/wall*100:.1f}%);  "
              f"stage_out alone {tot['stage_out']:.1f} ms ({tot['stage_out']/wall*100:.1f}%);  "
              f"deep-K path {tot['partacc']+tot['partcopy']:.1f} ms "
              f"({(tot['partacc']+tot['partcopy'])/wall*100:.1f}%)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")

if __name__ == "__main__":
    main()
