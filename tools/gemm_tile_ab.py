#!/usr/bin/env python3
"""Interleaved A/B of candidate GEMM tiles against the shipping one.

The re-sweep measures each configuration once, which is enough to rank them but
not enough to promote one. The incumbent measured 13.14 TOPS standalone and
12.163 TOPS inside the sweep -- an 8% spread on the SAME configuration -- so a
gap of that order has to be re-measured before it means anything.

Configurations are alternated within each round rather than run in blocks,
because this machine drifts 10-30% between runs and a block-ordered comparison
charges that drift to whichever ran last. Every invocation does its own warmup,
so no sample includes a cold context, and each carries the design's own PASS/FAIL
correctness check.
"""
import argparse, csv, json, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemm_tile_resweep import run_one, INCUMBENT, PROD, ENV, l1_now  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", required=True,
                    help="candidate tiles as mxkxn; the incumbent is added")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="artifacts/gemm-tile/tile_ab.csv")
    a = ap.parse_args()

    cfgs = [INCUMBENT] + [tuple(int(v) for v in t.split("x")) for t in a.tiles
                          if tuple(int(v) for v in t.split("x")) != INCUMBENT]
    print(f"interleaved A/B, {a.rounds} rounds x {a.iters} iters, "
          f"M={PROD['M']} K={PROD['K']} N={PROD['N']} {PROD['dtype_in']}->"
          f"{PROD['dtype_out']} cols={PROD['cols']} b_col_maj={PROD['b_col_maj']}")
    print(f"env {ENV}   incumbent {INCUMBENT[0]}x{INCUMBENT[1]}x{INCUMBENT[2]}\n")

    samples = {c: [] for c in cfgs}
    rows = []
    for r in range(a.rounds):
        for c in cfgs:
            rec = run_one(*c, a.warm, a.iters)
            rec["round"] = r
            rows.append(rec)
            if rec["status"] == "PASS":
                samples[c].append(rec["npu_us"])
            print(f"  round {r}  {c[0]:>3}x{c[1]:<3}x{c[2]:<3}  "
                  f"{rec['status']:>12}  {rec.get('npu_us', float('nan')):>9.1f} us"
                  f"  {rec.get('tops', float('nan')):>7.3f} TOPS", flush=True)

    ops = 2 * PROD["M"] * PROD["K"] * PROD["N"]
    print(f"\n  {'tile':>14}{'L1':>8}{'n':>4}{'median us':>11}{'sd':>8}"
          f"{'TOPS':>8}{'vs incumbent':>14}")
    base = st.median(samples[INCUMBENT]) if samples[INCUMBENT] else None
    summary = []
    for c in cfgs:
        s = samples[c]
        if not s:
            print(f"  {f'{c[0]}x{c[1]}x{c[2]}':>14}   no passing samples")
            continue
        med = st.median(s)
        tops = ops / (med / 1e6) / 1e12
        gain = base / med if base else float("nan")
        tag = "  <-- INCUMBENT" if c == INCUMBENT else ""
        print(f"  {f'{c[0]}x{c[1]}x{c[2]}':>14}{l1_now(*c):>8}{len(s):>4}"
              f"{med:>11.1f}{st.pstdev(s):>8.1f}{tops:>8.3f}{gain:>13.3f}x{tag}")
        summary.append(dict(m=c[0], k=c[1], n=c[2], l1_bytes=l1_now(*c),
                            samples=len(s), median_us=round(med, 1),
                            sd_us=round(st.pstdev(s), 1),
                            min_us=round(min(s), 1),
                            tops=round(tops, 3),
                            speedup_vs_incumbent=round(gain, 4)))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in keys})
    Path(a.out).with_suffix(".json").write_text(json.dumps(
        dict(coordinate=PROD, env=ENV, incumbent=list(INCUMBENT),
             rounds=a.rounds, iters=a.iters, summary=summary), indent=2) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
