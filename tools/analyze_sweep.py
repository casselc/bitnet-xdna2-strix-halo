#!/usr/bin/env python3
"""Summarize the scheduling sweep.

Reports medians (not means) because this machine carries background load that
produces occasional slow outliers in one direction only. Also reports the spread,
because a configuration that is slightly slower but far more predictable is the
better choice for a latency-sensitive controller.
"""
import csv, statistics as st, sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1
                                else "artifacts/next-pass/sweep.csv")))
for r in rows:
    for k in ("tok_s", "bg_load", "device_ms", "wait_ms", "sync_in_ms", "sync_out_ms",
              "dispatches", "resident_mib"):
        r[k] = float(r[k])
    for k in ("prompt", "ub", "threads", "rep"):
        r[k] = int(r[k])

g = defaultdict(list)
for r in rows:
    g[(r["prompt"], r["ub"], r["threads"], r["tiles"])].append(r)

print(f"{len(rows)} runs, {len(g)} configs\n")

# Best config per (prompt, threads), and the CPU-only reference it must beat.
print("=== best NPU share per (prompt, ub, threads), vs CPU-only ===")
print(f"{'prompt':>6} {'ub':>5} {'thr':>4} {'best':>10} {'tok/s':>8} {'cpu-only':>9} "
      f"{'gain':>6} {'spread':>7}")
by_pt = defaultdict(dict)
for (p, ub, th, tiles), rs in g.items():
    by_pt[(p, ub, th)][tiles] = st.median([r["tok_s"] for r in rs])
wins = []
for (p, ub, th), shares in sorted(by_pt.items()):
    cpu = shares.get("cpu")
    npu = {k: v for k, v in shares.items() if k != "cpu"}
    if not cpu or not npu:
        continue
    best_k = max(npu, key=npu.get)
    spread = ""
    rs = g[(p, ub, th, best_k)]
    if len(rs) > 1:
        vals = [r["tok_s"] for r in rs]
        spread = f"{100*(max(vals)-min(vals))/st.median(vals):.0f}%"
    gain = npu[best_k] / cpu
    wins.append((gain, p, ub, th, best_k, npu[best_k], cpu))
    print(f"{p:>6} {ub:>5} {th:>4} {('tiles='+str(best_k)):>10} {npu[best_k]:>8.1f} "
          f"{cpu:>9.1f} {gain:>5.2f}x {spread:>7}")

print("\n=== top 5 configurations by absolute prefill throughput ===")
allc = sorted(((st.median([r['tok_s'] for r in rs]), k) for k, rs in g.items()),
              reverse=True)[:5]
for v, (p, ub, th, tiles) in allc:
    print(f"  {v:8.1f} tok/s   prompt={p} ub={ub} threads={th} tiles={tiles}")

print("\n=== does -ub matter? (median tok/s at best share, threads=15) ===")
for p in sorted({r['prompt'] for r in rows}):
    line = []
    for ub in sorted({r['ub'] for r in rows if r['prompt'] == p}):
        shares = by_pt.get((p, ub, 15), {})
        npu = {k: v for k, v in shares.items() if k != 'cpu'}
        if npu:
            line.append(f"ub{ub}={max(npu.values()):.0f}")
    if line:
        print(f"  prompt {p:>5}: " + "  ".join(line))

print("\n=== stability: hybrid vs CPU-only run-to-run spread ===")
for label, pred in (("CPU-only", lambda k: k[3] == "cpu"),
                    ("hybrid",   lambda k: k[3] not in ("cpu", "0"))):
    sp = []
    for k, rs in g.items():
        if pred(k) and len(rs) > 1:
            v = [r["tok_s"] for r in rs]
            m = st.median(v)
            if m: sp.append(100 * (max(v) - min(v)) / m)
    if sp:
        print(f"  {label:<9} median spread {st.median(sp):.1f}%  "
              f"(n={len(sp)} configs, worst {max(sp):.0f}%)")
