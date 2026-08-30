#!/usr/bin/env python3
"""Pareto view of the whole-system benchmark.

A controller configuration is not just its own latency: the cores it takes come
out of the co-tenant. So the useful output is the frontier -- which (threads,
NPU-share) points are not dominated on BOTH controller latency and co-tenant
throughput.
"""
import csv, statistics as st, sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1
                                else "artifacts/next-pass/whole_system.csv")))
for r in rows:
    for k in ("ttft_ms", "total_ms", "pp_tok_s", "tg_tok_s", "cotenant_it_s"):
        r[k] = float(r[k])
    for k in ("bg_workers", "threads", "rep"):
        r[k] = int(r[k])

g = defaultdict(list)
for r in rows:
    g[(r["bg_workers"], r["threads"], r["tiles"])].append(r)

for bg in sorted({r["bg_workers"] for r in rows}):
    print(f"\n=== co-tenant workers: {bg} ===")
    print(f"  {'thr':>4} {'tiles':>6} {'TTFT ms':>9} {'total ms':>9} {'co-tenant it/s':>15}")
    pts = []
    for (b, th, tiles), rs in sorted(g.items()):
        if b != bg: continue
        ttft = st.median([r["ttft_ms"] for r in rs])
        tot = st.median([r["total_ms"] for r in rs])
        co = st.median([r["cotenant_it_s"] for r in rs])
        pts.append((th, tiles, ttft, tot, co))
        print(f"  {th:>4} {tiles:>6} {ttft:>9.0f} {tot:>9.0f} {co:>15.1f}")

    if bg:
        # Pareto: minimize TTFT, maximize co-tenant throughput.
        front = []
        for p in pts:
            if not any(q[2] <= p[2] and q[4] >= p[4] and q != p and
                       (q[2] < p[2] or q[4] > p[4]) for q in pts):
                front.append(p)
        print(f"\n  PARETO FRONTIER (lower TTFT / higher co-tenant is better):")
        for th, tiles, ttft, tot, co in sorted(front, key=lambda x: x[2]):
            print(f"    threads={th:<3} tiles={tiles:<4} TTFT {ttft:>7.0f} ms   "
                  f"co-tenant {co:>7.1f} it/s")
