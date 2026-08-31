#!/usr/bin/env python3
"""Measure how many warm state domains the controller actually holds.

Residency is NOT bounded by slot count. `llama-server` keeps a second tier,
`server_prompt_cache` (server-task.cpp), enabled by default at --cache-ram 8192
MiB: `prompt_save` parks a displaced slot's sequence state there and
`prompt_load` moves it back out (states.erase + shrink_to_fit), so the cache
holds roughly (domains visited - slots) entries and process RSS stays flat.
Eviction is plain LRU via states.pop_front().

The probe order matters. Walking domains in the order they were warmed measures
0% warm at any working set larger than the cache -- that is LRU thrash, not
capacity: revisiting an evicted domain rebuilds it and pop_fronts the entry you
were about to visit next. Probing NEWEST -> OLDEST and stopping at the first run
of misses measures the resident set with almost no cascade.

Reproduces the numbers in artifacts/controller-state-scheduler/RESULTS.md S10.
"""
import argparse, statistics as st, sys
sys.path.insert(0, "tools")
from service_bench import run_controller, write_rows
from prefix_bench import STABLE

# KV bytes per token, from model geometry: n_kv_heads x head_dim x n_layers
# x 2 (K and V) x 2 bytes (f16). BitNet-b1.58-2B-4T: 5 x 128 x 30 x 2 x 2.
KV_KIB_PER_TOKEN = 5 * 128 * 30 * 2 * 2 / 1024


def domain(d, n_topo=50, n_state=30, query="ACTION:"):
    """A distinct state domain: shared STABLE prefix, then domain-specific
    topology and canonical state, then a query suffix."""
    return (STABLE + f"\nDOMAIN-{d}-ROOT\n"
            + "".join(f"- d{d}.svc{i}: region=r{(i+d)%5} "
                      f"budget={80+((i*31)+d*997)%400}ms\n" for i in range(n_topo))
            + f"\nCANONICAL STATE d{d}\n"
            + "".join(f"- d{d}.svc{i}: p95={40+((i*13)+d*13)%500}ms "
                      f"err={(i+d)%9}\n" for i in range(n_state))
            + f"\nQUERY d{d}\n{query}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", type=int, default=128,
                    help="distinct domains to warm; must exceed capacity")
    ap.add_argument("--base", type=int, default=60000, help="domain id base")
    ap.add_argument("--cache-ram-mib", type=int, default=8192,
                    help="the server's --cache-ram, for the implied-size check")
    ap.add_argument("--stop-after-misses", type=int, default=3)
    ap.add_argument("--out", default="artifacts/controller-state-scheduler/residency_knee.csv")
    a = ap.parse_args()

    for i in range(a.domains):
        run_controller("w", 8, 1, 1, domain(a.base + i), cache=True)
        if (i + 1) % 32 == 0:
            print(f"  warmed {i+1}", flush=True)

    rows, miss = [], 0
    for i in range(a.domains - 1, -1, -1):
        r = run_controller("r", 8, 1, 2, domain(a.base + i), cache=True).row()
        warm = (r.get("eval_n") or 0) <= 8
        rows.append(dict(age_from_newest=a.domains - 1 - i, idx=i,
                         eval_n=r.get("eval_n"), reused_n=r.get("reused_n"),
                         ttft_ms=r.get("ttft_ms"), warm=int(warm)))
        miss = 0 if warm else miss + 1
        if miss >= a.stop_after_misses:
            break

    write_rows(a.out, rows)
    warm_n = sum(x["warm"] for x in rows)
    print(f"\n  probed newest->oldest: {len(rows)} domains, {warm_n} warm "
          f"before {a.stop_after_misses} consecutive misses")
    print("  " + "".join("." if x["warm"] else "X" for x in rows))
    wt = [x["ttft_ms"] for x in rows if x["warm"]]
    ct = [x["ttft_ms"] for x in rows if not x["warm"]]
    if wt:
        print(f"  warm TTFT median {st.median(wt):.0f} ms ({len(wt)} samples)")
    if ct:
        print(f"  cold TTFT median {st.median(ct):.0f} ms ({len(ct)} samples)")
    if not warm_n:
        print("  nothing warm: capacity is below the probe, or the cache is disabled")
        return
    tok = rows[0]["eval_n"] + rows[0]["reused_n"]
    implied = a.cache_ram_mib / warm_n
    print(f"\n  resident capacity ~= {warm_n} domains of ~{tok} tokens")
    print(f"  implied per-state size = {a.cache_ram_mib} MiB / {warm_n} = {implied:.0f} MiB"
          f" = {implied*1024/max(tok,1):.1f} KiB/token")
    print(f"  vs {KV_KIB_PER_TOKEN:.1f} KiB/token derived from model geometry"
          f"  ({abs(implied*1024/max(tok,1) - KV_KIB_PER_TOKEN)/KV_KIB_PER_TOKEN*100:.1f}% apart)")


if __name__ == "__main__":
    main()
