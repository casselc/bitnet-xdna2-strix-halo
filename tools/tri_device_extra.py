#!/usr/bin/env python3
"""Two arms the main matrix does not cover.

B11 -- separating memory CAPACITY cost from ACTIVITY cost. The NPU runtime keeps
~2.0 GiB of expanded int8 weights resident in the same LPDDR5X the GPU uses. A
DECODE-ONLY controller run loads the model and uploads those weights but issues
ZERO NPU dispatches (decode takes the CPU GEMV path), so:

    A   GPU alone, no NPU process        -> no NPU footprint at all
    F   GPU + controller decoding        -> weights resident, NPU idle
    E   GPU + controller prefilling      -> weights resident, NPU active

A->F is the cost of the footprint. F->E is the cost of the work.

B7 -- a CPU harness tenant shaped like the eventual control plane (structured
Clojure/SCI evaluation with invariant checking) rather than a synthetic loop.
"""
import argparse, csv, json, os, statistics as st, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tri_device import (Child, run_arm, controller_cmd, worker_cmd, REPO,
                        CTRL_BIN, CTRL_MODEL)

TENANT = REPO / "tools" / "cpu_tenant.clj"


def decode_ctrl(threads, n=128):
    """Model loaded, weights uploaded, zero NPU dispatches."""
    return Child("controller",
                 [str(CTRL_BIN), "-m", str(CTRL_MODEL), "-p", "0", "-n", str(n),
                  "-t", str(threads), "-ngl", "0", "-ub", "2048", "-r", "2"],
                 env={"BITNET_XDNA": "1", "BITNET_XDNA_STATS": "1"})


def tenant(secs):
    return Child("tenant", ["bb", str(TENANT), "secs", str(secs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tenant-secs", type=int, default=25)
    ap.add_argument("--out", default="artifacts/gpu-cotenancy/tri_device_extra.csv")
    a = ap.parse_args()

    def work():
        return Child("worker", worker_cmd(512, 64, 1), group="render")

    def prefill_ctrl(t):
        return Child("controller", controller_cmd(t, 2048, 3),
                     env={"BITNET_XDNA": "1", "BITNET_XDNA_STATS": "1"})

    arms = [
        ("tenant alone",            lambda: [tenant(a.tenant_secs)]),
        ("F  gpu + ctrl DECODE",    lambda: [work(), decode_ctrl(a.threads)]),
        (f"G  gpu + ctrl t{a.threads} + tenant",
         lambda: [work(), prefill_ctrl(a.threads), tenant(a.tenant_secs)]),
        ("H  ctrl + tenant (no gpu)",
         lambda: [prefill_ctrl(a.threads), tenant(a.tenant_secs)]),
    ]

    rows = []
    print(f"extra arms: {a.rounds} rounds, controller t{a.threads}\n")
    for r in range(a.rounds):
        for label, mk in arms:
            rec = run_arm(label, mk())
            rec["round"] = r
            # the tenant prints one JSON line on stdout
            for c in mk():
                pass
            rows.append(rec)
            tn = rec.get("tenant") or {}
            if tn:
                print(f"       tenant {tn.get('ops_per_s',0):8.1f} ops/s  "
                      f"p50 {tn.get('p50_ms',0):.3f} ms  p95 {tn.get('p95_ms',0):.3f} ms")
            c = rec.get("controller") or {}
            w = rec.get("worker") or {}
            cs = f"{c['pp2048'][0]:7.1f}" if isinstance(c, dict) and "pp2048" in c else \
                 (f"tg{c['tg128'][0]:6.2f}" if isinstance(c, dict) and "tg128" in c else "      -")
            wp = f"{w['pp512'][0]:7.1f}" if isinstance(w, dict) and "pp512" in w else "      -"
            wt = f"{w['tg64'][0]:6.2f}" if isinstance(w, dict) and "tg64" in w else "     -"
            print(f"  r{r} {label:30s} ctrl {cs}  gpu_pp {wp}  gpu_tg {wt}  "
                  f"{rec['wall_s']:6.1f}s  {rec['watts'] or 0:5.1f}W", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    flat = []
    for r in rows:
        d = {k: v for k, v in r.items()
             if k not in ("controller", "worker", "tenant", "pids")}
        for who in ("controller", "worker"):
            v = r.get(who)
            if isinstance(v, dict):
                for test, (ts, sd) in v.items():
                    d[f"{who}_{test}"] = ts
        tn = r.get("tenant")
        if isinstance(tn, dict):
            for k2 in ("ops", "ops_per_s", "p50_ms", "p95_ms"):
                d[f"tenant_{k2}"] = tn.get(k2)
        flat.append(d)
    keys, seen = [], set()
    for d in flat:
        for k in d:
            if k not in seen:
                seen.add(k); keys.append(k)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for d in flat: w.writerow({k: d.get(k, "") for k in keys})
    Path(a.out).with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
