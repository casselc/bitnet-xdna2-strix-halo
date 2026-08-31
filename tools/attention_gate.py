#!/usr/bin/env python3
"""The final attention economic gate, computed from measured files only.

Every input is read from an artifact produced by a measurement on this machine;
nothing is typed in. The point is that the verdict can be re-derived by rerunning
this script, and that a changed measurement changes the verdict automatically
rather than leaving a stale hand-copied number in prose.

  cpu_oracle.csv            per-prompt CPU attention time, in situ, 15 threads
  npu_mha_4k.csv            stock d=64 proxy, burdened, per layer
  gemm_mha_switch_*.json    real GEMM<->MHA context alternation

Budget:      budget(T) = CPU_attention(T) - switch_tax(T)
Falsification: an IMPOSSIBLE bound in which moving d=64 -> d=128 halves the
entire kernel. It overstates the benefit, because the proxy is FLOP-equivalent
for QK/PV -- only the softmax row count halves -- so QK and PV cannot shrink at
all. If even this bound loses, no d=128 port is defensible.
"""
import argparse, csv, json, sys
from pathlib import Path


def load_cpu(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[int(r["prompt"])] = float(r["attn_ms"])
    return out


def load_npu(path, layers):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r.get("burdened_ms"):
                continue
            out[int(r["seq_len"])] = dict(
                pad=int(r["seq_pad"]),
                per_layer=float(r["burdened_ms"]),
                prefill=float(r["burdened_ms"]) * layers,
                rel_l2=float(r["rel_l2"]) if r.get("rel_l2") else None,
                bad_frac=float(r["bad_frac"]) if r.get("bad_frac") else None,
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/attention-feasibility")
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--transitions-per-layer", type=int, default=2,
                    help="GEMM->MHA and MHA->GEMM; two transitions is one "
                         "measured alternating pair")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.dir)

    cpu = load_cpu(d / "cpu_oracle.csv")
    npu = load_npu(d / "npu_mha_4k.csv", a.layers)

    sw = {}
    for p in sorted(d.glob("gemm_mha_switch_s*.json")):
        j = json.loads(p.read_text())
        sw[int(j["seq"])] = j

    pairs_per_layer = a.transitions_per_layer / 2.0
    print(f"layers={a.layers}  transitions/layer={a.transitions_per_layer} "
          f"({pairs_per_layer:g} measured pair(s)/layer)\n")

    print(f"{'T':>6}{'CPU attn':>10}{'switch tax':>12}{'budget':>10}"
          f"{'NPU d64':>10}{'0.5x bound':>12}{'+tax':>10}{'verdict':>12}")
    rows = []
    for T in sorted(npu):
        if T not in cpu:
            continue
        # Use the switch measurement taken at this length if one exists,
        # otherwise the nearest measured one (it is flat across lengths).
        key = T if T in sw else min(sw, key=lambda s: abs(s - T)) if sw else None
        pair_ms = sw[key]["incremental_pair_ms"] if key else 0.0
        tax = pair_ms * pairs_per_layer * a.layers
        budget = cpu[T] - tax
        meas = npu[T]["prefill"]
        bound = 0.5 * meas
        burdened_bound = bound + tax
        ok = burdened_bound < cpu[T]
        rows.append(dict(T=T, seq_pad=npu[T]["pad"], cpu_attn_ms=cpu[T],
                         switch_pair_ms=pair_ms, switch_pair_src=key,
                         switch_tax_ms=round(tax, 1),
                         npu_budget_ms=round(budget, 1),
                         npu_measured_ms=round(meas, 1),
                         npu_measured_vs_cpu=round(meas / cpu[T], 3),
                         optimistic_half_bound_ms=round(bound, 1),
                         optimistic_plus_tax_ms=round(burdened_bound, 1),
                         optimistic_vs_cpu=round(burdened_bound / cpu[T], 3),
                         survives=ok))
        print(f"{T:>6}{cpu[T]:>10.1f}{tax:>12.1f}{budget:>10.1f}{meas:>10.1f}"
              f"{bound:>12.1f}{burdened_bound:>10.1f}"
              f"{('SURVIVES' if ok else 'FAILS'):>12}")

    any_survive = any(r["survives"] for r in rows)
    print()
    if any_survive:
        print("  At least one context survives the impossible-halving bound.")
        print("  -> proceed to the stage-bottleneck discriminator (Task 5).")
    else:
        print("  No context survives even the impossible-halving bound.")
        print("  -> CASE A: ATTENTION CLOSED, no d=128 port is defensible.")

    out = a.out or str(d / "attention_gate.json")
    Path(out).write_text(json.dumps(
        dict(layers=a.layers, transitions_per_layer=a.transitions_per_layer,
             rows=rows, any_survives=any_survive), indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0 if True else 1


if __name__ == "__main__":
    sys.exit(main())
