#!/usr/bin/env python3
"""The d=128 geometry economic gate, computed from measured files only.

Inputs, all produced by measurements on this machine:
  cpu_oracle.csv           CPU attention wall time, in situ, 15 threads
  gemm_mha_switch_*.json   real GEMM<->MHA context alternation
  d128_pipeline_model.json stage model, validated on held-out sequence lengths

    budget(T)     = CPU_attention(T) - switch_tax(T)
    d128_burdened = predicted_d128_prefill + switch_tax

The forecast is reported as a BAND, not a point. C_qblock covers work done once
per q block -- streaming Q in, init_scale_buffer, rescale_O over the whole O
tile, writing O out -- and the O tile doubles at d=128, so scaling it by 2 is
the physically motivated case and leaving it unchanged is a generous bound. The
gate is decided on the generous end.

Also reported: the per-core QK rate a d=128 kernel would need to reach the
budget, since the model says QK becomes the limiting stage at d=128. That turns
the residual requirement into a number about a primitive rather than a vibe.
"""
import argparse, csv, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/attention-feasibility")
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--material-gain", type=float, default=0.15,
                    help="fully burdened attention-path improvement at 4K below "
                         "which a full d=128 port is not worth building")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.dir)

    cpu = {}
    with open(d / "cpu_oracle.csv") as f:
        for r in csv.DictReader(f):
            cpu[int(r["prompt"])] = float(r["attn_ms"])

    sw = {}
    for p in sorted(d.glob("gemm_mha_switch_s*.json")):
        j = json.loads(p.read_text())
        sw[int(j["seq"])] = j["incremental_pair_ms"]
    # one alternating pair per layer = two transitions (GEMM->MHA, MHA->GEMM)
    tax = {T: sw[T] * a.layers for T in sw}

    model = json.loads((d / "d128_pipeline_model.json").read_text())
    if not model.get("model_usable"):
        print("stage model did not validate; no gate can be computed from it.")
        return
    fc = model["forecast_d128"]

    # measured stock d=64, for the reference column
    d64 = {r["S_pad"]: r["measured_ms"] for r in model["reconstruction"]}

    print(f"layers={a.layers}   switch tax = incremental pair x 1 pair/layer x "
          f"{a.layers} layers")
    print(f"model: fitted on S={model['fit']['fit_on']}, held out "
          f"S={model['fit']['holdout']}, worst holdout error "
          f"{model['worst_holdout_error']*100:.2f}%\n")

    # The design pads S to a multiple of B_q * pipelines = 512 and executes the
    # padded length, so the CPU's T=3968 is compared against the model's
    # S_pad=4096 row. Joining on the requested length would silently drop it.
    def pad_of(T, block=512):
        return ((T + block - 1) // block) * block

    rows = []
    for T in sorted(cpu):
        cand = [r for r in fc if r["S_pad"] == pad_of(T)]
        if not cand:
            continue
        key = T if T in tax else min(tax, key=lambda s: abs(s - T))
        tx = tax[key]
        budget = cpu[T] - tx
        best = min(cand, key=lambda r: r["prefill_ms_30L"])
        worst = max(cand, key=lambda r: r["prefill_ms_30L"])
        stock = d64.get(cand[0]["S_pad"], 0) * a.layers
        b_burd, w_burd = best["prefill_ms_30L"] + tx, worst["prefill_ms_30L"] + tx
        gain = 1 - b_burd / cpu[T]
        rows.append(dict(
            T=T, S_pad=cand[0]["S_pad"], S_executed=cand[0]["S_pad"],
            cpu_attn_ms=cpu[T],
            switch_tax_ms=round(tx, 1), budget_ms=round(budget, 1),
            stock_d64_prefill_ms=round(stock, 1),
            d128_best_ms=best["prefill_ms_30L"],
            d128_best_cfg=f"{best['b_q']}x{best['b_kv']} Cq{best['cq_scale']:.0f}x",
            d128_worst_ms=worst["prefill_ms_30L"],
            d128_best_burdened_ms=round(b_burd, 1),
            d128_worst_burdened_ms=round(w_burd, 1),
            best_over_budget=round(best["prefill_ms_30L"] / budget, 3)
            if budget > 0 else None,
            burdened_vs_cpu=round(b_burd / cpu[T], 3),
            attention_path_gain=round(gain, 4),
            limiting_stage=best["limiting"]))

    print(f"{'T':>6}{'CPU attn':>10}{'tax':>7}{'budget':>9}{'stock d64':>11}"
          f"{'d128 best':>10}{'d128 worst':>11}{'best+tax':>10}{'vs CPU':>8}{'gain':>8}")
    for r in rows:
        print(f"{r['T']:>6}{r['cpu_attn_ms']:>10.1f}{r['switch_tax_ms']:>7.0f}"
              f"{r['budget_ms']:>9.1f}{r['stock_d64_prefill_ms']:>11.1f}"
              f"{r['d128_best_ms']:>10.1f}{r['d128_worst_ms']:>11.1f}"
              f"{r['d128_best_burdened_ms']:>10.1f}"
              f"{r['burdened_vs_cpu']:>8.2f}x{r['attention_path_gain']*100:>7.1f}%")

    four_k = max(rows, key=lambda r: r["T"])
    gain = four_k["attention_path_gain"]
    if gain >= a.material_gain:
        verdict = "C. MATERIAL POSSIBILITY"
    elif gain > 0:
        verdict = "B. POSSIBLE BUT MARGINAL"
    else:
        verdict = "A. CLEAR NEGATIVE"
    print(f"\n  at T={four_k['T']}: best-case burdened d=128 is "
          f"{four_k['burdened_vs_cpu']:.2f}x the CPU "
          f"({gain*100:+.1f}% attention-path gain)")
    print(f"  classification: {verdict}")

    # What would have to be true. At d=128 the model says QK sets the rate, so
    # express the residual as the QK primitive rate required to reach budget.
    best4k = min((r for r in fc if r["S_pad"] == four_k["S_pad"]),
                 key=lambda r: r["prefill_ms_30L"])
    per_layer_budget = four_k["budget_ms"] / a.layers
    cq_ms = model["fit"]["c_qblock_us"] / 1e3 * best4k["cq_scale"]
    fixed = cq_ms * best4k["qblocks_per_pipe"]
    need_service_us = (per_layer_budget - fixed) / best4k["pairs_per_pipe"] * 1e3
    have_service_us = best4k["service_us"]
    print(f"\n  to reach the 4K budget at the best block ({best4k['b_q']}x"
          f"{best4k['b_kv']}, Cq{best4k['cq_scale']:.0f}x):")
    print(f"    per-pair service would have to fall to {need_service_us:.2f} us "
          f"from the modelled {have_service_us:.2f} us"
          f"  ({have_service_us/need_service_us:.2f}x)")

    out = dict(layers=a.layers, material_gain_threshold=a.material_gain,
               rows=rows, verdict=verdict,
               required_service_us=round(need_service_us, 3),
               modelled_service_us=round(have_service_us, 3),
               required_speedup_of_limiting_stage=round(
                   have_service_us / need_service_us, 3))
    path = a.out or str(d / "geometry_gate.json")
    Path(path).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
