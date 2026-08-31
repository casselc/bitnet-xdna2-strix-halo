#!/usr/bin/env python3
"""Economic gate for a FUSED attention core, from the measured fusion cost.

Replaces the projection that motivated this branch. That one assumed a fused
core would take the SUM of the stock stage times (16.147 us/pair) and scale to
32 cores, giving 2.64x. Both halves are now measured instead of assumed:

  * a fused core takes 18.845 us/pair, not 16.147 -- fusing costs 1.167x more
    than the naive sum, because a single core cannot overlap what three cores
    overlapped;
  * it must also perform two per-pair layout transforms that the stock design
    gets FREE from its inter-core DMA (memA carries a_dims, memP carries
    q_dims), measured here as a traffic-only lower bound of +1.2%;
  * the per-q-block cost is 93.7 us, higher than the 68.4 us the pipeline model
    fitted from the stock kernel.

d=128 stage times come from the geometry probe, scaled by the SAME measured
fusion penalty, since that penalty is a property of serialising three stages on
one core rather than of the geometry.
"""
import argparse, csv, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/attention-feasibility")
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--cores", type=int, default=32)
    ap.add_argument("--material-gain", type=float, default=0.15)
    ap.add_argument("--no-relayout", action="store_true",
                    help="model a tile-major-aware softmax, which would make "
                         "both inter-stage transforms cancel: the QK matmul's "
                         "(r,t) output tiling is already the layout the PV "
                         "matmul wants for its A operand")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.dir)

    fused = json.loads((d / "fused_core.json").read_text())["summary"]
    model = json.loads((d / "d128_pipeline_model.json").read_text())
    geom = {(r["stage"], int(float(r["d"])), int(float(r["b_q"])),
             int(float(r["b_kv"]))):
            float(r["macs"]) / (float(r["kernel_ms"]) / 1e3) / int(float(r["cores"]))
            for r in csv.DictReader(open(d / "geometry_qk_pv_c8.csv"))
            if r.get("kernel_ms") and not r.get("build_error")}
    sm_rows = list(csv.DictReader(open(d / "geometry_softmax.csv")))
    best = max(sm_rows, key=lambda r: float(r["elems"]))
    sm_rate = float(best["elems"]) / (float(best["kernel_ms"]) / 1e3)

    cpu = {int(r["prompt"]): float(r["attn_ms"])
           for r in csv.DictReader(open(d / "cpu_oracle.csv"))}
    tax = {int(json.loads(p.read_text())["seq"]):
           json.loads(p.read_text())["incremental_pair_ms"] * a.layers
           for p in sorted(d.glob("gemm_mha_switch_s*.json"))}

    st64 = model["d64_stage_us"]
    naive_sum = st64["t_qk"] + st64["t_sm"] + st64["t_pv"]
    measured = fused["slope_us_per_pair"]
    penalty = measured / naive_sum
    relayout = fused.get("relayout_slope_us_per_pair", measured) / measured
    if a.no_relayout:
        relayout = 1.0
    c_qblock = fused["intercept_us"]
    stock_service = fused["stock_service_us"]

    print(f"measured fusion cost")
    print(f"  naive sum of stock stage times   {naive_sum:8.3f} us/pair")
    print(f"  measured fused core              {measured:8.3f} us/pair "
          f"({penalty:.3f}x -- fusing is not free)")
    print(f"  + inter-stage relayout traffic   "
          f"{fused.get('relayout_slope_us_per_pair', measured):8.3f} us/pair "
          f"({relayout:.3f}x, LOWER BOUND)")
    print(f"  per-q-block cost                 {c_qblock:8.3f} us "
          f"(stock pipeline model fitted {model['fit']['c_qblock_us']:.2f})")
    print(f"  stock, per pair of core-time     {3*stock_service:8.3f} us (3 cores)\n")

    def pairs_qblocks(S, bq, bkv, heads):
        nq = S // bq
        return (sum((((i + 1) * bq - 1) // bkv) + 1 for i in range(nq)) * heads,
                nq * heads)

    def predict(S, dd, bq, bkv, heads, cq_scale=1.0):
        mm = bq * bkv * dd
        service = (mm / geom[("qk", dd, bq, bkv)] +
                   (bq * bkv) / sm_rate +
                   mm / geom[("pv", dd, bq, bkv)]) * 1e6      # us, naive sum
        service *= penalty * relayout
        p, q = pairs_qblocks(S, bq, bkv, heads)
        return (p * service + q * c_qblock * cq_scale) / a.cores / 1e3   # ms/layer

    print(f"{'T':>6}{'config':>18}{'Cq':>5}{'ms/layer':>10}{'prefill':>9}"
          f"{'+tax':>8}{'CPU':>8}{'vs CPU':>8}{'gain':>8}")
    rows = []
    for T, S in ((2048, 2048), (3968, 4096)):
        tx = tax.get(T, tax[min(tax, key=lambda s: abs(s - T))])
        for lbl, dd, bq, bkv, h in (("d64  64x64", 64, 64, 64, 40),
                                    ("d128 64x32", 128, 64, 32, 20),
                                    ("d128 32x64", 128, 32, 64, 20)):
            if ("qk", dd, bq, bkv) not in geom:
                continue
            for cq in (1.0, 2.0) if dd == 128 else (1.0,):
                ms = predict(S, dd, bq, bkv, h, cq)
                pre = ms * a.layers
                burd = pre + tx
                gain = 1 - burd / cpu[T]
                rows.append(dict(T=T, S_exec=S, config=lbl, d=dd, b_q=bq,
                                 b_kv=bkv, heads=h, cq_scale=cq,
                                 ms_per_layer=round(ms, 3),
                                 prefill_ms=round(pre, 1),
                                 switch_tax_ms=round(tx, 1),
                                 burdened_ms=round(burd, 1),
                                 cpu_ms=cpu[T],
                                 burdened_vs_cpu=round(burd / cpu[T], 4),
                                 attention_path_gain=round(gain, 4)))
                print(f"{T:>6}{lbl:>18}{cq:>5.0f}{ms:>10.2f}{pre:>9.0f}"
                      f"{burd:>8.0f}{cpu[T]:>8.0f}{burd/cpu[T]:>7.2f}x"
                      f"{gain*100:>7.1f}%")

    four = [r for r in rows if r["T"] == 3968]
    best4k = max(four, key=lambda r: r["attention_path_gain"])
    g = best4k["attention_path_gain"]
    verdict = ("C. MATERIAL POSSIBILITY" if g >= a.material_gain else
               "B. POSSIBLE BUT MARGINAL" if g > 0 else "A. CLEAR NEGATIVE")
    print(f"\n  best at 4K: {best4k['config']} Cq{best4k['cq_scale']:.0f}x -> "
          f"{best4k['burdened_vs_cpu']:.2f}x the CPU ({g*100:+.1f}%)")
    print(f"  classification: {verdict}")

    out = dict(fusion_penalty=round(penalty, 4),
               relayout_factor=round(relayout, 4),
               measured_us_per_pair=measured, c_qblock_us=c_qblock,
               cores=a.cores, layers=a.layers, rows=rows,
               best_4k=best4k, verdict=verdict)
    path = a.out or str(d / "fused_gate.json")
    Path(path).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
