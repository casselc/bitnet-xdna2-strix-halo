#!/usr/bin/env python3
"""Reconstruct AMD's d=64 MHA pipeline from measured stage rates, then forecast
d=128 -- replacing "d128_time = 0.5 * d64_time" with a measured stage model.

That bound was intentionally favourable but it is not a lower bound if geometry
changes execution efficiency per unit of QK/PV work.
tools/attention_geometry_probe.py measures that efficiency directly; this turns
the per-stage numbers into a whole-attention prediction and validates itself
against the already-measured stock d=64 kernel before forecasting anything.

Pipeline structure, read from iron/operators/mha/design.py rather than assumed:

    number_of_pipelines = 8, each a spatial chain of three cores
        QK core (row 2) -> softmax core (row 3) -> PV core (row 4)
    B_q = B_kv = 64, of_depth = 2 on every ObjectFifo
    q blocks distributed across pipelines; causal, so q block i
    consumes kv blocks 0..i

Two terms, both with a physical referent:

    t_layer = pairs_per_pipeline * service + q_blocks_per_pipeline * C_qblock

  service   steady-state cost of one (q block, kv block) pair. A spatial
            pipeline runs at its slowest stage, so this is max(QK, softmax, PV)
            times a calibration factor.
  C_qblock  work done once per q block rather than per pair: streaming in Q,
            init_scale_buffer, rescale_O over the whole O tile, and writing O
            out. A pure steady-state model omits this, which is why it
            underpredicts, and underpredicts most at small S where the ratio of
            q blocks to pairs is highest.

The model is fitted on the two LARGEST sequence lengths and then predicts the
two smallest as a holdout, so agreement there is a test rather than a fit.
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np


def load_csv(path):
    with open(path) as f:
        return [r for r in csv.DictReader(f)]


class StageRates:
    """Per-core MAC/s for each measured (stage, d, block) geometry.

    The geometry probe runs a whole-array GEMM, so the aggregate is divided by
    core count to get a per-core figure the 8-cores-per-stage MHA pipeline can be
    built from. Whether that division is legitimate is what the core-count
    control tests; the calibration factor below absorbs what it does not."""

    def __init__(self, rows):
        self.by_key = {}
        for r in rows:
            if r.get("build_error") or not r.get("kernel_ms"):
                continue
            key = (r["stage"], int(float(r["d"])), int(float(r["b_q"])),
                   int(float(r["b_kv"])))
            self.by_key[key] = dict(
                per_core_macs_per_s=float(r["macs"]) /
                (float(r["kernel_ms"]) / 1e3) / int(float(r["cores"])),
                cores=int(float(r["cores"])), tflops=float(r["tflops"]),
                label=r["label"])

    def has(self, stage, d, b_q, b_kv):
        return (stage, d, b_q, b_kv) in self.by_key

    def rate(self, stage, d, b_q, b_kv):
        return self.by_key[(stage, d, b_q, b_kv)]["per_core_macs_per_s"]


def counts(S_pad, b_q, b_kv, heads, pipelines):
    """(q block, kv block) pairs and q blocks each pipeline must process.

    Causal on the token grid, so it stays exact when b_q != b_kv."""
    nq = S_pad // b_q
    pairs = sum((((i + 1) * b_q - 1) // b_kv) + 1 for i in range(nq))
    return pairs * heads / pipelines, nq * heads / pipelines


def stage_times(d, b_q, b_kv, rates, sm_elems_per_s):
    mm = b_q * b_kv * d                                   # QK and PV each
    return dict(t_qk=mm / rates.rate("qk", d, b_q, b_kv),
                t_sm=(b_q * b_kv) / sm_elems_per_s,
                t_pv=mm / rates.rate("pv", d, b_q, b_kv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/attention-feasibility")
    ap.add_argument("--geom-csv", default="geometry_qk_pv_c8.csv")
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--pipelines", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.dir)

    rates = StageRates(load_csv(d / a.geom_csv))
    sm_rows = load_csv(d / "geometry_softmax.csv")
    sm_best = max(sm_rows, key=lambda r: float(r["elems"]))
    sm_rate = float(sm_best["elems"]) / (float(sm_best["kernel_ms"]) / 1e3)

    # Keyed by EXECUTED length, not requested: the design pads S to a multiple
    # of B_q * pipelines and runs the padded length, so 3968 and 4096 are the
    # same experiment and would make the fit singular. Where both exist, the
    # entry whose request equals what it executed is the one kept.
    mha = {}
    for r in load_csv(d / "npu_mha_4k.csv"):
        if not r.get("kernel_ms"):
            continue
        pad, S = int(r["seq_pad"]), int(r["seq_len"])
        rec = dict(pad=pad, seq_len=S, kernel_ms=float(r["kernel_ms"]),
                   heads=int(r["heads"]))
        if pad not in mha or S == pad:
            mha[pad] = rec

    print(f"per-core stage rates ({a.geom_csv}):")
    for k, v in sorted(rates.by_key.items()):
        print(f"  {v['label']:16s} {v['per_core_macs_per_s']/1e9:8.2f} GMAC/s/core"
              f"  ({v['tflops']:.3f} TFLOPS over {v['cores']} cores)")
    print(f"softmax_simple_bf16: {sm_rate/1e6:.1f} Melem/s per core "
          f"(rows={sm_best['rows']}, {sm_best['kernel_ms']} ms)")

    st64 = stage_times(64, 64, 64, rates, sm_rate)
    NAME = {"t_qk": "QK", "t_sm": "SOFTMAX", "t_pv": "PV"}
    print(f"\nd=64 stage times per 64x64 block, from the standalone probes:")
    print(f"  QK {st64['t_qk']*1e6:7.3f} us   softmax {st64['t_sm']*1e6:7.3f} us"
          f"   PV {st64['t_pv']*1e6:7.3f} us   -> limiting: "
          f"{NAME[max(st64, key=st64.get)]}")

    # --- fit on the two largest S, hold out the two smallest ---------------
    keys = sorted(mha)
    fit_S, hold_S = keys[-2:], keys[:-2]
    A, y = [], []
    for S in fit_S:
        p, q = counts(S, 64, 64, mha[S]["heads"], a.pipelines)
        A.append([p, q]); y.append(mha[S]["kernel_ms"])
    service_ms, cq_ms = np.linalg.solve(np.array(A), np.array(y))

    print(f"\nfit on S={fit_S} only, S={hold_S} held out:")
    print(f"  service    {service_ms*1e3:8.3f} us / (q,kv) pair")
    print(f"  C_qblock   {cq_ms*1e3:8.2f} us / q block")
    print(f"  {'S':>6}{'pairs/pipe':>12}{'qblk/pipe':>11}{'measured':>10}"
          f"{'predicted':>11}{'error':>9}{'role':>10}")
    recon = []
    for S in keys:
        m = mha[S]
        p, q = counts(S, 64, 64, m["heads"], a.pipelines)
        pred = service_ms * p + cq_ms * q
        err = pred / m["kernel_ms"] - 1
        role = "fit" if S in fit_S else "HOLDOUT"
        recon.append(dict(seq_len=m["seq_len"], S_pad=S, pairs_per_pipe=round(p, 1),
                          qblocks_per_pipe=round(q, 1),
                          measured_ms=m["kernel_ms"], predicted_ms=round(pred, 3),
                          rel_error=round(err, 4), role=role))
        print(f"  {S:>6}{p:>12.0f}{q:>11.0f}{m['kernel_ms']:>10.3f}{pred:>11.3f}"
              f"{err*100:>8.2f}%{role:>10}")

    worst_hold = max(abs(r["rel_error"]) for r in recon if r["role"] == "HOLDOUT")
    usable = worst_hold <= 0.10
    print(f"\n  worst HOLDOUT error {worst_hold*100:.2f}%  -> model "
          f"{'USABLE' if usable else 'NOT USABLE'} (gate: 10%)")

    # Is the fitted service physically the softmax stage, or a free parameter?
    mx = max(st64.values())
    calib = service_ms / 1e3 / mx
    which = NAME[max(st64, key=st64.get)]
    print(f"  fitted service is {calib:.3f}x the standalone {which} stage "
          f"({mx*1e6:.3f} us),")
    print(f"  and {service_ms*1e3/ (st64['t_qk']*1e6):.2f}x QK / "
          f"{service_ms*1e3/(st64['t_pv']*1e6):.2f}x PV -- so no combination of"
          f" QK and PV\n  accounts for it. The limiting stage is {which}.")

    out = dict(geom_csv=a.geom_csv, pipelines=a.pipelines,
               softmax_melem_per_s_per_core=round(sm_rate / 1e6, 2),
               d64_stage_us={k: round(v * 1e6, 4) for k, v in st64.items()},
               limiting_stage_d64=which,
               fit=dict(fit_on=fit_S, holdout=hold_S,
                        service_us=round(service_ms * 1e3, 4),
                        c_qblock_us=round(cq_ms * 1e3, 3),
                        calibration_vs_standalone=round(calib, 4)),
               reconstruction=recon, worst_holdout_error=round(worst_hold, 4),
               model_usable=bool(usable))

    if not usable:
        print("\n  model rejected; no d=128 forecast is made from it.")
    else:
        # d=128 has HALF the heads (20 vs 40) at the same total Q/KV width, so
        # softmax rows halve while QK/PV arithmetic is unchanged. The block must
        # shrink because d=128 does not fit L1 at 64x64; both escape routes are
        # forecast. C_qblock is scaled by a sensitivity band because the O tile
        # it rescales doubles at d=128.
        print(f"\nd=128 forecast (20 Q heads / 5 KV heads), calibration "
              f"{calib:.3f}x applied to service:")
        print(f"  {'S':>6}{'block':>8}{'Cq x':>6}{'t_QK':>8}{'t_SM':>8}{'t_PV':>8}"
              f"{'limit':>8}{'d64 meas':>10}{'d128':>9}{'ratio':>7}")
        fc = []
        for S in keys:
            m = mha[S]
            for (bq, bkv) in ((32, 64), (64, 32)):
                if not (rates.has("qk", 128, bq, bkv) and rates.has("pv", 128, bq, bkv)):
                    continue
                st = stage_times(128, bq, bkv, rates, sm_rate)
                svc = calib * max(st.values())
                p, q = counts(S, bq, bkv, m["heads"] // 2, a.pipelines)
                for cq_scale in (1.0, 2.0):
                    pred = svc * 1e3 * p + cq_ms * cq_scale * q
                    rec = dict(seq_len=m["seq_len"], S_pad=S, b_q=bq, b_kv=bkv,
                               cq_scale=cq_scale, heads=m["heads"] // 2,
                               pairs_per_pipe=round(p, 1),
                               qblocks_per_pipe=round(q, 1),
                               t_qk_us=round(st["t_qk"] * 1e6, 3),
                               t_sm_us=round(st["t_sm"] * 1e6, 3),
                               t_pv_us=round(st["t_pv"] * 1e6, 3),
                               limiting=NAME[max(st, key=st.get)],
                               service_us=round(svc * 1e6, 3),
                               d64_measured_ms=m["kernel_ms"],
                               predicted_ms=round(pred, 3),
                               ratio_vs_d64=round(pred / m["kernel_ms"], 4),
                               prefill_ms_30L=round(pred * a.layers, 1))
                    fc.append(rec)
                    print(f"  {S:>6}{f'{bq}x{bkv}':>8}{cq_scale:>6.0f}"
                          f"{rec['t_qk_us']:>8.2f}{rec['t_sm_us']:>8.2f}"
                          f"{rec['t_pv_us']:>8.2f}{rec['limiting']:>8}"
                          f"{m['kernel_ms']:>10.3f}{pred:>9.2f}"
                          f"{rec['ratio_vs_d64']:>7.3f}")
        out["forecast_d128"] = fc

    path = a.out or str(d / "d128_pipeline_model.json")
    Path(path).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
