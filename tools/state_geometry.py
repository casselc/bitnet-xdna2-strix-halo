#!/usr/bin/env python3
"""Predict per-domain warm-state bytes from a model's config, before measuring.

The prediction is stated first so the measurement can confirm or refute it
rather than be rationalised afterwards. Two components, and the split is the
whole point of the comparison:

  token-dependent   ordinary attention KV. Grows with the spine, so it sets how
                    much a LONGER spine costs.
  fixed-per-sequence recurrent/conv/SSM state. Independent of sequence length,
                    so it sets the FLOOR under every domain no matter how short.

An architecture can win on one and lose on the other. Qwen3.5 and LFM2.5 have
identical 12 KiB/token attention components and differ by two orders of
magnitude in the fixed part, which is invisible if only bytes/token is quoted.

llama.cpp's KV cache is f16 by default regardless of weight quantisation, so
the token-dependent term does not move with the GGUF quant level.
"""
import argparse, csv, json, sys
from pathlib import Path

KIB = 1024.0
MIB = 1024.0 * 1024.0


def geom(cfg):
    """Return (per-token bytes, fixed bytes, notes) from a HF config dict."""
    t = cfg.get("text_config", cfg)
    arch = (cfg.get("architectures") or t.get("architectures") or ["?"])[0]
    mt = t.get("model_type", cfg.get("model_type", "?"))
    n_layer = t.get("num_hidden_layers")
    hidden = t.get("hidden_size")
    n_kv = t.get("num_key_value_heads")
    n_head = t.get("num_attention_heads")
    head_dim = t.get("head_dim") or (hidden // n_head if hidden and n_head else None)
    notes = []

    lt = t.get("layer_types")
    pattern = t.get("hybrid_override_pattern")

    # --- how many layers actually hold an attention KV cache
    if lt:
        n_attn = sum(1 for x in lt if x == "full_attention")
        n_lin = sum(1 for x in lt if x == "linear_attention")
        n_conv = sum(1 for x in lt if x == "conv")
    elif pattern:
        # Nemotron-H: '*' attention, 'M' mamba, '-' MLP
        n_attn = pattern.count("*")
        n_lin = pattern.count("M")
        n_conv = 0
    else:
        n_attn, n_lin, n_conv = n_layer or 0, 0, 0

    per_tok = 0.0
    if n_attn and n_kv and head_dim:
        per_tok = n_attn * n_kv * head_dim * 2 * 2  # K and V, f16
        notes.append(f"{n_attn} attn x {n_kv} kv x {head_dim} hd x2(KV) x2B")

    fixed = 0.0
    # --- Qwen3.5 / Qwen3-Next style gated DeltaNet linear attention
    if n_lin and t.get("linear_num_value_heads"):
        vh = t["linear_num_value_heads"]
        kd = t.get("linear_key_head_dim")
        vd = t.get("linear_value_head_dim")
        # state dtype: these configs pin the recurrent state to fp32
        sb = 4 if str(t.get("mamba_ssm_dtype", "float32")).startswith("float32") else 2
        s = n_lin * vh * kd * vd * sb
        fixed += s
        notes.append(f"{n_lin} deltanet x {vh}h x {kd}x{vd} x{sb}B = {s/MIB:.1f} MiB")
        ck = t.get("linear_conv_kernel_dim")
        if ck:
            cdim = (vh * kd) * 2 + (vh * vd)   # q,k,v conv channels
            c = n_lin * (ck - 1) * cdim * 2
            fixed += c
            notes.append(f"deltanet conv {c/KIB:.0f} KiB")
    # --- Nemotron-H / Mamba2
    elif n_lin and t.get("mamba_num_heads"):
        mh = t["mamba_num_heads"]; mhd = t.get("mamba_head_dim")
        ss = t.get("ssm_state_size"); ck = t.get("conv_kernel", 4)
        ng = t.get("n_groups", 1)
        s = n_lin * mh * mhd * ss * 2                      # bf16 ssm state
        fixed += s
        notes.append(f"{n_lin} mamba2 x {mh}h x {mhd} x {ss} x2B = {s/MIB:.1f} MiB")
        cdim = (t.get("expand", 2) * hidden) + 2 * ng * ss
        c = n_lin * (ck - 1) * cdim * 2
        fixed += c
        notes.append(f"mamba conv {c/MIB:.2f} MiB")
    # --- LFM2 short-conv blocks
    if n_conv:
        cl = t.get("conv_L_cache", 3)
        cdim = t.get("conv_dim") or hidden
        c = n_conv * cl * cdim * 2
        fixed += c
        notes.append(f"{n_conv} conv x {cl} x {cdim} x2B = {c/KIB:.0f} KiB")

    return per_tok, fixed, {
        "arch": arch, "model_type": mt, "layers": n_layer, "hidden": hidden,
        "n_attn_layers": n_attn, "n_recurrent_layers": n_lin, "n_conv_layers": n_conv,
        "kv_heads": n_kv, "head_dim": head_dim,
        "vocab": t.get("vocab_size"), "ctx": t.get("max_position_embeddings"),
        "notes": "; ".join(notes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--spine", type=int, default=1600)
    ap.add_argument("--total", type=int, default=1735)
    a = ap.parse_args()

    rows = []
    for p in sorted(Path(a.cfg_dir).glob("*.json")):
        cfg = json.loads(p.read_text())
        per_tok, fixed, info = geom(cfg)
        name = p.stem.replace("_", "/", 1)
        at_spine = per_tok * a.spine + fixed
        at_total = per_tok * a.total + fixed
        row = {"model": name, **info,
               "kv_bytes_per_token": int(per_tok),
               "kv_kib_per_token": round(per_tok / KIB, 2),
               "fixed_state_bytes": int(fixed),
               "fixed_state_mib": round(fixed / MIB, 3),
               f"mib_at_{a.spine}tok": round(at_spine / MIB, 2),
               f"mib_at_{a.total}tok": round(at_total / MIB, 2)}
        for gib in (8, 16, 32):
            row[f"domains_at_{gib}gib"] = int((gib * 1024 * MIB) // at_total) if at_total else 0
        rows.append(row)

    # BitNet baseline is measured on frozen branches, not derived from a HF config
    rows.append({
        "model": "microsoft/BitNet-b1.58-2B-4T", "arch": "BitNetForCausalLM",
        "model_type": "bitnet-b1.58", "layers": 30, "hidden": 2560,
        "n_attn_layers": 30, "n_recurrent_layers": 0, "n_conv_layers": 0,
        "kv_heads": 5, "head_dim": 128, "vocab": 128256, "ctx": 4096,
        "notes": "30 attn x 5 kv x 128 hd x2(KV) x2B (controller-state-envelope 6)",
        "kv_bytes_per_token": 76800, "kv_kib_per_token": 75.0,
        "fixed_state_bytes": 0, "fixed_state_mib": 0.0,
        f"mib_at_{a.spine}tok": round(76800 * a.spine / MIB, 2),
        f"mib_at_{a.total}tok": round(76800 * a.total / MIB, 2),
        "domains_at_8gib": int(8 * 1024 * MIB // (76800 * a.total)),
        "domains_at_16gib": int(16 * 1024 * MIB // (76800 * a.total)),
        "domains_at_32gib": int(32 * 1024 * MIB // (76800 * a.total)),
    })

    cols = list(rows[0].keys())
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)

    w = max(len(r["model"]) for r in rows)
    print(f"{'model':<{w}}  {'KiB/tok':>8} {'fixed MiB':>10} "
          f"{'MiB@'+str(a.total):>10} {'@8GiB':>7} {'@32GiB':>7}")
    for r in sorted(rows, key=lambda x: x[f"mib_at_{a.total}tok"]):
        print(f"{r['model']:<{w}}  {r['kv_kib_per_token']:>8} {r['fixed_state_mib']:>10} "
              f"{r[f'mib_at_{a.total}tok']:>10} {r['domains_at_8gib']:>7} {r['domains_at_32gib']:>7}")
    print(f"\nwrote {a.out_csv}")


if __name__ == "__main__":
    main()
