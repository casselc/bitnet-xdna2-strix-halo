#!/usr/bin/env python3
"""Is restricted-logit training REALLY equivalent? Per-parameter, not aggregate.

The previous pass compared scalar loss and a single aggregate gradient norm and
called the paths "equivalent". Two different gradient vectors can share a norm,
so that evidence did not support the claim it was used for. This checks what the
claim actually requires:

  1. per-element gradient agreement over every trainable tensor
     (max |delta| and relative L2 over the concatenated gradient)
  2. an end-to-end optimizer step from IDENTICAL adapter and optimizer state,
     comparing every resulting trainable parameter

(2) is the one that matters operationally: gradients feed an optimizer, and it
is the parameters after a step that a training campaign actually carries
forward.

Both paths run on the same model object with the same inputs; only
`logits_to_keep` differs.
"""
import argparse, copy, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_controller_sft import build_examples, masked_loss
from train_scaling import resolve_targets, attach_lora, load_model, peak_mem_mib


def grads_of(model):
    import torch
    return {n: p.grad.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None}


def zero(model):
    for p in model.parameters():
        p.grad = None


def compare(a, b):
    import torch
    keys = sorted(set(a) & set(b))
    missing = sorted(set(a) ^ set(b))
    if not keys:
        return {"n_tensors": 0, "err": "no common trainable tensors"}
    fa = torch.cat([a[k].flatten().float() for k in keys])
    fb = torch.cat([b[k].flatten().float() for k in keys])
    d = (fa - fb).abs()
    return {"n_tensors": len(keys), "n_elements": int(fa.numel()),
            "missing_tensors": missing[:5],
            "max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "rel_l2": float((fa - fb).norm() / max(float(fa.norm()), 1e-12)),
            "identical": bool(float(d.max()) == 0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--logits-keep", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-action-tokens", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-impl", default="eager")
    a = ap.parse_args()

    import torch
    dev = a.device
    model_id = a.model if not os.path.isabs(a.model) else os.path.basename(a.model.rstrip("/"))
    tok, base, load_s = load_model(a.model, a.dtype, dev, a.attn_impl)
    targets, skipped, _ = resolve_targets(base)
    model, trainable, total = attach_lora(base, a.lora_r, a.lora_r * 2, targets)
    data = build_examples(tok, a.seq_len, 4, a.max_action_tokens, seed=a.seq_len)
    ids = torch.tensor([data[0][0]], device=dev)
    lab = torch.tensor([data[0][1]], device=dev)

    res = {"label": a.label, "model": model_id, "seq_len": a.seq_len,
           "logits_keep": a.logits_keep, "trainable_params": trainable,
           "lora_targets": targets}

    # ---- 1. gradients, per element
    zero(model)
    lf = masked_loss(model, ids, lab, 0); lf.backward()
    gf = grads_of(model); mem_full = peak_mem_mib()
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    zero(model)
    lr_ = masked_loss(model, ids, lab, a.logits_keep); lr_.backward()
    gr = grads_of(model); mem_restricted = peak_mem_mib()
    zero(model)

    res["loss_full"] = float(lf); res["loss_restricted"] = float(lr_)
    res["loss_delta"] = abs(float(lf) - float(lr_))
    res["gradient"] = compare(gf, gr)
    res["peak_mem_full_mib"] = mem_full
    res["peak_mem_restricted_mib"] = mem_restricted
    res["peak_mem_saved_pct"] = (round(100 * (mem_full - mem_restricted) / mem_full, 2)
                                 if mem_full else None)
    g = res["gradient"]
    print(f"[{a.label}] loss delta {res['loss_delta']:.3e}", flush=True)
    print(f"[{a.label}] gradient over {g['n_tensors']} tensors / {g['n_elements']:,} elements: "
          f"max|d|={g['max_abs']:.3e} rel_L2={g['rel_l2']:.3e} identical={g['identical']}",
          flush=True)
    print(f"[{a.label}] peak mem {mem_full} -> {mem_restricted} MiB "
          f"({res['peak_mem_saved_pct']}%)", flush=True)

    # ---- 2. one optimizer step from IDENTICAL state, compare parameters
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    snapshot = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def step(keep):
        # restore identical adapter state, then a fresh identical optimizer
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(snapshot[n])
        zero(model)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
        loss = masked_loss(model, ids, lab, keep)
        loss.backward()
        opt.step()
        return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    pf = step(0)
    pr = step(a.logits_keep)
    res["parameters_after_one_step"] = compare(pf, pr)
    p = res["parameters_after_one_step"]
    print(f"[{a.label}] parameters after one AdamW step: max|d|={p['max_abs']:.3e} "
          f"rel_L2={p['rel_l2']:.3e} identical={p['identical']}", flush=True)

    res["equivalent"] = bool(res["gradient"].get("identical") and
                             res["parameters_after_one_step"].get("identical"))
    res["verdict"] = ("RESTRICTED LOGITS BIT-IDENTICAL (gradients and post-step parameters)"
                      if res["equivalent"] else
                      "RESTRICTED LOGITS NOT BIT-IDENTICAL -- see max_abs / rel_l2")
    print(f"[{a.label}] VERDICT: {res['verdict']}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
