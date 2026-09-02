#!/usr/bin/env python3
"""Controller SFT: long context, tiny supervised action completion.

CORRECTS `tools/train_scaling.py`, which measured something real but not this.
Its generator grew the state text until the state ALONE already exceeded
seq_len, then appended "ACTION: <verb>", then truncated to seq_len+1 -- so the
action was always cut off. Loss was then applied to every position. Those
numbers are valid FULL-SEQUENCE LM THROUGHPUT and are kept under that name;
they are not the controller objective.

Here every example is:

    [ context: stable spine + dynamic state ] [ "\nACTION:" ] [ 1-4 action tokens ]

built so the action tokens are GUARANTEED inside the sequence, with the context
trimmed to make room rather than the action being pushed out. Labels are -100
on everything except the action completion, and every example is verified to
carry at least one and at most `--max-action-tokens` supervised positions. A
violation fails loudly instead of quietly training on nothing.

Because only the last few positions are supervised, the loss is computed here
rather than by passing `labels=` to the model, which lets `logits_to_keep`
materialise logits for just those positions. That matters most for Qwen3.5,
whose 248,320-token vocabulary makes a full [B, T, V] logits tensor the
dominant memory term. `--check-logits` asserts the restricted path reproduces
the full-logits loss and gradient before any timing is reported.
"""
import argparse, json, math, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_scaling import (PowerMeter, resolve_targets, attach_lora, load_model,
                           peak_mem_mib)

ACTIONS = ["HOLD", "SCALE", "ROLLBACK", "RESTART", "PAGE"]
MARKER = "\nACTION:"


def build_examples(tok, seq_len, n_seq, max_action_tokens=4, seed=0):
    """Exactly seq_len tokens, action completion guaranteed present.

    Returns (input_ids, labels) pairs. The context is TRIMMED to fit the action,
    which is the opposite of the superseded generator's behaviour.
    """
    rng = random.Random(seed)
    marker_ids = tok(MARKER, add_special_tokens=False)["input_ids"]
    out = []
    for i in range(n_seq):
        verb = ACTIONS[i % len(ACTIONS)]
        act_ids = tok(" " + verb, add_special_tokens=False)["input_ids"][:max_action_tokens]
        if not act_ids:
            raise SystemExit(f"tokenizer produced no action tokens for {verb!r}")
        budget = seq_len - len(marker_ids) - len(act_ids)
        if budget < 8:
            raise SystemExit(f"seq_len={seq_len} too small for marker+action")
        parts = [f"CONTROLLER DOMAIN {i:04d}\nSTATE SPINE\n"]
        ctx_ids = tok("".join(parts), add_special_tokens=False)["input_ids"]
        while len(ctx_ids) < budget:
            parts.append(
                f"  svc{rng.randrange(999):03d}: p95={rng.randrange(10,900)}ms "
                f"err={rng.randrange(50)} cpu={rng.randrange(100)}% "
                f"q={rng.randrange(300)} dep=svc{rng.randrange(999):03d}\n")
            if len(parts) % 8 == 0:
                ctx_ids = tok("".join(parts), add_special_tokens=False)["input_ids"]
        ctx_ids = tok("".join(parts), add_special_tokens=False)["input_ids"][:budget]

        ids = ctx_ids + marker_ids + act_ids
        labels = [-100] * (len(ctx_ids) + len(marker_ids)) + list(act_ids)
        assert len(ids) == seq_len == len(labels), (len(ids), seq_len, len(labels))
        n_sup = sum(1 for x in labels if x != -100)
        if not (1 <= n_sup <= max_action_tokens):
            raise SystemExit(f"example {i}: {n_sup} supervised tokens, "
                             f"expected 1..{max_action_tokens}")
        out.append((ids, labels))
    return out


def verify_corpus(data, seq_len, max_action_tokens):
    sup = [sum(1 for x in lab if x != -100) for _, lab in data]
    if min(sup) < 1 or max(sup) > max_action_tokens:
        raise SystemExit(f"corpus violates action-token bounds: min={min(sup)} max={max(sup)}")
    if any(len(i) != seq_len for i, _ in data):
        raise SystemExit("corpus contains a wrong-length example")
    return {"prompt_tokens": seq_len - int(sum(sup) / len(sup)),
            "supervised_tokens_mean": sum(sup) / len(sup),
            "supervised_tokens_min": min(sup), "supervised_tokens_max": max(sup),
            "supervised_fraction": (sum(sup) / len(sup)) / seq_len}


def masked_loss(model, ids, labels, logits_to_keep=0):
    """Next-token loss over the supervised positions only.

    Computed here rather than via `labels=` so `logits_to_keep` can restrict
    logit materialisation to the tail. `logits_to_keep=k` returns logits for the
    LAST k positions; predicting label[t] uses logits[t-1], so k must cover one
    position before the first supervised label.
    """
    import torch
    import torch.nn.functional as F
    kw = {}
    if logits_to_keep:
        kw["logits_to_keep"] = logits_to_keep
    out = model(input_ids=ids, **kw)
    logits = out.logits                       # [B, K, V] with K = kept positions
    K = logits.shape[1]
    tgt = labels[:, -K:]                      # align labels to the kept window
    # shift: logits at position p predict token p+1
    lg = logits[:, :-1, :]
    tg = tgt[:, 1:]
    mask = tg != -100
    if not bool(mask.any()):
        raise SystemExit("no supervised token inside the kept logits window; "
                         "increase --logits-keep")
    lg = lg.reshape(-1, lg.shape[-1])
    tg = tg.reshape(-1)
    return F.cross_entropy(lg.float(), tg, ignore_index=-100)


def run_arm(model, dev, data, seq_len, micro, accum, steps, warmup, lr,
            logits_keep, label=""):
    import torch
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    tokens_per_update = micro * seq_len * accum

    def one_update(idx):
        fwd = bwd = 0.0
        tot = 0.0
        for g in range(accum):
            s = ((idx * accum + g) * micro) % max(1, len(data) - micro)
            chunk = data[s:s + micro]
            if len(chunk) < micro:
                chunk = data[:micro]
            ids = torch.tensor([c[0] for c in chunk], device=dev)
            lab = torch.tensor([c[1] for c in chunk], device=dev)
            t0 = time.time()
            loss = masked_loss(model, ids, lab, logits_keep) / accum
            if dev == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            loss.backward()
            if dev == "cuda":
                torch.cuda.synchronize()
            t2 = time.time()
            fwd += t1 - t0; bwd += t2 - t1; tot += float(loss) * accum
        opt.step(); opt.zero_grad(set_to_none=True)
        if dev == "cuda":
            torch.cuda.synchronize()
        return fwd, bwd, tot / accum

    for i in range(warmup):
        one_update(i)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times, fwds, bwds, losses = [], [], [], []
    with PowerMeter() as pw:
        for i in range(steps):
            t0 = time.time()
            f, b, l = one_update(warmup + i)
            times.append(time.time() - t0); fwds.append(f); bwds.append(b); losses.append(l)
    med = sorted(times)[len(times) // 2]
    return {"seq_len": seq_len, "microbatch": micro, "grad_accum": accum,
            "tokens_per_update": tokens_per_update, "logits_to_keep": logits_keep,
            "steps_timed": steps, "step_s_median": round(med, 4),
            "tokens_per_s": round(tokens_per_update / med, 1),
            "fwd_s_median": round(sorted(fwds)[len(fwds) // 2], 4),
            "bwd_s_median": round(sorted(bwds)[len(bwds) // 2], 4),
            "peak_mem_mib": peak_mem_mib(), "watts": pw.watts,
            "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4)}


def check_logits_equivalence(model, dev, data, seq_len, keep, tol=2e-3):
    """Restricted logits must reproduce the full-logits loss and gradient."""
    import torch
    ids = torch.tensor([data[0][0]], device=dev)
    lab = torch.tensor([data[0][1]], device=dev)
    res = {}
    for name, k in (("full", 0), ("restricted", keep)):
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        loss = masked_loss(model, ids, lab, k)
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters()
                       if p.requires_grad and p.grad is not None])
        res[name] = {"loss": float(loss), "grad_norm": float(g.norm()),
                     "peak_mem_mib": peak_mem_mib()}
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    dl = abs(res["full"]["loss"] - res["restricted"]["loss"])
    dg = abs(res["full"]["grad_norm"] - res["restricted"]["grad_norm"])
    rel = dg / max(res["full"]["grad_norm"], 1e-9)
    res["delta_loss"] = dl
    res["delta_grad_norm"] = dg
    res["rel_grad_norm_delta"] = rel
    res["equivalent"] = bool(dl <= tol and rel <= tol)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-lens", default="512,1024")
    ap.add_argument("--token-budget", type=int, default=4096)
    ap.add_argument("--microbatches", default="",
                    help="comma list; default = derive from budget/seq_len")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-action-tokens", type=int, default=4)
    ap.add_argument("--logits-keep", type=int, default=8)
    ap.add_argument("--no-restrict-logits", action="store_true")
    ap.add_argument("--check-logits", action="store_true")
    ap.add_argument("--attn-impl", default="eager")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    dev = a.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("FAIL: no ROCm device visible to torch", file=sys.stderr)
        return 2

    model_id = a.model if not os.path.isabs(a.model) else os.path.basename(a.model.rstrip("/"))
    tok, base, load_s = load_model(a.model, a.dtype, dev, a.attn_impl)
    targets, skipped, present = resolve_targets(base)
    model, trainable, total = attach_lora(base, a.lora_r, a.lora_r * 2, targets)

    # Task 11: say exactly how much of the architecture the adapter reaches.
    import torch.nn as nn
    n_conv = sum(1 for _, m in model.named_modules() if isinstance(m, nn.Conv1d))
    info = {"label": a.label, "model": model_id, "device": dev, "dtype": a.dtype,
            "attn_implementation": a.attn_impl,
            "torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
            "load_s": load_s, "lora_r": a.lora_r, "lora_targets": targets,
            "conv_modules_not_adapted": skipped, "n_conv1d_modules": n_conv,
            "linear_module_names_present": present,
            "trainable_params": trainable, "total_params": total,
            "trainable_pct": round(100 * trainable / total, 4),
            "objective": "controller-SFT: loss on action completion only",
            "token_budget_per_update": a.token_budget,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "arms": []}
    print(f"[{a.label}] targets={targets} conv1d_modules={n_conv} (not adapted)", flush=True)
    print(f"[{a.label}] trainable {trainable:,}/{total:,} ({info['trainable_pct']}%)", flush=True)

    keep = 0 if a.no_restrict_logits else a.logits_keep

    for sl in [int(x) for x in a.seq_lens.split(",")]:
        micros = ([int(x) for x in a.microbatches.split(",") if x]
                  or [max(1, a.token_budget // sl)])
        data = build_examples(tok, sl, 64, a.max_action_tokens, seed=sl)
        shape = verify_corpus(data, sl, a.max_action_tokens)
        info.setdefault("corpus", {})[str(sl)] = shape
        print(f"[{a.label}] seq={sl} corpus ok: {shape['supervised_tokens_mean']:.1f} "
              f"supervised of {sl} ({100*shape['supervised_fraction']:.3f}%)", flush=True)

        if a.check_logits and keep:
            chk = check_logits_equivalence(model, dev, data, sl, keep)
            info.setdefault("logits_check", {})[str(sl)] = chk
            print(f"[{a.label}] seq={sl} logits check: equivalent={chk['equivalent']} "
                  f"dloss={chk['delta_loss']:.2e} dgrad_rel={chk['rel_grad_norm_delta']:.2e} "
                  f"mem full={chk['full']['peak_mem_mib']} restricted={chk['restricted']['peak_mem_mib']}",
                  flush=True)
            if not chk["equivalent"]:
                print(f"[{a.label}] REFUSING restricted logits: not equivalent", flush=True)
                keep = 0

        for mb in micros:
            accum = max(1, int(round(a.token_budget / (mb * sl))))
            try:
                r = run_arm(model, dev, data, sl, mb, accum, a.steps, a.warmup,
                            a.lr, keep, a.label)
                info["arms"].append(r)
                print(f"[{a.label}] seq={sl:<5} mb={mb:<2} ga={accum:<3} "
                      f"tok/upd={r['tokens_per_update']:<6} {r['tokens_per_s']:>8} tok/s  "
                      f"step {r['step_s_median']:.3f}s  fwd {r['fwd_s_median']:.3f} "
                      f"bwd {r['bwd_s_median']:.3f}  peak {r['peak_mem_mib']} MiB  "
                      f"{r['watts']} W", flush=True)
            except Exception as e:
                info["arms"].append({"seq_len": sl, "microbatch": mb,
                                     "err": f"{type(e).__name__}: {e}"})
                print(f"[{a.label}] seq={sl} mb={mb} FAILED: {type(e).__name__}: {e}", flush=True)
                if dev == "cuda":
                    torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(info, open(a.out, "w"), indent=2)
    print(f"[{a.label}] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
