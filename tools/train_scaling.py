#!/usr/bin/env python3
"""Realistic LoRA training scaling on gfx1151, and a REAL checkpoint resume.

Ported in spirit from `tools/train_smoke.py` on the `halo-training-smoke`
branch (`f4c27323f819e7d62290ac88870cbb7ae42d7f0a`), which proved the stack
runs. It did not measure campaign throughput, and this is not a criticism of
it -- it says so itself. Three things are different here:

  SEQUENCE LENGTH IS A VARIABLE. The smoke test packed ~20 short examples into
  one fixed batch, so its 2819 tok/s at Qwen3-0.6B describes that batch and not
  a training campaign. Here seq len is swept and the TOKENS PER OPTIMIZER UPDATE
  are held roughly constant with microbatch x gradient accumulation, so the
  numbers across lengths are comparable.

  LoRA TARGETS FOLLOW THE ARCHITECTURE. q/k/v/o-only is not a meaningful
  adapter on a model that is mostly DeltaNet, short-conv or Mamba: on
  Qwen3.5 only 6 of 24 layers even HAVE q_proj. Targets are resolved by
  inspecting the module names actually present, and both the target list and
  the trainable-parameter count are recorded so the arms can be compared
  honestly rather than assumed equivalent.

  CHECKPOINTS CARRY OPTIMIZER STATE. The smoke test deliberately saved adapter
  weights only, exposing that AdamW moments were lost. Here the checkpoint holds
  adapter + optimizer + scheduler + step + RNG, the process is destroyed, and
  the resumed loss is compared against a continuous run -- a fresh-optimizer
  spike is the thing being tested for.

Not tuned, and no quality claim: this measures throughput and memory only.
"""
import argparse, json, math, os, random, sys, time

RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj"


def read_energy():
    try:
        return int(open(RAPL).read().strip())
    except Exception:
        return None


class PowerMeter:
    def __enter__(self):
        self.t0, self.e0 = time.time(), read_energy()
        return self

    def __exit__(self, *a):
        e1, dt = read_energy(), time.time() - self.t0
        self.watts = (round((e1 - self.e0) / 1e6 / dt, 1)
                      if self.e0 is not None and e1 is not None and e1 >= self.e0
                      and dt > 0 else None)


# ---------------------------------------------------------------- data

def synth_corpus(tok, seq_len, n_seq, seed=0):
    """Controller-shaped token sequences of EXACTLY seq_len.

    Built by tokenising structured text and then slicing to length, so every
    microbatch carries the same token count and a throughput comparison across
    sequence lengths is not secretly a comparison of padding.
    """
    rng = random.Random(seed)
    actions = ["HOLD", "SCALE", "ROLLBACK", "RESTART", "PAGE"]
    out = []
    for i in range(n_seq):
        parts = [f"CONTROLLER DOMAIN {i:04d}\nSTATE SPINE\n"]
        while True:
            parts.append(
                f"  svc{rng.randrange(999):03d}: p95={rng.randrange(10,900)}ms "
                f"err={rng.randrange(50)} cpu={rng.randrange(100)}% "
                f"q={rng.randrange(300)} dep=svc{rng.randrange(999):03d}\n")
            if len(parts) % 16 == 0:
                ids = tok("".join(parts), add_special_tokens=False)["input_ids"]
                if len(ids) >= seq_len + 1:
                    break
        parts.append(f"ACTION: {actions[i % len(actions)]}\n")
        ids = tok("".join(parts), add_special_tokens=False)["input_ids"]
        out.append(ids[:seq_len + 1])
    return out


# ------------------------------------------------- architecture-aware LoRA

# Names are matched as SUFFIXES of module paths. Grouped by family so the
# choice is visible in the artifact rather than buried in a heuristic.
CANDIDATE_TARGETS = [
    # attention (present in every family, but only in SOME layers of hybrids)
    "q_proj", "k_proj", "v_proj", "o_proj",
    # dense MLP, Llama/Qwen naming
    "gate_proj", "up_proj", "down_proj",
    # dense MLP, LFM2 SwiGLU naming. Omitting these is not a neutral choice:
    # it adapted Qwen3.5's MLP while leaving LFM2's untrained, which made a
    # throughput comparison between them a comparison of two different jobs.
    "w1", "w2", "w3",
    # Qwen3.5 / Qwen3-Next gated DeltaNet linear-attention blocks
    "in_proj_qkvz", "in_proj_ba", "out_proj",
    # LFM2 short-conv blocks / Nemotron-H / Mamba2
    "in_proj", "out_proj",
]


def resolve_targets(model, extra=None, exclude_conv=True):
    """Pick LoRA targets from the module names actually present.

    1-D convolutions are excluded by default: peft's Linear adapter does not
    apply to them, and silently dropping them from the target list would
    otherwise be invisible in the trainable-parameter count.
    """
    import torch.nn as nn
    present, kinds = set(), {}
    for name, mod in model.named_modules():
        leaf = name.split(".")[-1]
        if isinstance(mod, nn.Linear):
            present.add(leaf); kinds[leaf] = "Linear"
        elif isinstance(mod, nn.Conv1d):
            kinds.setdefault(leaf, "Conv1d")
    want = set(extra or CANDIDATE_TARGETS)
    tgt = sorted(present & want)
    skipped = sorted({k for k, v in kinds.items() if v == "Conv1d"}) if exclude_conv else []
    return tgt, skipped, sorted(present)


def attach_lora(model, r, alpha, targets):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                     task_type="CAUSAL_LM", target_modules=targets)
    m = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    return m, trainable, total


def load_model(model_id, dtype, device, attn_impl=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"dtype": getattr(torch, dtype), "trust_remote_code": True}
    if attn_impl:
        kw["attn_implementation"] = attn_impl
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).to(device)
    return tok, model, round(time.time() - t0, 1)


def peak_mem_mib():
    import torch
    return round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None


# ------------------------------------------------------------- one arm

def run_arm(model, tok, dev, seq_len, budget, micro, steps, warmup, lr, args):
    """One (seq_len) measurement at a fixed tokens-per-update budget."""
    import torch
    accum = max(1, int(round(budget / (micro * seq_len))))
    tokens_per_update = micro * seq_len * accum
    data = synth_corpus(tok, seq_len, micro * accum * (warmup + steps) + micro * accum,
                        seed=seq_len)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()

    def one_update(idx):
        fwd = bwd = 0.0
        tot = 0.0
        for g in range(accum):
            chunk = data[(idx * accum + g) * micro:(idx * accum + g + 1) * micro]
            if len(chunk) < micro:
                chunk = data[:micro]
            ids = torch.tensor([c[:seq_len] for c in chunk], device=dev)
            lab = torch.tensor([c[1:seq_len + 1] for c in chunk], device=dev)
            t0 = time.time()
            out = model(input_ids=ids, labels=lab)
            loss = out.loss / accum
            if dev == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            loss.backward()
            if dev == "cuda":
                torch.cuda.synchronize()
            t2 = time.time()
            fwd += t1 - t0; bwd += t2 - t1; tot += float(out.loss)
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
            times.append(time.time() - t0)
            fwds.append(f); bwds.append(b); losses.append(l)
    med = sorted(times)[len(times) // 2]
    return {
        "seq_len": seq_len, "microbatch": micro, "grad_accum": accum,
        "tokens_per_update": tokens_per_update, "steps_timed": steps,
        "step_s_median": round(med, 4),
        "tokens_per_s": round(tokens_per_update / med, 1),
        "fwd_s_median": round(sorted(fwds)[len(fwds) // 2], 4),
        "bwd_s_median": round(sorted(bwds)[len(bwds) // 2], 4),
        "peak_mem_mib": peak_mem_mib(), "watts": pw.watts,
        "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-lens", default="256,512,1024,2048")
    ap.add_argument("--token-budget", type=int, default=4096)
    ap.add_argument("--microbatch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--targets", default="")
    ap.add_argument("--attn-impl", default="eager")
    a = ap.parse_args()

    import torch
    dev = a.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("FAIL: no ROCm device visible to torch", file=sys.stderr)
        return 2

    tok, model, load_s = load_model(a.model, a.dtype, dev, a.attn_impl)
    extra = [x for x in a.targets.split(",") if x] or None
    targets, skipped_conv, present = resolve_targets(model, extra)
    model, trainable, total = attach_lora(model, a.lora_r, a.lora_alpha, targets)

    # A local checkpoint directory is passed as an absolute path, which carries
    # the operator's home directory into an artifact bound for a public repo.
    # Keep the leaf, which is what identifies the weights.
    model_id = a.model if not os.path.isabs(a.model) else os.path.basename(
        a.model.rstrip("/"))

    info = {
        "label": a.label, "model": model_id, "device": dev, "dtype": a.dtype,
        "torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
        "load_s": load_s, "lora_r": a.lora_r, "lora_alpha": a.lora_alpha,
        "lora_targets": targets, "conv_modules_not_adapted": skipped_conv,
        "linear_module_names_present": present,
        "trainable_params": trainable, "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
        "token_budget_per_update": a.token_budget,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arms": [],
    }
    print(f"[{a.label}] targets={targets} conv_skipped={skipped_conv}", flush=True)
    print(f"[{a.label}] trainable {trainable:,} / {total:,} "
          f"({info['trainable_pct']}%)", flush=True)

    for sl in [int(x) for x in a.seq_lens.split(",")]:
        try:
            r = run_arm(model, tok, dev, sl, a.token_budget, a.microbatch,
                        a.steps, a.warmup, a.lr, a)
            info["arms"].append(r)
            print(f"[{a.label}] seq={sl:<5} mb={r['microbatch']} ga={r['grad_accum']:<3} "
                  f"tok/upd={r['tokens_per_update']:<6} {r['tokens_per_s']:>8} tok/s  "
                  f"step {r['step_s_median']:.3f}s  peak {r['peak_mem_mib']} MiB  "
                  f"{r['watts']} W", flush=True)
        except Exception as e:
            info["arms"].append({"seq_len": sl, "err": f"{type(e).__name__}: {e}"})
            print(f"[{a.label}] seq={sl} FAILED: {type(e).__name__}: {e}", flush=True)
            if dev == "cuda":
                torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(info, open(a.out, "w"), indent=2)
    print(f"[{a.label}] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
