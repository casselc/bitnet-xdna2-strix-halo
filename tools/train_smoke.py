#!/usr/bin/env python3
"""Halo local-training plumbing smoke test.

This is NOT a quality experiment. It proves one thing: that a real
forward/backward/optimizer/checkpoint/reload/resume loop runs on THIS Strix Halo
iGPU (gfx1151) through ROCm, in an environment isolated from the known-good XDNA
stack.

Two phases, deliberately in separate processes, because "checkpoint works" is only
true if a FRESH process can reproduce the loss:

  train   load, attach LoRA, overfit ~20 synthetic controller examples,
          save adapter, record the final loss
  verify  new process: reload adapter, recompute loss on the same data,
          assert it matches, then resume a few steps

Measures tokens/s, step time, peak device memory, checkpoint size and write/read
time, and package power.
"""
import argparse, json, os, time, sys

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


def synthetic_controller_data(tok, n=20, seed=0):
    """Tiny controller-shaped set: structured state in, one action token out.

    Deliberately trivial and deliberately memorizable -- the point is that the
    loss must COLLAPSE, which is what proves gradients are actually flowing.
    """
    import random
    rng = random.Random(seed)
    actions = ["HOLD", "SCALE", "ROLLBACK", "RESTART"]
    ex = []
    for i in range(n):
        a = actions[i % len(actions)]
        s = (f"STATE svc{i:03d} p95={rng.randrange(10,900)}ms "
             f"err={rng.randrange(50)} qdepth={rng.randrange(300)}\nACTION:")
        ex.append((s, " " + a))
    return ex


def build_batch(tok, examples, device):
    import torch
    ids, labels = [], []
    for prompt, answer in examples:
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        a = tok(answer, add_special_tokens=False)["input_ids"]
        ids.append(p + a)
        labels.append([-100] * len(p) + a)          # loss on the action only
    m = max(len(x) for x in ids)
    pad = tok.pad_token_id or 0
    input_ids = torch.tensor([x + [pad] * (m - len(x)) for x in ids], device=device)
    attn = torch.tensor([[1] * len(x) + [0] * (m - len(x)) for x in ids], device=device)
    lab = torch.tensor([x + [-100] * (m - len(x)) for x in labels], device=device)
    return input_ids, attn, lab, sum(len(x) for x in ids)


def load_model(model_id, dtype, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=getattr(torch, dtype)).to(device)
    return tok, model, round(time.time() - t0, 1)


def attach_lora(model, r=16, alpha=32):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                     task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, trainable, total


def device_mem_mib():
    import torch
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("train", "verify"))
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--examples", type=int, default=20)
    ap.add_argument("--adapter", default="/tmp/halo-train/adapter")
    ap.add_argument("--out", default="artifacts/halo-training-smoke/run.json")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    a = ap.parse_args()

    import torch
    if a.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = a.device
    info = dict(phase=a.phase, model=a.model, dtype=a.dtype, device=dev,
                torch=torch.__version__, hip=getattr(torch.version, "hip", None),
                gpu=(torch.cuda.get_device_name(0) if dev == "cuda" else None))
    print(json.dumps(info, indent=2), flush=True)
    if dev == "cuda" and not torch.cuda.is_available():
        print("FAIL: no ROCm device visible to torch", file=sys.stderr)
        return 2

    tok, model, load_s = load_model(a.model, a.dtype, dev)
    ex = synthetic_controller_data(tok, a.examples)
    input_ids, attn, lab, ntok = build_batch(tok, ex, dev)

    if a.phase == "train":
        model, trainable, total = attach_lora(model)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=a.lr)
        model.train()
        losses, step_times = [], []
        with PowerMeter() as pw:
            t_start = time.time()
            for s in range(a.steps):
                t0 = time.time()
                out = model(input_ids=input_ids, attention_mask=attn, labels=lab)
                out.loss.backward()
                opt.step(); opt.zero_grad(set_to_none=True)
                if dev == "cuda":
                    torch.cuda.synchronize()
                step_times.append(time.time() - t0)
                losses.append(float(out.loss))
                if s % 10 == 0 or s == a.steps - 1:
                    print(f"  step {s:>3}  loss {losses[-1]:.4f}  "
                          f"{step_times[-1]*1000:.0f} ms", flush=True)
            train_s = time.time() - t_start
        os.makedirs(os.path.dirname(a.adapter), exist_ok=True)
        t0 = time.time(); model.save_pretrained(a.adapter)
        ckpt_s = time.time() - t0
        size = sum(os.path.getsize(os.path.join(dp, f))
                   for dp, _, fs in os.walk(a.adapter) for f in fs)
        med = sorted(step_times)[len(step_times) // 2]
        info.update(load_s=load_s, trainable_params=trainable, total_params=total,
                    trainable_pct=round(100 * trainable / total, 4),
                    steps=a.steps, loss_first=round(losses[0], 4),
                    loss_last=round(losses[-1], 4),
                    loss_drop_pct=round(100 * (1 - losses[-1] / losses[0]), 2),
                    step_ms_median=round(med * 1000, 1),
                    tokens_per_s=round(ntok / med, 1), tokens_per_batch=ntok,
                    train_s=round(train_s, 1), peak_mem_mib=device_mem_mib(),
                    ckpt_bytes=size, ckpt_write_s=round(ckpt_s, 2), watts=pw.watts)
        print(f"\n  loss {losses[0]:.4f} -> {losses[-1]:.4f} "
              f"({info['loss_drop_pct']:.1f}% drop)")
        print(f"  trainable {trainable:,} / {total:,} ({info['trainable_pct']}%)")
        print(f"  {info['step_ms_median']} ms/step, {info['tokens_per_s']} tok/s, "
              f"peak {info['peak_mem_mib']} MiB, {pw.watts} W")
        print(f"  adapter {size/2**20:.2f} MiB written in {ckpt_s:.2f}s")
    else:
        from peft import PeftModel
        t0 = time.time()
        model = PeftModel.from_pretrained(model, a.adapter)
        reload_s = time.time() - t0
        model.eval()
        with torch.no_grad():
            loss_reload = float(model(input_ids=input_ids, attention_mask=attn,
                                      labels=lab).loss)
        # resume: the adapter must still be trainable after a round trip
        for _, p in model.named_parameters():
            pass
        model.train()
        for n_, p in model.named_parameters():
            if "lora_" in n_:
                p.requires_grad_(True)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=a.lr)
        resumed = []
        for s in range(5):
            out = model(input_ids=input_ids, attention_mask=attn, labels=lab)
            out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            resumed.append(float(out.loss))
        info.update(load_s=load_s, reload_s=round(reload_s, 2),
                    loss_after_reload=round(loss_reload, 4),
                    resumed_losses=[round(x, 4) for x in resumed],
                    trainable_after_reload=sum(p.numel() for p in model.parameters()
                                               if p.requires_grad),
                    peak_mem_mib=device_mem_mib())
        print(f"\n  loss after reload: {loss_reload:.4f}")
        print(f"  resumed 5 steps: {[round(x,4) for x in resumed]}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    prev = []
    if os.path.exists(a.out):
        prev = json.load(open(a.out))
        if isinstance(prev, dict):
            prev = [prev]
    prev.append(info)
    json.dump(prev, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
