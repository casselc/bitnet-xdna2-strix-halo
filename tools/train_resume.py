#!/usr/bin/env python3
"""A checkpoint that actually resumes: adapter + optimizer + scheduler + RNG.

`halo-training-smoke` saved adapter weights only, and said so: reloading gave a
fresh AdamW, so the first steps after a "resume" were not a continuation. This
fixes the HARNESS, not that branch's record.

The test is a three-way comparison, because a resume that merely runs is not a
resume that is correct:

  continuous   train N+M updates in one process, record every loss
  interrupted  train N updates, checkpoint, DESTROY THE PROCESS
  resumed      new process: reload, train M more

`resumed` must track the tail of `continuous`. The specific failure being
tested for is the fresh-optimizer spike: AdamW's first and second moments are
what smooth the first steps, and without them the loss jumps even though the
weights are correct. Comparing against a continuous run is what makes the spike
visible -- an isolated resumed loss curve looks fine on its own.

The adapter-only export is still produced, as a separate deployment artifact.
It is the right thing to ship; it is just not a training checkpoint.

Phases are separate processes ON PURPOSE. `save` and `resume` in one process
would share RNG and allocator state and could pass while a real restart fails.
"""
import argparse, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_scaling import (synth_corpus, resolve_targets, attach_lora,
                           load_model, peak_mem_mib)


def make_batches(tok, seq_len, n, seed):
    return synth_corpus(tok, seq_len, n, seed=seed)


def build_opt(model, lr, total_steps):
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    sch = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.1,
                                            total_iters=max(1, total_steps))
    return opt, sch


def one_update(model, data, idx, micro, accum, seq_len, dev, opt, sch):
    import torch
    tot = 0.0
    for g in range(accum):
        s = (idx * accum + g) * micro
        chunk = data[s:s + micro] or data[:micro]
        if len(chunk) < micro:
            chunk = data[:micro]
        ids = torch.tensor([c[:seq_len] for c in chunk], device=dev)
        lab = torch.tensor([c[1:seq_len + 1] for c in chunk], device=dev)
        out = model(input_ids=ids, labels=lab)
        (out.loss / accum).backward()
        tot += float(out.loss)
    opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
    if dev == "cuda":
        torch.cuda.synchronize()
    return tot / accum


def save_full(path, model, opt, sch, step, dev):
    """Everything needed to continue, not just what is needed to serve."""
    import torch
    os.makedirs(path, exist_ok=True)
    t0 = time.time()
    model.save_pretrained(os.path.join(path, "adapter"))     # deployment artifact
    state = {
        "optimizer": opt.state_dict(),
        "scheduler": sch.state_dict(),
        "step": step,
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state_all() if dev == "cuda" else None),
        },
    }
    torch.save(state, os.path.join(path, "train_state.pt"))
    write_s = time.time() - t0
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(path) for f in fs)
    adapter = sum(os.path.getsize(os.path.join(dp, f))
                  for dp, _, fs in os.walk(os.path.join(path, "adapter")) for f in fs)
    return {"ckpt_bytes": size, "adapter_bytes": adapter,
            "optimizer_bytes": size - adapter,
            "ckpt_write_s": round(write_s, 3)}


def load_full(path, model, opt, sch, dev):
    import torch
    from peft import PeftModel
    t0 = time.time()
    model = PeftModel.from_pretrained(model, os.path.join(path, "adapter"),
                                      is_trainable=True)
    st = torch.load(os.path.join(path, "train_state.pt"), weights_only=False)
    read_s = time.time() - t0
    return model, st, round(read_s, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("continuous", "save", "resume"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="/tmp/halo-train/ckpt")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--microbatch", type=int, default=1)
    ap.add_argument("--token-budget", type=int, default=4096)
    ap.add_argument("--steps-before", type=int, default=12)
    ap.add_argument("--steps-after", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    import torch
    dev = a.device
    random.seed(a.seed); torch.manual_seed(a.seed)
    accum = max(1, int(round(a.token_budget / (a.microbatch * a.seq_len))))
    total = a.steps_before + a.steps_after

    tok, base, load_s = load_model(a.model, a.dtype, dev, "eager")
    targets, skipped, _ = resolve_targets(base)
    data = make_batches(tok, a.seq_len, a.microbatch * accum * (total + 2), a.seed)

    rec = {"label": a.label, "model": a.model, "phase": a.phase, "device": dev,
           "seq_len": a.seq_len, "grad_accum": accum, "microbatch": a.microbatch,
           "tokens_per_update": a.microbatch * a.seq_len * accum,
           "lora_targets": targets, "seed": a.seed,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    if a.phase in ("continuous", "save"):
        model, trainable, tot_p = attach_lora(base, a.lora_r, a.lora_r * 2, targets)
        opt, sch = build_opt(model, a.lr, total)
        model.train()
        rec["trainable_params"] = trainable
        n = total if a.phase == "continuous" else a.steps_before
        losses = []
        for i in range(n):
            losses.append(one_update(model, data, i, a.microbatch, accum,
                                     a.seq_len, dev, opt, sch))
            print(f"  step {i:>3} loss {losses[-1]:.5f}", flush=True)
        rec["losses"] = [round(x, 6) for x in losses]
        rec["peak_mem_mib"] = peak_mem_mib()
        if a.phase == "save":
            rec.update(save_full(a.ckpt, model, opt, sch, a.steps_before, dev))
            print(f"  checkpoint {rec['ckpt_bytes']/2**20:.2f} MiB "
                  f"(adapter {rec['adapter_bytes']/2**20:.2f}, "
                  f"optimizer {rec['optimizer_bytes']/2**20:.2f}) "
                  f"in {rec['ckpt_write_s']}s", flush=True)
    else:
        model, st, read_s = load_full(a.ckpt, base, None, None, dev)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        opt, sch = build_opt(model, a.lr, total)
        opt.load_state_dict(st["optimizer"])
        sch.load_state_dict(st["scheduler"])
        # restoring RNG is what makes the resumed batches line up with the
        # continuous run; without it the comparison is against different data.
        random.setstate(st["rng"]["python"])
        torch.set_rng_state(st["rng"]["torch"])
        if dev == "cuda" and st["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(st["rng"]["cuda"])
        start = st["step"]
        model.train()
        rec.update(ckpt_read_s=read_s, resumed_from_step=start,
                   trainable_params=trainable,
                   optimizer_state_entries=len(st["optimizer"].get("state", {})))
        losses = []
        for i in range(a.steps_after):
            losses.append(one_update(model, data, start + i, a.microbatch, accum,
                                     a.seq_len, dev, opt, sch))
            print(f"  step {start+i:>3} loss {losses[-1]:.5f}", flush=True)
        rec["losses"] = [round(x, 6) for x in losses]
        rec["peak_mem_mib"] = peak_mem_mib()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    prev = []
    if os.path.exists(a.out):
        try:
            prev = json.load(open(a.out))
            if isinstance(prev, dict):
                prev = [prev]
        except Exception:
            prev = []
    prev.append(rec)
    json.dump(prev, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
