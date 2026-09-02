#!/usr/bin/env python3
"""TASK 9 -- what GPU LoRA training costs the warm CPU controller, and vice versa.

The controller runs on CPU and training runs on the iGPU, but they share one
memory system, one power budget and one thermal envelope on this part, so
"different devices" is not an argument that they do not interfere. This
measures it.

Three arms, in this order, so the controller baseline is taken on a machine in
the same state the loaded arm will start from:

  controller alone   warm state-spine turns, no training running
  both               the same turns while a LoRA job trains at seq 1024
  training alone     already measured by train_scaling.py; referenced, not redone

Reporting rule taken from the mission and from this project's history: if the
loaded and unloaded numbers land on top of each other, that is reported as
"no material penalty resolved", NOT as a speedup and NOT as a percentage of a
near-zero interference. `gpu-cotenancy` was explicit that the TAIL is what
degrades, so p95 is reported beside p50 and neither is dropped.
"""
import argparse, json, os, statistics as st, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_domain import Domain
from model_bakeoff import (_req, calibrate_spine, calibrate_delta, turn, stats)


def read_energy():
    try:
        return int(open("/sys/class/powercap/intel-rapl:0/energy_uj").read().strip())
    except Exception:
        return None


def controller_arm(base, dom, dl, predict, turns, warmup, t0_turn=0):
    for i in range(warmup):
        turn(base, dom.prompt(t0_turn + i, dl), predict, cache=True)
    e0, w0 = read_energy(), time.time()
    rows = []
    for i in range(turns):
        r = turn(base, dom.prompt(t0_turn + warmup + i, dl), predict, cache=True)
        rows.append(r)
    e1, dt = read_energy(), time.time() - w0
    ok = [r for r in rows if not r["err"]]
    watts = (round((e1 - e0) / 1e6 / dt, 1)
             if e0 is not None and e1 is not None and e1 >= e0 and dt > 0 else None)
    return {
        "turns": len(rows), "errors": len(rows) - len(ok),
        "ttft_ms": stats([r["ttft_ms"] for r in ok]),
        "total_ms": stats([r["total_ms"] for r in ok]),
        "decode_tps": stats([r["predicted_per_second"] for r in ok]),
        "cache_n": stats([r["cache_n"] for r in ok]),
        "wall_s": round(dt, 1), "package_watts": watts,
        "req_per_s": round(len(ok) / dt, 3) if dt else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-model", required=True)
    ap.add_argument("--train-label", required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--predict", type=int, default=4)
    ap.add_argument("--train-steps", type=int, default=40)
    ap.add_argument("--train-out", default="/tmp/halo-train/coexist_train.json")
    ap.add_argument("--trainer", default="")
    ap.add_argument("--microbatch", type=int, default=1)
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}"
    nt, ns, sp = calibrate_spine(base, 1600)
    dom = Domain(0, 0xC0FFEE, n_topo=nt, n_state=ns)
    dl, dn, pn = calibrate_delta(base, dom, 135)
    res = {"controller_port": a.port, "spine_tokens": sp, "delta_tokens": dn,
           "train_model": a.train_model, "train_seq_len": a.seq_len,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(f"spine={sp} delta={dn}", flush=True)

    # ---- arm 1: controller alone
    res["controller_alone"] = controller_arm(base, dom, dl, a.predict, a.turns, a.warmup, 0)
    c = res["controller_alone"]
    print(f"ALONE     ttft p50={c['ttft_ms']['p50']:.1f} p95={c['ttft_ms']['p95']:.1f} "
          f"total p50={c['total_ms']['p50']:.1f} {c['package_watts']} W", flush=True)

    # ---- arm 2: controller while the GPU trains
    root = Path(__file__).resolve().parent.parent
    # Supersedes the full-sequence LM workload this originally drove. The
    # controller-SFT objective is what a real campaign runs, and it has a
    # different memory and utilisation profile, so the earlier +48% figure
    # cannot simply be assumed to carry over.
    trainer = a.trainer or str(root / "tools/train_controller_sft.py")
    cmd = [str(root / "tools/halo_rocm_env.sh"), "exec",
           str(root / ".venv-train/bin/python"), trainer,
           "--model", a.train_model, "--label", a.train_label,
           "--out", a.train_out, "--seq-lens", str(a.seq_len),
           "--microbatches", str(a.microbatch),
           "--steps", str(a.train_steps), "--warmup", "2"]
    env = dict(os.environ, HALO_PYTHON=str(root / ".venv-train/bin/python"))
    print(f"starting training: {a.train_model} seq={a.seq_len}", flush=True)
    proc = subprocess.Popen(cmd, cwd=str(root), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Wait for the job to actually be ON the GPU. Measuring during model load
    # would credit the controller with an idle-GPU window and understate the
    # interference we are trying to find.
    t_wait = time.time()
    while time.time() - t_wait < 600:
        if proc.poll() is not None:
            break
        try:
            import torch  # noqa: F401
        except Exception:
            pass
        # cheap proxy: the training process has allocated device memory
        out = subprocess.run(["sg", "render", "-c", "rocm-smi --showmemuse"],
                             capture_output=True, text=True)
        if "VRAM%" in out.stdout:
            pct = [l for l in out.stdout.splitlines() if "VRAM%" in l]
            try:
                if pct and int(pct[0].split(":")[-1].strip()) > 5:
                    break
            except Exception:
                pass
        time.sleep(5)
    time.sleep(20)   # let it settle into steady-state stepping
    res["train_running_at_measure"] = proc.poll() is None
    print(f"training on GPU (running={res['train_running_at_measure']}); "
          f"measuring controller under load", flush=True)

    res["controller_loaded"] = controller_arm(base, dom, dl, a.predict, a.turns,
                                              a.warmup, 5000)
    l = res["controller_loaded"]
    print(f"LOADED    ttft p50={l['ttft_ms']['p50']:.1f} p95={l['ttft_ms']['p95']:.1f} "
          f"total p50={l['total_ms']['p50']:.1f} {l['package_watts']} W", flush=True)
    res["still_training_after"] = proc.poll() is None

    try:
        proc.wait(timeout=1800)
    except Exception:
        proc.kill()
    if os.path.exists(a.train_out):
        try:
            res["training_under_load"] = json.load(open(a.train_out))
        except Exception:
            pass

    # Task 16 asks for controller-alone AFTER as well, so a drift during the
    # window cannot be mistaken for interference.
    res["controller_alone_after"] = controller_arm(base, dom, dl, a.predict, a.turns,
                                                   a.warmup, 9000)
    aft = res["controller_alone_after"]
    print(f"ALONE-AFTER ttft p50={aft['ttft_ms']['p50']:.1f} p95={aft['ttft_ms']['p95']:.1f} "
          f"total p50={aft['total_ms']['p50']:.1f} {aft['package_watts']} W", flush=True)

    def d(k, stat="p50"):
        x, y = c[k][stat], l[k][stat]
        return round(100.0 * (y - x) / x, 1) if x else None
    res["delta_pct"] = {"ttft_p50": d("ttft_ms"), "ttft_p95": d("ttft_ms", "p95"),
                        "total_p50": d("total_ms"), "total_p95": d("total_ms", "p95"),
                        "req_per_s": (round(100.0 * (l["req_per_s"] - c["req_per_s"])
                                            / c["req_per_s"], 1)
                                      if c.get("req_per_s") else None)}
    print(f"DELTA     ttft p50 {res['delta_pct']['ttft_p50']:+}%  "
          f"p95 {res['delta_pct']['ttft_p95']:+}%  "
          f"total p50 {res['delta_pct']['total_p50']:+}%", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
