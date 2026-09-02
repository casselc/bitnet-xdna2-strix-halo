#!/usr/bin/env python3
"""Would persisting context checkpoints actually fix hybrid restore?

The upstream fix for #28194 serialises `slot.prompt.checkpoints` alongside the
sequence state. Reimplementing it is ~750 lines and the author's branch is not
public, so this asks the question the patch would answer, using only what the
stock server already does.

The insight is that a checkpoint only has to be PERSISTED if it works in the
first place. So drop save/restore entirely and exercise the checkpoint path
in-process:

    turn 1:  A_prefix + delta_1     (populates slot.prompt.checkpoints)
    turn 2:  A_prefix + delta_2     (shares the 1575-token prefix)

Turn 2 is exactly the situation a restored slot would be in if checkpoints had
survived: same sequence, same prefix, a checkpoint available. Whether it reuses,
and whether it is CORRECT when it does, decides the value of the patch:

  reuse + correct   -> persisting checkpoints is a real fix; the blocker is
                       purely that they are not serialised.
  reuse + wrong     -> the checkpoint mechanism is itself unsound for hybrid
                       memory, and persisting it would propagate the fault into
                       restored sessions. Fixing #28194 alone would not be safe.
  no reuse          -> checkpoints are not what gates hybrid reuse here, and the
                       issue's diagnosis does not transfer to this build.

Ground truth is a full recompute on a `-ctxcp 0` server, since checkpointing
perturbs hybrid state (see RESTORE.md 1) and a same-server reference would be
contaminated.
"""
import argparse, json, subprocess, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_domain import Domain

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "refs/BitNet/build-xdna/bin/llama-server"


def wait_health(port, timeout=300):
    for _ in range(timeout):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as f:
                if b"ok" in f.read():
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def post(port, path, payload, timeout=900):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                               data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read().decode())


class Server:
    def __init__(self, model, port, ctxcp, logdir):
        self.port = port
        Path(logdir).mkdir(parents=True, exist_ok=True)
        self.log = Path(logdir) / f"server_cp{ctxcp}.log"
        # --slot-save-path is required for /slots actions at all; without it
        # the erase endpoint returns HTTP 501 and the arms silently share state.
        sd = Path(logdir) / f"state_cp{ctxcp}"; sd.mkdir(parents=True, exist_ok=True)
        cmd = [str(BIN), "-m", str(model), "-t", "4", "-ngl", "0", "-c", "40960",
               "-np", "8", "-b", "4096", "-ub", "4096", "-tb", "16",
               "-ctxcp", str(ctxcp), "--slot-save-path", str(sd) + "/",
               "--host", "127.0.0.1", "--port", str(port), "--no-webui"]
        self.lf = open(self.log, "w")
        self.p = subprocess.Popen(cmd, stdout=self.lf, stderr=self.lf,
                                  stdin=subprocess.DEVNULL,
                                  env={"BITNET_XDNA": "0", "PATH": "/usr/bin:/bin"})

    def __enter__(self):
        if not wait_health(self.port):
            raise SystemExit(f"server on {self.port} did not become healthy")
        return self

    def __exit__(self, *a):
        self.p.terminate()
        try:
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()
        self.lf.close()


def ask(port, prompt, topk, n_predict=1, slot=0):
    r = post(port, "/completion",
             {"prompt": prompt, "n_predict": n_predict, "temperature": 0,
              "top_k": 1, "seed": 1234, "n_probs": topk,
              "cache_prompt": True, "id_slot": slot})
    tm = r.get("timings", {}) or {}
    cp = r.get("completion_probabilities") or []
    top = {}
    if cp:
        for e in (cp[0].get("top_logprobs") or []):
            top[e.get("id")] = float(e.get("logprob", 0.0))
    return {"content": r.get("content", ""), "top": top,
            "cache_n": tm.get("cache_n"), "prompt_n": tm.get("prompt_n"),
            "ttft_ms": tm.get("prompt_ms")}


def diff(ref, got):
    common = sorted(set(ref["top"]) & set(got["top"]))
    if not common:
        return None
    return max(abs(ref["top"][t] - got["top"][t]) for t in common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--spine-tokens", type=int, default=1600)
    ap.add_argument("--delta-tokens", type=int, default=135)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--workdir", default="/tmp/bitnet-ckptprobe")
    a = ap.parse_args()

    wd = Path(a.workdir) / a.label
    from model_bakeoff import calibrate_spine, calibrate_delta

    # calibrate + ground truth on a clean (-ctxcp 0) server
    with Server(a.model, a.port, 0, wd) as s:
        base = f"http://127.0.0.1:{s.port}"
        nt, ns, sp = calibrate_spine(base, a.spine_tokens)
        A = Domain(1, 0xC0FFEE, n_topo=nt, n_state=ns)
        dl, dn, _ = calibrate_delta(base, A, a.delta_tokens)
        p1, p2 = A.prompt(11, dl), A.prompt(12, dl)
        post(a.port, f"/slots/0?action=erase", {})
        ref = ask(a.port, p2, a.topk)
        # and the same two-turn sequence with checkpoints OFF, as the control
        post(a.port, f"/slots/0?action=erase", {})
        t1_cp0 = ask(a.port, p1, a.topk)
        t2_cp0 = ask(a.port, p2, a.topk)

    res = {"label": a.label, "model": Path(a.model).name, "spine_tokens": sp,
           "delta_tokens": dn, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "reference_ctxcp0_fullrecompute": {k: ref[k] for k in
                                              ("content", "cache_n", "prompt_n", "ttft_ms")}}
    print(f"[{a.label}] spine={sp} delta={dn}; reference cache_n={ref['cache_n']} "
          f"ttft={ref['ttft_ms']:.1f}", flush=True)

    res["ctxcp0_two_turn"] = {
        "turn2_cache_n": t2_cp0["cache_n"], "turn2_ttft_ms": t2_cp0["ttft_ms"],
        "turn2_max_dlogprob": diff(ref, t2_cp0),
        "turn2_content_same": t2_cp0["content"] == ref["content"]}
    c0 = res["ctxcp0_two_turn"]
    print(f"[{a.label}] ctxcp=0  turn2 cache_n={c0['turn2_cache_n']:<6} "
          f"ttft={c0['turn2_ttft_ms']:7.1f} max|d|={c0['turn2_max_dlogprob']}", flush=True)

    # the real question: same two turns with checkpoints ENABLED
    with Server(a.model, a.port, 32, wd) as s:
        post(a.port, f"/slots/0?action=erase", {})
        t1 = ask(a.port, p1, a.topk)
        t2 = ask(a.port, p2, a.topk)
    res["ctxcp32_two_turn"] = {
        "turn1_cache_n": t1["cache_n"], "turn1_ttft_ms": t1["ttft_ms"],
        "turn2_cache_n": t2["cache_n"], "turn2_ttft_ms": t2["ttft_ms"],
        "turn2_max_dlogprob": diff(ref, t2),
        "turn2_content_same": t2["content"] == ref["content"]}
    c32 = res["ctxcp32_two_turn"]
    print(f"[{a.label}] ctxcp=32 turn2 cache_n={c32['turn2_cache_n']:<6} "
          f"ttft={c32['turn2_ttft_ms']:7.1f} max|d|={c32['turn2_max_dlogprob']} "
          f"same={c32['turn2_content_same']}", flush=True)

    reused = (c32["turn2_cache_n"] or 0) >= sp * 0.9
    correct = (c32["turn2_max_dlogprob"] is not None
               and c32["turn2_max_dlogprob"] < 1e-6)
    res["in_process_checkpoint_reuse"] = bool(reused)
    res["in_process_checkpoint_correct"] = bool(correct)
    if reused and correct:
        v = ("PERSISTING CHECKPOINTS WOULD FIX RESTORE — the mechanism gives "
             "correct reuse in-process; only serialisation is missing")
    elif reused and not correct:
        v = ("PERSISTING CHECKPOINTS WOULD NOT BE SAFE — the mechanism reuses "
             "but is numerically wrong for this architecture, so fixing #28194 "
             "alone would carry the fault into restored sessions")
    else:
        v = ("CHECKPOINTS DO NOT GATE REUSE HERE — no in-process reuse either, "
             "so the issue's diagnosis does not fully transfer to this build")
    res["verdict"] = v
    print(f"[{a.label}] VERDICT: {v}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
