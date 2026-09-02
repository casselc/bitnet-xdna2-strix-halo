#!/usr/bin/env python3
"""Does a smaller micro-batch unlock hybrid prefix reuse, and is it CORRECT?

A previous pass on this project reported a large hybrid latency win and later
had to withdraw it, because the speed came from reusing state that was not
right. So this tool refuses to report a reuse number without the matching
correctness number, measured against a full recompute.

The lever is `-ub`. Hybrid reuse is gated at `server-context.cpp:3252` by
`pos_min >= pos_min_thold`, where `pos_min` is the earliest position the memory
can still represent. For recurrent memory that is bounded by the last
micro-batch boundary, so a SMALLER `-ub` leaves a nearer roll-back point and
more of the prefix stays reusable. Notably this is not the context-checkpoint
path -- the server creates zero checkpoints in these runs.

Reference is a full recompute on a `-ctxcp 0` server, because context
checkpointing perturbs hybrid state (RESTORE.md 1) and a same-server reference
would be contaminated.

Reported per `-ub`: reused tokens, TTFT, and max |delta logprob| over the top-k
against that reference. A configuration is only useful if BOTH improve.
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
    def __init__(self, model, port, ub, ctxcp, logdir):
        self.port = port
        d = Path(logdir); d.mkdir(parents=True, exist_ok=True)
        sd = d / f"state_ub{ub}_cp{ctxcp}"; sd.mkdir(parents=True, exist_ok=True)
        self.log = d / f"server_ub{ub}_cp{ctxcp}.log"
        cmd = [str(BIN), "-m", str(model), "-t", "4", "-ngl", "0", "-c", "40960",
               "-np", "8", "-b", "4096", "-ub", str(ub), "-tb", "16",
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
            "ttft_ms": tm.get("prompt_ms"),
            "total_ms": (tm.get("prompt_ms") or 0) + (tm.get("predicted_ms") or 0)}


def maxdiff(ref, got):
    common = sorted(set(ref["top"]) & set(got["top"]))
    if not common:
        return None
    return max(abs(ref["top"][t] - got["top"][t]) for t in common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ubatches", default="4096,1024,512,256,128")
    ap.add_argument("--ctxcp", type=int, default=0)
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--spine-tokens", type=int, default=1600)
    ap.add_argument("--delta-tokens", type=int, default=135)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--decision-tokens", type=int, default=8)
    ap.add_argument("--workdir", default="/tmp/bitnet-ubprobe")
    a = ap.parse_args()

    wd = Path(a.workdir) / a.label
    from model_bakeoff import calibrate_spine, calibrate_delta

    # reference: full recompute, checkpoints off, largest ubatch
    with Server(a.model, a.port, 4096, 0, wd) as s:
        base = f"http://127.0.0.1:{s.port}"
        nt, ns, sp = calibrate_spine(base, a.spine_tokens)
        A = Domain(1, 0xC0FFEE, n_topo=nt, n_state=ns)
        dl, dn, _ = calibrate_delta(base, A, a.delta_tokens)
        p1, p2 = A.prompt(11, dl), A.prompt(12, dl)
        post(a.port, "/slots/0?action=erase", {})
        ref = ask(a.port, p2, a.topk)
        post(a.port, "/slots/0?action=erase", {})
        ref_dec = post(a.port, "/completion",
                       {"prompt": p2, "n_predict": a.decision_tokens, "temperature": 0,
                        "top_k": 1, "seed": 1234, "cache_prompt": True,
                        "id_slot": 0}).get("content", "")

    res = {"label": a.label, "model": Path(a.model).name, "spine_tokens": sp,
           "delta_tokens": dn, "ctxcp": a.ctxcp,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "reference": {"cache_n": ref["cache_n"], "prompt_n": ref["prompt_n"],
                         "ttft_ms": ref["ttft_ms"], "decision": ref_dec},
           "arms": []}
    print(f"[{a.label}] spine={sp} delta={dn} reference ttft={ref['ttft_ms']:.0f}ms "
          f"decision={ref_dec!r}", flush=True)

    for ub in [int(x) for x in a.ubatches.split(",")]:
        with Server(a.model, a.port, ub, a.ctxcp, wd) as s:
            post(a.port, "/slots/0?action=erase", {})
            t1 = ask(a.port, p1, a.topk)
            t2 = ask(a.port, p2, a.topk)
            dec = post(a.port, "/completion",
                       {"prompt": p2, "n_predict": a.decision_tokens, "temperature": 0,
                        "top_k": 1, "seed": 1234, "cache_prompt": True,
                        "id_slot": 0}).get("content", "")
            ncp = 0
            try:
                ncp = sum(1 for l in open(s.log, errors="replace")
                          if "created context checkpoint" in l)
            except Exception:
                pass
        d = maxdiff(ref, t2)
        arm = {"ubatch": ub, "turn1_cache_n": t1["cache_n"], "turn1_ttft_ms": t1["ttft_ms"],
               "turn2_cache_n": t2["cache_n"], "turn2_ttft_ms": t2["ttft_ms"],
               "turn2_total_ms": t2["total_ms"], "max_abs_dlogprob": d,
               "next_token_same": t2["content"] == ref["content"],
               "decision": dec, "decision_same": dec == ref_dec,
               "checkpoints_created": ncp,
               "reuse_fraction": round((t2["cache_n"] or 0) / max(1, sp), 3),
               # Correctness is judged on the logprob distribution of a single
               # controlled request. `decision_same` is recorded but NOT gated
               # on: the reference decision is generated from a clean slot while
               # the arm's follows two prior turns, so the two continuations
               # start from different cache states and can differ even when the
               # next-token distribution is bit-identical (observed on the
               # pure-attention control at 0.00000 divergence).
               "correct": bool(d is not None and d < 1e-6)}
        res["arms"].append(arm)
        print(f"[{a.label}] ub={ub:<5} turn2 cache_n={arm['turn2_cache_n']:<6} "
              f"ttft={arm['turn2_ttft_ms']:7.1f}ms  max|d|="
              f"{('n/a' if d is None else format(d,'.5f')):9s} decision_same={arm['decision_same']!s:5s} "
              f"ckpts={ncp}  CORRECT={arm['correct']}", flush=True)

    ok = [x for x in res["arms"] if x["correct"] and (x["turn2_cache_n"] or 0) > sp * 0.5]
    res["note_decision_same"] = ("recorded, not gated: reference decision starts "
                                 "from a clean slot, arm decision follows two turns")
    res["best_correct_reuse"] = (min(ok, key=lambda x: x["turn2_ttft_ms"]) if ok else None)
    res["verdict"] = ("SMALLER UBATCH GIVES CORRECT HYBRID REUSE" if ok else
                      "NO UBATCH SETTING GIVES BOTH REUSE AND CORRECTNESS")
    print(f"[{a.label}] VERDICT: {res['verdict']}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
