#!/usr/bin/env python3
"""Does enabling server context checkpoints change a CLEAN single request?

This is the control that decides how to read every restore-fidelity number on
`model-candidate-halo`. Those probes compared a restored state against a
"full-recompute reference" taken from the same server, with context checkpoints
at their default (`-ctxcp 32`). If checkpointing perturbs the reference itself,
the differences reported there were between two perturbed states rather than
between a correct one and a corrupted one.

Protocol, deliberately minimal so nothing else can explain a difference:

    fresh server process
    ONE request
    greedy, fixed seed
    no slot reuse, no restore, no prior traffic

run twice per model, once with `-ctxcp 32` and once with `-ctxcp 0`.

A pure-attention model is the control. If it is insensitive while the hybrids
are not, the effect is specific to recurrent/hybrid memory and the reference in
the earlier pass cannot be trusted.
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


def one_run(model, port, ctxcp, prompt, n_predict, topk, logdir):
    log = Path(logdir) / f"ctxcp{ctxcp}.log"
    sd = Path(logdir) / f"state{ctxcp}"; sd.mkdir(parents=True, exist_ok=True)
    cmd = [str(BIN), "-m", str(model), "-t", "4", "-ngl", "0", "-c", "40960",
           "-np", "8", "-b", "4096", "-ub", "4096", "-tb", "16",
           "-ctxcp", str(ctxcp), "--slot-save-path", str(sd) + "/",
           "--host", "127.0.0.1", "--port", str(port), "--no-webui"]
    env = {"BITNET_XDNA": "0", "PATH": "/usr/bin:/bin"}
    with open(log, "w") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=lf, stdin=subprocess.DEVNULL, env=env)
    try:
        if not wait_health(port):
            return {"err": "server did not become healthy", "log": str(log)}
        r = post(port, "/completion",
                 {"prompt": prompt, "n_predict": n_predict, "temperature": 0,
                  "top_k": 1, "seed": 1234, "n_probs": topk,
                  "cache_prompt": True, "id_slot": 0})
        tm = r.get("timings", {}) or {}
        cp = r.get("completion_probabilities") or []
        top = {}
        if cp:
            for e in (cp[0].get("top_logprobs") or []):
                top[e.get("id")] = float(e.get("logprob", 0.0))
        return {"content": r.get("content", ""), "cache_n": tm.get("cache_n"),
                "prompt_n": tm.get("prompt_n"), "top": top}
    finally:
        p.terminate()
        try:
            p.wait(timeout=30)
        except Exception:
            p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--spine-tokens", type=int, default=1600)
    ap.add_argument("--delta-tokens", type=int, default=135)
    ap.add_argument("--n-predict", type=int, default=8)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--workdir", default="/tmp/bitnet-ctxcp")
    a = ap.parse_args()

    wd = Path(a.workdir) / a.label
    wd.mkdir(parents=True, exist_ok=True)

    # Calibrate against a throwaway instance so both measured runs see the same
    # prompt text and neither is charged for calibration traffic.
    cal_log = wd / "calib"
    cal_log.mkdir(exist_ok=True)
    cmd = [str(BIN), "-m", a.model, "-t", "4", "-ngl", "0", "-c", "40960", "-np", "8",
           "-b", "4096", "-ub", "4096", "-tb", "16", "--host", "127.0.0.1",
           "--port", str(a.port + 1), "--no-webui"]
    with open(cal_log / "server.log", "w") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=lf, stdin=subprocess.DEVNULL,
                             env={"BITNET_XDNA": "0", "PATH": "/usr/bin:/bin"})
    try:
        if not wait_health(a.port + 1):
            print(f"[{a.label}] calibration server failed", file=sys.stderr)
            return 2
        from model_bakeoff import calibrate_spine, calibrate_delta
        import model_bakeoff as mb
        base = f"http://127.0.0.1:{a.port + 1}"
        nt, ns, sp = calibrate_spine(base, a.spine_tokens)
        A = Domain(1, 0xC0FFEE, n_topo=nt, n_state=ns)
        dl, dn, _ = calibrate_delta(base, A, a.delta_tokens)
    finally:
        p.terminate()
        try:
            p.wait(timeout=30)
        except Exception:
            p.kill()
    prompt = A.prompt(7, dl)
    print(f"[{a.label}] spine={sp} delta={dn}", flush=True)

    res = {"label": a.label, "model": Path(a.model).name, "spine_tokens": sp,
           "delta_tokens": dn, "n_predict": a.n_predict,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": {}}
    for ctxcp in (32, 0):
        r = one_run(a.model, a.port, ctxcp, prompt, a.n_predict, a.topk, wd)
        res["runs"][str(ctxcp)] = {k: v for k, v in r.items() if k != "top"}
        res["runs"][str(ctxcp)]["topk_size"] = len(r.get("top") or {})
        res["runs"][str(ctxcp)]["_top"] = r.get("top")
        print(f"[{a.label}] ctxcp={ctxcp:<3} cache_n={r.get('cache_n')} "
              f"prompt_n={r.get('prompt_n')} content={r.get('content')!r}", flush=True)

    t32 = res["runs"]["32"].pop("_top", None) or {}
    t0 = res["runs"]["0"].pop("_top", None) or {}
    common = sorted(set(t32) & set(t0))
    same = res["runs"]["32"].get("content") == res["runs"]["0"].get("content")
    res["identical_output"] = same
    res["max_abs_dlogprob"] = (max(abs(t32[t] - t0[t]) for t in common) if common else None)
    res["n_common_topk"] = len(common)
    res["verdict"] = ("INSENSITIVE to context checkpoints" if same and
                      (res["max_abs_dlogprob"] or 0) < 1e-6
                      else "SENSITIVE to context checkpoints")
    print(f"[{a.label}] {res['verdict']}  identical_output={same}  "
          f"max|dlogprob|={res['max_abs_dlogprob']}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
