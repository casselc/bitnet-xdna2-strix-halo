#!/usr/bin/env python3
"""Prove the controller is actually warm before any measurement window opens.

`/health` returning ok does NOT mean the XDNA path is ready. The runtime expands
ternary weights to int8 and uploads them to device BOs lazily, on the first
prefill whose token dimension clears the offload threshold. A benchmark started
at /health therefore charges that one-time residency cost to its first cell,
which is exactly the kind of contamination that shows up as an inexplicable p95
or max after a service restart.

This issues real ~1954-token controller requests until dispatches are non-zero
and latency has stopped moving, and records the cold / first-warm / steady
progression -- the cold residency cost is useful operational information in its
own right, not just something to discard.

Exit non-zero if the NPU never dispatched, so a misconfigured run fails loudly
instead of quietly measuring the CPU.
"""
import argparse, json, sys, time, urllib.request

sys.path.insert(0, "tools")


def _post(port, path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), (time.perf_counter() - t0) * 1000


def _get(port, path, timeout=30):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def dispatches(port):
    """NPU dispatch counter, exposed by the patched server's /props if present.
    Falls back to None when the build does not export it."""
    try:
        p = _get(port, "/props")
    except Exception:
        return None
    for k in ("xdna_dispatches", "bitnet_xdna_dispatches"):
        if k in p:
            return int(p[k])
    return None


def warm(port, prompt, n_predict, max_iters, tol_frac, verbose=True):
    """Issue requests until two consecutive latencies agree within tol_frac."""
    hist, prev = [], None
    for i in range(max_iters):
        j, wall = _post(port, "/completion",
                        {"prompt": prompt, "n_predict": n_predict,
                         "temperature": 0, "cache_prompt": False})
        t = j.get("timings", {})
        rec = dict(iter=i, wall_ms=round(wall, 1),
                   prompt_ms=round(t.get("prompt_ms", 0), 1),
                   prompt_n=t.get("prompt_n"),
                   predicted_ms=round(t.get("predicted_ms", 0), 1))
        hist.append(rec)
        if verbose:
            print(f"    warmup {i}: wall {rec['wall_ms']:8.1f} ms  "
                  f"prompt {rec['prompt_ms']:8.1f} ms  n={rec['prompt_n']}", flush=True)
        cur = rec["prompt_ms"]
        if prev is not None and cur > 0 and abs(cur - prev) / max(prev, 1e-9) < tol_frac:
            return hist, True
        prev = cur
    return hist, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--tokens", type=int, default=1954,
                    help="approximate prompt length; must clear the offload threshold")
    ap.add_argument("--n-predict", type=int, default=4)
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--tol", type=float, default=0.05,
                    help="two consecutive prompt_ms within this fraction = steady")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--require-dispatch", action="store_true", default=True)
    a = ap.parse_args()

    from service_bench import controller_prompt   # one prompt builder, shared
    prompt = controller_prompt(a.tokens)

    d0 = dispatches(a.port)
    hist, steady = warm(a.port, prompt, a.n_predict, a.max_iters, a.tol)
    d1 = dispatches(a.port)

    cold = hist[0]["prompt_ms"] if hist else None
    first_warm = hist[1]["prompt_ms"] if len(hist) > 1 else None
    steady_ms = hist[-1]["prompt_ms"] if hist else None
    rec = dict(label=a.label, port=a.port, tokens=a.tokens, iters=len(hist),
               steady=steady, cold_prompt_ms=cold, first_warm_prompt_ms=first_warm,
               steady_prompt_ms=steady_ms,
               cold_penalty_ms=(round(cold - steady_ms, 1)
                                if cold is not None and steady_ms is not None else None),
               dispatches_before=d0, dispatches_after=d1, history=hist)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=2)

    print(f"  warmup {a.label}: cold {cold} ms -> steady {steady_ms} ms "
          f"(one-time cost {rec['cold_penalty_ms']} ms), steady={steady}")

    if d0 is not None and d1 is not None:
        moved = d1 - d0
        print(f"  NPU dispatches during warmup: {moved}")
        if a.require_dispatch and moved <= 0:
            print("  FAIL: the NPU never dispatched -- check -ub >= the offload "
                  "threshold and BITNET_XDNA=1", file=sys.stderr)
            return 2
    else:
        # Not fatal: this build may not export the counter over HTTP. The
        # benchmark's own ne11/lease instrumentation still proves engagement,
        # and claiming a verified dispatch here would be dishonest.
        print("  note: server does not expose a dispatch counter over HTTP; "
              "relying on ne11/lease instrumentation instead")
    if not steady:
        print(f"  WARNING: latency still moving after {a.max_iters} iterations",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
