#!/usr/bin/env python3
"""Prove the controller is warm before any measurement window opens.

`/health` returning ok does not mean the XDNA path is ready: the runtime expands
ternary weights to int8 and uploads them to device buffers lazily, on the first
prefill that clears the offload threshold. A benchmark started at /health charges
that one-time cost to its first cell, which shows up as an inexplicable max or
p95 after a restart.

Issues real ~1954-token controller requests until two consecutive prefills agree
within a tolerance, and records the cold/steady progression -- the cold cost is
useful operational information, not just something to discard.
"""
import argparse, json, sys, time, urllib.request

sys.path.insert(0, "tools")


def post(port, path, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--tokens", type=int, default=1954)
    ap.add_argument("--n-predict", type=int, default=4)
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from service_bench import controller_prompt        # one shared prompt builder
    prompt = controller_prompt(a.tokens)

    hist, prev, steady = [], None, False
    for i in range(a.max_iters):
        j, wall = post(a.port, "/completion",
                       {"prompt": prompt, "n_predict": a.n_predict,
                        "temperature": 0, "cache_prompt": False})
        t = j.get("timings", {})
        rec = dict(iter=i, wall_ms=round(wall, 1),
                   prompt_ms=round(t.get("prompt_ms", 0), 1),
                   prompt_n=t.get("prompt_n"))
        hist.append(rec)
        print(f"    warmup {i}: wall {rec['wall_ms']:8.1f} ms  "
              f"prompt {rec['prompt_ms']:8.1f} ms  n={rec['prompt_n']}", flush=True)
        cur = rec["prompt_ms"]
        if prev is not None and cur > 0 and abs(cur - prev) / max(prev, 1e-9) < a.tol:
            steady = True
            break
        prev = cur

    cold = hist[0]["prompt_ms"] if hist else None
    last = hist[-1]["prompt_ms"] if hist else None
    out = dict(label=a.label, port=a.port, tokens=a.tokens, iters=len(hist),
               steady=steady, cold_prompt_ms=cold, steady_prompt_ms=last,
               cold_penalty_ms=(round(cold - last, 1)
                                if cold is not None and last is not None else None),
               history=hist)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
    print(f"  warmup {a.label}: cold {cold} ms -> steady {last} ms "
          f"(one-time {out['cold_penalty_ms']} ms), steady={steady}")
    if not steady:
        print(f"  WARNING: prefill still moving after {a.max_iters} iterations",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
