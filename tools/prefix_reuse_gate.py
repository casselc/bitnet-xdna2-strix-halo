#!/usr/bin/env python3
"""Task 9 -- what does REALISTIC prefix reuse buy the controller?

Deliberately not the easy version. Turning cache_prompt on for a prompt that is
byte-identical every time measures deduplication, not prefix reuse, and would
overstate the result badly.

Instead each request is built as:

    stable prefix   -- controller instructions, action schema, fixed topology
    volatile suffix -- genuinely different current state, DISTINCT PER REQUEST

at controlled reuse fractions (0 / 50 / 75 / 90%), with total length held near
1954 tokens so prefill work is comparable across arms.

The interesting threshold is the runtime's own: XDNA declines below 1024
evaluated tokens. At 90% reuse the uncached suffix is ~195 tokens, well under
it, so the NPU should drop out on its own. That would be a natural split --
cold/large-context requests take the CPU+NPU hybrid path, warm/small-delta
requests take CPU-only -- and it is measured here rather than forced.
"""
import argparse, random, statistics as st, subprocess, sys, time

sys.path.insert(0, "tools")
from service_bench import (run_controller, write_rows, pct, Ne11Window,
                           LeaseWindow, Power, CTRL, post)

NE11 = "/tmp/bitnet-service/ne11.csv"
LEASE = "/tmp/bitnet-service/lease.csv"

STABLE_HEAD = (
    "CONTROLLER INSTRUCTIONS\n"
    "You are a service controller. Read the system state and emit exactly one\n"
    "action from the schema below. Do not explain. Do not emit prose.\n\n"
    "ACTION SCHEMA\n"
    "  RESTART <service>   -- restart a single degraded service\n"
    "  SCALE <service> <n> -- change replica count\n"
    "  ROLLBACK <service>  -- revert to the previous known-good release\n"
    "  WAIT                -- take no action this tick\n\n"
)


def tokens_of(text, port=8081):
    d = post(f"http://127.0.0.1:{port}", "/tokenize", {"content": text})
    return len(d["tokens"])


def build(stable_lines, volatile_lines, seed):
    """Stable material is identical across requests; volatile material is not."""
    rng = random.Random(seed)
    s = [STABLE_HEAD, "FIXED TOPOLOGY\n"]
    for i in range(stable_lines):
        s.append(f"- svc{i:03d}: region=r{i%7} tier={i%4} "
                 f"deps=[svc{(i*3)%97:03d},svc{(i*7)%97:03d}] budget={80+(i*31)%400}ms\n")
    v = ["\nCURRENT STATE\n"]
    for i in range(volatile_lines):
        v.append(f"- svc{i:03d}: p95={rng.randint(10, 900)}ms err={rng.randint(0,40)} "
                 f"qdepth={rng.randint(0,500)} rps={rng.randint(1,9000)}\n")
    v.append("\nACTION:")
    return "".join(s), "".join(v)


def calibrate(target_tokens, reuse_frac, port):
    """Choose line counts so total ~= target and stable/total ~= reuse_frac."""
    # One probe of each line type, then solve; verified by re-tokenizing.
    s1, v1 = build(40, 40, 0)
    head_t = tokens_of(STABLE_HEAD, port)
    per_stable = (tokens_of(s1, port) - head_t) / 40.0
    per_vol = (tokens_of(v1, port) - tokens_of("\nCURRENT STATE\n\nACTION:", port)) / 40.0
    want_stable_t = target_tokens * reuse_frac
    want_vol_t = target_tokens * (1 - reuse_frac)
    sl = max(0, int(round((want_stable_t - head_t) / per_stable)))
    vl = max(1, int(round(want_vol_t / per_vol)))
    return sl, vl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--tokens", type=int, default=1954)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-predict", type=int, default=8)
    ap.add_argument("--fracs", default="0,0.5,0.75,0.9")
    ap.add_argument("--out", default="artifacts/service-batching-gate/prefix_reuse.csv")
    a = ap.parse_args()

    rows = []
    for frac in [float(x) for x in a.fracs.split(",")]:
        sl, vl = calibrate(a.tokens, frac, 8081)
        s, v = build(sl, vl, 0)
        total = tokens_of(s + v, 8081)
        stable_t = tokens_of(s, 8081)
        print(f"\n== reuse {frac:.0%}: stable {stable_t} tok + volatile "
              f"{total-stable_t} tok = {total} total", flush=True)
        for arm_idx, cache in enumerate((False, True)):
            # DISJOINT seeds per arm. Reusing the same seeds meant the
            # cache=True arm re-sent the exact prompts the cache=False arm had
            # just warmed, so it measured DEDUPLICATION of an identical prompt
            # rather than prefix reuse -- the trap this task exists to avoid.
            # Observed as eval=1 token at 0% reuse before the fix.
            seed_base = 1000 + arm_idx * 100000 + int(frac * 1e6)
            pw = Power()
            with Ne11Window(NE11) as nw, LeaseWindow(LEASE) as lw, pw:
                t0 = time.perf_counter()
                out = []
                for i in range(a.per_cell):
                    # A DISTINCT volatile suffix every request: this is prefix
                    # reuse, not deduplication of an identical prompt.
                    _, vi = build(sl, vl, seed_base + i)
                    r = run_controller(f"p{i}", a.threads, 1, a.n_predict, s + vi,
                                       cache=cache).row()
                    out.append(r)
                wall = time.perf_counter() - t0
                time.sleep(0.4)
            ok = [r for r in out if not r.get("err")]
            ev = [r.get("prompt_n") for r in ok if r.get("prompt_n") is not None]
            reused = [r.get("reused_n", 0) or 0 for r in ok]
            ttft = [r["client_ttft_ms"] for r in ok if r.get("client_ttft_ms")]
            pm = [r["prompt_ms"] for r in ok if r.get("prompt_ms")]
            rec = dict(reuse_frac=frac, cache_prompt=cache, stable_tokens=stable_t,
                       total_tokens=total, requests=len(ok),
                       evaluated_mean=round(st.mean(ev), 1) if ev else None,
                       reused_mean=round(st.mean(reused), 1) if reused else None,
                       ttft_p50=pct(ttft, .5), ttft_p95=pct(ttft, .95),
                       prompt_ms_mean=round(st.mean(pm), 1) if pm else None,
                       req_per_s=round(len(ok) / wall, 3), watts=pw.watts)
            rec.update(nw.delta()); rec.update(lw.delta())
            rows.append(rec)
            print(f"   cache={str(cache):<5} eval {rec['evaluated_mean']:>7} "
                  f"reused {rec['reused_mean']:>7} tok  "
                  f"ttft p50 {rec['ttft_p50']:>8.1f}  prompt {rec['prompt_ms_mean']:>8.1f}  "
                  f"offloaded {rec.get('ne11_nodes_offloaded')}  "
                  f"hist {rec.get('ne11_offloaded_hist') or '-'}  {rec['watts']}W",
                  flush=True)
            write_rows(a.out, rows)
    write_rows(a.out, rows)
    print(f"\nwrote {a.out} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
