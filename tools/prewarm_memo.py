#!/usr/bin/env python3
"""Background prewarm, completed-result memoization, and cache residency.

TASK 9  Does refreshing a state spine in the background before the foreground
        query arrives help, and does that background work hurt the GPU worker or
        the verifier tenant? Background acceleration that damages foreground
        service is not acceleration.

TASK 11 Memoization keyed on the SEMANTIC coordinate, not prompt bytes. The
        invalidation tests matter more than the hit: a cache that cannot be
        invalidated by a policy or authority change is a correctness bug wearing
        a performance costume, and cross-tenant reuse is an isolation failure
        even when the prompt bytes match.

TASK 12 How many warm state domains fit before the fixed slot pool evicts.
"""
import argparse, csv, hashlib, json, statistics as st, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from service_bench import run_controller, run_worker, write_rows, pct, summarize, Power
from prefix_bench import STABLE, topo_line, state_line


def domain_prompt(domain, epoch, n_topo=50, n_state=30):
    return (STABLE + f"\nDOMAIN {domain}\n"
            + "".join(topo_line(i + domain * 7) for i in range(n_topo))
            + "\nCANONICAL STATE\n"
            + "".join(state_line(i, epoch + domain * 101) for i in range(n_state))
            + f"\nQUERY domain {domain} epoch {epoch}\nACTION:")


class Memo:
    """Completed-result cache. The key is the coordinate under which reuse is
    SAFE, not the prompt text."""

    def __init__(self):
        self.store = {}
        self.hits = self.misses = 0

    @staticmethod
    def key(*, model, tokenizer, tenant, authority, policy_version,
            state_version, objective, tool_schema, grammar, temperature, seed,
            max_tokens):
        h = hashlib.sha256()
        for part in (model, tokenizer, tenant, authority, policy_version,
                     state_version, objective, tool_schema, grammar,
                     f"{temperature}", f"{seed}", f"{max_tokens}"):
            h.update(str(part).encode()); h.update(b"\x1f")
        return h.hexdigest()

    def get_or_run(self, k, fn):
        if k in self.store:
            self.hits += 1
            return self.store[k], True
        self.misses += 1
        v = fn()
        self.store[k] = v
        return v, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--predict", type=int, default=4)
    ap.add_argument("--outdir", default="artifacts/controller-state-scheduler")
    ap.add_argument("--skip-gpu", action="store_true")
    a = ap.parse_args()
    rows = []

    # ---------------- TASK 9: background prewarm --------------------------
    print("TASK 9 -- background state prewarm\n")
    print(f"{'arm':>34}{'fg TTFT':>10}{'fg total':>10}{'prewarm ms':>12}{'watts':>8}")
    def measure(label, prewarm, dom, epoch, bg=None):
        pw_ms = None
        with Power() as p:
            if bg: bg.start()
            if prewarm:
                t0 = time.time()
                run_controller("pw", a.threads, 1, 1,
                               domain_prompt(dom, epoch), cache=True)
                pw_ms = round((time.time() - t0) * 1e3, 1)
            r = run_controller("fg", a.threads, 1, a.predict,
                               domain_prompt(dom, epoch), cache=True).row()
            if bg: bg.join()
        rec = dict(arm=label, prewarmed=int(bool(prewarm)),
                   fg_ttft_ms=r.get("ttft_ms"), fg_total_ms=r.get("total_ms"),
                   fg_eval_n=r.get("eval_n"), prewarm_ms=pw_ms, watts=p.watts)
        rows.append(rec)
        print(f"{label:>34}{r.get('ttft_ms',0):>10.0f}{r.get('total_ms',0):>10.0f}"
              f"{(pw_ms or 0):>12.1f}{p.watts or 0:>8.1f}")
        return rec

    cold = measure("A foreground cold miss", False, 501, 1)
    warm = measure("B prewarmed just before query", True, 502, 1)
    if not a.skip_gpu:
        gpu = threading.Thread(target=lambda: run_worker("bgw", 1, 96))
        measure("C prewarm while GPU worker active", True, 503, 1, bg=gpu)
    ver = None
    import subprocess, os, signal
    ver = subprocess.Popen(["bb", str(Path(__file__).resolve().parent / "cpu_tenant.clj"),
                            "secs", "100000"], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, text=True,
                           start_new_session=True)
    d = measure("D prewarm while verifier active", True, 504, 1)
    os.killpg(os.getpgid(ver.pid), signal.SIGTERM)
    out = ver.communicate(timeout=20)[0] or ""
    vline = next((l for l in out.splitlines() if l.startswith("{")), None)
    if vline:
        vj = json.loads(vline)
        d.update({f"verifier_{k}": v for k, v in vj.items()})
        print(f"{'':>34}verifier {vj['ops_per_s']:.0f} ops/s  "
              f"p95 {vj['p95_ms']:.3f} ms  p99 {vj['p99_ms']:.3f} ms")
    write_rows(f"{a.outdir}/prewarm.csv", rows)

    # ---------------- TASK 11: memoization + invalidation ------------------
    print("\nTASK 11 -- completed-result memoization: invalidation is the point\n")
    memo = Memo()
    base = dict(model="bitnet-2b-i2s", tokenizer="bitnet-v1", tenant="A",
                authority="ctl-v1", policy_version="p1", state_version="s1",
                objective="scale?", tool_schema="ts1", grammar="none",
                temperature=0.0, seed=42, max_tokens=a.predict)
    def call(**over):
        cfg = {**base, **over}
        k = Memo.key(**cfg)
        v, hit = memo.get_or_run(k, lambda: run_controller(
            "m", a.threads, 1, a.predict,
            domain_prompt(600, hash(cfg["state_version"]) % 97), cache=True,
            capture_text=True).chain.get("text"))
        return hit
    cases = [("first call (populate)", {}, False),
             ("same everything", {}, True),
             ("different objective", dict(objective="restart?"), False),
             ("changed state version", dict(state_version="s2"), False),
             ("changed policy version", dict(policy_version="p2"), False),
             ("changed authority", dict(authority="ctl-v2"), False),
             ("different tenant", dict(tenant="B"), False),
             ("different grammar", dict(grammar="json"), False)]
    mrows = []
    for label, over, want_hit in cases:
        hit = call(**over)
        ok = (hit == want_hit)
        mrows.append(dict(case=label, hit=int(hit), expected_hit=int(want_hit),
                          correct=int(ok)))
        print(f"  {'PASS' if ok else 'FAIL'}  {label:26s} hit={hit!s:5} expected={want_hit}")
    print(f"  executions: {memo.misses}   hits: {memo.hits}")
    write_rows(f"{a.outdir}/memoization.csv", mrows)

    # ---------------- TASK 12: residency ----------------------------------
    print("\nTASK 12 -- how many warm state domains fit (8 slots)\n")
    rrows = []
    print(f"{'domains':>9}{'warm TTFT p50':>15}{'evicted TTFT p50':>18}")
    for nd in (1, 2, 4, 8, 12):
        for dcur in range(nd):                       # warm them all
            run_controller("w", a.threads, 1, 1, domain_prompt(700 + dcur, 1), cache=True)
        ts = []
        for dcur in range(nd):                       # revisit in the same order
            r = run_controller("r", a.threads, 1, a.predict,
                               domain_prompt(700 + dcur, 1), cache=True).row()
            ts.append((r.get("ttft_ms"), r.get("eval_n")))
        warm_t = pct([t for t, e in ts if e is not None and e < 200], .5)
        cold_t = pct([t for t, e in ts if e is not None and e >= 200], .5)
        nwarm = sum(1 for _, e in ts if e is not None and e < 200)
        rec = dict(domains=nd, revisits=len(ts), still_warm=nwarm,
                   evicted=len(ts) - nwarm, warm_ttft_p50=warm_t,
                   evicted_ttft_p50=cold_t)
        rrows.append(rec)
        print(f"{nd:>9}{str(warm_t):>15}{str(cold_t):>18}   warm={nwarm}/{len(ts)}")
    write_rows(f"{a.outdir}/residency.csv", rrows)
    print(f"\nwrote prewarm.csv, memoization.csv, residency.csv")


if __name__ == "__main__":
    main()
