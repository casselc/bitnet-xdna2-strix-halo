#!/usr/bin/env python3
"""Semantic singleflight and constrained controller output.

TASK 6. The previous pass submitted N IDENTICAL requests and measured them
queueing as independent work. For a deterministic controller that is the wrong
serving semantics: N identical requests should produce ONE model execution and N
identical answers.

The coalescing key is deliberately NOT just the prompt bytes. It is the tuple
that must match for reuse to be SAFE -- see Task 3 in RESULTS.md. Reusing a
result across a different policy version or security domain would be a
correctness and isolation bug, not an optimisation, so the key carries those
fields even though this benchmark only ever populates one domain.

TASK 10. Controller output should be tiny. Compares 32-token prose against
constrained short answers, since after prefix reuse decode is the dominant cost.
"""
import argparse, csv, hashlib, json, statistics as st, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from service_bench import (CTRL, post, run_controller, controller_prompt,
                           write_rows, pct)

# One explicit benchmark security domain. A real deployment would carry the
# tenant and authority version here; reuse must never cross either.
DOMAIN = dict(tenant="bench", policy_version="v1", authority="benchmark-only")


def semantic_key(prompt, model="bitnet-2b-i2s", prompt_version="v1",
                 temperature=0.0, seed=42, n_predict=32):
    """Minimum coordinate under which two requests may share one execution."""
    h = hashlib.sha256()
    for part in (model, prompt_version, DOMAIN["tenant"],
                 DOMAIN["policy_version"], DOMAIN["authority"],
                 f"{temperature}", f"{seed}", f"{n_predict}", prompt):
        h.update(part.encode()); h.update(b"\x1f")
    return h.hexdigest()


class SingleFlight:
    """One in-flight execution per key; later arrivals attach as waiters."""

    def __init__(self):
        self._lock = threading.Lock()
        self._inflight = {}          # key -> Event
        self._result = {}            # key -> row
        self.executions = 0
        self.coalesced = 0

    def do(self, key, fn):
        with self._lock:
            ev = self._inflight.get(key)
            if ev is not None:
                self.coalesced += 1
                waiter = ev
            else:
                waiter = None
                ev = threading.Event()
                self._inflight[key] = ev
                self.executions += 1
        if waiter is not None:
            waiter.wait(timeout=600)
            return self._result.get(key), True
        try:
            r = fn()
            self._result[key] = r
            return r, False
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            ev.set()


def burst(n, fn):
    out, lock = [], threading.Lock()
    def one(i):
        t0 = time.time()
        r = fn(i)
        with lock:
            out.append((r, (time.time() - t0) * 1e3))
    ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in ts: t.start()
    for t in ts: t.join()
    return out, (time.time() - t0) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bursts", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--predict", type=int, default=32)
    ap.add_argument("--outdir", default="artifacts/controller-cache-batching")
    a = ap.parse_args()
    p = controller_prompt()
    run_controller("warm", a.threads, 1, 4, p, cache=True)   # warm the prefix

    rows = []
    print("TASK 6 -- semantic singleflight on identical requests\n")
    print(f"{'mode':>14}{'burst':>7}{'executions':>12}{'completed':>11}"
          f"{'wall ms':>9}{'p50 ms':>9}{'p95 ms':>9}")
    for n in a.bursts:
        # A: ordinary server queue -- every client request is its own execution.
        res, wall = burst(n, lambda i: run_controller(
            f"q{i}", a.threads, n, a.predict, p, cache=True, capture_text=True))
        lat = sorted(x[1] for x in res)
        texts = {x[0].chain.get("text") for x in res}
        rows.append(dict(mode="server-queue", burst=n, executions=n,
                         completed=len(res), wall_ms=round(wall, 1),
                         p50_ms=pct(lat, .5), p95_ms=pct(lat, .95),
                         identical_outputs=int(len(texts) == 1)))
        print(f"{'server-queue':>14}{n:>7}{n:>12}{len(res):>11}{wall:>9.0f}"
              f"{pct(lat,.5):>9}{pct(lat,.95):>9}")

        # B: singleflight -- one execution, the rest attach as waiters.
        sf = SingleFlight()
        k = semantic_key(p, n_predict=a.predict)
        res2, wall2 = burst(n, lambda i: sf.do(
            k, lambda: run_controller(f"s{i}", a.threads, n, a.predict, p,
                                      cache=True, capture_text=True))[0])
        lat2 = sorted(x[1] for x in res2)
        texts2 = {x[0].chain.get("text") for x in res2 if x[0]}
        rows.append(dict(mode="singleflight", burst=n, executions=sf.executions,
                         completed=len(res2), coalesced=sf.coalesced,
                         wall_ms=round(wall2, 1), p50_ms=pct(lat2, .5),
                         p95_ms=pct(lat2, .95),
                         identical_outputs=int(len(texts2) == 1)))
        print(f"{'singleflight':>14}{n:>7}{sf.executions:>12}{len(res2):>11}"
              f"{wall2:>9.0f}{pct(lat2,.5):>9}{pct(lat2,.95):>9}"
              f"   coalesced={sf.coalesced} identical={len(texts2)==1}")
    write_rows(f"{a.outdir}/singleflight.csv", rows)

    print("\nTASK 10 -- controller output size (decode dominates once prefix is reused)\n")
    print(f"{'variant':>22}{'gen tokens':>12}{'decode ms':>11}{'total ms':>10}{'TTFT ms':>9}")
    orows = []
    variants = [
        ("A 32-token prose", a.predict, None),
        ("B action only (4 tok)", 4, None),
        ("C JSON grammar", 12,
         {"grammar": 'root ::= "{\\"action\\":\\"" act "\\"}" \n'
                     'act ::= "RESTART" | "SCALE" | "ROLLBACK" | "WAIT"'}),
        ("D single token", 1, None),
    ]
    for label, npred, extra in variants:
        rs = []
        for i in range(6):
            r = run_controller(f"o{i}", a.threads, 1, npred, p, cache=True,
                               capture_text=True, extra=extra).row()
            rs.append(r)
        f = lambda k: [x[k] for x in rs if x.get(k) is not None]
        rec = dict(variant=label, n_predict=npred,
                   gen_n_p50=pct(f("gen_n"), .5), gen_ms_p50=pct(f("gen_ms"), .5),
                   total_ms_p50=pct(f("total_ms"), .5),
                   ttft_ms_p50=pct(f("ttft_ms"), .5))
        orows.append(rec)
        print(f"{label:>22}{rec['gen_n_p50']:>12}{rec['gen_ms_p50']:>11}"
              f"{rec['total_ms_p50']:>10}{rec['ttft_ms_p50']:>9}")
    write_rows(f"{a.outdir}/short_output.csv", orows)
    print(f"\nwrote {a.outdir}/singleflight.csv and short_output.csv")


if __name__ == "__main__":
    main()
