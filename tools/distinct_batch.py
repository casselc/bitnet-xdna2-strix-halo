#!/usr/bin/env python3
"""Do truly DISTINCT shared-prefix suffixes batch, and does that re-engage XDNA?

The previous branch's c=8 cell sent the SAME prompt to every client thread. That
is concurrency class 1 (duplicate), and it cannot answer the class-2 question:
distinct requests sharing a stable prefix but each evaluating its own new tokens.

Two things are measured that throughput alone cannot separate:

  CORRECTNESS  every suffix is run alone first to build an oracle, then the same
               suffixes are run concurrently. Concurrent output must equal
               independent output, or the sequences are contaminating each other
               and no performance number means anything.

  BATCH SHAPE  the runtime now records ne11 -- the token dimension of the batch
               actually reaching each I2_S mul_mat -- in power-of-two buckets. If
               8 suffixes of 79 new tokens are combined, a 512-1023 bucket
               appears; if they are processed separately, only the 64-127 bucket
               moves. Concurrency scaling proves neither.
"""
import argparse, csv, json, statistics as st, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from service_bench import CTRL, post, run_controller, write_rows, pct, summarize
from prefix_bench import STABLE, topo_line, state_line

NE11 = Path("/tmp/bitnet-service/ne11.csv")


def ntok(text):
    return len(post(CTRL, "/tokenize", {"content": text}).get("tokens", []))


def make_family(n_stable_lines, n_suffix_lines, uid):
    """Shared prefix, then a suffix unique to this request.

    The uid is embedded in the suffix so no two concurrent requests can share a
    cache entry -- otherwise the 'distinct' test would silently become the
    duplicate test again."""
    return (STABLE
            + "".join(topo_line(i) for i in range(n_stable_lines))
            + f"\nCURRENT STATE (view {uid})\n"
            + "".join(state_line(i, uid * 97 + 13) for i in range(n_suffix_lines))
            + f"\nObjective {uid}: choose one action.\n")


def ne11_snapshot():
    """Last complete row. The file is appended from the hot path, so the final
    line can be a partial write; a torn row must be skipped, not parsed."""
    if not NE11.exists():
        return None
    good = None
    for r in csv.DictReader(open(NE11)):
        try:
            good = {k: int(v) for k, v in r.items()}
        except (ValueError, TypeError):
            continue          # skip a torn append, keep scanning
    return good


def ne11_delta(a, b):
    """Per-window bucket deltas.

    A torn append must be SKIPPED, not treated as end-of-file: the first version
    stopped at the first unparsable line, so one early torn row froze the
    snapshot and every window reported a zero delta.

    `max` is a running maximum since process start, so its delta is meaningless
    -- the first version reported ne11_max=2048 for every cell because an
    earlier warmup had already set it. The window's largest batch is taken from
    the highest bucket that actually moved."""
    if not a or not b:
        return {}
    d = {k: b.get(k, 0) - a.get(k, 0) for k in b if k != "wall_ns"}
    buckets = {k: v for k, v in d.items() if k.startswith("b") and v > 0}
    top = max((int(k[1:]) for k in buckets), default=0)
    return dict(ne11_calls=d.get("calls", 0), ne11_tokens=d.get("tokens", 0),
                ne11_top_bucket=top,
                ne11_buckets=";".join(f"{k}:{v}" for k, v in sorted(
                    buckets.items(), key=lambda x: int(x[0][1:]))))


def run_batch(prompts, threads, predict, cache=True):
    out, lock = [], threading.Lock()
    def one(i, p):
        r = run_controller(f"d{i}", threads, len(prompts), predict, p,
                           cache=cache, capture_text=True)
        with lock:
            out.append((i, r))
    ts = [threading.Thread(target=one, args=(i, p)) for i, p in enumerate(prompts)]
    t0 = time.time()
    for t in ts: t.start()
    for t in ts: t.join()
    return sorted(out), (time.time() - t0) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--predict", type=int, default=4)
    ap.add_argument("--conc", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--out", default="artifacts/controller-state-scheduler/distinct_batching.csv")
    a = ap.parse_args()

    # Suffix sizes chosen to straddle kMTile=1024 when aggregated:
    #   8 x  79 =  632  < 1024      (only aggregation could cross it)
    #   4 x 379 = 1516  > 1024
    #   2 x 759 = 1518  > 1024
    #   1 x1139 = 1139  > 1024      (crosses alone -- the control)
    fams = [("~79 new", 71, 4), ("~379 new", 56, 19),
            ("~759 new", 37, 38), ("~1139 new", 18, 57)]

    rows = []
    print(f"{'family':>11}{'conc':>5}{'new/req':>9}{'aggregate':>10}{'top bkt':>9}"
          f"{'NPU':>5}{'req/s':>8}{'ttft p50':>10}{'total p95':>10}  buckets")
    for label, ns, nsuf in fams:
        # Warm the shared prefix in every slot that will be used, using a uid
        # range disjoint from the measured one so warmup cannot itself be the hit.
        for u in range(max(a.conc)):
            run_controller("w", a.threads, 1, 2, make_family(ns, nsuf, 900 + u),
                           cache=True)
        for c in a.conc:
            uids = [1000 + c * 50 + i for i in range(c)]
            prompts = [make_family(ns, nsuf, u) for u in uids]

            # ---- concurrent FIRST, on suffixes no slot has seen -----------
            # Order matters and the first version got it wrong: running the
            # oracle first warms these exact prompts, so the concurrent pass
            # becomes a duplicate cache hit (eval=1) and measures class 1 again.
            # Concurrent-first keeps each suffix genuinely new.
            s0 = ne11_snapshot()
            res, wall = run_batch(prompts, a.threads, a.predict)
            s1 = ne11_snapshot()
            shape = ne11_delta(s0, s1)

            # ---- oracle AFTER: same prompts, run alone --------------------
            # These are now cached, but the previous branch established that
            # caching does not change output, so equality is still the right
            # correctness test for cross-sequence contamination.
            oracle = {}
            for i, p in enumerate(prompts):
                r = run_controller(f"o{i}", a.threads, 1, a.predict, p,
                                   cache=True, capture_text=True)
                oracle[i] = r.chain.get("text")

            rowsr = [r.row() for _, r in res]
            match = all(r.chain.get("text") == oracle[i] for i, r in res)
            newtok = pct([x["eval_n"] for x in rowsr if x.get("eval_n")], .5) or 0
            agg = newtok * c
            rec = dict(family=label, concurrency=c, predict=a.predict,
                       new_tokens_per_req=newtok, aggregate_new_tokens=agg,
                       outputs_match_oracle=int(match),
                       requests=len(rowsr), wall_ms=round(wall, 1),
                       req_per_s=round(len(rowsr) / (wall / 1e3), 3), **shape)
            rec.update(summarize(rowsr, "ttft_ms"))
            rec.update(summarize(rowsr, "total_ms"))
            npu = "YES" if shape.get("ne11_top_bucket", 0) >= 1024 else "no"
            rows.append(rec)
            print(f"{label:>11}{c:>5}{newtok:>9}{agg:>10}"
                  f"{shape.get('ne11_top_bucket',0):>9}"
                  f"{npu:>5}{rec['req_per_s']:>8}{rec.get('ttft_ms_p50',0):>10}"
                  f"{rec.get('total_ms_p95',0):>10}  {shape.get('ne11_buckets','')[:46]}"
                  f"{'' if match else '   *** OUTPUT MISMATCH ***'}", flush=True)
    write_rows(a.out, rows)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
