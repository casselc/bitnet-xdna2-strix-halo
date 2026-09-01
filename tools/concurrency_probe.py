#!/usr/bin/env python3
"""c=1 vs c=2 on the warm state-spine workload, per candidate.

Deliberately two points and no sweep. The question is narrow: does a second
concurrent domain buy throughput, or does it only move latency into the tail?
`service-cotenancy` already established for the incumbent that the TAIL is what
degrades first, so p95 is reported beside p50 and neither is dropped.

Each concurrent client owns its OWN domain and its OWN slot, which is what a
multi-domain controller actually looks like. Sharing one domain across clients
would measure deduplication instead of concurrency.

For hybrid models the per-turn sequence is restore-then-query, because ordinary
prefix reuse is unavailable to them (see STATE_ENVELOPE.md). The restore is
included in the reported decision latency.
"""
import argparse, json, statistics as st, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_domain import Domain
from model_bakeoff import (_req, calibrate_spine, calibrate_delta, turn,
                           state_save, state_restore, state_erase, stats)


def prepare(base, dom, save_dir, label, idx):
    """One checkpoint per domain, taken once, outside every measurement."""
    slot = idx
    state_erase(base, slot)
    _req(base, "/completion", {"prompt": dom._prefix, "n_predict": 1,
                               "temperature": 0, "cache_prompt": True, "id_slot": slot})
    fn = f"{label}.conc.{idx}.state"
    b, ntk, err = state_save(base, slot, fn, save_dir)
    return fn, b, err


def one_decision(base, dom, dl, fn, slot, t, predict, use_ckpt):
    t0 = time.time()
    if use_ckpt:
        state_restore(base, slot, fn)
    r = turn(base, dom.prompt(t, dl), predict, cache=True, slot_id=slot)
    r["decision_ms"] = (time.time() - t0) * 1000.0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", default="1,2")
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--predict", type=int, default=4)
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="use native prefix reuse instead (correct for pure-attention)")
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}"
    sd = Path(a.save_dir); sd.mkdir(parents=True, exist_ok=True)
    nt, ns, sp = calibrate_spine(base, 1600)
    probe = Domain(0, 0xC0FFEE, n_topo=nt, n_state=ns)
    dl, dn, _ = calibrate_delta(base, probe, 135)
    use_ckpt = not a.no_checkpoint
    res = {"label": a.label, "spine_tokens": sp, "delta_tokens": dn,
           "mode": "checkpoint_restore" if use_ckpt else "native_prefix_reuse",
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "arms": []}
    print(f"[{a.label}] spine={sp} delta={dn} mode={res['mode']}", flush=True)

    for c in [int(x) for x in a.conc.split(",")]:
        doms = [Domain(100 + i, 0xC0FFEE, n_topo=nt, n_state=ns) for i in range(c)]
        prep = [prepare(base, d, sd, a.label, i) for i, d in enumerate(doms)]

        def client(i, t):
            fn, _, _ = prep[i]
            return one_decision(base, doms[i], dl, fn, i, t, a.predict, use_ckpt)

        with ThreadPoolExecutor(max_workers=c) as ex:
            for t in range(a.warmup):
                list(ex.map(lambda i: client(i, t), range(c)))
            t0 = time.time()
            rows = []
            for t in range(a.warmup, a.warmup + a.turns):
                rows += list(ex.map(lambda i: client(i, t), range(c)))
            wall = time.time() - t0

        ok = [r for r in rows if not r.get("err")]
        arm = {"concurrency": c, "decisions": len(rows), "errors": len(rows) - len(ok),
               "wall_s": round(wall, 2),
               "decisions_per_s": round(len(ok) / wall, 3) if wall else None,
               "decision_ms": stats([r["decision_ms"] for r in ok]),
               "ttft_ms": stats([r["ttft_ms"] for r in ok]),
               "cache_n": stats([r["cache_n"] for r in ok])}
        res["arms"].append(arm)
        print(f"[{a.label}] c={c}  {arm['decisions_per_s']} dec/s  "
              f"decision p50={arm['decision_ms']['p50']:.1f} "
              f"p95={arm['decision_ms']['p95']:.1f}  errors={arm['errors']}", flush=True)

    if len(res["arms"]) == 2:
        a1, a2 = res["arms"]
        res["scaling"] = {
            "throughput_x": round(a2["decisions_per_s"] / a1["decisions_per_s"], 3),
            "p50_x": round(a2["decision_ms"]["p50"] / a1["decision_ms"]["p50"], 3),
            "p95_x": round(a2["decision_ms"]["p95"] / a1["decision_ms"]["p95"], 3)}
        s = res["scaling"]
        print(f"[{a.label}] c1->c2: throughput {s['throughput_x']}x  "
              f"p50 {s['p50_x']}x  p95 {s['p95_x']}x", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
