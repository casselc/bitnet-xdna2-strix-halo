#!/usr/bin/env python3
"""How CLOSE is a restored state to a recomputed one, in logprobs?

Argmax equality is a coarse instrument. Two states that differ enough to matter
can still agree on the top token for an easy prompt, so "the output matched" is
weak evidence of a correct restore -- and on this workload several candidates
emit the same short action regardless.

This compares the full next-token logprob vector after
`restore(spine) + delta` against the same distribution computed by processing
`spine + delta` from an empty slot. A bit-correct restore should reproduce it
to within floating-point noise. A large divergence means the recurrent state
was not restored, whether or not the argmax happened to survive.

Reported per arm:
  max |dlogprob|  over the tokens present in both top-k lists
  argmax agreement
  cache_n         -- an arm with cache_n ~ 0 fell back to a full reprocess and
                     proves nothing; it is reported as NOT EXERCISED.
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_domain import Domain
from model_bakeoff import (_req, calibrate_spine, calibrate_delta,
                           state_save, state_restore, state_erase)


def dist(base, prompt, slot, k=20):
    r = _req(base, "/completion",
             {"prompt": prompt, "n_predict": 1, "temperature": 0, "top_k": 1,
              "n_probs": k, "cache_prompt": True, "id_slot": slot})
    tm = r.get("timings", {}) or {}
    cp = r.get("completion_probabilities") or []
    top = {}
    if cp:
        for e in (cp[0].get("top_logprobs") or []):
            top[e.get("id", e.get("token"))] = float(e.get("logprob", 0.0))
    return {"cache_n": tm.get("cache_n"), "content": r.get("content", ""), "top": top}


def compare(ref, got):
    common = set(ref["top"]) & set(got["top"])
    if not common:
        return None, 0
    return max(abs(ref["top"][t] - got["top"][t]) for t in common), len(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spine-tokens", type=int, default=1600)
    ap.add_argument("--delta-tokens", type=int, default=135)
    ap.add_argument("--tol", type=float, default=0.01)
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}"
    sd = Path(a.save_dir); sd.mkdir(parents=True, exist_ok=True)
    nt, ns, sp = calibrate_spine(base, a.spine_tokens)
    A = Domain(1, 0xC0FFEE, n_topo=nt, n_state=ns)
    B = Domain(2, 0xC0FFEE, n_topo=nt, n_state=ns)
    dl, dn, _ = calibrate_delta(base, A, a.delta_tokens)

    state_erase(base, 0)
    ref = dist(base, A.prompt(7, dl), 0)
    print(f"[{a.label}] reference cache_n={ref['cache_n']} topk={len(ref['top'])} "
          f"argmax={ref['content']!r}", flush=True)

    fn = f"{a.label}.fid.A.state"
    state_erase(base, 0)
    _req(base, "/completion", {"prompt": A._prefix, "n_predict": 1, "temperature": 0,
                               "cache_prompt": True, "id_slot": 0})
    b, ntk, err = state_save(base, 0, fn, sd)

    rows = []
    for name, pollute in (("none", None), ("self_prefix", A._prefix),
                          ("foreign_prefix", B._prefix), ("none_after_foreign", None)):
        state_erase(base, 0)
        if pollute is not None:
            _req(base, "/completion", {"prompt": pollute, "n_predict": 1, "temperature": 0,
                                       "cache_prompt": True, "id_slot": 0})
        state_erase(base, 0)
        state_restore(base, 0, fn)
        g = dist(base, A.prompt(7, dl), 0)
        d, ncommon = compare(ref, g)
        exercised = (g["cache_n"] or 0) >= sp * 0.9
        rows.append({"arm": name, "cache_n": g["cache_n"], "restore_exercised": exercised,
                     "argmax_same": g["content"] == ref["content"],
                     "max_abs_dlogprob": d, "common_topk": ncommon,
                     "within_tol": (d is not None and d <= a.tol)})
        print(f"[{a.label}] {name:20s} cache_n={g['cache_n']:<6} "
              f"exercised={exercised!s:<6} argmax_same={g['content']==ref['content']!s:<6} "
              f"max|dlogprob|={'n/a' if d is None else format(d, '.5f')}", flush=True)

    ex = [r for r in rows if r["restore_exercised"]]
    worst = max((r["max_abs_dlogprob"] or 0) for r in ex) if ex else None
    res = {"label": a.label, "spine_tokens": sp, "delta_tokens": dn,
           "checkpoint_bytes": b, "tolerance": a.tol,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "reference": {k: ref[k] for k in ("cache_n", "content")},
           "arms": rows,
           "worst_dlogprob_over_exercised_arms": worst,
           "bit_faithful_restore": bool(ex) and all(r["within_tol"] for r in ex),
           "argmax_stable": all(r["argmax_same"] for r in ex)}
    print(f"[{a.label}] worst |dlogprob| over exercised arms = "
          f"{'n/a' if worst is None else format(worst, '.5f')}  "
          f"faithful={res['bit_faithful_restore']}  argmax_stable={res['argmax_stable']}",
          flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
