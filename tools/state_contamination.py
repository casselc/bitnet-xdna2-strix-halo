#!/usr/bin/env python3
"""Does erase+restore actually reset a slot's recurrent state?

This exists because a weaker version of the question passes for the wrong
reason. An evict/restore round trip that falls back to a FULL reprocess
(`cache_n == 0`) reproduces the reference output perfectly while testing
nothing -- it never exercised the restore at all. Only a turn that both
(a) reports a prefix hit and (b) matches the reference has demonstrated a
correct restore.

The probe holds the checkpoint fixed and varies ONLY what the slot held
beforehand:

    none                  -> restore A -> query A
    A's own prefix        -> restore A -> query A
    a FOREIGN prefix (B)  -> restore A -> query A
    none, again           -> restore A -> query A     (does it recover?)

Every arm restores the identical file and asks the identical question, so any
difference in the answer is residue from the previous occupant -- which for a
controller means one domain's state changing another domain's decision.

The trailing repeat matters as much as the foreign arm: a corruption that
clears itself is a performance bug, one that persists is a correctness bug.
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_domain import Domain
from model_bakeoff import (_req, calibrate_spine, calibrate_delta,
                           state_save, state_restore, state_erase)
from state_semantics import greedy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spine-tokens", type=int, default=1600)
    ap.add_argument("--delta-tokens", type=int, default=135)
    ap.add_argument("--predict", type=int, default=8)
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}"
    sd = Path(a.save_dir); sd.mkdir(parents=True, exist_ok=True)
    nt, ns, sp = calibrate_spine(base, a.spine_tokens)
    A = Domain(1, 0xC0FFEE, n_topo=nt, n_state=ns)
    B = Domain(2, 0xC0FFEE, n_topo=nt, n_state=ns)
    dl, dn, _ = calibrate_delta(base, A, a.delta_tokens)

    res = {"label": a.label, "port": a.port, "spine_tokens": sp, "delta_tokens": dn,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    state_erase(base, 0)
    ref = greedy(base, A.prompt(7, dl), a.predict, slot_id=0)
    res["reference"] = ref
    print(f"[{a.label}] REF cache_n={ref['cache_n']} {ref['content']!r}", flush=True)

    fn = f"{a.label}.contam.A.state"
    state_erase(base, 0)
    _req(base, "/completion", {"prompt": A._prefix, "n_predict": 1, "temperature": 0,
                               "cache_prompt": True, "id_slot": 0})
    b, ntk, err = state_save(base, 0, fn, sd)
    res["checkpoint"] = {"bytes": b, "tokens": ntk, "err": err}

    arms = [("none", None), ("self_prefix", A._prefix),
            ("foreign_prefix", B._prefix), ("none_after_foreign", None)]
    rows = []
    for name, pollute in arms:
        state_erase(base, 0)
        if pollute is not None:
            _req(base, "/completion", {"prompt": pollute, "n_predict": 1, "temperature": 0,
                                       "cache_prompt": True, "id_slot": 0})
        state_erase(base, 0)
        _, rerr = state_restore(base, 0, fn)
        g = greedy(base, A.prompt(7, dl), a.predict, slot_id=0)
        row = {"arm": name, "cache_n": g["cache_n"], "prompt_n": g["prompt_n"],
               "content": g["content"], "restore_err": rerr,
               "matches_reference": g["content"] == ref["content"],
               # a hit is only meaningful if the restore was actually used
               "restore_exercised": bool((g["cache_n"] or 0) >= sp * 0.9)}
        rows.append(row)
        print(f"[{a.label}] {name:20s} cache_n={row['cache_n']:<6} "
              f"match={row['matches_reference']!s:<6} "
              f"exercised={row['restore_exercised']!s:<6} {row['content']!r}", flush=True)
    res["arms"] = rows

    exercised = [r for r in rows if r["restore_exercised"]]
    bad = [r for r in exercised if not r["matches_reference"]]
    res["verdict"] = {
        "arms_exercising_restore": len(exercised),
        "arms_wrong": [r["arm"] for r in bad],
        "contamination_observed": bool(bad),
        "persists_after_clean_restore": any(
            r["arm"] == "none_after_foreign" and not r["matches_reference"]
            for r in exercised),
    }
    v = res["verdict"]
    res["deployment"] = (
        "STATE-SPINE DEPLOYMENT BLOCKED IN THIS RUNTIME (slot residue changes the decision)"
        if v["contamination_observed"] else
        "STATE RESTORE CLEAN ACROSS DOMAIN SWITCHES")
    print(f"[{a.label}] VERDICT: {res['deployment']}", flush=True)
    if v["contamination_observed"]:
        print(f"[{a.label}]   wrong arms: {v['arms_wrong']}  "
              f"persists={v['persists_after_clean_restore']}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
