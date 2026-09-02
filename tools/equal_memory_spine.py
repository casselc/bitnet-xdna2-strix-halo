#!/usr/bin/env python3
"""At an equal per-domain state budget, how much more context can a hybrid hold?

Task 13. The incumbent costs ~127 MiB/domain at a ~1750-token spine. The
question is whether a hybrid should spend its smaller per-token state on MORE
DOMAINS or on a LONGER SPINE per domain.

Scope, stated up front: this measures **capacity and latency**, never semantic
quality. And one half of the original question cannot be answered on this build.
It asked whether a longer spine can be held "while keeping each decision cheap",
which presumes warm reuse. Hybrid prefix reuse is either absent or numerically
wrong here (see RESTORE.md), so every hybrid decision is a full prefill and its
cost GROWS with spine length. The capacity finding stands on its own; the
latency finding is reported as what it is -- the cost of a full prefill at that
length -- and not dressed up as a warm decision.

Measured per model, per spine length:
  serialized state bytes  (POST /slots/{id}?action=save -- bytes on disk)
  prefill latency for the spine
  decision latency for spine + delta + 4 tokens
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_domain import Domain
from model_bakeoff import (_req, ntok, calibrate_spine, calibrate_delta,
                           state_save, state_erase)


def spine_for_tokens(base, target, cache={}):
    """Scale (n_topo, n_state) to hit a spine token target."""
    ratio = 22.0 / 30.0
    lo, hi, best = 4, 4096, None
    while lo <= hi:
        mid = (lo + hi) // 2
        d = Domain(0, 0xC0FFEE, n_topo=mid, n_state=max(1, int(round(mid * ratio))))
        n = ntok(base, d._prefix)
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (mid, n)
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
    nt = best[0]
    return nt, max(1, int(round(nt * ratio))), best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spines", default="1600,3200,6400,9600")
    ap.add_argument("--delta-tokens", type=int, default=135)
    ap.add_argument("--predict", type=int, default=4)
    ap.add_argument("--budget-mib", type=float, default=127.18)
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}"
    sd = Path(a.save_dir); sd.mkdir(parents=True, exist_ok=True)
    res = {"label": a.label, "budget_mib": a.budget_mib,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "arms": []}

    for target in [int(x) for x in a.spines.split(",")]:
        try:
            nt, ns, sp = spine_for_tokens(base, target)
            dom = Domain(0, 0xC0FFEE, n_topo=nt, n_state=ns)
            dl, dn, pn = calibrate_delta(base, dom, a.delta_tokens)

            state_erase(base, 0)
            t0 = time.time()
            r = _req(base, "/completion", {"prompt": dom._prefix, "n_predict": 0,
                                           "temperature": 0, "cache_prompt": True,
                                           "id_slot": 0}, timeout=1800)
            prefill_ms = (r.get("timings", {}) or {}).get("prompt_ms")
            b, ntk, err = state_save(base, 0, f"{a.label}.spine{target}.state", sd)

            state_erase(base, 0)
            t0 = time.time()
            r2 = _req(base, "/completion", {"prompt": dom.prompt(3, dl),
                                            "n_predict": a.predict, "temperature": 0,
                                            "cache_prompt": True, "id_slot": 0},
                      timeout=1800)
            wall = (time.time() - t0) * 1000.0
            tm2 = r2.get("timings", {}) or {}
            arm = {"spine_target": target, "spine_tokens": sp, "delta_tokens": dn,
                   "total_tokens": pn + dn,
                   "state_bytes": b, "state_mib": round((b or 0) / 2**20, 2),
                   "bytes_per_spine_token": round((b or 0) / max(1, sp), 1),
                   "spine_prefill_ms": round(prefill_ms or 0, 1),
                   "decision_ttft_ms": round(tm2.get("prompt_ms") or 0, 1),
                   "decision_total_ms": round((tm2.get("prompt_ms") or 0) +
                                              (tm2.get("predicted_ms") or 0), 1),
                   "decision_wall_ms": round(wall, 1),
                   "within_budget": bool((b or 0) / 2**20 <= a.budget_mib)}
            res["arms"].append(arm)
            print(f"[{a.label}] spine={sp:<6} total={arm['total_tokens']:<6} "
                  f"state={arm['state_mib']:>7.2f} MiB  prefill={arm['spine_prefill_ms']:>8.1f}ms  "
                  f"decision={arm['decision_total_ms']:>8.1f}ms  "
                  f"within_{a.budget_mib}MiB={arm['within_budget']}", flush=True)
            # stop once we have clearly exceeded the budget
            if (b or 0) / 2**20 > a.budget_mib * 1.15:
                print(f"[{a.label}] budget exceeded; stopping sweep", flush=True)
                break
        except Exception as e:
            res["arms"].append({"spine_target": target, "err": f"{type(e).__name__}: {e}"})
            print(f"[{a.label}] spine={target} FAILED: {type(e).__name__}: {e}", flush=True)
            break

    ok = [x for x in res["arms"] if x.get("within_budget")]
    res["longest_spine_within_budget"] = (max(ok, key=lambda x: x["spine_tokens"])
                                          if ok else None)
    if res["longest_spine_within_budget"]:
        L = res["longest_spine_within_budget"]
        print(f"[{a.label}] longest spine within {a.budget_mib} MiB: "
              f"{L['spine_tokens']} tokens at {L['state_mib']} MiB, "
              f"decision {L['decision_total_ms']} ms", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
