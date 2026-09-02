#!/usr/bin/env python3
"""Exact-boundary restore discriminator: is hybrid state restore actually wrong?

Supersedes the probes on `model-candidate-halo`, which reported a restore defect
without separating three different things that all look alike from outside:

  1. the checkpoint not covering exactly the prefix,
  2. the model's sequence state being restored incorrectly,
  3. the SERVER's per-slot bookkeeping (`slot.prompt.checkpoints`) being stale.

(1) turned out to be a non-issue -- see `--assert-exact-boundary`. The saved
token count exceeds a bare `/tokenize` count by exactly one because the server
prepends BOS, not because a token was generated. `n_predict=0` and
`n_predict=1` produce byte-identical saved state, which this tool asserts rather
than assumes.

(3) is the live hypothesis, and it is visible in the pinned build's source:
`SLOT_RESTORE` restores the model state and replaces `slot.prompt.tokens`, but
never touches `slot.prompt.checkpoints`; `prompt_clear()` (used by
`SLOT_ERASE`) also leaves them. So a slot carries context checkpoints belonging
to whatever it processed previously, and the reuse path at
`server-context.cpp:3302` can load one into an unrelated restored sequence.

That predicts a specific, testable thing: running with `--ctx-checkpoints 0`
should remove the divergence, because there is then no stale checkpoint to load.
This tool is the instrument for that A/B; it does not need a patched server.

Correctness is judged against a FULL RECOMPUTE of the same prompt, on the
logprob distribution rather than the argmax, with the pure-attention model as
the numerical floor. Small floating differences are not called corruption: the
control establishes what "no difference" measures as on this machine.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_domain import Domain
from model_bakeoff import (_req, ntok, calibrate_spine, calibrate_delta,
                           state_save, state_restore, state_erase)


def dist(base, prompt, slot, k=100, n_predict=1):
    """Next-token distribution plus the cache accounting for one request."""
    t0 = time.time()
    r = _req(base, "/completion",
             {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "top_k": 1,
              "n_probs": k, "cache_prompt": True, "id_slot": slot})
    wall = (time.time() - t0) * 1000.0
    tm = r.get("timings", {}) or {}
    cp = r.get("completion_probabilities") or []
    top = {}
    if cp:
        for e in (cp[0].get("top_logprobs") or []):
            top[e.get("id")] = float(e.get("logprob", 0.0))
    return {"content": r.get("content", ""), "top": top,
            "cache_n": tm.get("cache_n"), "prompt_n": tm.get("prompt_n"),
            "ttft_ms": tm.get("prompt_ms"),
            "total_ms": (tm.get("prompt_ms") or 0) + (tm.get("predicted_ms") or 0),
            "wall_ms": wall}


def compare(ref, got):
    common = sorted(set(ref["top"]) & set(got["top"]))
    if not common:
        return {"n_common": 0, "max_abs": None, "mean_abs": None,
                "top1_same": None, "next_token_same": got["content"] == ref["content"]}
    d = [abs(ref["top"][t] - got["top"][t]) for t in common]
    ref_top1 = max(ref["top"], key=ref["top"].get)
    got_top1 = max(got["top"], key=got["top"].get)
    return {"n_common": len(common), "max_abs": max(d), "mean_abs": st.fmean(d),
            "top1_same": ref_top1 == got_top1,
            "next_token_same": got["content"] == ref["content"]}


def make_checkpoint(base, prefix, slot, fn, sd, n_predict=0):
    state_erase(base, slot)
    r = _req(base, "/completion", {"prompt": prefix, "n_predict": n_predict,
                                   "temperature": 0, "cache_prompt": True,
                                   "id_slot": slot})
    tm = r.get("timings", {}) or {}
    b, ntk, err = state_save(base, slot, fn, sd)
    return {"bytes": b, "saved_tokens": ntk, "prompt_n": tm.get("prompt_n"),
            "predicted_n": tm.get("predicted_n"), "err": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server-log", default="")
    ap.add_argument("--spine-tokens", type=int, default=1600)
    ap.add_argument("--delta-tokens", type=int, default=135)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--decision-tokens", type=int, default=8)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}"
    sd = Path(a.save_dir); sd.mkdir(parents=True, exist_ok=True)
    nt, ns, sp = calibrate_spine(base, a.spine_tokens)
    A = Domain(1, 0xC0FFEE, n_topo=nt, n_state=ns)
    B = Domain(2, 0xC0FFEE, n_topo=nt, n_state=ns)
    dl, dn, _ = calibrate_delta(base, A, a.delta_tokens)

    res = {"label": a.label, "note": a.note, "port": a.port,
           "spine_tokens": sp, "delta_tokens": dn, "topk": a.topk,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ---- Task 1 acceptance: the checkpoint must cover exactly the prefix.
    # A bare /tokenize omits BOS while the server prepends it, so the target is
    # the with-BOS count, not the bare one. Comparing against the bare count is
    # what produced the earlier "checkpoint is one token too long" reading.
    n_bare = len(_req(base, "/tokenize", {"content": A._prefix, "add_special": False})["tokens"])
    n_bos = len(_req(base, "/tokenize", {"content": A._prefix, "add_special": True})["tokens"])
    ck0 = make_checkpoint(base, A._prefix, 0, f"{a.label}.rm.A.np0.state", sd, n_predict=0)
    ck1 = make_checkpoint(base, A._prefix, 0, f"{a.label}.rm.A.np1.state", sd, n_predict=1)
    res["boundary"] = {
        "tokenize_no_bos": n_bare, "tokenize_with_bos": n_bos,
        "n_predict0": ck0, "n_predict1": ck1,
        "exact_boundary": ck0["saved_tokens"] == n_bos,
        "np0_np1_identical": (ck0["saved_tokens"] == ck1["saved_tokens"]
                              and ck0["bytes"] == ck1["bytes"]),
    }
    bd = res["boundary"]
    print(f"[{a.label}] boundary: /tokenize {n_bare} (no BOS) / {n_bos} (BOS); "
          f"saved np0={ck0['saved_tokens']} np1={ck1['saved_tokens']}  "
          f"exact={bd['exact_boundary']}  np0==np1={bd['np0_np1_identical']}", flush=True)

    fn = f"{a.label}.rm.A.np0.state"          # the exact-boundary checkpoint
    res["checkpoint_bytes"] = ck0["bytes"]

    # ---- A. full-recompute reference
    state_erase(base, 0)
    ref = dist(base, A.prompt(7, dl), 0, a.topk)
    state_erase(base, 0)
    ref_dec = _req(base, "/completion", {"prompt": A.prompt(7, dl), "n_predict": a.decision_tokens,
                                         "temperature": 0, "top_k": 1, "cache_prompt": True,
                                         "id_slot": 0}).get("content", "")
    res["reference"] = {k: ref[k] for k in ("content", "cache_n", "prompt_n", "ttft_ms", "total_ms")}
    res["reference"]["decision"] = ref_dec
    print(f"[{a.label}] reference cache_n={ref['cache_n']} prompt_n={ref['prompt_n']} "
          f"topk={len(ref['top'])} decision={ref_dec!r}", flush=True)

    log_before = 0
    if a.server_log and Path(a.server_log).exists():
        log_before = Path(a.server_log).stat().st_size

    arms = [("B_clean", None), ("C_foreign", B._prefix), ("D_clean_after_foreign", None)]
    rows = []
    for name, pollute in arms:
        state_erase(base, 0)
        if pollute is not None:
            _req(base, "/completion", {"prompt": pollute, "n_predict": 0, "temperature": 0,
                                       "cache_prompt": True, "id_slot": 0})
        state_erase(base, 0)
        t0 = time.time()
        rr, rerr = state_restore(base, 0, fn)
        restore_ms = (time.time() - t0) * 1000.0
        got = dist(base, A.prompt(7, dl), 0, a.topk)
        dec = _req(base, "/completion", {"prompt": A.prompt(7, dl), "n_predict": a.decision_tokens,
                                         "temperature": 0, "top_k": 1, "cache_prompt": True,
                                         "id_slot": 0}).get("content", "")
        cmp = compare(ref, got)
        row = {**cmp,
               "arm": name, "restore_ms": round(restore_ms, 2),
               "restored_tokens": (rr or {}).get("n_restored") or (rr or {}).get("n_tokens"),
               "restore_err": rerr,
               "cache_n": got["cache_n"], "prompt_n": got["prompt_n"],
               "ttft_ms": got["ttft_ms"], "total_ms": got["total_ms"],
               "content": got["content"], "decision": dec,
               "decision_same": dec == ref_dec,
               "reused": (got["cache_n"] or 0) >= sp * 0.9}
        rows.append(row)
        print(f"[{a.label}] {name:22s} restore={row['restore_ms']:6.1f}ms "
              f"cache_n={row['cache_n']:<6} ttft={row['ttft_ms']:7.1f} "
              f"max|d|={('n/a' if cmp['max_abs'] is None else format(cmp['max_abs'],'.5f')):9s} "
              f"top1={cmp['top1_same']!s:5s} decision_same={row['decision_same']}", flush=True)
    res["arms"] = rows

    if a.server_log and Path(a.server_log).exists():
        with open(a.server_log, "r", errors="replace") as f:
            f.seek(log_before)
            tail = f.read()
        keep = [l for l in tail.splitlines()
                if any(s in l for s in ("checkpoint", "re-processing", "restore", "cache"))]
        res["server_log_excerpt"] = keep[-40:]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"[{a.label}] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
