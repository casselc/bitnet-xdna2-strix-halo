#!/usr/bin/env python3
"""Exact-spine restore: is the reuse both LARGE and NUMERICALLY RIGHT?

Samizdat never needs arbitrary prefix rollback. Every turn is:

    restore the state saved at exactly the stable spine S
    append the changing delta D
    emit a tiny action A

so the restored state is ALREADY the state after S. Nothing has to be rewound.
The generic server path does not know that, and gates hybrid reuse behind a
context-checkpoint search that either declines (cache_n = 0, full re-prefill) or
perturbs recurrent state.

This measures one server binary against its OWN full-recompute reference, so a
stock build and a patched build can be compared without assuming the two builds
agree a priori -- the reference is recomputed inside each run and is itself
checked across binaries.

Acceptance, both required. A previous pass on this project reported a large
hybrid speedup that came from reusing state the model never computed, so speed
alone is not evidence:

    reuse:      cache_n ~ N, prompt_n ~ the delta only
    correctness: max |delta logprob| at or below the floor the pure-attention
                 control measures on the same protocol

Arms:
  A  full recompute of spine+delta from an empty slot        (the oracle)
  R  erase -> eval spine with n_predict=0 -> save -> erase
     -> restore -> spine+delta                               (the path under test)
"""
import argparse, json, statistics as st, subprocess, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_domain import Domain


def wait_health(port, timeout=300):
    for _ in range(timeout):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as f:
                if b"ok" in f.read():
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def post(port, path, payload, timeout=1800):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                               data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read().decode())


class Server:
    def __init__(self, binpath, model, port, ctxcp, workdir, ub=4096, extra=None):
        self.port = port
        d = Path(workdir); d.mkdir(parents=True, exist_ok=True)
        self.state = d / "state"; self.state.mkdir(exist_ok=True)
        self.log = d / "server.log"
        cmd = [str(binpath), "-m", str(model), "-t", "4", "-ngl", "0", "-c", "40960",
               "-np", "8", "-b", "4096", "-ub", str(ub), "-tb", "16",
               "-ctxcp", str(ctxcp), "--slot-save-path", str(self.state) + "/",
               "--host", "127.0.0.1", "--port", str(port), "--no-webui"]
        if extra:
            cmd += list(extra)
        self.cmd = cmd
        self.lf = open(self.log, "w")
        self.p = subprocess.Popen(cmd, stdout=self.lf, stderr=self.lf,
                                  stdin=subprocess.DEVNULL,
                                  env={"BITNET_XDNA": "0", "PATH": "/usr/bin:/bin"})

    def __enter__(self):
        if not wait_health(self.port):
            tail = ""
            try:
                tail = "\n".join(open(self.log, errors="replace").read().splitlines()[-15:])
            except Exception:
                pass
            raise SystemExit(f"server on {self.port} unhealthy\n{tail}")
        return self

    def __exit__(self, *a):
        self.p.terminate()
        try:
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()
        self.lf.close()


def ask(port, prompt, topk=100, n_predict=1, slot=0):
    t0 = time.time()
    r = post(port, "/completion",
             {"prompt": prompt, "n_predict": n_predict, "temperature": 0,
              "top_k": 1, "seed": 1234, "n_probs": topk,
              "cache_prompt": True, "id_slot": slot})
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


def cmp_dist(ref, got):
    common = sorted(set(ref["top"]) & set(got["top"]))
    if not common:
        return {"n_common": 0, "max_abs": None, "mean_abs": None, "top1_same": None}
    d = [abs(ref["top"][t] - got["top"][t]) for t in common]
    return {"n_common": len(common), "max_abs": max(d), "mean_abs": st.fmean(d),
            "top1_same": max(ref["top"], key=ref["top"].get) ==
                         max(got["top"], key=got["top"].get)}


def token_exact_boundary(port, spine_text, full_texts):
    """Longest token prefix common to the spine AND every full prompt.

    Text-level prefixing is NOT token-level prefixing: BPE merges across the
    spine/delta seam, so tokenising the spine alone yields a final token that
    does not appear when a delta follows. Measured on LFM2.5-1.2B, the spine is
    1576 tokens but only 1575 of them survive as a prefix of spine+delta -- the
    last token changes from 708 to 509.

    The exact-spine contract requires token-for-token identity, so the boundary
    has to be the common prefix over the deltas that will actually be appended,
    and the state must be saved at exactly that many TOKENS. Feeding token ids
    rather than text is what makes that expressible.
    """
    spine = post(port, "/tokenize", {"content": spine_text, "add_special": True})["tokens"]
    n = len(spine)
    for t in full_texts:
        full = post(port, "/tokenize", {"content": t, "add_special": True})["tokens"]
        k = 0
        for a, b in zip(spine, full):
            if a != b:
                break
            k += 1
        n = min(n, k)
    return spine[:n], len(spine)


def run(port, dom, dl, topk, decision_tokens, state_dir, fname):
    """Arms A and R against one live server."""
    prompt = dom.prompt(7, dl)

    prompt_ids = post(port, "/tokenize", {"content": prompt, "add_special": True})["tokens"]
    post(port, "/slots/0?action=erase", {})
    A = ask(port, prompt_ids, topk)
    post(port, "/slots/0?action=erase", {})
    A_dec = post(port, "/completion",
                 {"prompt": prompt_ids, "n_predict": decision_tokens, "temperature": 0,
                  "top_k": 1, "seed": 1234, "cache_prompt": True,
                  "id_slot": 0}).get("content", "")

    # exact spine boundary, in TOKENS not text -- see token_exact_boundary()
    spine_ids, spine_text_tokens = token_exact_boundary(
        port, dom._prefix, [prompt, dom.prompt(8, dl), dom.prompt(9, dl)])
    post(port, "/slots/0?action=erase", {})
    sp_r = post(port, "/completion", {"prompt": spine_ids, "n_predict": 0,
                                      "temperature": 0, "cache_prompt": True,
                                      "id_slot": 0})
    spine_prompt_n = (sp_r.get("timings", {}) or {}).get("prompt_n")
    sv = post(port, f"/slots/0?action=save", {"filename": fname})
    saved_tokens = sv.get("n_saved")
    saved_bytes = (Path(state_dir) / fname).stat().st_size

    # Arm S: split evaluation WITHOUT save/restore -- spine, then the full prompt,
    # in the same live slot. This separates two candidate causes of divergence
    # that arm R would otherwise confound:
    #   * evaluating 1575+135 in two batches instead of 1710 in one, which can
    #     differ numerically for a chunked recurrent scan even when nothing is
    #     wrong, and
    #   * the save/restore round trip itself.
    # If S already diverges, the round trip is not implicated.
    post(port, "/slots/0?action=erase", {})
    post(port, "/completion", {"prompt": spine_ids, "n_predict": 0, "temperature": 0,
                               "cache_prompt": True, "id_slot": 0})
    S = ask(port, prompt_ids, topk)
    S_dec = post(port, "/completion",
                 {"prompt": prompt_ids, "n_predict": decision_tokens, "temperature": 0,
                  "top_k": 1, "seed": 1234, "cache_prompt": True,
                  "id_slot": 0}).get("content", "")

    post(port, "/slots/0?action=erase", {})
    t0 = time.time()
    rs = post(port, f"/slots/0?action=restore", {"filename": fname})
    restore_ms = (time.time() - t0) * 1000.0
    R = ask(port, prompt_ids, topk)
    R_dec = post(port, "/completion",
                 {"prompt": prompt_ids, "n_predict": decision_tokens, "temperature": 0,
                  "top_k": 1, "seed": 1234, "cache_prompt": True,
                  "id_slot": 0}).get("content", "")

    return {
        "spine_tokens_text_only": spine_text_tokens,
        "spine_tokens_exact_boundary": len(spine_ids),
        "boundary_lost_to_bpe_merge": spine_text_tokens - len(spine_ids),
        "spine_prompt_n": spine_prompt_n, "saved_tokens": saved_tokens,
        "saved_bytes": saved_bytes, "restore_ms": round(restore_ms, 2),
        "restored_tokens": rs.get("n_restored") or rs.get("n_tokens"),
        "A": {k: A[k] for k in ("content", "cache_n", "prompt_n", "ttft_ms", "total_ms")},
        "A_decision": A_dec,
        "R": {k: R[k] for k in ("content", "cache_n", "prompt_n", "ttft_ms", "total_ms")},
        "R_decision": R_dec,
        "S": {k: S[k] for k in ("content", "cache_n", "prompt_n", "ttft_ms", "total_ms")},
        "S_decision": S_dec,
        "S_vs_A": cmp_dist(A, S),
        "S_decision_same": S_dec == A_dec,
        "R_vs_A": cmp_dist(A, R),
        "R_vs_S": cmp_dist(S, R),
        "decision_same": R_dec == A_dec,
        "_A_top": A["top"], "_R_top": R["top"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--ctxcp", type=int, default=0)
    ap.add_argument("--ub", type=int, default=4096)
    ap.add_argument("--spine-tokens", type=int, default=1600)
    ap.add_argument("--delta-tokens", type=int, default=135)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--decision-tokens", type=int, default=8)
    ap.add_argument("--server-arg", action="append", default=[])
    ap.add_argument("--workdir", default="/tmp/bitnet-exactspine")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    wd = Path(a.workdir) / a.label
    from model_bakeoff import calibrate_spine, calibrate_delta

    with Server(a.bin, a.model, a.port, a.ctxcp, wd, a.ub, a.server_arg) as s:
        base = f"http://127.0.0.1:{s.port}"
        nt, ns, sp = calibrate_spine(base, a.spine_tokens)
        dom = Domain(1, 0xC0FFEE, n_topo=nt, n_state=ns)
        dl, dn, _ = calibrate_delta(base, dom, a.delta_tokens)
        r = run(a.port, dom, dl, a.topk, a.decision_tokens, s.state,
                f"{a.label}.spine.state")

    top_a = r.pop("_A_top"); top_r = r.pop("_R_top")
    res = {"label": a.label, "note": a.note, "binary": str(a.bin),
           "server_args": a.server_arg, "ctxcp": a.ctxcp, "ub": a.ub,
           "model": Path(a.model).name, "spine_tokens": sp, "delta_tokens": dn,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **r}
    res["reference_top_logprobs"] = {str(k): v for k, v in
                                     sorted(top_a.items(), key=lambda x: -x[1])[:10]}

    reused = (r["R"]["cache_n"] or 0) >= sp * 0.9
    c = r["R_vs_A"]
    exact = c["max_abs"] is not None and c["max_abs"] < 1e-6
    res["reuse_ok"] = bool(reused)
    res["numerically_exact"] = bool(exact)
    res["verdict"] = ("REUSE + EXACT" if reused and exact else
                      "EXACT BUT NO REUSE" if exact else
                      "REUSE BUT NUMERICALLY WRONG" if reused else
                      "NO REUSE AND NOT EXACT")
    print(f"[{a.label}] spine={sp} delta={dn} saved={r['saved_tokens']} tok "
          f"({r['saved_bytes']/2**20:.2f} MiB)", flush=True)
    print(f"[{a.label}] A  cache_n={r['A']['cache_n']:<6} prompt_n={r['A']['prompt_n']:<6} "
          f"ttft={r['A']['ttft_ms']:8.1f}ms", flush=True)
    print(f"[{a.label}] R  cache_n={r['R']['cache_n']:<6} prompt_n={r['R']['prompt_n']:<6} "
          f"ttft={r['R']['ttft_ms']:8.1f}ms  restore={r['restore_ms']:.1f}ms", flush=True)
    cs = r["S_vs_A"]; crs = r["R_vs_S"]
    print(f"[{a.label}] S  cache_n={r['S']['cache_n']:<6} prompt_n={r['S']['prompt_n']:<6} "
          f"ttft={r['S']['ttft_ms']:8.1f}ms  (split eval, no save/restore)", flush=True)
    print(f"[{a.label}] S vs A max|d|={('n/a' if cs['max_abs'] is None else format(cs['max_abs'],'.6f'))}  "
          f"R vs S max|d|={('n/a' if crs['max_abs'] is None else format(crs['max_abs'],'.6f'))}", flush=True)
    print(f"[{a.label}] max|d|={('n/a' if c['max_abs'] is None else format(c['max_abs'],'.6f'))} "
          f"mean|d|={('n/a' if c['mean_abs'] is None else format(c['mean_abs'],'.6f'))} "
          f"top1={c['top1_same']} decision_same={r['decision_same']}", flush=True)
    print(f"[{a.label}] VERDICT: {res['verdict']}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
