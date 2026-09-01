#!/usr/bin/env python3
"""Hardware-suitability bakeoff for warm state-spine controller candidates.

This measures the ONE workload the controller actually runs -- a large stable
state spine, a small changing delta, a tiny constrained decision -- across
models with different architectures, and it measures the state footprint by
SERIALIZING it rather than by estimating from RSS.

Why serialization and not RSS:

  RSS folds weights, the allocator's slack, the prompt-cache pool and the KV
  cache into one number, and for hybrid models it hides the question that
  matters -- whether the recurrent state is token-dependent at all. The server's
  `POST /slots/{id}?action=save` writes exactly the bytes required to reconstruct
  one sequence, so the file size IS the per-domain cost. It also proves the
  runtime can serialize that state, which for a DeltaNet/Mamba/conv model is a
  deployment precondition, not a detail.

Every token count comes from the server's own /tokenize. Nothing here estimates
chars-per-token: the spine is calibrated per model to a token target, so a model
with a larger vocabulary is not silently handed a shorter prompt.
"""
import argparse, json, os, statistics as st, sys, time, urllib.request, urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_domain import Domain, ACTIONS


# ---------------------------------------------------------------- HTTP

def _req(base, path, payload=None, timeout=600, method=None):
    url = base + path
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read().decode())


def ntok(base, text):
    return len(_req(base, "/tokenize", {"content": text}, timeout=180)["tokens"])


def props(base):
    try:
        return _req(base, "/props", timeout=30)
    except Exception:
        return {}


# ------------------------------------------------- spine calibration

def calibrate_spine(base, target_tokens, seed_root=0xC0FFEE):
    """Scale (n_topo, n_state) so the STABLE prefix lands near target_tokens.

    The prefix grows monotonically in both counts, so a single ratio-guided
    bisection on a fixed 30:22 shape is enough and keeps every model's spine
    the same *shape*, differing only in how many lines it takes to reach the
    same token budget.
    """
    ratio = 22.0 / 30.0
    lo, hi = 4, 512
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        d = Domain(0, seed_root, n_topo=mid, n_state=max(1, int(round(mid * ratio))))
        n = ntok(base, d._prefix)
        if best is None or abs(n - target_tokens) < abs(best[1] - target_tokens):
            best = (mid, n)
        if n < target_tokens:
            lo = mid + 1
        else:
            hi = mid - 1
    n_topo = best[0]
    return n_topo, max(1, int(round(n_topo * ratio))), best[1]


def calibrate_delta(base, dom, target_tokens):
    """Delta-line count whose volatile suffix lands near target_tokens."""
    prefix_n = ntok(base, dom._prefix)
    lo, hi = 0, 4
    while ntok(base, dom.prompt(1, hi)) - prefix_n < target_tokens and hi < 8192:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if ntok(base, dom.prompt(1, mid)) - prefix_n < target_tokens:
            lo = mid + 1
        else:
            hi = mid
    return lo, ntok(base, dom.prompt(1, lo)) - prefix_n, prefix_n


# ------------------------------------------------------------ one turn

def turn(base, prompt, n_predict, cache=True, slot_id=None):
    """One controller decision. Returns server timings plus client wall.

    `err` is carried explicitly. An earlier pass on this project recorded 24
    silent HTTP 400s as ~2 ms completions and produced a total-below-TTFT
    result; a failed turn must never look fast.
    """
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0,
            "cache_prompt": cache, "stream": False}
    if slot_id is not None:
        body["id_slot"] = slot_id
    t0 = time.time()
    try:
        r = _req(base, "/completion", body)
    except urllib.error.HTTPError as e:
        return {"err": f"HTTP {e.code}: {e.read()[:200].decode('utf8','replace')}"}
    except Exception as e:
        return {"err": f"{type(e).__name__}: {e}"}
    wall = (time.time() - t0) * 1000.0
    tm = r.get("timings", {}) or {}
    return {
        "err": "",
        "content": r.get("content", ""),
        "wall_ms": wall,
        "prompt_n": tm.get("prompt_n"),
        "prompt_ms": tm.get("prompt_ms"),
        "predicted_n": tm.get("predicted_n"),
        "predicted_ms": tm.get("predicted_ms"),
        "predicted_per_second": tm.get("predicted_per_second"),
        "cache_n": tm.get("cache_n", r.get("tokens_cached")),
        "ttft_ms": tm.get("prompt_ms"),
        "total_ms": (tm.get("prompt_ms") or 0) + (tm.get("predicted_ms") or 0),
    }


def pct(v, p):
    if not v:
        return None
    s = sorted(v)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {}
    return {"n": len(vals), "p50": pct(vals, 50), "p95": pct(vals, 95),
            "mean": st.fmean(vals), "min": min(vals), "max": max(vals)}


# ------------------------------------------------------- state size

def state_save(base, slot_id, filename, save_dir):
    """Serialize one slot's sequence state; return bytes ACTUALLY ON DISK.

    Returns (bytes_on_disk, n_tokens_reported, err).

    The size is taken from the file, not from the server's JSON. The server's
    `n_saved` field is a TOKEN count, not a byte count -- reading it as bytes
    reports a ~1 byte/token state for every model and silently erases the whole
    comparison this tool exists to make.
    """
    try:
        r = _req(base, f"/slots/{slot_id}?action=save", {"filename": filename}, timeout=600)
    except urllib.error.HTTPError as e:
        return None, None, f"HTTP {e.code}: {e.read()[:200].decode('utf8','replace')}"
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"
    p = Path(save_dir) / filename
    if not p.exists():
        return None, r.get("n_saved"), "file not found on disk"
    return p.stat().st_size, r.get("n_saved"), ""


def state_restore(base, slot_id, filename):
    try:
        r = _req(base, f"/slots/{slot_id}?action=restore", {"filename": filename}, timeout=600)
        return r, ""
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read()[:200].decode('utf8','replace')}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def state_erase(base, slot_id):
    try:
        _req(base, f"/slots/{slot_id}?action=erase", {}, timeout=120)
        return ""
    except Exception as e:
        return f"{type(e).__name__}: {e}"


# ------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spine-tokens", type=int, default=1600)
    ap.add_argument("--delta-tokens", type=int, default=135)
    ap.add_argument("--short-delta-tokens", type=int, default=36)
    ap.add_argument("--predict", type=int, default=4)
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--slot-save-dir", default="")
    ap.add_argument("--skip-state", action="store_true")
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}"
    out = {"label": a.label, "port": a.port, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    pr = props(base)
    out["props"] = {k: pr.get(k) for k in ("model_path", "n_ctx", "total_slots",
                                           "default_generation_settings", "build_info")
                    if k in pr}
    # These artifacts are committed to a public repository. The server reports an
    # absolute model path, which carries the operator's home directory; keep the
    # basename, which is the only part that identifies the weights.
    if out["props"].get("model_path"):
        out["props"]["model_path"] = os.path.basename(out["props"]["model_path"])
    dgs = (pr.get("default_generation_settings") or {})
    out["n_ctx_slot"] = dgs.get("n_ctx")

    # --- calibrate to a TOKEN budget, per model
    n_topo, n_state, spine_n = calibrate_spine(base, a.spine_tokens)
    dom = Domain(0, 0xC0FFEE, n_topo=n_topo, n_state=n_state)
    dl, delta_n, prefix_n = calibrate_delta(base, dom, a.delta_tokens)
    dl_s, delta_n_s, _ = calibrate_delta(base, dom, a.short_delta_tokens)
    out["calibration"] = {"n_topo": n_topo, "n_state": n_state,
                          "spine_tokens": spine_n, "delta_lines": dl,
                          "delta_tokens": delta_n, "total_tokens": prefix_n + delta_n,
                          "short_delta_lines": dl_s, "short_delta_tokens": delta_n_s}
    print(f"[{a.label}] spine={spine_n} tok (topo={n_topo} state={n_state})  "
          f"delta={delta_n} tok  total={prefix_n+delta_n} tok", flush=True)

    # --- COLD: a domain the server has never seen, cache disabled
    cold_dom = Domain(9999, 0xFEEDBEEF, n_topo=n_topo, n_state=n_state)
    c = turn(base, cold_dom.prompt(1, dl), a.predict, cache=False)
    out["cold"] = c
    print(f"[{a.label}] cold  ttft={c.get('ttft_ms')} total={c.get('total_ms')} err={c.get('err')}", flush=True)

    # --- WARM: same domain, prefix reused, delta changes every turn
    for t in range(a.warmup):
        turn(base, dom.prompt(t, dl), a.predict, cache=True)

    rows = []
    for t in range(a.warmup, a.warmup + a.turns):
        r = turn(base, dom.prompt(t, dl), a.predict, cache=True)
        r["turn"] = t
        rows.append(r)
    out["warm_rows"] = rows
    ok = [r for r in rows if not r["err"]]
    out["warm"] = {
        "turns": len(rows), "errors": len(rows) - len(ok),
        "ttft_ms": stats([r["ttft_ms"] for r in ok]),
        "total_ms": stats([r["total_ms"] for r in ok]),
        "wall_ms": stats([r["wall_ms"] for r in ok]),
        "decode_tps": stats([r["predicted_per_second"] for r in ok]),
        "prompt_n": stats([r["prompt_n"] for r in ok]),
        "cache_n": stats([r["cache_n"] for r in ok]),
    }
    w = out["warm"]
    print(f"[{a.label}] warm  ttft p50={w['ttft_ms'].get('p50')} p95={w['ttft_ms'].get('p95')}  "
          f"total p50={w['total_ms'].get('p50')}  errors={w['errors']}", flush=True)

    # --- short delta probe
    srows = []
    for t in range(500, 500 + max(4, a.turns // 3)):
        r = turn(base, dom.prompt(t, dl_s), a.predict, cache=True)
        if not r["err"]:
            srows.append(r)
    out["short_delta"] = {"delta_tokens": delta_n_s,
                          "ttft_ms": stats([r["ttft_ms"] for r in srows]),
                          "total_ms": stats([r["total_ms"] for r in srows])}

    # --- STATE FOOTPRINT (the headline measurement)
    if not a.skip_state and a.slot_save_dir:
        sd = Path(a.slot_save_dir); sd.mkdir(parents=True, exist_ok=True)
        st_out = {}
        # spine-only (~spine_tokens) and spine+delta (~spine+delta)
        for name, prompt_txt, want in (
                ("spine", dom._prefix, spine_n),
                ("spine_delta", dom.prompt(4242, dl), prefix_n + delta_n)):
            turn(base, prompt_txt, 1, cache=True, slot_id=0)
            fn = f"{a.label}.{name}.state"
            b, ntk, err = state_save(base, 0, fn, sd)
            st_out[name] = {"tokens_target": want, "bytes": b, "err": err}
            if b:
                st_out[name]["bytes_per_token"] = b / float(want)
            print(f"[{a.label}] state {name}: {b} bytes at ~{want} tok  err={err}", flush=True)
        out["state"] = st_out

    # --- EXPLICIT SPINE CHECKPOINT / RESTORE
    #
    # Ordinary prefix reuse is unavailable to hybrid/recurrent models: the
    # runtime says so itself ("forcing full prompt re-processing due to lack of
    # cache data (likely due to SWA or hybrid/recurrent memory)"). The reason is
    # architectural, not a configuration mistake -- a recurrent state after N
    # tokens is one fixed-size object with no way to rewind it to position P<N,
    # so a turn that shares only the spine cannot drop the previous delta.
    #
    # The deployment answer is to checkpoint the state AT the spine boundary and
    # restore it before each delta, so the common prefix always equals the whole
    # restored state and nothing has to be removed. This measures whether that
    # actually works and what it costs.
    if not a.skip_state and a.slot_save_dir:
        sd = Path(a.slot_save_dir); sd.mkdir(parents=True, exist_ok=True)
        ck = {}
        fn = f"{a.label}.ckpt.state"
        state_erase(base, 0)
        seed = turn(base, dom._prefix, 1, cache=True, slot_id=0)
        b, ntk, err = state_save(base, 0, fn, sd)
        ck["checkpoint_bytes"] = b
        ck["checkpoint_tokens"] = ntk
        ck["save_err"] = err
        t0 = time.time(); _b2, _n2, _e2 = state_save(base, 0, fn + ".t", sd)
        ck["save_ms"] = (time.time() - t0) * 1000.0

        crows = []
        for t in range(900, 900 + a.turns):
            state_erase(base, 0)
            t0 = time.time()
            rr, rerr = state_restore(base, 0, fn)
            restore_ms = (time.time() - t0) * 1000.0
            if rerr:
                crows.append({"err": rerr}); continue
            r = turn(base, dom.prompt(t, dl), a.predict, cache=True, slot_id=0)
            r["restore_ms"] = restore_ms
            r["turn"] = t
            crows.append(r)
        ok2 = [r for r in crows if not r.get("err")]
        ck["turns"] = len(crows); ck["errors"] = len(crows) - len(ok2)
        ck["restore_ms"] = stats([r["restore_ms"] for r in ok2])
        ck["ttft_ms"] = stats([r["ttft_ms"] for r in ok2])
        ck["total_ms"] = stats([r["total_ms"] for r in ok2])
        ck["cache_n"] = stats([r["cache_n"] for r in ok2])
        ck["prompt_n"] = stats([r["prompt_n"] for r in ok2])
        # end-to-end cost of one decision INCLUDING the restore
        ck["decision_ms"] = stats([(r["restore_ms"] or 0) + (r["total_ms"] or 0) for r in ok2])
        ck["rows"] = crows[:5]
        out["checkpoint"] = ck
        print(f"[{a.label}] ckpt  bytes={b} restore p50={ck['restore_ms'].get('p50')} "
              f"ttft p50={ck['ttft_ms'].get('p50')} cache_n p50={ck['cache_n'].get('p50')} "
              f"decision p50={ck['decision_ms'].get('p50')}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"[{a.label}] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
