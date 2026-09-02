#!/usr/bin/env python3
"""Alternating domains through ONE slot: does exact-spine reuse stay correct?

Task 5. A single successful restore proves very little -- an earlier pass on
this project accepted one and later had to withdraw the result, because the
failure only appeared once a *different* domain had occupied the slot. So this
cycles many domains through one physical slot and re-checks correctness
throughout, not just at the start.

Per turn:
    restore domain d's spine checkpoint into slot 0
    send d's spine + a fresh delta (as token ids, so the prefix is exact)
    check the emitted action carries no OTHER domain's 64-bit tag

On sampled turns it additionally recomputes the same prompt from an empty slot
on a second slot id and compares the full next-token distribution, which is the
only check that can distinguish "restored" from "plausible".

Drift is reported over time rather than as a single pass/fail: a defect that
appears at turn 60 is the one this exists to catch.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_domain import Domain
from exact_spine_probe import Server, post, ask, cmp_dist, token_exact_boundary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--domains", type=int, default=4)
    ap.add_argument("--turns", type=int, default=100)
    ap.add_argument("--verify-every", type=int, default=10)
    ap.add_argument("--ctxcp", type=int, default=0)
    ap.add_argument("--spine-tokens", type=int, default=1600)
    ap.add_argument("--delta-tokens", type=int, default=135)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--predict", type=int, default=4)
    ap.add_argument("--server-arg", action="append", default=[])
    ap.add_argument("--workdir", default="/tmp/bitnet-alternating")
    a = ap.parse_args()

    from model_bakeoff import calibrate_spine, calibrate_delta
    wd = Path(a.workdir) / a.label
    res = {"label": a.label, "model": Path(a.model).name, "domains": a.domains,
           "turns": a.turns, "ctxcp": a.ctxcp,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    with Server(a.bin, a.model, a.port, a.ctxcp, wd, extra=a.server_arg) as s:
        base = f"http://127.0.0.1:{a.port}"
        nt, ns, sp = calibrate_spine(base, a.spine_tokens)
        doms = [Domain(100 + i, 0xC0FFEE, n_topo=nt, n_state=ns) for i in range(a.domains)]
        dl, dn, _ = calibrate_delta(base, doms[0], a.delta_tokens)
        tags = [d.tag for d in doms]
        res["tags"] = tags

        # one exact-boundary checkpoint per domain
        ck = []
        for i, d in enumerate(doms):
            spine_ids, _ = token_exact_boundary(
                a.port, d._prefix, [d.prompt(t, dl) for t in (1, 2, 3)])
            post(a.port, "/slots/0?action=erase", {})
            post(a.port, "/completion", {"prompt": spine_ids, "n_predict": 0,
                                         "temperature": 0, "cache_prompt": True,
                                         "id_slot": 0})
            fn = f"{a.label}.d{i}.state"
            sv = post(a.port, f"/slots/0?action=save", {"filename": fn})
            ck.append({"file": fn, "n": len(spine_ids),
                       "bytes": (s.state / fn).stat().st_size,
                       "saved_tokens": sv.get("n_saved")})
        res["checkpoints"] = ck
        print(f"[{a.label}] {a.domains} domains, spine~{sp} tok, "
              f"{ck[0]['bytes']/2**20:.2f} MiB each", flush=True)

        rows, mismatches, contaminated = [], [], []
        for t in range(a.turns):
            i = t % a.domains
            d = doms[i]
            prompt_ids = post(a.port, "/tokenize",
                              {"content": d.prompt(1000 + t, dl), "add_special": True})["tokens"]
            post(a.port, "/slots/0?action=erase", {})
            t0 = time.time()
            post(a.port, f"/slots/0?action=restore", {"filename": ck[i]["file"]})
            restore_ms = (time.time() - t0) * 1000.0
            g = ask(a.port, prompt_ids, a.topk, n_predict=a.predict, slot=0)

            # any FOREIGN 64-bit tag in the output is a real leak, not a guess
            foreign = [tg for j, tg in enumerate(tags) if j != i and tg in g["content"]]
            if foreign:
                contaminated.append({"turn": t, "domain": i, "foreign": foreign,
                                     "content": g["content"]})

            row = {"turn": t, "domain": i, "restore_ms": round(restore_ms, 2),
                   "cache_n": g["cache_n"], "prompt_n": g["prompt_n"],
                   "ttft_ms": g["ttft_ms"], "total_ms": g["total_ms"]}

            if t % a.verify_every == 0:
                # full recompute of the identical prompt on a DIFFERENT slot
                post(a.port, "/slots/1?action=erase", {})
                ref = ask(a.port, prompt_ids, a.topk, n_predict=a.predict, slot=1)
                c = cmp_dist(ref, g)
                row["verify"] = {"max_abs": c["max_abs"], "mean_abs": c["mean_abs"],
                                 "top1_same": c["top1_same"],
                                 "content_same": g["content"] == ref["content"]}
                if c["max_abs"] is None or c["max_abs"] >= 1e-6:
                    mismatches.append({"turn": t, "domain": i, **row["verify"]})
                print(f"[{a.label}] turn {t:>3} dom {i} cache_n={row['cache_n']:<6} "
                      f"ttft={row['ttft_ms']:7.1f} max|d|="
                      f"{('n/a' if c['max_abs'] is None else format(c['max_abs'],'.6f'))}",
                      flush=True)
            rows.append(row)

    ok = [r for r in rows if r["cache_n"]]
    res["rows"] = rows[:20]
    res["summary"] = {
        "turns": len(rows),
        "reuse_min": min((r["cache_n"] or 0) for r in rows),
        "reuse_max": max((r["cache_n"] or 0) for r in rows),
        "ttft_p50": sorted(r["ttft_ms"] for r in rows)[len(rows)//2],
        "ttft_p95": sorted(r["ttft_ms"] for r in rows)[int(len(rows)*0.95)],
        "restore_ms_p50": sorted(r["restore_ms"] for r in rows)[len(rows)//2],
        "verified_turns": sum(1 for r in rows if "verify" in r),
        "numerical_mismatches": len(mismatches),
        "contaminated_turns": len(contaminated),
    }
    res["mismatches"] = mismatches[:10]
    res["contamination"] = contaminated[:10]
    su = res["summary"]
    res["verdict"] = ("STABLE ACROSS ALTERNATING DOMAINS"
                      if not mismatches and not contaminated else
                      "DRIFT OR CONTAMINATION OBSERVED")
    print(f"[{a.label}] {su['turns']} turns, reuse {su['reuse_min']}..{su['reuse_max']}, "
          f"ttft p50={su['ttft_p50']:.1f} p95={su['ttft_p95']:.1f}, "
          f"verified={su['verified_turns']}, mismatches={su['numerical_mismatches']}, "
          f"contaminated={su['contaminated_turns']}", flush=True)
    print(f"[{a.label}] VERDICT: {res['verdict']}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
