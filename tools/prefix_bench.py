#!/usr/bin/env python3
"""Production-shaped prefix reuse: stable prefix + changing suffix.

The identical-prompt A/B is only a mechanism test -- it proves the cache works,
not that a real controller benefits. A real controller sends a stable context
(instructions, action vocabulary, topology) followed by CHANGED state, so the
question is how reuse degrades as the shared fraction falls.

Shared fraction is controlled in TOKENS, not characters: the two differ by ~2x
on this structured text, and a character-ratio family would be mislabelled.
Token counts come from the server's own tokenizer via /tokenize, so the reported
fraction is the one the cache actually sees.

The stable prefix is FIRST, because the cache matches a common PREFIX -- a
design that varies the head and shares the tail reuses nothing, and that is
worth demonstrating rather than assuming (see the `suffix-stable` control).
"""
import argparse, csv, json, statistics as st, sys, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from service_bench import CTRL, post, run_controller, write_rows, summarize, pct

def ntok(text):
    return len(post(CTRL, "/tokenize", {"content": text}).get("tokens", []))

STABLE = ("CONTROLLER POLICY v1\n"
          "You supervise a service mesh. Choose exactly one action from the\n"
          "vocabulary: RESTART, SCALE, ROLLBACK, WAIT.\n"
          "Rules: never ROLLBACK during an active deploy; prefer WAIT when\n"
          "error rate is falling; SCALE only when latency p95 exceeds budget\n"
          "and CPU headroom remains; RESTART only for wedged instances.\n"
          "Tool schema: {\"action\": <verb>, \"target\": <service-id>}\n"
          "TOPOLOGY\n")

def topo_line(i):
    return (f"- svc{i}: region=r{i%5} tier={'edge' if i%3 else 'core'} "
            f"deps=[svc{(i+1)%200},svc{(i+2)%200}] budget_p95={80+(i*7)%400}ms\n")

def state_line(i, epoch):
    return (f"- svc{i}: p95={40+((i*13)+epoch*31)%500}ms err={(i+epoch)%9} "
            f"cpu={20+((i*7)+epoch)%70}% deploy={'yes' if (i+epoch)%11==0 else 'no'}\n")

def build(n_topo, n_state, epoch):
    return (STABLE
            + "".join(topo_line(i) for i in range(n_topo))
            + "\nCURRENT STATE\n"
            + "".join(state_line(i, epoch) for i in range(n_state))
            + "\nChoose one action.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--predict", type=int, default=32)
    ap.add_argument("--target-tokens", type=int, default=1900)
    ap.add_argument("--out", default="artifacts/controller-cache-batching/shared_prefix.csv")
    a = ap.parse_args()

    # Auto-size from a MEASURED tokens-per-line rather than a guess: the first
    # attempt assumed ~16 tokens/line, produced 3367-token prompts and was
    # rejected by the server's per-slot context. Sizing is done against the
    # server's own tokenizer so the family labels are honest.
    head = ntok(STABLE)
    per_topo = (ntok(STABLE + "".join(topo_line(i) for i in range(20))) - head) / 20
    per_state = (ntok(STABLE + "".join(state_line(i, 0) for i in range(20))) - head) / 20
    budget = a.target_tokens - head - 12          # 12 for the section markers
    n_lines = int(budget / ((per_topo + per_state) / 2))
    print(f"  tokens/line: topo {per_topo:.1f}  state {per_state:.1f}  "
          f"-> {n_lines} lines for a ~{a.target_tokens}-token prompt\n")
    families = []
    for frac, label in ((0.95, "~95% shared"), (0.75, "~75% shared"),
                        (0.50, "~50% shared"), (0.25, "~25% shared"),
                        (0.05, "~5% shared")):
        nt = max(1, int(n_lines * frac))
        families.append((label, nt, max(1, n_lines - nt)))

    rows = []
    print(f"{'family':>14}{'supplied':>10}{'reused':>8}{'eval':>7}"
          f"{'TTFT p50':>10}{'total p50':>11}{'prompt_ms':>11}")
    for label, nt, ns in families:
        p0 = build(nt, ns, 0)
        tot = ntok(p0)
        stable_tok = ntok(STABLE + "".join(topo_line(i) for i in range(nt)))
        # Warm the slot with epoch 0, then measure changing epochs.
        run_controller("w", a.threads, 1, 4, p0, cache=True)
        recs = []
        for e in range(1, a.reps + 1):
            r = run_controller(f"p{e}", a.threads, 1, a.predict,
                               build(nt, ns, e), cache=True).row()
            recs.append(r)
        f = lambda k: [x[k] for x in recs if x.get(k) is not None]
        row = dict(family=label, n_topo=nt, n_state=ns,
                   supplied_tokens=tot, stable_tokens=stable_tok,
                   stable_frac=round(stable_tok / tot, 3),
                   reps=len(recs),
                   reused_p50=pct(f("reused_n"), .5),
                   eval_p50=pct(f("eval_n"), .5),
                   prompt_ms_p50=pct(f("prompt_ms"), .5))
        row.update(summarize(recs, "ttft_ms"))
        row.update(summarize(recs, "total_ms"))
        rows.append(row)
        print(f"{label:>14}{tot:>10}{row['reused_p50']:>8}{row['eval_p50']:>7}"
              f"{row['ttft_ms_p50']:>10}{row['total_ms_p50']:>11}"
              f"{row['prompt_ms_p50']:>11}")

    # Control: same content, but the CHANGING part first. The cache matches a
    # prefix, so this should reuse nothing however much text is shared.
    print("\n  control -- shared text placed AFTER the changing part:")
    half = max(1, n_lines // 2)
    run_controller("wc", a.threads, 1, 4,
                   "".join(state_line(i, 0) for i in range(half)) + STABLE, cache=True)
    ctl = []
    for e in range(1, 5):
        bad = ("".join(state_line(i, e) for i in range(half)) + STABLE
               + "".join(topo_line(i) for i in range(half)) + "\nChoose one action.\n")
        ctl.append(run_controller(f"c{e}", a.threads, 1, a.predict, bad,
                                  cache=True).row())
    g = lambda k: [x[k] for x in ctl if x.get(k) is not None]
    crow = dict(family="suffix-stable CONTROL", supplied_tokens=ntok(bad),
                reps=len(ctl), reused_p50=pct(g("reused_n"), .5),
                eval_p50=pct(g("eval_n"), .5),
                prompt_ms_p50=pct(g("prompt_ms"), .5))
    crow.update(summarize(ctl, "ttft_ms")); crow.update(summarize(ctl, "total_ms"))
    rows.append(crow)
    print(f"{'stable-last':>14}{crow['supplied_tokens']:>10}{crow['reused_p50']:>8}"
          f"{crow['eval_p50']:>7}{crow['ttft_ms_p50']:>10}"
          f"{crow['total_ms_p50']:>11}{crow['prompt_ms_p50']:>11}")

    write_rows(a.out, rows)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
