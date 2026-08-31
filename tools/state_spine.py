#!/usr/bin/env python3
"""State spine: authoritative state is truth, model KV is disposable acceleration.

Three things are tested and they are different questions:

  TASK 7  50 sequential state updates under three regimes -- full rebuild each
          turn, spine reuse, and spine reuse with periodic rebase.

  TASK 8  ephemeral query forks. A controller query must NOT become part of the
          future state projection. After query A and query B against spine S,
          appending delta D must give S+D, not S+A+decisionA+B+decisionB+D.

  The invariant that matters more than any timing: losing the KV cache must
  never lose durable state. Every prompt here is reconstructed from the
  authoritative event list the client holds, so a rebase is always possible and
  is checked rather than assumed.
"""
import argparse, csv, hashlib, json, statistics as st, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from service_bench import run_controller, write_rows, pct, summarize
from prefix_bench import STABLE, topo_line, state_line


class Authoritative:
    """External truth. The model never holds anything that is not derivable
    from here -- that is what makes the KV cache disposable."""

    def __init__(self, n_topo=50, n_state=30):
        self.n_topo, self.n_state = n_topo, n_state
        self.events = []                    # ordered, append-only

    def apply(self, ev):
        self.events.append(ev)

    def version(self):
        h = hashlib.sha256()
        for e in self.events:
            h.update(e.encode()); h.update(b"\x1f")
        return h.hexdigest()[:16]

    def spine_prompt(self):
        """Canonical projection: stable policy + topology + snapshot + events.
        Reconstructable at any time from self.events alone."""
        return (STABLE
                + "".join(topo_line(i) for i in range(self.n_topo))
                + "\nCANONICAL STATE\n"
                + "".join(state_line(i, 0) for i in range(self.n_state))
                + "\nEVENT LOG\n" + "".join(self.events))

    def query_prompt(self, objective):
        """An ephemeral fork: spine + one question. Never appended to events."""
        return self.spine_prompt() + f"\nQUERY: {objective}\nACTION:"


def turn_row(label, turn, r, extra=None):
    d = dict(regime=label, turn=turn)
    row = r.row()
    for k in ("supplied_n", "reused_n", "eval_n", "prompt_ms", "ttft_ms",
              "total_ms"):
        if row.get(k) is not None:
            d[k] = row[k]
    if extra:
        d.update(extra)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--turns", type=int, default=50)
    ap.add_argument("--predict", type=int, default=4)
    ap.add_argument("--rebase-every", type=int, nargs="+", default=[10, 25])
    ap.add_argument("--outdir", default="artifacts/controller-state-scheduler")
    a = ap.parse_args()

    def event(i):
        return (f"- t{i}: svc{i%50} p95={60+(i*17)%400}ms err={(i*3)%9} "
                f"deploy={'start' if i%13==0 else 'none'}\n")

    rows = []

    def run_regime(label, rebase_every=None, reuse=True):
        auth = Authoritative()
        prev_ver = None
        for t in range(1, a.turns + 1):
            auth.apply(event(t))
            rebased = False
            if rebase_every and t % rebase_every == 0:
                # Throw the KV acceleration away and rebuild from authority.
                # Modelled by sending the canonical prompt with cache disabled,
                # which is exactly "reconstruct without relying on cached state".
                r = run_controller(f"rb{t}", a.threads, 1, 2,
                                   auth.spine_prompt(), cache=False)
                rebased = True
            r = run_controller(f"q{t}", a.threads, 1, a.predict,
                               auth.query_prompt(f"turn {t} decision"),
                               cache=reuse, capture_text=True)
            rows.append(turn_row(label, t, r,
                                 dict(rebased=int(rebased),
                                      state_version=auth.version())))
        return auth

    print(f"TASK 7 -- {a.turns} state updates, three regimes\n")
    run_regime("A rebuild (no reuse)", reuse=False)
    run_regime("B spine reuse", reuse=True)
    for n in a.rebase_every:
        run_regime(f"C rebase/{n}", rebase_every=n, reuse=True)

    print(f"{'regime':>22}{'eval p50':>10}{'reused p50':>12}{'TTFT p50':>10}"
          f"{'TTFT p95':>10}{'total p50':>11}")
    summ = []
    for label in sorted({r["regime"] for r in rows}):
        sub = [r for r in rows if r["regime"] == label]
        rec = dict(regime=label, turns=len(sub),
                   eval_p50=pct([x["eval_n"] for x in sub if x.get("eval_n")], .5),
                   reused_p50=pct([x.get("reused_n", 0) for x in sub], .5),
                   eval_total=sum(x.get("eval_n", 0) for x in sub))
        rec.update(summarize(sub, "ttft_ms")); rec.update(summarize(sub, "total_ms"))
        summ.append(rec)
        print(f"{label:>22}{rec['eval_p50']:>10}{rec['reused_p50']:>12}"
              f"{rec['ttft_ms_p50']:>10}{rec['ttft_ms_p95']:>10}"
              f"{rec['total_ms_p50']:>11}")
    write_rows(f"{a.outdir}/state_spine.csv", rows)
    write_rows(f"{a.outdir}/state_spine_summary.csv", summ)

    # ---- TASK 8: ephemeral query forks --------------------------------------
    print("\nTASK 8 -- ephemeral query forks must not enter the state projection\n")
    auth = Authoritative()
    for i in range(1, 6):
        auth.apply(event(i))
    v_before = auth.version()
    base = run_controller("f0", a.threads, 1, a.predict,
                          auth.query_prompt("baseline"), cache=True,
                          capture_text=True)
    # Two ephemeral queries. Their prompts and outputs are deliberately NOT
    # appended to auth.events.
    qa = run_controller("fa", a.threads, 1, a.predict,
                        auth.query_prompt("fork A"), cache=True, capture_text=True)
    qb = run_controller("fb", a.threads, 1, a.predict,
                        auth.query_prompt("fork B"), cache=True, capture_text=True)
    v_after_forks = auth.version()
    auth.apply(event(6))
    v_after_delta = auth.version()

    # Independent reconstruction: a fresh authority replaying the same events.
    fresh = Authoritative()
    for i in list(range(1, 6)) + [6]:
        fresh.apply(event(i))
    same_projection = (fresh.spine_prompt() == auth.spine_prompt())
    after = run_controller("f1", a.threads, 1, a.predict,
                           auth.query_prompt("baseline"), cache=True,
                           capture_text=True)
    fresh_r = run_controller("f2", a.threads, 1, a.predict,
                             fresh.query_prompt("baseline"), cache=True,
                             capture_text=True)
    frows = [dict(check="state version unchanged by forks",
                  ok=int(v_before == v_after_forks),
                  detail=f"{v_before} -> {v_after_forks}"),
             dict(check="state version changed by authoritative delta",
                  ok=int(v_after_forks != v_after_delta),
                  detail=f"{v_after_forks} -> {v_after_delta}"),
             dict(check="projection equals independent replay S+D",
                  ok=int(same_projection), detail="prompt text identical"),
             dict(check="post-fork query matches fresh-replay query",
                  ok=int(after.chain.get("text") == fresh_r.chain.get("text")),
                  detail=repr((after.chain.get("text") or "")[:24]))]
    for f in frows:
        print(f"  {'PASS' if f['ok'] else 'FAIL'}  {f['check']:48s} {f['detail']}")
    write_rows(f"{a.outdir}/query_forks.csv", frows)
    print(f"\nwrote {a.outdir}/state_spine.csv, state_spine_summary.csv, query_forks.csv")


if __name__ == "__main__":
    main()
