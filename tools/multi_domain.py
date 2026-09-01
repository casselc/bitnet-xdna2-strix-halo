#!/usr/bin/env python3
"""Production-shaped multi-domain warm-state controller workload.

What earlier branches measured separately -- cold unrelated requests, duplicate
requests, shared-prefix requests, ONE state spine, and RAM cache capacity -- is
not the workload a real controller sees. This builds the missing one: MANY
independent warm domains, each with its own authoritative spine, each receiving
a genuinely different dynamic delta, each emitting a tiny constrained decision.

Domain construction:

  stable prefix (identical across every turn of a domain, so prefix reuse is
  real and not deduplication):
      objective, controller contract, policy version, action schema,
      WorkGraph summary, authoritative state spine, domain tag

  volatile suffix (different every turn):
      state version, changed cells, new events, new evidence,
      resource delta, verification delta

Tags are 64-bit random hex, NOT sequential. controller-cache-batching produced a
false cross-contamination signal because sequential tags were guessable and the
model simply predicted a neighbouring one; only unguessable tags can distinguish
"the model saw foreign state" from "the model guessed".
"""
import argparse, hashlib, json, random, statistics as st, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "tools")
from service_bench import (run_controller, write_rows, pct, summarize,
                           assert_timing_sane, preflight_length, count_tokens,
                           slot_context, Power, append_jsonl, CTRL)

ACTIONS = ["HOLD", "SCALE", "ROLLBACK", "RESTART", "PAGE"]


class Domain:
    """One independent controller domain with a stable identity."""

    def __init__(self, idx, seed_root=0xC0FFEE, n_topo=30, n_state=22):
        # 64-bit unguessable tag, derived deterministically from (root, idx) so
        # a run is reproducible without being predictable from a neighbour.
        h = hashlib.sha256(f"{seed_root}:{idx}".encode()).hexdigest()
        self.idx = idx
        self.tag = h[:16]                      # 64 bits
        self.policy_version = h[16:24]
        self.n_topo, self.n_state = n_topo, n_state
        self.version = 0
        self._prefix = self._build_prefix()

    def _build_prefix(self):
        rng = random.Random(int(self.tag, 16) & 0xFFFFFFFF)
        L = [
            "OBJECTIVE\n",
            "Maintain service health within policy. Emit exactly one action.\n\n",
            "CONTROLLER CONTRACT\n",
            "  - Read the authoritative state spine and the volatile delta.\n",
            "  - Emit one action token from the schema. No prose, no explanation.\n",
            f"  - Domain tag: {self.tag}\n",
            f"  - Policy version: {self.policy_version}\n\n",
            "ACTION SCHEMA\n",
        ]
        L += [f"  {a}\n" for a in ACTIONS]
        L.append("\nWORKGRAPH SUMMARY\n")
        for i in range(12):
            L.append(f"  stage{i:02d} -> stage{i+1:02d}  owner=svc{rng.randrange(999):03d} "
                     f"sla={50 + rng.randrange(900)}ms\n")
        L.append("\nAUTHORITATIVE STATE SPINE\n")
        for i in range(self.n_topo):
            L.append(f"  svc{i:03d}: region=r{i % 7} tier={i % 4} "
                     f"deps=[svc{(i*3) % 97:03d},svc{(i*7) % 97:03d}] "
                     f"budget={80 + (i * 31) % 400}ms owner=team{i % 11}\n")
        for i in range(self.n_state):
            L.append(f"  svc{i:03d}: baseline_p95={40 + (i * 13) % 500}ms "
                     f"replicas={1 + i % 6} slo=0.99{i % 9}\n")
        return "".join(L)

    def turn(self, t, n_delta_lines):
        """Volatile suffix for turn t. Different every turn, same prefix."""
        self.version = t
        rng = random.Random((int(self.tag, 16) ^ (t * 0x9E3779B97F4A7C15)) & 0xFFFFFFFF)
        V = [f"\nSV {self.tag}.{t:04d}\n"]
        for i in range(n_delta_lines):
            V.append(f"s{rng.randrange(self.n_state):03d} "
                     f"{rng.randrange(10, 900)} {rng.randrange(60)} "
                     f"{rng.randrange(500)} {rng.randrange(1, 9000)}\n")
        V.append(f"ev{rng.randrange(10**6):06d} dep svc{rng.randrange(self.n_state):03d}\n")
        V.append(f"res {rng.randrange(100)}/{rng.randrange(100)}/{rng.randrange(100)}\n")
        V.append(f"ver {rng.randrange(200)}/{rng.randrange(9)}\nACTION:")
        return "".join(V)

    def prompt(self, t, n_delta_lines):
        return self._prefix + self.turn(t, n_delta_lines)


def calibrate_delta(domain, target_tokens, base=None):
    """Pick the delta-line count that hits a target volatile-suffix size.

    Measured with the server's tokenizer, never a chars/token estimate -- a
    previous pass was off by 1.9x guessing at structured text.
    """
    prefix_n = count_tokens(domain._prefix, base)
    lo, hi = 0, 4
    while count_tokens(domain.prompt(1, hi), base) - prefix_n < target_tokens and hi < 4096:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if count_tokens(domain.prompt(1, mid), base) - prefix_n < target_tokens:
            lo = mid + 1
        else:
            hi = mid
    return lo, count_tokens(domain.prompt(1, lo), base) - prefix_n, prefix_n


def make_domains(n, n_topo=30, n_state=22, seed_root=0xC0FFEE):
    return [Domain(i, seed_root, n_topo, n_state) for i in range(n)]


def contamination_check(text, own_tag, all_tags):
    """Any FOREIGN 64-bit tag appearing in output is a real leak signal."""
    return sorted(t for t in all_tags if t != own_tag and t in (text or ""))


def cell(label, domains, turns, n_delta, conc, n_predict, threads,
         cache=True, jsonl=None, extra=None, turn0=1):
    """One measurement cell: `turns` requests spread over the domains.

    turn0 matters. The warm pass and the measured pass must NOT use the same
    turn numbers: with D domains and D requests, `turn = 1 + i//D` is 1 for
    every request, so a measured pass reusing turn 1 re-sends the exact prompt
    the warm pass just sent and reports eval=1. That is duplicate-request
    deduplication, not a fresh delta against a warm spine -- the precise trap
    this workload exists to avoid. Warm with turn0=0, measure with turn0=1.
    """
    tags = {d.tag for d in domains}
    pw = Power()
    rows = []
    with pw:
        t0 = time.perf_counter()

        def one(i):
            d = domains[i % len(domains)]
            t = turn0 + i // len(domains)
            r = run_controller(f"{label}-{i}", threads, conc, n_predict,
                               d.prompt(t, n_delta), cache=cache,
                               capture_text=True)
            row = r.row()
            row.update(domain=d.idx, tag=d.tag, turn=t)
            if not row.get("err"):
                txt = r.chain.get("text", "")
                foreign = contamination_check(txt, d.tag, tags)
                row["foreign_tags"] = ";".join(foreign)
                row["contaminated"] = int(bool(foreign))
            return row

        if conc <= 1:
            rows = [one(i) for i in range(turns)]
        else:
            with ThreadPoolExecutor(max_workers=conc) as ex:
                rows = [f.result() for f in [ex.submit(one, i) for i in range(turns)]]
        wall = time.perf_counter() - t0
    good, bad = assert_timing_sane(rows, label)
    if jsonl:
        append_jsonl(jsonl, rows)
    ev = [r["prompt_n"] for r in good if r.get("prompt_n") is not None]
    ru = [r.get("reused_n", 0) or 0 for r in good]
    tt = [r["ttft_ms"] for r in good]
    to = [r["total_ms"] for r in good]
    rec = dict(cell=label, domains=len(domains), requests=len(rows),
               usable=len(good), excluded=len(bad), concurrency=conc,
               n_predict=n_predict, delta_lines=n_delta, threads=threads,
               wall_s=round(wall, 2), req_per_s=round(len(good) / wall, 3),
               eval_mean=round(st.mean(ev), 1) if ev else None,
               reused_mean=round(st.mean(ru), 1) if ru else None,
               supplied_mean=round(st.mean([a + b for a, b in zip(ev, ru)]), 1) if ev else None,
               ttft_p50=pct(tt, .5), ttft_p95=pct(tt, .95),
               total_p50=pct(to, .5), total_p95=pct(to, .95),
               contaminated=sum(r.get("contaminated", 0) for r in good),
               watts=pw.watts, gpu_busy_med=pw.gpu_busy_med)
    if extra:
        rec.update(extra)
    return rec, rows
