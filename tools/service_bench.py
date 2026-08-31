#!/usr/bin/env python3
"""Warm persistent-service benchmark: BitNet controller + Qwen GPU worker.

The previous pass measured COMPONENT throughput with one-shot benchmark
processes. This measures REQUEST latency against long-lived services, which is
a different question: model load is excluded, requests queue, and the
single-flight NPU lease becomes a shared resource between concurrent requests.

Three quantities are kept distinct throughout, because blending them is how
service measurements mislead:

    queue wait     submitted -> the service began working on it
    service time   began -> finished
    total latency  submitted -> response in hand

llama.cpp's server reports its own prompt/predicted timings, so service time is
taken from the server rather than inferred from the client's wall clock; the
difference between client wall time and server service time IS the queue wait
plus transport, and is recorded rather than assumed to be zero.
"""
import argparse, csv, json, os, statistics as st, threading, time, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CTRL = "http://127.0.0.1:8081"
WORK = "http://127.0.0.1:8082"
RAPL = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
RAPL_MAX = Path("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")
GPU_BUSY = Path("/sys/class/drm/card0/device/gpu_busy_percent")


def read_int(p, d=0):
    try:
        return int(p.read_text().strip())
    except Exception:
        return d


def post(base, path, payload, timeout=600):
    req = urllib.request.Request(base + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def controller_prompt(approx_tokens=2000):  # ~2.0k tokens, measured
    """A controller-shaped input: structured state to reason over, not prose.
    Sized once and reused so every request is identical."""
    unit = ("- service {i}: state=degraded latency_p95={p}ms errors={e} "
            "region=r{r} deps=[svc{a},svc{b}]\n")
    lines = ["SYSTEM STATE REPORT\n"]
    for i in range(60):
        lines.append(unit.format(i=i, p=40 + (i * 7) % 300, e=(i * 13) % 9,
                                 r=i % 5, a=(i + 1) % 150, b=(i + 2) % 150))
    lines.append("\nChoose exactly one action: RESTART, SCALE, ROLLBACK, or WAIT. "
                 "Answer with the action and one sentence of justification.\n")
    return "".join(lines)


WORKER_PROMPT = (
    "Implement a Python function `merge_intervals(intervals)` that merges "
    "overlapping closed intervals and returns them sorted. Include a docstring "
    "and three assert-based tests."
)


class Req:
    """One timed request. Phase timestamps are captured at the boundaries the
    client can actually observe; server-side service time comes from the
    server's own timings block."""

    __slots__ = ("rid", "cls", "threads", "conc", "t_submit", "t_admit",
                 "t_first", "t_end", "server", "err", "chain")

    def __init__(self, rid, cls, threads, conc):
        self.rid, self.cls, self.threads, self.conc = rid, cls, threads, conc
        self.t_submit = self.t_admit = self.t_first = self.t_end = None
        self.server = {}
        self.chain = {}
        self.err = None

    def row(self):
        d = dict(rid=self.rid, cls=self.cls, threads=self.threads,
                 concurrency=self.conc, err=self.err or "")
        if self.t_submit and self.t_end:
            d["total_ms"] = round((self.t_end - self.t_submit) * 1e3, 2)
        s = self.server
        for k_out, k_in in (("prompt_n", "prompt_n"), ("gen_n", "predicted_n"),
                            ("prompt_ms", "prompt_ms"),
                            ("gen_ms", "predicted_ms")):
            if k_in in s:
                d[k_out] = round(s[k_in], 2) if isinstance(s[k_in], float) else s[k_in]
        if "prompt_ms" in s and "predicted_ms" in s:
            svc = s["prompt_ms"] + s["predicted_ms"]
            d["service_ms"] = round(svc, 2)
            if "total_ms" in d:
                # Everything the client waited that the server did not spend
                # working: admission queue plus HTTP transport.
                d["queue_ms"] = round(d["total_ms"] - svc, 2)
            # TTFT as the server sees it: prompt processing plus one token.
            per_tok = (s["predicted_ms"] / s["predicted_n"]
                       if s.get("predicted_n") else 0.0)
            d["ttft_ms"] = round(s["prompt_ms"] + per_tok, 2)
            if s.get("predicted_per_second"):
                d["gen_tok_s"] = round(s["predicted_per_second"], 3)
        d.update(self.chain)
        return d


def run_controller(rid, threads, conc, n_predict, prompt):
    r = Req(rid, "C", threads, conc)
    r.t_submit = time.time()
    try:
        d = post(CTRL, "/completion",
                 dict(prompt=prompt, n_predict=n_predict, temperature=0,
                      seed=42, cache_prompt=False))
        r.server = d.get("timings", {})
    except Exception as e:
        r.err = f"{type(e).__name__}: {str(e)[:80]}"
    r.t_end = time.time()
    return r


def run_worker(rid, conc, n_predict, prompt=WORKER_PROMPT, cls="W"):
    r = Req(rid, cls, 0, conc)
    r.t_submit = time.time()
    try:
        d = post(WORK, "/completion",
                 dict(prompt=prompt, n_predict=n_predict, temperature=0,
                      seed=42, cache_prompt=False))
        r.server = d.get("timings", {})
    except Exception as e:
        r.err = f"{type(e).__name__}: {str(e)[:80]}"
    r.t_end = time.time()
    return r


def run_chain(rid, threads, conc, ctrl_predict, work_predict, prompt):
    """Controller decides, then a deterministic payload goes to the worker.

    The worker payload is derived from the controller's OUTPUT but does not
    depend on it being CORRECT -- this measures queueing, not controller
    quality, and a benchmark whose timing depends on model semantics is not
    reproducible."""
    r = Req(rid, "CW", threads, conc)
    r.t_submit = time.time()
    try:
        t0 = time.time()
        c = post(CTRL, "/completion",
                 dict(prompt=prompt, n_predict=ctrl_predict, temperature=0,
                      seed=42, cache_prompt=False))
        t1 = time.time()
        ct = c.get("timings", {})
        action = (c.get("content", "") or "").strip().split()[:1]
        action = action[0] if action else "WAIT"
        payload = (f"The controller selected action '{action}'. "
                   + WORKER_PROMPT)
        t2 = time.time()
        w = post(WORK, "/completion",
                 dict(prompt=payload, n_predict=work_predict, temperature=0,
                      seed=42, cache_prompt=False))
        t3 = time.time()
        wt = w.get("timings", {})
        r.server = wt
        r.chain = dict(
            ctrl_total_ms=round((t1 - t0) * 1e3, 2),
            ctrl_prompt_ms=round(ct.get("prompt_ms", 0), 2),
            ctrl_gen_ms=round(ct.get("predicted_ms", 0), 2),
            ctrl_prompt_n=ct.get("prompt_n"),
            handoff_ms=round((t2 - t1) * 1e3, 3),
            work_total_ms=round((t3 - t2) * 1e3, 2),
            work_prompt_ms=round(wt.get("prompt_ms", 0), 2),
            work_gen_ms=round(wt.get("predicted_ms", 0), 2),
            work_gen_tok_s=round(wt.get("predicted_per_second", 0), 3),
            chain_ms=round((t3 - t0) * 1e3, 2))
    except Exception as e:
        r.err = f"{type(e).__name__}: {str(e)[:80]}"
    r.t_end = time.time()
    return r


def pct(v, p):
    if not v:
        return None
    v = sorted(v)
    return round(v[min(len(v) - 1, int(p * len(v)))], 2)


def summarize(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    if not v:
        return {}
    return {f"{key}_p50": pct(v, .50), f"{key}_p95": pct(v, .95),
            f"{key}_p99": pct(v, .99), f"{key}_mean": round(st.mean(v), 2),
            f"{key}_max": round(max(v), 2)}


class LeaseWindow:
    """Difference the runtime's lease CSV across a measurement window.

    The controller service appends cumulative snapshots from inside the lease
    release path, so a window's contention is the difference between the first
    and last snapshot that fall inside it. Reading the file rather than an API
    keeps the service process untouched."""

    def __init__(self, path):
        self.path = Path(path) if path else None

    def _rows(self):
        if not self.path or not self.path.exists():
            return []
        out = []
        with open(self.path) as f:
            for r in csv.DictReader(f):
                try:
                    out.append({k: int(v) for k, v in r.items()})
                except Exception:
                    pass
        return out

    def __enter__(self):
        self.before = self._rows()
        return self

    def __exit__(self, *a):
        self.after = self._rows()

    def delta(self):
        b = self.before[-1] if self.before else None
        a = self.after[-1] if self.after else None
        if not a:
            return {}
        if not b:
            b = {k: 0 for k in a}
        d = {}
        for k in ("acquisitions", "immediate", "waited", "wait_ns", "hold_ns"):
            d["lease_" + k] = a.get(k, 0) - b.get(k, 0)
        d["lease_wait_max_ns"] = a.get("wait_max_ns", 0)
        d["lease_hold_max_ns"] = a.get("hold_max_ns", 0)
        d["lease_waiters_max"] = a.get("waiters_max", 0)
        n = d["lease_acquisitions"]
        if n > 0:
            d["lease_wait_mean_us"] = round(d["lease_wait_ns"] / n / 1e3, 3)
            d["lease_hold_mean_us"] = round(d["lease_hold_ns"] / n / 1e3, 3)
            d["lease_contended_frac"] = round(d["lease_waited"] / n, 4)
        return d


class Power:
    """Package RAPL, accumulated across samples.

    The counter wraps at max_energy_range_uj -- 65533 J on this machine, which
    at ~110 W is every ~596 s. Taking one delta across a long window and adding
    a single wrap silently UNDERCOUNTS: with more than one wrap, or with a wrap
    that still leaves a positive raw delta, the correction never fires. That is
    exactly what happened to the first 900 s soak, which reported 43.2 W for a
    load independently measured at ~117 W.

    So energy is accumulated from the same 0.5 s poll that samples GPU busy,
    where each interval is far shorter than a wrap period and a negative delta
    unambiguously means one wrap."""

    def __enter__(self):
        self.t0 = time.time()
        self.wrap = read_int(RAPL_MAX, 0)
        self._e_prev = read_int(RAPL)
        self.energy_uj = 0
        self.gpu = []
        self._stop = False
        self._t = threading.Thread(target=self._poll, daemon=True)
        self._t.start()
        return self

    def _accumulate(self):
        e = read_int(RAPL)
        d = e - self._e_prev
        if d < 0:
            d += self.wrap
        self.energy_uj += d
        self._e_prev = e

    def _poll(self):
        while not self._stop:
            self.gpu.append(read_int(GPU_BUSY))
            self._accumulate()
            time.sleep(0.5)

    def __exit__(self, *a):
        self._stop = True
        self._t.join(timeout=2)
        self._accumulate()
        dt = time.time() - self.t0
        self.watts = round((self.energy_uj / 1e6) / dt, 1) if dt > 0 else None
        self.wall = round(dt, 2)
        self.gpu_busy_med = st.median(self.gpu) if self.gpu else None


def drive(make_req, n, concurrency):
    """Fixed-concurrency closed loop: `concurrency` workers each pulling from a
    shared counter until n requests are done. Closed-loop rather than open, so
    an overloaded service produces longer latencies rather than an unbounded
    backlog -- which is what a real caller with a connection pool sees."""
    out, lock = [], threading.Lock()
    counter = {"i": 0}

    def worker():
        while True:
            with lock:
                i = counter["i"]
                if i >= n:
                    return
                counter["i"] = i + 1
            r = make_req(i, concurrency)
            with lock:
                out.append(r)

    ts = [threading.Thread(target=worker) for _ in range(concurrency)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out, time.time() - t0


def start_verifier():
    """Launch the control-plane tenant for the duration of a load window.

    Started with a generous limit and STOPPED when the load finishes, so its
    measurement window matches the load exactly. The previous pass ran a fixed
    25 s tenant that outlived the benchmark and made average power
    incomparable; that mistake is not repeated."""
    import subprocess
    return dict(t0=time.time(),
                proc=subprocess.Popen(
                    ["bb", str(REPO / "tools" / "cpu_tenant.clj"),
                     "secs", "100000"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL, text=True, cwd=str(REPO),
                    start_new_session=True))


def stop_verifier(v):
    """SIGINT so babashka runs its shutdown and prints the summary line; the
    tenant loop is deadline-based, so a hard kill would lose the numbers."""
    import signal, os as _os
    p = v["proc"]
    try:
        _os.killpg(_os.getpgid(p.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        out = p.communicate(timeout=20)[0] or ""
    except Exception:
        p.kill(); out = ""
    line = next((l for l in out.splitlines()
                 if l.strip().startswith("{") and "ops_per_s" in l), None)
    if not line:
        return {"verifier": "no-output"}
    d = json.loads(line)
    return {f"verifier_{k}": v2 for k, v2 in d.items()}


def write_rows(path, rows):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def append_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def set_threads(n):
    """The controller service is restarted by the caller for a new width; this
    only records the intent. llama-server fixes -t at startup."""
    return n


def cell(label, make_req, n, conc, lease_csv, jsonl, threads):
    lw = LeaseWindow(lease_csv)
    with Power() as pw, lw:
        reqs, wall = drive(make_req, n, conc)
    rows = [r.row() for r in reqs]
    for r in rows:
        r["arm"] = label
    append_jsonl(jsonl, rows)
    ok = [r for r in rows if not r.get("err")]
    rec = dict(arm=label, threads=threads, concurrency=conc, requests=len(rows),
               errors=len(rows) - len(ok), wall_s=round(wall, 2),
               req_per_s=round(len(ok) / wall, 3) if wall else None,
               watts=pw.watts, gpu_busy_med=pw.gpu_busy_med)
    tot_gen = sum(r.get("gen_n", 0) or 0 for r in ok)
    rec["gen_tok_s_agg"] = round(tot_gen / wall, 2) if wall else None
    tot_pp = sum(r.get("prompt_n", 0) or 0 for r in ok)
    rec["prompt_tok_s_agg"] = round(tot_pp / wall, 2) if wall else None
    for k in ("total_ms", "ttft_ms", "queue_ms", "service_ms", "chain_ms"):
        rec.update(summarize(ok, k))
    rec.update(lw.delta())
    return rec, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["baseline", "concurrency", "chain",
                                     "mixed", "soak"])
    ap.add_argument("--mix", default="C:1",
                    help="request-class mixture, e.g. C:1,CW:1 or C:1,CW:2,W:1")
    ap.add_argument("--verifier", action="store_true",
                    help="run the Clojure/SCI control-plane tenant DURING the "
                         "load window, sized to the load rather than a fixed "
                         "wall time -- a fixed-duration tenant outliving the "
                         "benchmark is what made power incomparable last pass")
    ap.add_argument("--soak-s", type=int, default=1800)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--conc", type=int, nargs="+", default=[1])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ctrl-predict", type=int, default=32)
    ap.add_argument("--work-predict", type=int, default=128)
    ap.add_argument("--lease-csv", default=None)
    ap.add_argument("--outdir", default="artifacts/service-cotenancy")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    prompt = controller_prompt()
    jsonl = f"{a.outdir}/requests.jsonl"
    out = []

    if a.mode == "baseline":
        for cls in ("C", "W"):
            mk = ((lambda i, c: run_controller(f"C{i}", a.threads, c,
                                               a.ctrl_predict, prompt))
                  if cls == "C" else
                  (lambda i, c: run_worker(f"W{i}", c, a.work_predict)))
            rec, _ = cell(f"baseline-{cls}-t{a.threads}", mk, a.n, 1,
                          a.lease_csv, jsonl, a.threads)
            out.append(rec)
            print(f"  {rec['arm']:24s} n={rec['requests']} "
                  f"ttft_p50={rec.get('ttft_ms_p50')} "
                  f"total_p50={rec.get('total_ms_p50')} "
                  f"total_p95={rec.get('total_ms_p95')} "
                  f"{rec['watts']}W", flush=True)

    elif a.mode == "concurrency":
        for c in a.conc:
            mk = lambda i, cc: run_controller(f"C{i}", a.threads, cc,
                                              a.ctrl_predict, prompt)
            rec, _ = cell(f"conc-t{a.threads}-c{c}", mk, a.n, c,
                          a.lease_csv, jsonl, a.threads)
            out.append(rec)
            print(f"  t{a.threads} c={c:<2} req/s={rec['req_per_s']:<6} "
                  f"ttft p50={rec.get('ttft_ms_p50')} p95={rec.get('ttft_ms_p95')} "
                  f"total p50={rec.get('total_ms_p50')} p95={rec.get('total_ms_p95')} "
                  f"queue p95={rec.get('queue_ms_p95')} "
                  f"leasewait={rec.get('lease_wait_mean_us')}us "
                  f"cont={rec.get('lease_contended_frac')} {rec['watts']}W",
                  flush=True)

    elif a.mode == "chain":
        for c in a.conc:
            mk = lambda i, cc: run_chain(f"CW{i}", a.threads, cc,
                                         a.ctrl_predict, a.work_predict, prompt)
            rec, _ = cell(f"chain-t{a.threads}-c{c}", mk, a.n, c,
                          a.lease_csv, jsonl, a.threads)
            out.append(rec)
            print(f"  chain t{a.threads} c={c} chain p50={rec.get('chain_ms_p50')} "
                  f"p95={rec.get('chain_ms_p95')} {rec['watts']}W", flush=True)

    elif a.mode in ("mixed", "soak"):
        classes = []
        for part in a.mix.split(","):
            name, _, w = part.partition(":")
            classes += [name.strip()] * int(w or 1)

        def make(i, cc):
            cls = classes[i % len(classes)]
            if cls == "C":
                return run_controller(f"C{i}", a.threads, cc, a.ctrl_predict, prompt)
            if cls == "W":
                return run_worker(f"W{i}", cc, a.work_predict)
            return run_chain(f"CW{i}", a.threads, cc, a.ctrl_predict,
                             a.work_predict, prompt)

        for c in a.conc:
            ver = None
            if a.verifier:
                ver = start_verifier()
            rec, rows = cell(f"{a.mode}-t{a.threads}-c{c}-[{a.mix}]", make,
                             a.n, c, a.lease_csv, jsonl, a.threads)
            if ver:
                rec.update(stop_verifier(ver))
            rec["mix"] = a.mix
            # Per-class distributions: an aggregate hides which class suffered.
            for cls in sorted(set(classes)):
                sub = [r for r in rows if r["cls"] == cls and not r.get("err")]
                if not sub:
                    continue
                rec[f"{cls}_n"] = len(sub)
                for k in ("total_ms", "ttft_ms", "queue_ms", "chain_ms"):
                    for kk, vv in summarize(sub, k).items():
                        rec[f"{cls}_{kk}"] = vv
            out.append(rec)
            print(f"  {a.mode} t{a.threads} c={c} [{a.mix}] req/s={rec['req_per_s']} "
                  f"total p50={rec.get('total_ms_p50')} p95={rec.get('total_ms_p95')} "
                  f"{rec['watts']}W ver_ops={rec.get('verifier_ops_per_s')} "
                  f"ver_p95={rec.get('verifier_p95_ms')}", flush=True)

    tagged = f"_{a.tag}" if a.tag else ""
    write_rows(f"{a.outdir}/{a.mode}{tagged}.csv", out)
    print(f"\nwrote {a.outdir}/{a.mode}{tagged}.csv")


if __name__ == "__main__":
    main()
