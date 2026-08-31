#!/usr/bin/env python3
"""Tri-device co-tenancy: NPU controller + Vulkan GPU worker + CPU tenant.

The system question this pass exists to answer. Each device is fast in
isolation; what matters is whether they can run at the same time on one
LPDDR5X pool, and which split of CPU threads gives the best whole-system
result rather than the best isolated controller.

Arms are launched CONCURRENTLY and each reports its own throughput, so a
regression shows up on the resource that suffered rather than as a single
blended number. Power is package RAPL sampled across the whole arm.

Process discipline, because this project has repeatedly drawn false conclusions
from stale processes: every child is tracked by explicit PID and reaped by PID.
There is no pattern matching anywhere in this file -- `pkill -f` once killed
this project's own harness, and `pgrep -x` silently fails on names longer than
15 characters, which let two benchmark runs execute concurrently unnoticed.
"""
import argparse, csv, json, os, re, signal, statistics as st, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CTRL_MODEL = REPO / "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
WORK_MODEL = REPO / "models/worker/Qwen3.6-27B-UD-Q4_K_XL.gguf"
CTRL_BIN = REPO / "refs/BitNet/build-xdna/bin/llama-bench"
WORK_BIN = REPO / "refs/BitNet/build-vulkan/bin/llama-bench"

RAPL = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
RAPL_MAX = Path("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")
GPU_BUSY = Path("/sys/class/drm/card0/device/gpu_busy_percent")
GTT_USED = Path("/sys/class/drm/card0/device/mem_info_gtt_used")

TS_RE = re.compile(r"\|\s*([\d.]+)\s*±\s*([\d.]+)\s*\|\s*$")


def read_int(p, default=0):
    try:
        return int(p.read_text().strip())
    except Exception:
        return default


class Power:
    """Package RAPL. The `core` subdomain is unusable on this SoC (measured
    earlier in this project), so package is the only trustworthy counter."""

    def __init__(self):
        self.ok = RAPL.exists() and os.access(RAPL, os.R_OK)
        self.wrap = read_int(RAPL_MAX, 0)

    def __enter__(self):
        self.t0 = time.time()
        self.e0 = read_int(RAPL) if self.ok else 0
        return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t0
        e1 = read_int(RAPL) if self.ok else 0
        de = e1 - self.e0
        if de < 0 and self.wrap:
            de += self.wrap
        self.watts = (de / 1e6) / self.dt if self.ok and self.dt > 0 else None


def sample_system(stop_at, out):
    """Poll GPU busy / GTT while an arm runs. Cheap sysfs reads only."""
    while time.time() < stop_at:
        out["gpu_busy"].append(read_int(GPU_BUSY))
        out["gtt_gib"].append(read_int(GTT_USED) / 1073741824)
        time.sleep(0.5)


def parse_bench(text):
    """llama-bench table rows -> {test: (t/s, sd)}."""
    out = {}
    for line in text.splitlines():
        if not line.startswith("| ") or "t/s" in line or "---" in line:
            continue
        cols = [c.strip() for c in line.split("|")]
        m = TS_RE.search(line)
        if not m:
            continue
        test = next((c for c in cols if re.fullmatch(r"(pp|tg)\d+", c)), None)
        if test:
            out[test] = (float(m.group(1)), float(m.group(2)))
    return out


class Child:
    """A tracked subprocess. Reaped by PID, never by pattern."""

    def __init__(self, label, argv, env=None, group=None):
        self.label = label
        self.argv = (["sg", group, "-c", " ".join(argv)] if group else argv)
        self.env = {**os.environ, **(env or {})}
        self.proc = None
        self.out = ""

    def start(self):
        self.proc = subprocess.Popen(
            self.argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=self.env, text=True,
            cwd=str(REPO), start_new_session=True)
        return self.proc.pid

    def wait(self, timeout):
        try:
            self.out = self.proc.communicate(timeout=timeout)[0] or ""
        except subprocess.TimeoutExpired:
            self.kill()
            self.out = "TIMEOUT"
        return self.proc.returncode

    def kill(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                time.sleep(2)
                if self.proc.poll() is None:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


def controller_cmd(threads, prompt, reps):
    return [str(CTRL_BIN), "-m", str(CTRL_MODEL), "-p", str(prompt), "-n", "0",
            "-t", str(threads), "-ngl", "0", "-ub", "2048", "-r", str(reps)]


def worker_cmd(pp, tg, reps):
    return [str(WORK_BIN), "-m", str(WORK_MODEL), "-ngl", "99",
            "-p", str(pp), "-n", str(tg), "-r", str(reps)]


def run_arm(label, children, timeout=3600):
    """Launch every child at once, wait for all, sample the system throughout."""
    sysinfo = {"gpu_busy": [], "gtt_gib": []}
    pids = {}
    with Power() as pw:
        for c in children:
            pids[c.label] = c.start()
        import threading
        stop = time.time() + timeout
        t = threading.Thread(target=sample_system, args=(stop, sysinfo), daemon=True)
        t.start()
        for c in children:
            c.wait(timeout)
        stop_at = time.time()
    sysinfo["gpu_busy"] = [g for g in sysinfo["gpu_busy"]]
    rec = dict(arm=label, wall_s=round(pw.dt, 2),
               watts=round(pw.watts, 1) if pw.watts else None,
               pids=pids,
               gpu_busy_median=(st.median(sysinfo["gpu_busy"])
                                if sysinfo["gpu_busy"] else None),
               gpu_busy_max=(max(sysinfo["gpu_busy"])
                             if sysinfo["gpu_busy"] else None),
               gtt_gib_max=(round(max(sysinfo["gtt_gib"]), 2)
                            if sysinfo["gtt_gib"] else None))
    for c in children:
        if c.label == "tenant":
            # The CPU tenant reports one JSON line: ops, ops_per_s, p50, p95.
            # parse_bench only understands llama-bench tables, so without this
            # the tenant's own numbers are silently dropped.
            line = next((l for l in c.out.splitlines()
                         if l.strip().startswith("{") and "ops_per_s" in l), None)
            rec[c.label] = json.loads(line) if line else {}
        else:
            rec[c.label] = parse_bench(c.out)
        if "TIMEOUT" in c.out:
            rec[c.label] = "TIMEOUT"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--prompt", type=int, default=2048)
    ap.add_argument("--ctrl-reps", type=int, default=3)
    ap.add_argument("--worker-pp", type=int, default=512)
    ap.add_argument("--worker-tg", type=int, default=64)
    ap.add_argument("--worker-reps", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--render-group", default="render")
    ap.add_argument("--out", default="artifacts/gpu-cotenancy/tri_device.csv")
    a = ap.parse_args()

    def ctrl(t, npu=True):
        return Child("controller", controller_cmd(t, a.prompt, a.ctrl_reps),
                     env={"BITNET_XDNA": "1" if npu else "0",
                          "BITNET_XDNA_STATS": "1"})

    def work():
        return Child("worker", worker_cmd(a.worker_pp, a.worker_tg,
                                          a.worker_reps),
                     group=a.render_group)

    # Arms, in the brief's lettering. Baselines first so a drifting machine is
    # visible in the repeats rather than hidden in a single up-front block.
    arms = [("A  gpu alone", lambda: [work()])]
    for t in a.threads:
        arms.append((f"ctrl-npu t{t} alone", lambda t=t: [ctrl(t)]))
    arms.append(("B  gpu + ctrl-cpu t8", lambda: [work(), ctrl(8, npu=False)]))
    for t in a.threads:
        letter = {4: "C", 6: "D", 8: "E"}.get(t, "?")
        arms.append((f"{letter}  gpu + ctrl-npu t{t}",
                     lambda t=t: [work(), ctrl(t)]))

    rows = []
    print(f"tri-device matrix: {a.rounds} rounds, arms interleaved")
    print(f"  controller pp{a.prompt} -ub 2048 -r {a.ctrl_reps}   "
          f"worker pp{a.worker_pp}/tg{a.worker_tg} -r {a.worker_reps}\n")
    for r in range(a.rounds):
        for label, mk in arms:
            rec = run_arm(label, mk())
            rec["round"] = r
            rows.append(rec)
            c = rec.get("controller") or {}
            w = rec.get("worker") or {}
            cs = (f"{c[f'pp{a.prompt}'][0]:7.1f}"
                  if isinstance(c, dict) and f"pp{a.prompt}" in c else "      -")
            wp = (f"{w[f'pp{a.worker_pp}'][0]:7.1f}"
                  if isinstance(w, dict) and f"pp{a.worker_pp}" in w else "      -")
            wt = (f"{w[f'tg{a.worker_tg}'][0]:6.2f}"
                  if isinstance(w, dict) and f"tg{a.worker_tg}" in w else "     -")
            print(f"  r{r} {label:24s} ctrl {cs}  gpu_pp {wp}  gpu_tg {wt}  "
                  f"{rec['wall_s']:6.1f}s  {rec['watts'] or 0:5.1f}W  "
                  f"gtt {rec['gtt_gib_max'] or 0:5.1f}G", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    flat = []
    for r in rows:
        d = {k: v for k, v in r.items() if k not in ("controller", "worker", "pids")}
        for who in ("controller", "worker"):
            v = r.get(who)
            if isinstance(v, dict):
                for test, (ts, sd) in v.items():
                    d[f"{who}_{test}"] = ts
                    d[f"{who}_{test}_sd"] = sd
            elif v:
                d[f"{who}_status"] = v
        flat.append(d)
    keys, seen = [], set()
    for d in flat:
        for k in d:
            if k not in seen:
                seen.add(k); keys.append(k)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for d in flat: w.writerow({k: d.get(k, "") for k in keys})
    Path(a.out).with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
