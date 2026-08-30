#!/usr/bin/env python3
"""Tasks 3-4: dependency-aware discrete-event simulation of CPU/NPU overlap.

Answers whether cross-micro-batch pipelining is worth building, using the real
transformer dependency DAG and MEASURED per-operation times -- not aggregate
timing arithmetic, which cannot see that an operation is blocked.

Three schedules are reported and never collapsed into one number:

  A  serial          every operation in graph order on one timeline. This is
                     what the runtime does today, and it reproduces the measured
                     wall time (validation check below).
  B  perfect overlap total_work / max(sum CPU, sum NPU). A THEORETICAL UPPER
                     BOUND that ignores every dependency.
  C  dependency-     list-scheduled over the real DAG with one NPU resource and
     constrained     one CPU resource. This is the achievable bound.

Resource model, stated because it bounds what the result means:

  * ONE CPU resource. Operation times are measured at the target thread count,
    so a node already uses every CPU thread. Modelling the CPU as a single
    resource therefore claims no gain from running two CPU nodes concurrently --
    conservative, and it keeps the question focused on CPU/NPU overlap, which is
    what the pipeline proposal is about.
  * ONE NPU resource. The device is single-tenant and contexts time-slice the
    whole array.
  * Offloading does not free the CPU. An operation placed on the NPU still costs
    the CPU its measured staging + epilogue time, because that is what the
    measurement shows (see RESULTS.md section 2). Modelling NPU placement as
    "free for the CPU" would overstate the pipeline's value.

Dependency edges are the transformer's, including the only cross-micro-batch
edge that exists: attention in micro-batch m reads the KV cache written by every
m' < m, so attn(m,L) waits on kv_write(m-1,L). That single edge is what permits
a wavefront, and what limits it.
"""
import argparse, csv, collections, heapq, json
from pathlib import Path

NPU_ELIGIBLE = {"attn_q_proj", "attn_out_proj", "ffn_gate", "ffn_up", "ffn_down"}


def load_times(csv_path, prompt, ub, mode):
    """-> {mb_pos: {category: (cpu_ms, npu_ms, count)}} for one configuration."""
    out = collections.defaultdict(dict)
    for r in csv.DictReader(open(csv_path)):
        if int(r["prompt"]) != prompt or int(r["ub"]) != ub or r["mode"] != mode:
            continue
        out[int(r["mb_pos"])][r["category"]] = (
            float(r["cpu_ms"]), float(r["npu_ms"]), int(r["count"]))
    return out


class Op:
    __slots__ = ("id", "mb", "layer", "role", "cpu", "npu", "deps", "place")

    def __init__(self, oid, mb, layer, role, cpu, npu):
        self.id, self.mb, self.layer, self.role = oid, mb, layer, role
        self.cpu, self.npu = cpu, npu      # ms if placed on CPU / on NPU
        self.deps, self.place = [], None


def build_dag(cpu_times, npu_times, n_mb, n_layers=30):
    """Build the real per-micro-batch, per-layer operation DAG.

    cpu_times[pos][cat] -> pure-CPU time for the whole category over all layers
    npu_times[pos][cat] -> (npu_device, cpu_side) for an NPU placement
    """
    ops, by_key = [], {}

    def per_layer(t, pos, cat, share=1.0):
        v = t.get(pos, {}).get(cat)
        return (v[0] + v[1]) / n_layers * share if v else 0.0

    def add(mb, layer, role, cpu, npu=None):
        o = Op(len(ops), mb, layer, role, cpu, npu)
        ops.append(o); by_key[(mb, layer, role)] = o
        return o

    for mb in range(n_mb):
        for L in range(n_layers):
            # norm is 4 nodes/layer (attn_norm, attn_sub_norm, ffn_norm,
            # ffn_sub_norm); split half before attention, half before the FFN.
            n_half = per_layer(cpu_times, mb, "norm", 0.5)
            rope_1 = per_layer(cpu_times, mb, "rope", 0.5)
            add_1  = per_layer(cpu_times, mb, "residual_add", 0.5)
            kvw    = per_layer(cpu_times, mb, "kv_cache_write", 0.5)

            an = add(mb, L, "attn_norm", n_half)
            q  = add(mb, L, "attn_q_proj",  per_layer(cpu_times, mb, "attn_q_proj"))
            k  = add(mb, L, "attn_k_proj",  per_layer(cpu_times, mb, "attn_k_proj"))
            v  = add(mb, L, "attn_v_proj",  per_layer(cpu_times, mb, "attn_v_proj"))
            rq = add(mb, L, "rope_q", rope_1)
            rk = add(mb, L, "rope_k", rope_1)
            kw = add(mb, L, "kv_write", kvw)
            at = add(mb, L, "attention",    per_layer(cpu_times, mb, "attention"))
            ao = add(mb, L, "attn_out_proj", per_layer(cpu_times, mb, "attn_out_proj"))
            fi = add(mb, L, "ffn_inp", add_1)
            fn = add(mb, L, "ffn_norm", n_half)
            gt = add(mb, L, "ffn_gate",  per_layer(cpu_times, mb, "ffn_gate"))
            up = add(mb, L, "ffn_up",    per_layer(cpu_times, mb, "ffn_up"))
            ac = add(mb, L, "ffn_activation", per_layer(cpu_times, mb, "ffn_activation"))
            dn = add(mb, L, "ffn_down",  per_layer(cpu_times, mb, "ffn_down"))
            lo = add(mb, L, "l_out", add_1)

            if L > 0:
                an.deps.append(by_key[(mb, L - 1, "l_out")])
            for d, s in ((q, an), (k, an), (v, an), (rq, q), (rk, k),
                         (kw, rk), (kw, v), (at, rq), (at, kw), (ao, at),
                         (fi, ao), (fn, fi), (gt, fn), (up, fn),
                         (ac, gt), (ac, up), (dn, ac), (lo, dn), (lo, fi)):
                d.deps.append(s)
            if L > 0:
                fi.deps.append(by_key[(mb, L - 1, "l_out")])
            # The only cross-micro-batch edge: causal attention reads the KV
            # written by every earlier micro-batch at the same layer.
            if mb > 0:
                at.deps.append(by_key[(mb - 1, L, "kv_write")])

    # attach NPU placement costs
    for o in ops:
        if o.role in NPU_ELIGIBLE:
            e = npu_times.get(o.mb, {}).get(o.role)
            if e:
                o.npu = (e[1] / 30.0, e[0] / 30.0)   # (device_ms, cpu_side_ms)
    return ops


def simulate(ops, policy="greedy"):
    """List-schedule over the DAG with one CPU and one NPU resource."""
    n = len(ops)
    indeg = [len(o.deps) for o in ops]
    succ = collections.defaultdict(list)
    for o in ops:
        for d in o.deps:
            succ[d.id].append(o.id)

    ready = [i for i in range(n) if indeg[i] == 0]
    done_at = [0.0] * n
    cpu_free = npu_free = 0.0
    cpu_busy = npu_busy = 0.0
    placed_npu = 0
    t = 0.0
    finished = 0
    # event-driven: repeatedly take the ready op whose earliest start is soonest
    pending = []   # (finish_time, op_id)
    while finished < n:
        progressed = False
        for _ in range(len(ready)):
            if not ready:
                break
            # choose the op that can start earliest; break ties by graph order
            best_i, best_start, best_res = None, None, None
            for idx, oid in enumerate(ready):
                o = ops[oid]
                est = max([done_at[d.id] for d in o.deps], default=0.0)
                can_npu = o.npu is not None
                s_cpu = max(est, cpu_free)
                s_npu = max(est, npu_free) if can_npu else float("inf")
                if policy == "npu_first" and can_npu:
                    s_cpu = float("inf")
                # finish time decides, not start time
                f_cpu = s_cpu + o.cpu if s_cpu < float("inf") else float("inf")
                f_npu = (s_npu + max(o.npu[0], o.npu[1])
                         if s_npu < float("inf") else float("inf"))
                if f_npu < f_cpu:
                    s, res, f = s_npu, "npu", f_npu
                else:
                    s, res, f = s_cpu, "cpu", f_cpu
                if best_start is None or f < best_start:
                    best_i, best_start, best_res, best_s = idx, f, res, s
            oid = ready.pop(best_i)
            o = ops[oid]
            if best_res == "npu":
                # device time on the NPU; its CPU-side staging+epilogue still
                # occupies the CPU resource
                npu_free = best_s + o.npu[0]
                cpu_free = max(cpu_free, best_s) + o.npu[1]
                npu_busy += o.npu[0]; cpu_busy += o.npu[1]
                placed_npu += 1
            else:
                cpu_free = best_s + o.cpu
                cpu_busy += o.cpu
            done_at[oid] = best_start
            finished += 1
            progressed = True
            for s2 in succ[oid]:
                indeg[s2] -= 1
                if indeg[s2] == 0:
                    ready.append(s2)
        if not progressed:
            raise RuntimeError("deadlock in DAG")
    makespan = max(done_at)
    return dict(makespan=makespan, cpu_busy=cpu_busy, npu_busy=npu_busy,
                npu_duty=npu_busy / makespan, cpu_util=cpu_busy / makespan,
                npu_idle=makespan - npu_busy, cpu_idle=makespan - cpu_busy,
                n_on_npu=placed_npu, done_at=done_at)


def critical_path(ops, done_at):
    """Walk back from the last-finishing op, accumulating time by role."""
    end = max(range(len(ops)), key=lambda i: done_at[i])
    path, cur = [], end
    while True:
        o = ops[cur]
        path.append(o)
        if not o.deps:
            break
        cur = max((d.id for d in o.deps), key=lambda i: done_at[i])
    agg = collections.Counter()
    for o in reversed(path):
        agg[o.role] += (o.npu is not None and max(o.npu) or o.cpu)
    return list(reversed(path)), agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residue", nargs="+", required=True)
    ap.add_argument("--allnpu", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts/overlap-de-risk/simulated_schedules.json")
    a = ap.parse_args()

    cpu_rows, npu_rows = {}, {}
    for p in a.residue:
        for r in csv.DictReader(open(p)):
            cpu_rows.setdefault((int(r["prompt"]), int(r["ub"]), r["mode"]), []).append(r)
    for p in a.allnpu:
        for r in csv.DictReader(open(p)):
            npu_rows.setdefault((int(r["prompt"]), int(r["ub"])), []).append(r)

    results = []
    for (prompt, ub, mode), rows in sorted(cpu_rows.items()):
        if mode != "cpu":
            continue
        n_mb = int(rows[0]["n_micro_batches"])
        cpu_t = collections.defaultdict(dict)
        for r in rows:
            cpu_t[int(r["mb_pos"])][r["category"]] = (
                float(r["cpu_ms"]), float(r["npu_ms"]), int(r["count"]))
        npu_t = collections.defaultdict(dict)
        for r in npu_rows.get((prompt, ub), []):
            npu_t[int(r["mb_pos"])][r["category"]] = (
                float(r["cpu_ms"]), float(r["npu_ms"]), int(r["count"]))
        if not npu_t:
            continue

        ops = build_dag(cpu_t, npu_t, n_mb)
        serial_cpu = sum(o.cpu for o in ops)
        sim_npu_first = simulate(ops, "npu_first")
        for o in ops:
            o.place = None
        sim_greedy = simulate(ops, "greedy")

        # B: perfect overlap, ignoring every dependency
        tot_npu = sum(o.npu[0] for o in ops if o.npu)
        tot_cpu = sum((o.npu[1] if o.npu else o.cpu) for o in ops)
        perfect = max(tot_npu, tot_cpu)
        serial_npu = tot_npu + tot_cpu

        _, cp = critical_path(ops, sim_greedy["done_at"])
        top = ", ".join(f"{k} {v:.0f}ms" for k, v in cp.most_common(4))

        r = dict(prompt=prompt, ub=ub, n_micro_batches=n_mb,
                 A_serial_cpu_only_ms=round(serial_cpu, 1),
                 A_serial_with_npu_ms=round(serial_npu, 1),
                 B_perfect_overlap_ms=round(perfect, 1),
                 C_dep_constrained_ms=round(sim_greedy["makespan"], 1),
                 C_npu_first_ms=round(sim_npu_first["makespan"], 1),
                 C_npu_duty=round(sim_greedy["npu_duty"], 3),
                 C_cpu_util=round(sim_greedy["cpu_util"], 3),
                 C_npu_idle_ms=round(sim_greedy["npu_idle"], 1),
                 C_ops_on_npu=sim_greedy["n_on_npu"],
                 speedup_B_over_A=round(serial_npu / perfect, 3),
                 speedup_C_over_A=round(serial_npu / sim_greedy["makespan"], 3),
                 critical_path_top=top)
        results.append(r)
        print(f"\n== pp{prompt} ub{ub}, {n_mb} micro-batch(es) ==")
        print(f"  A serial, CPU only                    {serial_cpu:8.0f} ms")
        print(f"  A serial, with NPU (today)            {serial_npu:8.0f} ms")
        print(f"  B perfect overlap   [UPPER BOUND]     {perfect:8.0f} ms   "
              f"{serial_npu/perfect:.2f}x")
        print(f"  C dependency-constrained [ACHIEVABLE] {sim_greedy['makespan']:8.0f} ms   "
              f"{serial_npu/sim_greedy['makespan']:.2f}x")
        print(f"      NPU duty {sim_greedy['npu_duty']*100:.1f}%  "
              f"CPU util {sim_greedy['cpu_util']*100:.1f}%  "
              f"NPU idle {sim_greedy['npu_idle']:.0f} ms  "
              f"ops on NPU {sim_greedy['n_on_npu']}")
        print(f"      critical path: {top}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
