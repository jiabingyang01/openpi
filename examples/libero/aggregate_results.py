"""Aggregate per-worker JSON results from main_parallel.py.

Combines the per-task raw timing lists across all workers and reports the
two standard VLA benchmark metrics alongside success rate:

  * VLA inference latency  -- wall time of one client.infer() call (ms)
  * Control frequency      -- env.step() rate as seen by the policy (Hz)

plus per-episode wall time and total rollout throughput.

Outputs both a human-readable summary.txt and a machine-readable combined.json.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from statistics import mean, median, pstdev


def _pct(x):
    xs = sorted(x)
    if not xs:
        return 0.0
    k = int(round(0.95 * (len(xs) - 1)))
    return xs[k]


def _stats(xs):
    if not xs:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(xs),
        "mean": mean(xs),
        "median": median(xs),
        "p95": _pct(xs),
        "std": pstdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default=None,
                   help="glob for per-worker JSONs, e.g. 'logs/EVAL-*-worker*.json'")
    p.add_argument("--files", nargs="*", default=None)
    p.add_argument("--out", default=None, help="combined JSON output")
    p.add_argument("--log", default=None, help="human-readable summary output")
    args = p.parse_args()

    files = list(args.files or [])
    if args.pattern:
        files.extend(sorted(glob.glob(args.pattern)))
    files = sorted(set(files))
    if not files:
        print("No input files.", file=sys.stderr)
        sys.exit(2)

    per_task_succ = defaultdict(int)
    per_task_eps = defaultdict(int)
    per_task_name = {}
    per_task_lat = defaultdict(list)       # ms
    per_task_ep_walls = defaultdict(list)  # s
    per_task_ctrl_steps = defaultdict(int)
    per_task_ctrl_wall = defaultdict(float)
    per_task_skipvla_skip = defaultdict(int)
    per_task_skipvla_call = defaultdict(int)
    per_task_mlp_lat = defaultdict(list)

    worker_totals = {}  # wid -> (succ, eps)
    run_id = suite = None
    num_trials = None
    action_chunk_len = None   # actions the policy returns per infer() (e.g. 50 for pi0)
    replan_steps = None       # actions actually consumed per infer() before re-query (e.g. 5)

    for path in files:
        with open(path) as f:
            data = json.load(f)
        run_id = run_id or data.get("run_id")
        suite = suite or data.get("task_suite_name")
        num_trials = num_trials or data.get("num_trials_per_task")
        wid = data["worker_id"]
        worker_totals[wid] = (data["total_successes"], data["total_episodes"])
        for tid_str, rec in data["per_task"].items():
            tid = int(tid_str)
            per_task_succ[tid] += rec["successes"]
            per_task_eps[tid] += rec["episodes"]
            per_task_name.setdefault(tid, rec.get("task_description", ""))
            per_task_lat[tid].extend(rec.get("vla_latencies_ms", []))
            per_task_ep_walls[tid].extend(rec.get("episode_walls_s", []))
            per_task_ctrl_steps[tid] += rec.get("control_steps", 0)
            per_task_ctrl_wall[tid] += rec.get("control_wall_s", 0.0)
            per_task_skipvla_skip[tid] += rec.get("skipvla_skip", 0)
            per_task_skipvla_call[tid] += rec.get("skipvla_call", 0)
            per_task_mlp_lat[tid].extend(rec.get("mlp_latencies_ms", []))
            if action_chunk_len is None:
                action_chunk_len = rec.get("action_chunk_len")
            if replan_steps is None:
                replan_steps = rec.get("replan_steps")

    total_s = sum(per_task_succ.values())
    total_e = sum(per_task_eps.values())
    all_lat = []
    for v in per_task_lat.values():
        all_lat.extend(v)
    all_ep_wall = []
    for v in per_task_ep_walls.values():
        all_ep_wall.extend(v)
    total_ctrl_steps = sum(per_task_ctrl_steps.values())
    total_ctrl_wall = sum(per_task_ctrl_wall.values())
    total_vla_calls = len(all_lat) + len(files)  # aggregator drops 1 lat/worker (warmup); approximate
    # More accurate: per-task vla_call_count = len(per_task_lat[tid]) + episodes_in_task (warmup drops)
    total_vla_calls = sum(len(v) for v in per_task_lat.values()) + (len(files) * num_trials if num_trials else 0)
    # Even simpler: infer from actual data
    total_skipvla_skip = sum(per_task_skipvla_skip.values())
    total_skipvla_call = sum(per_task_skipvla_call.values())
    # True VLA call count is what's recorded in per_task_lat (but with 1-per-episode warmup drop).
    # Reconstruct: count_in_lat + episodes_counted_by_warmup. If skip stats present, prefer skipvla_call.
    if total_skipvla_call + total_skipvla_skip > 0:
        total_vla_calls = total_skipvla_call
    else:
        total_vla_calls = sum(len(v) for v in per_task_lat.values())

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"=== Parallel LIBERO eval aggregate ===")
    emit(f"run_id: {run_id}")
    emit(f"suite:  {suite}   workers: {len(files)}   trials/task: {num_trials}")
    if action_chunk_len is not None:
        emit(f"chunk:  action_chunk_len={action_chunk_len} (policy-returned)  "
             f"replan_steps={replan_steps} (consumed before re-query)")
    emit("")

    # Throughput (OFT convention): actions per second delivered to the control loop
    #   = (actions consumed per VLA call) / (VLA latency)
    # For baseline this = replan_steps / latency. For Skip-VLA it rises because each
    # VLA call effectively covers more steps (skipped boundaries don't trigger infer()).
    # Formula: actions_per_call = total_control_steps / total_vla_calls
    def _throughput(lat_mean_ms, k):
        if lat_mean_ms <= 0 or k is None or k == 0:
            return 0.0
        return k / (lat_mean_ms / 1000.0)

    header = (f"  {'task':>4}  {'succ/N':>8}  {'%':>5}  "
              f"{'VLA_lat mean(p50|p95) ms':>26}  "
              f"{'eff_lat / throughput':>24}  "
              f"{'ctrl_Hz':>7}  "
              f"{'ep_wall':>8}")
    emit("Per-subtask:")
    emit(header)
    for tid in sorted(per_task_eps):
        s, e = per_task_succ[tid], per_task_eps[tid]
        lat = _stats(per_task_lat[tid])
        ctrl_hz = (per_task_ctrl_steps[tid] / per_task_ctrl_wall[tid]) if per_task_ctrl_wall[tid] > 0 else 0.0
        epw = _stats(per_task_ep_walls[tid])
        pct = (s / e * 100) if e else 0.0
        lat_col = f"{lat['mean']:5.1f} ({lat['median']:5.1f}|{lat['p95']:5.1f})"
        # OFT convention: effective_latency = weighted avg of VLA and MLP call times.
        n_vla = per_task_skipvla_call[tid] if per_task_skipvla_call[tid] > 0 else len(per_task_lat[tid])
        n_mlp = per_task_skipvla_skip[tid]
        mlp_lat_mean = (sum(per_task_mlp_lat[tid]) / len(per_task_mlp_lat[tid])) if per_task_mlp_lat[tid] else 0.0
        if n_vla + n_mlp > 0:
            eff_lat = (lat['mean'] * n_vla + mlp_lat_mean * n_mlp) / (n_vla + n_mlp)
        else:
            eff_lat = lat['mean']
        thru_oft = _throughput(eff_lat, action_chunk_len)
        emit(f"  {tid:>4}  {s:>3}/{e:<4}  {pct:>4.1f}%  "
             f"{lat_col:>26}  "
             f"eff_lat={eff_lat:5.1f}ms  thru={thru_oft:>6.1f}a/s  "
             f"{ctrl_hz:>5.2f}Hz  "
             f"{epw['mean']:>6.2f}s   -- {per_task_name.get(tid, '')}")

    emit("")
    overall_pct = (total_s / total_e * 100) if total_e else 0.0
    overall_lat = _stats(all_lat)
    overall_hz = (total_ctrl_steps / total_ctrl_wall) if total_ctrl_wall > 0 else 0.0
    overall_epw = _stats(all_ep_wall)
    # OFT convention: effective per-call latency is a weighted avg across VLA and MLP calls.
    # throughput = action_chunk_len / effective_latency. For baseline (no skip), MLP count = 0
    # and effective_latency == VLA latency.
    all_mlp_lat = [x for v in per_task_mlp_lat.values() for x in v]
    mlp_lat_mean = (sum(all_mlp_lat) / len(all_mlp_lat)) if all_mlp_lat else 0.0
    n_vla_total = total_vla_calls  # already set below; we'll recompute here for clarity
    n_vla_total = total_skipvla_call if (total_skipvla_call + total_skipvla_skip) > 0 else len(all_lat)
    n_mlp_total = total_skipvla_skip
    if n_vla_total + n_mlp_total > 0:
        effective_latency_ms = (overall_lat['mean'] * n_vla_total + mlp_lat_mean * n_mlp_total) / (n_vla_total + n_mlp_total)
    else:
        effective_latency_ms = overall_lat['mean']
    thru_oft = _throughput(effective_latency_ms, action_chunk_len)
    thru_baseline = _throughput(overall_lat['mean'], action_chunk_len)
    # Real end-to-end throughput: how many actions actually executed per wall second.
    # Unlike `replan_steps / VLA_latency`, this counts skipped steps too, so Skip-VLA
    # acceleration actually shows up here.
    effective_replan = (total_ctrl_steps / total_vla_calls) if total_vla_calls else 0.0
    vla_call_rate = (total_vla_calls / total_ctrl_steps * 100) if total_ctrl_steps else 0.0
    skip_rate_global = (total_skipvla_skip / max(total_skipvla_skip + total_skipvla_call, 1)) * 100

    emit(f"OVERALL:")
    emit(f"  success             : {total_s}/{total_e} ({overall_pct:.2f}%)")
    emit(f"  VLA latency         : mean={overall_lat['mean']:.2f} ms   "
         f"median={overall_lat['median']:.2f}   "
         f"p95={overall_lat['p95']:.2f}   "
         f"std={overall_lat['std']:.2f}   n={overall_lat['n']}")
    emit(f"  REAL control Hz     : {overall_hz:.2f} Hz = "
         f"{total_ctrl_steps} steps / {total_ctrl_wall:.1f} s "
         f"(aggregate over all workers; this is the true end-to-end action throughput)")
    emit(f"  VLA call rate       : {vla_call_rate:.2f}% of steps trigger a fresh infer()  "
         f"(effective replan_steps = {effective_replan:.2f}; baseline w/o skip ≈ {replan_steps})")
    if total_skipvla_skip + total_skipvla_call > 0:
        emit(f"  Skip-VLA skip rate  : {skip_rate_global:.2f}% of chunk boundaries "
             f"({total_skipvla_skip}/{total_skipvla_skip + total_skipvla_call})")
    emit(f"  effective latency   : {effective_latency_ms:.2f} ms  "
         f"(= (VLA_lat × {n_vla_total} + MLP_lat × {n_mlp_total}) / {n_vla_total + n_mlp_total}; "
         f"MLP_lat_mean = {mlp_lat_mean:.3f} ms)")
    emit(f"  throughput (OFT)    : {thru_oft:.1f} actions/s "
         f"(= action_chunk_len={action_chunk_len} / effective_latency); "
         f"baseline-equivalent (no skip) would be {thru_baseline:.1f} a/s")
    emit(f"  episode wall        : mean={overall_epw['mean']:.2f} s   "
         f"median={overall_epw['median']:.2f}   "
         f"p95={overall_epw['p95']:.2f}")

    emit("")
    emit("Worker contributions:")
    for wid in sorted(worker_totals):
        s, e = worker_totals[wid]
        pct = (s / e * 100) if e else 0.0
        emit(f"  w{wid}: {s}/{e} ({pct:.1f}%)")

    if num_trials is not None:
        incomplete = [tid for tid in per_task_eps if per_task_eps[tid] != num_trials]
        if incomplete:
            emit("")
            emit(f"WARNING: task(s) {incomplete} did not reach num_trials_per_task={num_trials}")

    if args.out:
        combined = {
            "run_id": run_id,
            "task_suite_name": suite,
            "num_trials_per_task": num_trials,
            "total_successes": total_s,
            "total_episodes": total_e,
            "overall_success_rate": overall_pct / 100.0,
            "action_chunk_len": action_chunk_len,
            "replan_steps": replan_steps,
            "overall_vla_latency_ms": overall_lat,
            "overall_throughput_actions_per_s": {
                "oft_effective": thru_oft,  # action_chunk_len / effective_latency (true OFT throughput)
                "oft_baseline": thru_baseline,  # action_chunk_len / mean VLA latency (model-side, no skip)
            },
            "overall_control_hz": overall_hz,
            "overall_episode_wall_s": overall_epw,
            "per_task": {
                str(tid): {
                    "task_description": per_task_name.get(tid, ""),
                    "successes": per_task_succ[tid],
                    "episodes": per_task_eps[tid],
                    "success_rate": (per_task_succ[tid] / per_task_eps[tid]) if per_task_eps[tid] else 0.0,
                    "vla_latency_ms": _stats(per_task_lat[tid]),
                    "control_hz": (per_task_ctrl_steps[tid] / per_task_ctrl_wall[tid])
                                   if per_task_ctrl_wall[tid] > 0 else 0.0,
                    "episode_wall_s": _stats(per_task_ep_walls[tid]),
                }
                for tid in sorted(per_task_eps)
            },
            "workers": {str(wid): {"successes": s, "episodes": e} for wid, (s, e) in worker_totals.items()},
            "source_files": files,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nWrote combined results to {args.out}")

    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)) or ".", exist_ok=True)
        with open(args.log, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Wrote summary to {args.log}")


if __name__ == "__main__":
    main()
