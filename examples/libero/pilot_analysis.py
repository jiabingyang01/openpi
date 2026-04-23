"""Skip-VLA pilot analysis.

Reads per-worker trajectory pickles dumped by main_parallel.py (when run with
--pilot-dump) and computes:

  1. Oracle skip rate: for each step t>=2, linear extrapolation a_hat[t] = 2*a[t-1] - a[t-2].
     err[t] = ||a[t] - a_hat[t]||_1 (or L-inf). Oracle skip rate at tolerance δ =
     fraction of steps with err < δ. Sweep δ → Pareto-style table.

  2. Scheme A confidence C_A(t) = exp(-α * ||Δ²a||_2 / (||Δa||_2 + ε))
     and its false-positive rate: among steps where C_A > τ, what fraction have
     err > δ (i.e. Scheme A thinks it's safe to skip but linear extrap is actually bad).

  3. Gripper-change step statistics: how often gripper state flips within a trajectory,
     what fraction of total steps are within K of a gripper change (these are the
     "critical" steps that gripper-guard should catch).

Usage:
    python examples/libero/pilot_analysis.py LOG_DIR
    python examples/libero/pilot_analysis.py LOG_DIR --action-dim 7 --alpha 10.0
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np


def _norm(x, ord):
    if x.ndim == 1:
        return np.linalg.norm(x, ord=ord)
    return np.linalg.norm(x, ord=ord, axis=-1)


def compute_errors(actions: np.ndarray):
    """Return (err_L1, err_Linf, delta_a_norm, delta2_a_norm) each of shape (T-2,).
    Aligned so entry i corresponds to step t = i+2 in the original trajectory."""
    T, D = actions.shape
    if T < 3:
        return None
    a_hat = 2.0 * actions[1:-1] - actions[:-2]  # predicted a[t] from a[t-1], a[t-2]
    err = actions[2:] - a_hat                   # shape (T-2, D)
    err_l1 = np.linalg.norm(err, ord=1, axis=-1)
    err_linf = np.linalg.norm(err, ord=np.inf, axis=-1)
    # For Scheme A: Δa[t-1] = a[t-1] - a[t-2],  Δ²a[t] = Δa[t] - Δa[t-1]
    delta_a = actions[1:] - actions[:-1]        # (T-1, D)
    delta_a_norm = np.linalg.norm(delta_a[1:], ord=2, axis=-1)  # (T-2,)
    delta2_a = delta_a[1:] - delta_a[:-1]       # (T-2, D)
    delta2_a_norm = np.linalg.norm(delta2_a, ord=2, axis=-1)    # (T-2,)
    return err_l1, err_linf, delta_a_norm, delta2_a_norm


def gripper_change_mask(actions: np.ndarray, gripper_dim: int = -1, K: int = 3):
    """Return boolean mask (T-2,) for steps within K of a gripper change.
    Uses the sign of the last action dim (LIBERO gripper: -1=open, +1=close)."""
    T = actions.shape[0]
    if T < 3:
        return np.zeros(0, dtype=bool)
    g = np.sign(actions[:, gripper_dim])
    flips = np.where(np.diff(g) != 0)[0]  # index i means flip between t=i and t=i+1
    mask = np.zeros(T, dtype=bool)
    for f in flips:
        lo = max(0, f - K)
        hi = min(T, f + K + 1)
        mask[lo:hi] = True
    return mask[2:]  # align with err arrays


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log_dir")
    p.add_argument("--alpha", type=float, default=10.0,
                   help="Scheme A confidence exponent: C_A = exp(-α · ||Δ²a||/(||Δa||+ε))")
    p.add_argument("--gripper-neighborhood", type=int, default=3,
                   help="Count a step as 'near-gripper-change' if within K of a flip")
    p.add_argument("--success-only", action="store_true",
                   help="Only include episodes that ended successfully (cleaner signal)")
    p.add_argument("--out", default=None, help="Optional path to save raw arrays (.npz)")
    args = p.parse_args()

    trace_files = sorted(glob.glob(os.path.join(args.log_dir, "*traces-worker*.pkl")))
    if not trace_files:
        print(f"no trace pickles in {args.log_dir} (expected *traces-worker*.pkl)",
              file=sys.stderr)
        sys.exit(2)

    all_l1 = []
    all_linf = []
    all_delta_a = []
    all_delta2_a = []
    all_gripper_near = []
    all_is_vla = []
    all_task_id = []

    n_ep = 0
    n_ep_kept = 0
    n_steps_total = 0
    per_task_eps = {}
    per_task_l1 = {}

    for tp in trace_files:
        with open(tp, "rb") as f:
            episodes = pickle.load(f)
        for ep in episodes:
            n_ep += 1
            if args.success_only and not ep["success"]:
                continue
            actions = ep["actions"]
            is_vla = ep["is_vla"]
            task_id = ep["task_id"]
            if actions.ndim != 2 or actions.shape[0] < 3:
                continue
            res = compute_errors(actions)
            if res is None:
                continue
            err_l1, err_linf, dA, d2A = res
            grip = gripper_change_mask(actions, gripper_dim=-1, K=args.gripper_neighborhood)
            all_l1.append(err_l1)
            all_linf.append(err_linf)
            all_delta_a.append(dA)
            all_delta2_a.append(d2A)
            all_gripper_near.append(grip)
            all_is_vla.append(is_vla[2:])  # align with err (skip first two steps)
            all_task_id.append(np.full_like(err_l1, task_id, dtype=np.int32))
            n_ep_kept += 1
            n_steps_total += len(err_l1)
            per_task_eps[task_id] = per_task_eps.get(task_id, 0) + 1
            per_task_l1.setdefault(task_id, []).append(err_l1)

    if n_ep_kept == 0:
        print("no usable episodes after filtering", file=sys.stderr)
        sys.exit(2)

    err_l1 = np.concatenate(all_l1)
    err_linf = np.concatenate(all_linf)
    dA = np.concatenate(all_delta_a)
    d2A = np.concatenate(all_delta2_a)
    grip_mask = np.concatenate(all_gripper_near)
    is_vla_mask = np.concatenate(all_is_vla)
    task_arr = np.concatenate(all_task_id)

    # Scheme A confidence
    eps_num = 1e-8
    C_A = np.exp(-args.alpha * (d2A / (dA + eps_num)))

    # ---------- Report ----------
    print(f"=== Skip-VLA pilot analysis ({args.log_dir}) ===")
    print(f"trace files: {len(trace_files)}")
    print(f"episodes scanned: {n_ep}   kept: {n_ep_kept}   total analysable steps: {n_steps_total}")
    print(f"action dim: {all_l1[0].shape if False else '-'}   (per step err = L1 over action dims)")
    print()

    # Action statistics
    print(f"Action L1-err stats (linear extrapolation):")
    for q in [5, 25, 50, 75, 90, 95, 99]:
        print(f"  p{q:2d}: {np.percentile(err_l1, q):.4f}")
    print(f"  mean: {err_l1.mean():.4f}   max: {err_l1.max():.4f}")
    print()

    # Oracle skip rate at different δ
    print("Oracle skip rate (linear-extrap L1 err < δ):")
    print(f"  {'δ':>6}  {'skip rate':>10}  {'example fraction-of-trajectory':>32}")
    for d in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        rate = (err_l1 < d).mean()
        print(f"  {d:>6.3f}  {rate*100:>9.2f}%  ({int(rate*n_steps_total):,}/{n_steps_total:,} steps)")
    print()

    # Scheme A: choose τ via quantile of C_A, measure FP rate at each
    print("Scheme A confidence-gated skip (C_A = exp(-α·||Δ²a||/(||Δa||+ε)), α={}):".format(args.alpha))
    print("  for each τ: 'gated_skip_rate' = frac of steps with C_A > τ")
    print("              'fp_rate@δ=0.05' = among those, frac with L1 err > 0.05")
    print("              'fp_rate@δ=0.10' = same with δ=0.10")
    print(f"  {'τ':>6}  {'gated_skip':>12}  {'fp@δ=0.05':>12}  {'fp@δ=0.10':>12}  {'fp@δ=0.20':>12}")
    for tau in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
        mask = C_A > tau
        n = mask.sum()
        if n == 0:
            print(f"  {tau:>6.2f}  {0.0:>11.2f}%  {'-':>12}  {'-':>12}  {'-':>12}")
            continue
        gated = n / n_steps_total
        fp_005 = (err_l1[mask] > 0.05).mean()
        fp_010 = (err_l1[mask] > 0.10).mean()
        fp_020 = (err_l1[mask] > 0.20).mean()
        print(f"  {tau:>6.2f}  {gated*100:>11.2f}%  {fp_005*100:>11.2f}%  {fp_010*100:>11.2f}%  {fp_020*100:>11.2f}%")
    print()

    # Gripper-neighborhood step fraction
    near = grip_mask.mean()
    print(f"Steps within ±{args.gripper_neighborhood} of a gripper flip: "
          f"{near*100:.2f}% of all steps ({grip_mask.sum():,}/{n_steps_total:,})")
    err_near = err_l1[grip_mask] if grip_mask.any() else err_l1[:0]
    err_far = err_l1[~grip_mask]
    if len(err_near) > 0:
        print(f"  mean L1 err near gripper change: {err_near.mean():.4f}  "
              f"(vs {err_far.mean():.4f} elsewhere; ratio {err_near.mean()/max(err_far.mean(),1e-8):.2f}×)")
    print()

    # Fraction of steps that were actually VLA calls in this run (reference: replan_steps)
    if is_vla_mask.any():
        vla_frac = is_vla_mask.mean()
        print(f"Baseline VLA call frequency (fresh infer()/step): {vla_frac*100:.2f}% "
              f"(= 1/replan_steps if replan_steps is fixed)")
        print()

    # Per-task oracle skip rate at δ=0.05 (quick summary)
    print("Per-task oracle skip rate @ δ=0.05 (quick peek):")
    for tid in sorted(per_task_l1):
        vals = np.concatenate(per_task_l1[tid])
        rate = (vals < 0.05).mean() * 100
        print(f"  task{tid} ({per_task_eps[tid]} eps, {len(vals)} steps):  {rate:5.2f}%")

    # ---------- Go/no-go verdict ----------
    oracle_at_005 = (err_l1 < 0.05).mean() * 100
    print()
    print("=" * 60)
    print("GO / NO-GO verdict (doc §5.1 criteria):")
    print(f"  Oracle skip rate @ δ=0.05: {oracle_at_005:.2f}%")
    if oracle_at_005 >= 40:
        print("    ✓ >= 40% → idea viable, proceed with implementation")
    elif oracle_at_005 >= 30:
        print("    ⚠  30-40% → borderline; try MLP predictor instead of linear extrap")
    else:
        print("    ✗ < 30% → ABORT")

    # Scheme A FP rate at its most "confident" τ that still yields reasonable skip coverage
    # Pick the smallest τ yielding gated_skip_rate in [30%, 60%].
    chosen_tau = None
    for tau in [0.95, 0.9, 0.8, 0.7, 0.5, 0.3]:
        mask = C_A > tau
        gs = mask.mean() * 100
        if 30 <= gs <= 60:
            chosen_tau = tau
            fp = (err_l1[mask] > 0.05).mean() * 100
            print(f"  Scheme A @ τ={tau} (gated_skip={gs:.1f}%): fp@δ=0.05 = {fp:.2f}%")
            if fp <= 5:
                print("    ✓ Scheme A alone is clean enough")
            elif fp <= 15:
                print("    ⚠  Scheme B (learned gate) likely needed")
            else:
                print("    ✗ Scheme A badly mis-gated; Scheme B + guards mandatory")
            break
    if chosen_tau is None:
        print("  (no τ in 0.3..0.95 yielded a 30-60% gated_skip_rate; "
              "Scheme A coverage is off-spec, re-tune α)")
    print("=" * 60)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        np.savez_compressed(
            args.out,
            err_l1=err_l1, err_linf=err_linf,
            delta_a=dA, delta2_a=d2A, C_A=C_A,
            gripper_near=grip_mask, is_vla=is_vla_mask, task_id=task_arr,
        )
        print(f"\nraw arrays saved → {args.out}")


if __name__ == "__main__":
    main()
