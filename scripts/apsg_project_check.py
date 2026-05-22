"""Phase-0 sanity script for APSG: write a per-episode mp4 showing three
projected reference points on each agentview frame so you can debug both the
camera calibration AND the action-interpretation choice in one pass.

Three dots are drawn:
  - BLUE  = projection of the *current* EE (state[:3]). Validates camera.
  - GREEN = projection of the *true future* EE (state[:3] at frame t+H from
            the demo). This is the ground-truth attention target.
  - RED   = projection of whatever the APSG training pipeline actually feeds
            the model, given the current configuration. Should track GREEN
            if your action-format interpretation is correct.

Usage:
    python scripts/apsg_project_check.py \
        --num-episodes=3 --action-horizon=50 \
        --out-dir=./apsg_viz

This bypasses the openpi data pipeline (so no norm_stats needed) and reads
LeRobot directly so we can render real demonstration sequences.
"""

from __future__ import annotations

import argparse
import math
import pathlib

import imageio.v2 as imageio
import numpy as np

import lerobot.common.datasets.lerobot_dataset as L
import openpi.models_pytorch.apsg as apsg_mod


def _draw_dot(img: np.ndarray, u: float, v: float, color=(255, 0, 0), radius=4):
    h, w, _ = img.shape
    cu = int(round(u)); cv = int(round(v))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            x = cu + dx; y = cv + dy
            if 0 <= x < w and 0 <= y < h:
                img[y, x] = color


def _draw_legend(img: np.ndarray):
    """Tiny legend in top-left: BLUE=now, GREEN=future-state, RED=apsg-target."""
    pad = 3
    for i, (label, col) in enumerate([("now (blue)", (50, 100, 255)),
                                       ("future (green)", (50, 220, 80)),
                                       ("apsg (red)", (255, 60, 60))]):
        y = pad + 8 + i * 9
        for x in range(pad, pad + 6):
            for yy in range(y, y + 6):
                if 0 <= yy < img.shape[0] and 0 <= x < img.shape[1]:
                    img[yy, x] = col


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="physical-intelligence/libero")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=50,
                        help="H used to compute the future EE reference")
    parser.add_argument("--action-xyz-scale", type=float, default=1.0,
                        help="Scale factor when integrating per-step deltas")
    parser.add_argument("--apsg-mode", default="future_state_direct",
                        choices=("future_state_direct",
                                 "state_plus_action_endpoint",
                                 "state_plus_integrated_action",
                                 "action_endpoint_absolute"))
    parser.add_argument("--rotate-180", action="store_true", default=False,
                        help="LeRobot already stores LIBERO frames right-side-up. "
                             "Only set this if your dataset stores raw mujoco frames "
                             "(upside-down) instead.")
    parser.add_argument("--out-dir", default="./apsg_viz")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = apsg_mod.libero_agentview_projector()
    apsg_cfg_for_render = apsg_mod.APSGConfig(
        enabled=True,
        target_mode=args.apsg_mode,
        action_xyz_scale=args.action_xyz_scale,
    )

    ds = L.LeRobotDataset(args.repo_id)
    n_eps = ds.num_episodes
    rng = np.random.default_rng(0)
    pick = rng.choice(n_eps, size=min(args.num_episodes, n_eps), replace=False)

    summary_lines = []
    for ep_rank, ep in enumerate(pick.tolist()):
        fr_from = int(ds.episode_data_index["from"][ep])
        fr_to = int(ds.episode_data_index["to"][ep])
        H = args.action_horizon
        frames = []
        in_bounds_now = 0
        in_bounds_apsg = 0
        in_bounds_future = 0

        for t in range(fr_from, fr_to):
            sample = ds[t]
            img = sample["image"]  # CHW float [0,1] or HWC depending on dataset
            if hasattr(img, "numpy"):
                img = img.numpy()
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)

            # Match the inference-time 180-deg rotation (main.py does it before
            # sending to the policy). All projections must be drawn in this
            # rotated frame; our camera already has `flip_uv=True` to compensate.
            if args.rotate_180:
                img = np.ascontiguousarray(img[::-1, ::-1])

            img = np.ascontiguousarray(img)

            state = sample["state"].numpy()
            cur_xyz = state[:3].astype(np.float32)
            # Future EE: take state H steps ahead, clipped to episode end.
            fut_t = min(t + H, fr_to - 1)
            fut_xyz = ds[fut_t]["state"].numpy()[:3].astype(np.float32)
            if args.apsg_mode == "future_state_direct":
                # Match what training does: target = state[t+H-1][:3] from the demo.
                # Note: this is identical to the GREEN dot for H steps ahead, so
                # RED should sit on top of GREEN by construction.
                apsg_tgt = fut_xyz
            else:
                chunk = []
                for k in range(H):
                    kk = min(t + k, fr_to - 1)
                    chunk.append(ds[kk]["actions"].numpy())
                chunk = np.stack(chunk, axis=0).astype(np.float32)
                chunk_delta = chunk.copy()
                chunk_delta[:, :6] -= state[:6]  # matches extra_delta_transform=True
                apsg_tgt, _ = apsg_mod._compute_target_world_ee_np(state, chunk_delta, apsg_cfg_for_render)

            (cur_u, cur_v), cur_ib = cam.project_np(cur_xyz)
            (fut_u, fut_v), fut_ib = cam.project_np(fut_xyz)
            (apsg_u, apsg_v), apsg_ib = cam.project_np(apsg_tgt)
            in_bounds_now += int(cur_ib); in_bounds_future += int(fut_ib); in_bounds_apsg += int(apsg_ib)

            HH, WW, _ = img.shape
            _draw_dot(img, float(cur_u) * WW, float(cur_v) * HH, color=(50, 100, 255), radius=4)
            _draw_dot(img, float(fut_u) * WW, float(fut_v) * HH, color=(50, 220, 80), radius=4)
            _draw_dot(img, float(apsg_u) * WW, float(apsg_v) * HH, color=(255, 60, 60), radius=4)
            _draw_legend(img)
            frames.append(img)

        n_frames = len(frames)
        task = ds[fr_from]["task"] if isinstance(ds[fr_from]["task"], str) else str(ds[fr_from]["task"])
        safe_task = task.replace(" ", "_").replace("/", "_")[:80]
        out_path = out_dir / f"ep{ep_rank:02d}_{ep}_{safe_task}.mp4"
        imageio.mimwrite(out_path, frames, fps=args.fps, codec="libx264", quality=8)
        msg = (
            f"ep {ep_rank} (idx={ep}, {n_frames}fr) -> {out_path}  "
            f"in_bounds: now={in_bounds_now/n_frames:.0%} "
            f"future={in_bounds_future/n_frames:.0%} "
            f"apsg={in_bounds_apsg/n_frames:.0%}"
        )
        print(msg); summary_lines.append(msg)

    print("\n=== SUMMARY ===")
    for s in summary_lines:
        print(s)
    print("\nInterpretation:")
    print("  - BLUE on gripper      => camera is calibrated.")
    print("  - GREEN leads BLUE     => future-EE ground truth is sane.")
    print("  - RED tracks GREEN     => APSG target_mode/action-scale are right.")
    print("If RED is stuck near a fixed point (e.g. world origin) or wildly off "
          "GREEN, change --apsg-mode and/or --action-xyz-scale and rerun.")


if __name__ == "__main__":
    main()
