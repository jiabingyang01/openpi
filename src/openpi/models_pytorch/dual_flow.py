"""Dual Flow Matching: co-evolving action generation and object motion prediction.

Adds a parallel mask trajectory flow to Pi0's action flow matching. Both flows
share the same transformer backbone and interact via self-attention, enabling
action generation to be grounded by object motion prediction and vice versa.

At inference the mask output can be discarded (or used for visualization);
the core benefit comes from co-denoising during training.
"""

from __future__ import annotations

import dataclasses
import os

import cv2
import numpy as np

import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class DualFlowConfig:
    enabled: bool = False
    lambda_mask: float = 0.1
    warmup_steps: int = 0
    mask_dir: str = "./data/igca_masks"
    patch_grid: int = 16


def dual_flow_lambda(cfg: DualFlowConfig, step: int | None) -> float:
    if not cfg.enabled:
        return 0.0
    if cfg.warmup_steps <= 0 or step is None:
        return float(cfg.lambda_mask)
    progress = min(1.0, max(0.0, step / cfg.warmup_steps))
    return float(cfg.lambda_mask * progress)


@dataclasses.dataclass(frozen=True)
class DualFlowMaskSeqTransform(_transforms.DataTransformFn):
    """Load a SEQUENCE of masks covering the action horizon.

    Produces ``dual_flow_mask_seq`` with shape ``[action_horizon, patch_grid, patch_grid]``.
    """

    mask_dir: str = "./data/igca_masks"
    patch_grid: int = 16
    action_horizon: int = 50

    def __call__(self, data: dict) -> dict:
        episode_idx = int(data.get("episode_index", -1))
        frame_idx = int(data.get("frame_index", 0))

        mask_path = os.path.join(self.mask_dir, f"episode_{episode_idx:06d}.npz")

        if os.path.exists(mask_path):
            masks_all = np.load(mask_path)["masks"]  # [T, H, W]
            T = len(masks_all)
            seq = []
            for i in range(self.action_horizon):
                fi = min(frame_idx + i, T - 1)
                mask = masks_all[fi]
                mask_patch = cv2.resize(
                    mask, (self.patch_grid, self.patch_grid),
                    interpolation=cv2.INTER_AREA,
                )
                mask_patch = (mask_patch > 0.5).astype(np.float32)
                seq.append(mask_patch)
            mask_seq = np.stack(seq, axis=0)  # [H_action, pg, pg]
        else:
            mask_seq = np.zeros(
                (self.action_horizon, self.patch_grid, self.patch_grid),
                dtype=np.float32,
            )

        data["dual_flow_mask_seq"] = mask_seq
        return data
