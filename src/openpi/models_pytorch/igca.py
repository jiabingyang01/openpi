"""IGCA: Instruction-Grounded Contrastive Attention.

Adds a contrastive attention module to Pi0 that learns to separate
task-relevant object features from background features, guided by
GT segmentation masks during training.

Design:
- A lightweight attention sub-net predicts a soft attention map Φ+ from
  visual tokens, supervised by GT mask via MSE loss.
- Φ- = 1 - Φ+ gives the complementary background attention.
- Region-level contrastive loss pulls h_full toward h_obj and pushes
  h_full away from h_bg.
- At inference time, the learned attention sub-net predicts Φ+ without
  GT mask. No extra input needed, zero inference overhead.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import openpi.transforms as _transforms


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class IGCAConfig:
    enabled: bool = False
    # Contrastive loss weights
    lambda_contrast: float = 0.01
    margin: float = 10.0
    # Attention supervision weight
    lambda_att: float = 0.1
    # Warmup
    warmup_steps: int = 0
    # Mask directory
    mask_dir: str = "./data/igca_masks"


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class IGCAModule(nn.Module):
    """Contrastive attention sub-net + loss computation.

    Takes visual tokens [B, N, D] and optional GT mask [B, N],
    produces contrastive + attention losses.
    """

    def __init__(self, config: IGCAConfig, feature_dim: int = 2048):
        super().__init__()
        self.config = config
        # Attention sub-net: predicts per-patch relevance score
        self.attention_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 1),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,   # [B, N, D]
        gt_mask: torch.Tensor | None = None,  # [B, H_patch, W_patch] float, 0 or 1
    ) -> dict:
        """Compute IGCA losses.

        Returns dict with:
            igca_contrast_loss: scalar
            igca_att_loss: scalar
            igca_phi: [B, N, 1] learned attention map (for visualization)
        """
        B, N, D = visual_tokens.shape

        # 1. Generate soft attention map
        phi_plus = torch.sigmoid(self.attention_head(visual_tokens))  # [B, N, 1]
        phi_minus = 1.0 - phi_plus

        # 2. Weighted features
        f_obj = visual_tokens * phi_plus    # [B, N, D]
        f_bg = visual_tokens * phi_minus    # [B, N, D]

        # 3. Global average pooling -> feature vectors
        h_full = visual_tokens.mean(dim=1)  # [B, D]
        h_obj = f_obj.mean(dim=1)           # [B, D]
        h_bg = f_bg.mean(dim=1)             # [B, D]

        # 4. Contrastive loss: pull h_full<->h_obj, push h_full<->h_bg
        loss_pull = F.mse_loss(h_full, h_obj)
        dist_bg = (h_full - h_bg).pow(2).sum(dim=-1)  # [B]
        loss_push = F.relu(self.config.margin - dist_bg).mean()
        igca_contrast_loss = loss_pull + loss_push

        # 5. Attention supervision loss (only when GT mask is available)
        igca_att_loss = torch.tensor(0.0, device=visual_tokens.device)
        if gt_mask is not None:
            # gt_mask: [B, H, W] -> flatten to [B, N]
            gt_flat = gt_mask.reshape(B, -1).unsqueeze(-1).float()  # [B, N, 1]
            igca_att_loss = F.mse_loss(phi_plus, gt_flat)

        return {
            "igca_contrast_loss": igca_contrast_loss,
            "igca_att_loss": igca_att_loss,
            "igca_phi": phi_plus.detach(),
        }


# ---------------------------------------------------------------------------
# Lambda warmup (same pattern as APSG)
# ---------------------------------------------------------------------------


def igca_lambda(cfg: IGCAConfig, step: int | None) -> float:
    if not cfg.enabled:
        return 0.0
    if cfg.warmup_steps <= 0 or step is None:
        return 1.0
    progress = min(1.0, max(0.0, step / cfg.warmup_steps))
    return float(progress)


# ---------------------------------------------------------------------------
# Data transform: loads pre-computed masks
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class IGCAMaskTransform(_transforms.DataTransformFn):
    """Load pre-computed segmentation mask and add to data dict.

    Reads mask from .npz file, indexes by frame, resizes to patch grid
    (16x16 for ViT with 224px input / 14px patches).
    """

    mask_dir: str = "./data/igca_masks"
    patch_grid: int = 16  # ViT patch grid size (224/14 = 16)

    def __call__(self, data: dict) -> dict:
        episode_idx = int(data.get("episode_index", -1))
        frame_idx = int(data.get("frame_index", 0))

        mask_path = os.path.join(self.mask_dir, f"episode_{episode_idx:06d}.npz")

        if os.path.exists(mask_path):
            masks_all = np.load(mask_path)["masks"]  # [T, H, W]
            # Clamp frame index for episodes where mask is shorter
            fi = min(frame_idx, len(masks_all) - 1)
            mask = masks_all[fi]  # [H, W] uint8

            # Resize to patch grid (16x16)
            import cv2
            mask_patch = cv2.resize(
                mask, (self.patch_grid, self.patch_grid),
                interpolation=cv2.INTER_AREA
            )
            # Binarize after resize (INTER_AREA may produce non-binary values)
            mask_patch = (mask_patch > 0.5).astype(np.float32)
        else:
            # No mask available: use zeros (will be ignored by loss)
            mask_patch = np.zeros((self.patch_grid, self.patch_grid), dtype=np.float32)

        data["igca_mask"] = mask_patch  # [16, 16]
        return data
