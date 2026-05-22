"""APSG: Action-Projected Self-Grounding for VLA cross-attention.

Auxiliary training loss that pulls action-expert -> vision cross-attention toward
the *image-plane projection of the future end-effector position* derived directly
from the demonstration action chunk. Inference is untouched (no extra modules,
no extra compute) -- the loss only shapes attention during training.

This implementation targets pi0 / pi0.5 (PyTorch path) and is opt-in via
`APSGConfig.enabled`. When disabled the model behaves identically to baseline.

Design contract
---------------
* Camera projection happens *in the data pipeline* (before Normalize), via the
  ``APSGProjection`` transform. This avoids the model needing access to raw
  un-normalized state. The transform writes three optional fields into the
  data dict: ``apsg_target_uv`` (shape [2]), ``apsg_target_in_bounds`` (bool),
  ``apsg_sigma_patches`` (scalar). Those flow through ``Observation`` to the
  model.
* The model captures attention weights for selected layers via an opt-in
  attribute on ``PaliGemmaWithExpertModel``. The capture mechanism is dormant
  (zero-cost) unless an APSG-enabled ``forward`` call activates it.
* Camera projection is pluggable through ``Projector`` so different
  datasets/robots can supply their own intrinsics/extrinsics. The bundled
  ``libero_agentview_projector`` encodes LIBERO's fixed third-person camera.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import math
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

import openpi.transforms as _transforms


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class APSGConfig:
    """Configuration for APSG. ``enabled=False`` => no-op (baseline behavior)."""

    enabled: bool = False

    # Loss weight: L_total = L_action + lambda_attn * L_apsg.
    lambda_attn: float = 0.05

    # Layers of the joint self-attention to supervise. Indices refer to the
    # PaliGemma/Gemma layer stack (both share depth). Defaults to first + last,
    # matching the prescription in the APSG document (seed grounding + counter
    # the well-known deep-layer visual decay).
    layer_indices: tuple[int, ...] = (0, 17)

    # Gaussian width: sigma_patches = alpha * displacement_meters + sigma_min.
    # 'displacement' is the L2 distance between current EE xyz and the
    # chunk-end EE xyz, in meters. Defaults match the doc.
    sigma_alpha_per_m: float = 15.0  # ~5 patches at 30cm reach
    sigma_min_patches: float = 1.5

    # Which view to supervise. Must match a key in the model's preprocessed
    # image dict. For LIBERO this is always 'base_0_rgb' (third-person agentview).
    image_key: str = "base_0_rgb"

    # Image / patch geometry. The defaults match SigLIP @ 224x224 / patch 14
    # which is what pi0 / pi05 use today.
    image_size: int = 224
    patch_size: int = 14

    # How to derive the target world-frame EE xyz.
    # 'future_state_direct' (recommended): use state[t+H][:3] from the demo
    #   trajectory. Requires the data pipeline to chunk state as well as actions
    #   (handled automatically by LeRobotLiberoAPSGDataConfig).
    # 'state_plus_action_endpoint' / 'state_plus_integrated_action' /
    # 'action_endpoint_absolute' are legacy modes that derive the target from
    # the action chunk; they are sensitive to action format and only work for
    # datasets whose action units are calibrated.
    target_mode: str = "future_state_direct"

    # Slice of `state` carrying EE xyz in world frame.
    state_xyz_slice: tuple[int, int] = (0, 3)
    # Slice of `actions` carrying the xyz component.
    action_xyz_slice: tuple[int, int] = (0, 3)
    # Optional scalar for `target_mode='state_plus_integrated_action'`. Maps the
    # per-step controller delta to a metric displacement. Tune via the Phase-0
    # visualization script when the action units don't already match meters.
    action_xyz_scale: float = 1.0

    # Linear warm-up of lambda_attn from 0 over this many steps. Useful when the
    # base policy is being fine-tuned and you want the attention prior to kick
    # in gradually. 0 = no warmup.
    warmup_steps: int = 0

    # Loss reduction across supervised layers: 'mean' or 'sum'.
    layer_reduction: str = "mean"


# ---------------------------------------------------------------------------
# Camera projection
# ---------------------------------------------------------------------------


class Projector(Protocol):
    """Maps world-frame xyz tensors to normalized image coords in [0, 1].

    Returns (uv_norm, in_bounds) where uv_norm is [B, 2] with column 0 = u (x)
    and column 1 = v (y), and in_bounds is [B] bool indicating whether the
    projection landed inside the image.
    """

    def __call__(self, xyz_world: Tensor) -> tuple[Tensor, Tensor]: ...


@dataclasses.dataclass(frozen=True)
class PinholeCamera:
    """A static pinhole camera, intrinsics + extrinsics in numpy-friendly form.

    The image-plane convention matches an upright image where u grows right and
    v grows down (standard pinhole). Apply the optional ``flip_uv`` if the
    downstream pipeline rotates the rendered frame by 180 degrees (LIBERO does).
    """

    # Intrinsics
    image_w: int
    image_h: int
    fovy_deg: float

    # Extrinsics: camera pose in world frame. We stash position + a 3x3 rotation
    # matrix mapping camera-frame vectors into world frame. Apply the inverse
    # during projection.
    cam_pos_world: tuple[float, float, float]
    cam_rot_world: tuple[tuple[float, float, float], ...]  # 3x3 row-major

    # If True, applies the same 180-degree image rotation used at policy time
    # (LIBERO's `main.py` rotates `agentview_image[::-1, ::-1]`).
    flip_uv: bool = True
    # Per-axis flips for cases where only one of U/V is inverted (more common
    # with mujoco camera-axis convention mismatches). When set non-None they
    # take precedence over `flip_uv`.
    flip_u: bool | None = None
    flip_v: bool | None = None

    def project_np(self, xyz_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """xyz_world: shape (..., 3) -> ((u_norm, v_norm) shape (..., 2),
        in_bounds bool shape (...)). Used in the data transform.
        """
        fy = 0.5 * self.image_h / math.tan(math.radians(self.fovy_deg) * 0.5)
        fx = fy
        cx = self.image_w / 2.0
        cy = self.image_h / 2.0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        R_wc = np.array(self.cam_rot_world, dtype=np.float64)
        t_wc = np.array(self.cam_pos_world, dtype=np.float64)
        R_cw = R_wc.T

        # World -> mujoco camera frame.
        x_cam = (xyz_world.astype(np.float64) - t_wc) @ R_cw.T
        # Mujoco cam looks down -Z with +Y up. Pinhole expects +Z with +Y down.
        x_cam = x_cam * np.array([1.0, -1.0, -1.0])
        z = np.maximum(x_cam[..., 2], 1e-3)
        u = (K[0, 0] * x_cam[..., 0] + K[0, 2] * z) / z
        v = (K[1, 1] * x_cam[..., 1] + K[1, 2] * z) / z
        do_flip_u = self.flip_uv if self.flip_u is None else self.flip_u
        do_flip_v = self.flip_uv if self.flip_v is None else self.flip_v
        if do_flip_u:
            u = self.image_w - u
        if do_flip_v:
            v = self.image_h - v
        u_norm = (u / self.image_w).astype(np.float32)
        v_norm = (v / self.image_h).astype(np.float32)
        uv = np.stack([u_norm, v_norm], axis=-1)
        in_bounds = (u_norm >= 0) & (u_norm < 1) & (v_norm >= 0) & (v_norm < 1)
        return uv, in_bounds

    def __call__(self, xyz_world):
        """Convenience: dispatches based on tensor type."""
        if isinstance(xyz_world, torch.Tensor):
            arr = xyz_world.detach().cpu().numpy()
            uv, ib = self.project_np(arr)
            return (
                torch.from_numpy(uv).to(xyz_world.device),
                torch.from_numpy(ib).to(xyz_world.device),
            )
        return self.project_np(xyz_world)


def _quat_to_R(wxyz: tuple[float, float, float, float]) -> tuple[tuple[float, float, float], ...]:
    w, x, y, z = wxyz
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)),
        (2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)),
    )


# Empirical LIBERO 'agentview' params, dumped from `env.sim.model.cam_pos /
# cam_quat / cam_fovy` per task suite (see scripts/dump_libero_cam.py).
# libero_spatial / libero_goal / libero_10 share one camera; libero_object is
# placed lower and pitched more steeply down.
_LIBERO_CAM_DEFAULT = dict(
    cam_pos_world=(0.6586, 0.0, 1.6104),
    cam_quat_wxyz=(0.6380, 0.3048, 0.3048, 0.6380),
    fovy_deg=45.0,
)
_LIBERO_CAM_OBJECT = dict(
    cam_pos_world=(0.8966, 0.0, 0.6500),
    cam_quat_wxyz=(0.6182, 0.3432, 0.3432, 0.6182),
    fovy_deg=45.0,
)


def libero_agentview_projector(suite: str = "default") -> PinholeCamera:
    """Returns LIBERO 'agentview' as a PinholeCamera.

    Use ``suite='object'`` for ``libero_object`` tasks (different camera);
    everything else gets the shared spatial/goal/10 camera. Per-task camera
    lookup can be added later if the dataset mixes object with the others.
    """
    spec = _LIBERO_CAM_OBJECT if suite == "object" else _LIBERO_CAM_DEFAULT
    R_wc = _quat_to_R(spec["cam_quat_wxyz"])
    return PinholeCamera(
        image_w=224, image_h=224, fovy_deg=spec["fovy_deg"],
        cam_pos_world=spec["cam_pos_world"],
        cam_rot_world=R_wc,
        # LeRobot data is right-side-up (V axis matches pinhole), but the
        # mujoco-renderer's framebuffer-X is mirrored relative to the pinhole
        # convention, so U needs flipping.
        flip_uv=False,
        flip_u=True,
        flip_v=False,
    )


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------


def compute_target_world_ee(
    state: Tensor,
    actions: Tensor,
    cfg: APSGConfig,
) -> tuple[Tensor, Tensor]:
    """Returns (target_world_xyz [B, 3], current_world_xyz [B, 3])."""
    s_a, s_b = cfg.state_xyz_slice
    a_a, a_b = cfg.action_xyz_slice
    cur = state[..., s_a:s_b].to(torch.float32)
    if cfg.target_mode == "state_plus_action_endpoint":
        # LIBERO with extra_delta_transform=True: action[h, :3] is already
        # (world_target_xyz - current_state_xyz). Endpoint = state + action[-1].
        tgt = cur + actions[:, -1, a_a:a_b].to(torch.float32)
    elif cfg.target_mode == "state_plus_integrated_action":
        # Raw delta-per-step actions (e.g. pi05 without extra_delta_transform).
        integrated = actions[:, :, a_a:a_b].to(torch.float32).sum(dim=1) * cfg.action_xyz_scale
        tgt = cur + integrated
    elif cfg.target_mode == "action_endpoint_absolute":
        # Actions are absolute world-frame EE positions per step.
        tgt = actions[:, -1, a_a:a_b].to(torch.float32)
    else:
        raise ValueError(f"unknown target_mode: {cfg.target_mode}")
    return tgt, cur


def make_gaussian_target(
    uv_norm: Tensor,            # [B, 2], normalized image coords in [0,1]
    sigma_patches: Tensor,       # [B] in patch units
    grid_h: int,
    grid_w: int,
    in_bounds: Tensor,           # [B] bool
) -> tuple[Tensor, Tensor]:
    """Returns (target_dist [B, grid_h * grid_w], valid_mask [B] bool).

    The distribution is normalized to sum to 1 over the patch grid. Samples
    with `in_bounds=False` are zeroed (still safe to KL against, but the caller
    should mask them).
    """
    device = uv_norm.device
    dtype = torch.float32
    rows = torch.arange(grid_h, device=device, dtype=dtype)
    cols = torch.arange(grid_w, device=device, dtype=dtype)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")  # [H, W]

    # Pixel center -> patch grid coords (0..grid-1). uv_norm * grid yields
    # the *patch-index* corresponding to the projected pixel (center-aligned).
    r_star = uv_norm[:, 1].to(dtype) * grid_h - 0.5
    c_star = uv_norm[:, 0].to(dtype) * grid_w - 0.5

    sigma = sigma_patches.to(dtype).clamp(min=0.5)
    d2 = (grid_r[None] - r_star[:, None, None]) ** 2 + (grid_c[None] - c_star[:, None, None]) ** 2
    logits = -d2 / (2.0 * sigma[:, None, None] ** 2)
    # Numerically stable softmax over the patch grid.
    logits_flat = logits.reshape(logits.shape[0], -1)
    target_dist = torch.softmax(logits_flat, dim=-1)
    target_dist = target_dist * in_bounds.to(dtype).unsqueeze(-1)
    return target_dist, in_bounds


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def compute_apsg_loss(
    attn_per_layer: dict[int, Tensor],   # layer_idx -> [B, num_heads, num_action_q, num_vis_tokens]
    target_dist: Tensor,                  # [B, num_vis_tokens]
    in_bounds: Tensor,                    # [B] bool
    layer_indices: Sequence[int],
    reduction: str = "mean",
) -> Tensor:
    """KL(attn || target). Forward KL: penalizes attention mass outside the
    target region. We average attention over heads and action queries to get
    a per-token distribution, then normalize across the visual tokens.
    """
    if in_bounds.sum() == 0 or len(attn_per_layer) == 0:
        return torch.zeros((), device=target_dist.device, dtype=torch.float32)

    eps = 1e-8
    losses = []
    valid = in_bounds
    target = target_dist[valid].to(torch.float32)
    # Ensure target sums to 1 (it already does for in_bounds samples).
    target = target / target.sum(dim=-1, keepdim=True).clamp(min=eps)

    for layer_idx in layer_indices:
        if layer_idx not in attn_per_layer:
            continue
        attn = attn_per_layer[layer_idx]            # [B, H, Q, V]
        attn = attn.to(torch.float32).mean(dim=(1, 2))  # [B, V]
        attn = attn[valid]
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=eps)
        # KL(target || attn): sum_i t_i * (log t_i - log a_i)
        kl = (target * (torch.log(target.clamp(min=eps)) - torch.log(attn.clamp(min=eps)))).sum(dim=-1)
        losses.append(kl.mean())

    if not losses:
        return torch.zeros((), device=target_dist.device, dtype=torch.float32)

    stacked = torch.stack(losses)
    return stacked.mean() if reduction == "mean" else stacked.sum()


def apsg_lambda(cfg: APSGConfig, step: int | None) -> float:
    if not cfg.enabled:
        return 0.0
    if cfg.warmup_steps <= 0 or step is None:
        return float(cfg.lambda_attn)
    progress = min(1.0, max(0.0, step / cfg.warmup_steps))
    return float(cfg.lambda_attn * progress)


# ---------------------------------------------------------------------------
# Attention-capture lifecycle helpers
# ---------------------------------------------------------------------------


class AttentionCapture:
    """Context manager that toggles `_apsg_capture_layers` / `_apsg_attn_cache`
    on a PaliGemmaWithExpertModel. Outside this context the model is a no-op
    overhead-wise (the capture branch in `compute_layer_complete` is a single
    `if not self._apsg_capture_layers: pass`).
    """

    def __init__(self, paligemma_with_expert, layer_indices: Sequence[int], prefix_len_provider):
        self.model = paligemma_with_expert
        self.layers = set(int(i) for i in layer_indices)
        self.prefix_len_provider = prefix_len_provider

    def __enter__(self):
        self.model._apsg_attn_cache = {}
        self.model._apsg_capture_layers = self.layers
        self.model._apsg_prefix_len = None
        self.model._apsg_set_prefix_len = self.prefix_len_provider
        return self.model._apsg_attn_cache

    def __exit__(self, exc_type, exc, tb):
        self.model._apsg_capture_layers = set()
        self.model._apsg_attn_cache = {}
        self.model._apsg_prefix_len = None
        self.model._apsg_set_prefix_len = None
        return False


# ---------------------------------------------------------------------------
# Data transform: compute projection BEFORE normalization
# ---------------------------------------------------------------------------


def _compute_target_world_ee_np(state: np.ndarray, actions: np.ndarray, cfg: APSGConfig) -> tuple[np.ndarray, np.ndarray]:
    s_a, s_b = cfg.state_xyz_slice
    a_a, a_b = cfg.action_xyz_slice
    cur = state[..., s_a:s_b].astype(np.float32)
    if cfg.target_mode == "state_plus_action_endpoint":
        tgt = cur + actions[-1, a_a:a_b].astype(np.float32)
    elif cfg.target_mode == "state_plus_integrated_action":
        integrated = actions[:, a_a:a_b].astype(np.float32).sum(axis=0) * cfg.action_xyz_scale
        tgt = cur + integrated
    elif cfg.target_mode == "action_endpoint_absolute":
        tgt = actions[-1, a_a:a_b].astype(np.float32)
    else:
        raise ValueError(f"unknown target_mode: {cfg.target_mode}")
    return tgt, cur


@dataclasses.dataclass(frozen=True)
class APSGProjection(_transforms.DataTransformFn):
    """Computes the APSG target (u,v,in_bounds,sigma) and stashes it in the
    data dict.

    Expects ``state`` to arrive as a *chunk* of shape [H, S] when target_mode is
    ``future_state_direct`` (i.e. ``state`` was added to
    ``action_sequence_keys``). The chunk is consumed here: only state[0] is
    written back to ``data['state']``, state[-1][:3] becomes the projection
    target, so downstream transforms see normal-shaped state.

    For the legacy action-based modes, ``state`` is expected to be unchunked
    [S] and ``actions`` to be the standard [H, A] chunk.
    """

    cfg: APSGConfig
    camera: PinholeCamera

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if self.cfg.target_mode == "future_state_direct":
            # state must be a chunk [H, S]; we consume it back to [S].
            if state.ndim != 2:
                raise ValueError(
                    "APSG future_state_direct mode requires state chunking; "
                    "add 'state' to action_sequence_keys in your DataConfig."
                )
            cur_xyz = state[0, self.cfg.state_xyz_slice[0]:self.cfg.state_xyz_slice[1]].astype(np.float32)
            target_xyz = state[-1, self.cfg.state_xyz_slice[0]:self.cfg.state_xyz_slice[1]].astype(np.float32)
            # Restore unchunked state for downstream.
            data["state"] = state[0]
        else:
            actions = np.asarray(data["actions"])
            target_xyz, cur_xyz = _compute_target_world_ee_np(state, actions, self.cfg)

        uv, in_bounds = self.camera.project_np(target_xyz)
        displacement = float(np.linalg.norm(target_xyz - cur_xyz))
        sigma = float(self.cfg.sigma_alpha_per_m * displacement + self.cfg.sigma_min_patches)
        data["apsg_target_uv"] = uv.astype(np.float32)
        data["apsg_target_in_bounds"] = np.bool_(bool(in_bounds))
        data["apsg_sigma_patches"] = np.float32(sigma)
        return data
