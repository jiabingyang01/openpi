"""DynaActVAE transforms for integrating frozen Action VAE into pi0/pi0.5 VLA training.

Paper: DynaActVAE (Section 3.3)
- Training: raw actions → frozen VAE encoder → normalized latent z̄_a (becomes the target)
- Inference: VLA generates latent z̃_a → frozen VAE decoder → raw actions

The Action VAE encoder/decoder are FROZEN during VLA training. Only the VLA policy is trained.
"""

import dataclasses
import logging
from pathlib import Path

import numpy as np
import torch

from openpi import transforms

logger = logging.getLogger(__name__)


def _load_action_vae(checkpoint_path: str, action_dim: int, chunk_size: int, z_dim: int,
                     hidden_dims: tuple = (256, 512, 512), num_res_blocks: int = 8,
                     device: str = "cpu"):
    """Load a trained Action VAE from checkpoint."""
    import sys
    # Add action-traj-vae to path so we can import ActionVAE
    vae_project = Path(checkpoint_path).parent
    while vae_project.name and not (vae_project / "vidact_vae").exists():
        vae_project = vae_project.parent
    if (vae_project / "vidact_vae").exists():
        sys.path.insert(0, str(vae_project))

    from vidact_vae.models.action_vae import ActionVAE

    model = ActionVAE(
        action_dim=action_dim,
        chunk_size=chunk_size,
        z_dim=z_dim,
        hidden_dims=hidden_dims,
        num_res_blocks=num_res_blocks,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Handle different checkpoint formats
    state_dict = ckpt.get("model", ckpt)
    # Filter to only action_vae keys if the checkpoint contains the full VidACT model
    vae_keys = {k: v for k, v in state_dict.items() if k.startswith("action_vae.")}
    if vae_keys:
        state_dict = {k.replace("action_vae.", ""): v for k, v in vae_keys.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    logger.info(f"Loaded frozen Action VAE from {checkpoint_path} "
                f"(action_dim={action_dim}, chunk_size={chunk_size}, z_dim={z_dim})")
    return model


@dataclasses.dataclass(frozen=True)
class DynaActVAEEncodeActions(transforms.DataTransformFn):
    """Training-time transform: encode raw actions into normalized VAE latents.

    Replaces the "actions" key with the encoded latent target z̄_a (Eq. 7 in paper).
    The VLA policy will learn to predict these latents instead of raw actions.
    """

    checkpoint_path: str = ""
    action_dim: int = 14
    chunk_size: int = 10
    z_dim: int = 16
    hidden_dims: tuple = (256, 512, 512)
    num_res_blocks: int = 8
    # Pre-computed latent normalization stats (mean, std per dim)
    latent_mean: np.ndarray | None = None
    latent_std: np.ndarray | None = None

    _model: object = dataclasses.field(default=None, init=False, repr=False, compare=False)

    def _get_model(self):
        if self._model is None:
            model = _load_action_vae(
                self.checkpoint_path, self.action_dim, self.chunk_size, self.z_dim,
                self.hidden_dims, self.num_res_blocks, device="cpu",
            )
            object.__setattr__(self, "_model", model)
        return self._model

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data

        model = self._get_model()
        actions = np.asarray(data["actions"], dtype=np.float32)  # (T, action_dim)

        # Encode: (1, T, D) → mu (1, z_dim, T')
        with torch.no_grad():
            actions_t = torch.from_numpy(actions).unsqueeze(0)  # (1, T, D)
            _, mu, _ = model.encode(actions_t)  # (z, mu, logvar) → use mu
            # Use deterministic mean (Eq. 7: z̄_a = sg(Norm(µ_φ(a))))
            latent = mu.squeeze(0).numpy()  # (z_dim, T')

        # Flatten to (T', z_dim) to match openpi's (action_horizon, action_dim) convention
        latent = latent.T  # (T', z_dim)

        # Normalize (Eq. 7: Norm(·))
        if self.latent_mean is not None and self.latent_std is not None:
            latent = (latent - self.latent_mean) / (self.latent_std + 1e-6)

        data["actions"] = latent.astype(np.float32)
        return data


@dataclasses.dataclass(frozen=True)
class DynaActVAEDecodeActions(transforms.DataTransformFn):
    """Inference-time transform: decode generated latent actions back to raw actions.

    Takes the VLA-generated latent z̃_a and decodes it through the frozen VAE decoder
    to produce executable robot actions (Eq. 10 in paper).
    """

    checkpoint_path: str = ""
    action_dim: int = 14
    chunk_size: int = 10
    z_dim: int = 16
    hidden_dims: tuple = (256, 512, 512)
    num_res_blocks: int = 8
    latent_mean: np.ndarray | None = None
    latent_std: np.ndarray | None = None

    _model: object = dataclasses.field(default=None, init=False, repr=False, compare=False)

    def _get_model(self):
        if self._model is None:
            model = _load_action_vae(
                self.checkpoint_path, self.action_dim, self.chunk_size, self.z_dim,
                self.hidden_dims, self.num_res_blocks, device="cpu",
            )
            object.__setattr__(self, "_model", model)
        return self._model

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data

        model = self._get_model()
        latent = np.asarray(data["actions"], dtype=np.float32)  # (T', z_dim)

        # Unnormalize
        if self.latent_mean is not None and self.latent_std is not None:
            latent = latent * (self.latent_std + 1e-6) + self.latent_mean

        # Decode: (1, z_dim, T') → (1, T, D)
        with torch.no_grad():
            latent_t = torch.from_numpy(latent.T).unsqueeze(0)  # (1, z_dim, T')
            actions = model.decode(latent_t)  # (1, T, D)
            actions = actions.squeeze(0).numpy()  # (T, D)

        data["actions"] = actions[:, :self.action_dim].astype(np.float32)
        return data
