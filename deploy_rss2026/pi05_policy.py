"""RSS2026 Post-Training Challenge — pi0.5 deployment wrapper.

Wraps a trained openpi pi05 checkpoint as a `policy_deployment` BasePolicy so the
organizers' eval client can drive it over WebSocket.

Deploy:
  1. Clone https://github.com/posttraining-for-robotics/policy_deployment
  2. Copy (or symlink) this file into policy_deployment/examples/pi05_policy.py
  3. Launch with the openpi venv (needs openpi + jax importable), e.g.:

     cd /path/to/policy_deployment
     /path/to/openpi/.venv/bin/python -m pip install -q websockets msgpack   # if missing
     CKPT=/path/to/openpi/checkpoints/pi05_insert-mouse-battery/pi05_insert-mouse-battery/40000
     PYTHONPATH=.:/path/to/openpi/src \
       /path/to/openpi/.venv/bin/python scripts/launch.py \
         --policy examples.pi05_policy:Pi05Policy \
         --policy-kwargs config_name=pi05_insert-mouse-battery \
         --policy-kwargs checkpoint_dir=$CKPT \
         --policy-kwargs default_prompt="Insert the battery to the mouse." \
         --port 8000

The openpi YamInputs transform reads obs["images"] (dict camera->CHW uint8),
obs["state"] (14,), obs["prompt"] directly at inference (repack is NOT applied
at inference), which is exactly the deployment InferenceRequest format — so this
wrapper passes the observation through almost unchanged.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from server.policy import BasePolicy
from server.schema import InferenceResponse, ServerMetadata

import openpi.training.config as _config
from openpi.policies import policy_config as _policy_config


class Pi05Policy(BasePolicy):
    def __init__(
        self,
        config_name: str,
        checkpoint_dir: str,
        default_prompt: str = "",
        action_horizon: int = 50,
        # Native dataset resolution (H=180, W=320). The model resizes-with-pad to 224
        # internally; advertising the native size keeps the eval client's aspect ratio
        # consistent with training. Verify against the organizer's eval client.
        image_h: int = 180,
        image_w: int = 320,
    ) -> None:
        train_config = _config.get_config(config_name)
        self._policy = _policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
            default_prompt=default_prompt or None,
        )
        self._action_horizon = int(action_horizon)
        self._default_prompt = default_prompt
        self._image_shape = [3, int(image_h), int(image_w)]
        self._cameras = ("cam_high", "cam_left_wrist", "cam_right_wrist")
        self._warmup()

    def _warmup(self) -> None:
        """Trigger JAX compilation so the first real observation isn't slow."""
        dummy = {
            "images": {
                k: np.zeros((3, self._image_shape[1], self._image_shape[2]), dtype=np.uint8)
                for k in self._cameras
            },
            "state": np.zeros(14, dtype=np.float32),
            "prompt": self._default_prompt or "do the task",
        }
        try:
            self._policy.infer(dummy)
        except Exception as e:  # noqa: BLE001 - warmup is best-effort
            print(f"[Pi05Policy] warmup inference failed (non-fatal): {e}")

    @property
    def metadata(self) -> ServerMetadata:
        return {
            "protocol_version": "1.0",
            "policy_name": "pi05-rss2026",
            "control_mode": "joints",
            "action_horizon": self._action_horizon,
            "action_dim": 14,
            "state_dim": 14,
            "image_keys": list(self._cameras),
            "image_shape": self._image_shape,
            "expects_prompt": True,
        }

    def infer(self, obs: Dict[str, Any]) -> InferenceResponse:
        inp: Dict[str, Any] = {
            "images": {k: np.asarray(v) for k, v in obs["images"].items()},
            "state": np.asarray(obs["state"], dtype=np.float32),
        }
        prompt = obs.get("prompt") or self._default_prompt
        if prompt:
            inp["prompt"] = prompt
        out = self._policy.infer(inp)
        actions = np.asarray(out["actions"], dtype=np.float32)  # (action_horizon, 14)
        return {"actions": actions}

    def reset(self) -> None:
        return None
