"""ManipArena bimanual policy transforms for OpenPI.

ManipArena provides 14D end-effector state/actions (7D left + 7D right),
with 3 cameras (front, left wrist, right wrist). The dataset stores 56D
(real) or 28D (sim) vectors, but only the first 14D (EE) are used here.

Dimension layout per arm (7D):
  [0:3] position (x, y, z)
  [3:6] rotation (roll, pitch, yaw)
  [6]   gripper (open/close)
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Number of EE dimensions: 7 per arm × 2 arms = 14
ACTION_DIM = 14


def make_maniparena_example() -> dict:
    """Creates a random input example for the ManipArena policy."""
    return {
        "images": {
            "front": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
            "left_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
            "right_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        },
        "state": np.random.rand(14).astype(np.float32),
        "prompt": "pick up the cup",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # LeRobot stores as (C, H, W); convert to (H, W, C) if needed.
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ManipArenaInputs(transforms.DataTransformFn):
    """Convert ManipArena observations to pi0 model input format.

    Expected input keys (after repack):
      - images/front, images/left_wrist, images/right_wrist: camera images
      - state: robot state (56D real / 28D sim / 14D EE-only)
      - actions: action array (only during training)
      - prompt: task instruction text
    """

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # Extract 14D EE state from potentially larger state vector.
        state = np.asarray(data["state"], dtype=np.float32)
        state = state[:ACTION_DIM]

        # Parse camera images.
        images = data.get("images", {})
        front_img = _parse_image(images["front"]) if "front" in images else np.zeros((480, 640, 3), dtype=np.uint8)
        left_img = _parse_image(images["left_wrist"]) if "left_wrist" in images else np.zeros_like(front_img)
        right_img = _parse_image(images["right_wrist"]) if "right_wrist" in images else np.zeros_like(front_img)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": front_img,
                "left_wrist_0_rgb": left_img,
                "right_wrist_0_rgb": right_img,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            # Extract first 14D (EE) from each timestep.
            inputs["actions"] = actions[..., :ACTION_DIM]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ManipArenaOutputs(transforms.DataTransformFn):
    """Convert pi0 model output back to ManipArena action format.

    The model outputs (action_horizon, 32) but ManipArena only uses the first 14 dims.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
