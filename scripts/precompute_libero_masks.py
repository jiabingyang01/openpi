#!/usr/bin/env python3
"""Pre-compute segmentation masks for the LIBERO LeRobot dataset.

Uses original LIBERO HDF5 demo files (with full sim states) to render
aligned segmentation masks. Matches HDF5 demos to LeRobot episodes
via first-frame image similarity.

Usage:
    cd /DATA/disk0/yjb/projects/VLA/openpi
    PYTHONPATH=third_party/libero:$PYTHONPATH \
    examples/libero/.venv/bin/python scripts/precompute_libero_masks.py

Output:
    data/igca_masks/
        episode_{idx:06d}.npz   # "masks": uint8 [T, 256, 256], binary
        mapping.json            # {episode_idx: {hdf5_file, demo_idx, mse}}
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

LIBERO_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "libero")
if LIBERO_SRC not in sys.path:
    sys.path.insert(0, LIBERO_SRC)

BDDL_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "libero", "libero", "libero", "bddl_files"
)

# Task description -> BDDL suite mapping
TASK_TO_SUITE = {}  # populated at runtime


def find_bddl_file(task_stem: str) -> str:
    """Find BDDL file matching a task stem across all suites."""
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]:
        suite_dir = os.path.join(BDDL_ROOT, suite)
        if not os.path.isdir(suite_dir):
            continue
        for fname in os.listdir(suite_dir):
            if not fname.endswith(".bddl"):
                continue
            if task_stem.lower() in fname.lower().replace(".bddl", ""):
                return os.path.join(suite_dir, fname)
    raise FileNotFoundError(f"No BDDL for: {task_stem}")


def match_demos_to_episodes(hdf5_path: str, episode_indices: list, dataset_root: str) -> dict:
    """Match HDF5 demos to LeRobot episodes via first-frame image similarity.

    Returns: {episode_idx: demo_idx}
    """
    import h5py
    from PIL import Image
    import io
    import pyarrow.parquet as pq

    # Load all HDF5 first frames
    with h5py.File(hdf5_path, "r") as h:
        demo_keys = sorted(
            [k for k in h["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )
        hdf5_frames = []
        for dk in demo_keys:
            img = h[f"data/{dk}/obs/agentview_rgb"][0][::-1, ::-1]
            hdf5_frames.append(cv2.resize(img, (256, 256)))

    # Load LeRobot first frames
    lr_frames = {}
    for ep_idx in episode_indices:
        for chunk in ["chunk-000", "chunk-001"]:
            fp = os.path.join(dataset_root, "data", chunk, f"episode_{ep_idx:06d}.parquet")
            if os.path.exists(fp):
                df = pq.read_table(fp, columns=["image"]).to_pandas()
                img = np.array(Image.open(io.BytesIO(df["image"].iloc[0]["bytes"])))
                lr_frames[ep_idx] = img
                break

    # Match
    mapping = {}
    for ep_idx, lr_img in lr_frames.items():
        best_mse, best_demo = float("inf"), -1
        for di, hdf5_img in enumerate(hdf5_frames):
            mse = float(np.mean((lr_img.astype(float) - hdf5_img.astype(float)) ** 2))
            if mse < best_mse:
                best_mse, best_demo = mse, di
        mapping[ep_idx] = (best_demo, best_mse)

    return mapping


def render_masks_for_demo(env, hdf5_path: str, demo_idx: int, instance_ids: list, resolution: int) -> np.ndarray:
    """Render segmentation masks by replaying from init_state + actions.

    Uses init_state (stored in HDF5 demo attrs) to set exact initial scene,
    then steps with recorded actions. env.step() triggers proper re-rendering,
    unlike direct state manipulation + _get_observations() which caches.
    """
    import h5py

    with h5py.File(hdf5_path, "r") as h:
        init_state = h[f"data/demo_{demo_idx}"].attrs["init_state"]
        actions = h[f"data/demo_{demo_idx}/actions"][:]

    # Set initial state and step through
    env.reset()
    env.env.sim.set_state_from_flattened(init_state)
    env.env.sim.forward()

    masks = []
    for frame_idx in range(len(actions)):
        obs, _, done, _ = env.step(actions[frame_idx])

        seg = obs["agentview_segmentation_instance"].squeeze(-1)[::-1, ::-1]

        mask = np.zeros(seg.shape, dtype=np.uint8)
        for obj_id in instance_ids:
            mask[seg == obj_id] = 1

        # Resize from 128x128 to target resolution
        if resolution != 128:
            mask = cv2.resize(mask, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
        masks.append(mask)

        # Don't break on done — keep stepping to get masks for all frames.
        # The env may signal done early but we need masks for the full episode.

    return np.stack(masks)


def process_task(hdf5_path: str, task_episodes: dict, dataset_root: str, output_dir: str, resolution: int):
    """Process one task: match demos, render masks, save per-episode."""
    from libero.libero.envs import OffScreenRenderEnv

    task_stem = os.path.basename(hdf5_path).replace("_demo.hdf5", "")
    episode_indices = task_episodes.get(task_stem, [])
    if not episode_indices:
        print(f"    No LeRobot episodes for {task_stem}, skipping")
        return {}

    # Check which episodes still need masks
    remaining = [ei for ei in episode_indices if not os.path.exists(os.path.join(output_dir, f"episode_{ei:06d}.npz"))]
    if not remaining:
        print(f"    All {len(episode_indices)} episodes already done, skipping")
        return {}

    # Match HDF5 demos to LeRobot episodes
    print(f"    Matching {len(remaining)} episodes to HDF5 demos...")
    mapping = match_demos_to_episodes(hdf5_path, remaining, dataset_root)

    good = sum(1 for _, (_, mse) in mapping.items() if mse < 500)
    print(f"    Matches: {good}/{len(mapping)} good (MSE<500)")

    # Create env at 128x128 (original LIBERO recording resolution)
    # Masks will be resized to target resolution afterward
    bddl_file = find_bddl_file(task_stem)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=128,
        camera_widths=128,
        camera_segmentations="instance",
    )
    env.seed(0)
    env.reset()

    # Build instance ID list for objects of interest
    # After reset, SegmentationRenderEnv assigns instance IDs sequentially
    # We need to figure out which IDs correspond to obj_of_interest
    obj_of_interest = env.env.obj_of_interest
    seg_id_map = {}
    for i, name in enumerate(list(env.env.model.instances_to_ids.keys())):
        if name not in ["Panda0", "RethinkMount0", "PandaGripper0"]:
            seg_id_map[i] = name
    instance_to_id = {v: k + 1 for k, v in seg_id_map.items()}
    instance_ids = [instance_to_id[obj] for obj in obj_of_interest if obj in instance_to_id]
    print(f"    OOI: {obj_of_interest}, instance IDs: {instance_ids}")

    # Render masks for each matched episode
    result_mapping = {}
    rendered_demos = {}  # cache: demo_idx -> masks array

    for ep_idx, (demo_idx, mse) in mapping.items():
        if mse > 1000:
            print(f"    ep {ep_idx}: MSE={mse:.0f} too high, skipping")
            continue

        if demo_idx not in rendered_demos:
            masks = render_masks_for_demo(env, hdf5_path, demo_idx, instance_ids, resolution)
            rendered_demos[demo_idx] = masks

        masks = rendered_demos[demo_idx]
        out_path = os.path.join(output_dir, f"episode_{ep_idx:06d}.npz")
        np.savez_compressed(out_path, masks=masks)
        result_mapping[str(ep_idx)] = {
            "hdf5_file": os.path.basename(hdf5_path),
            "demo_idx": demo_idx,
            "mse": round(mse, 1),
            "num_frames": len(masks),
        }
        print(f"    ep {ep_idx} <- demo_{demo_idx} (MSE={mse:.0f}), {len(masks)} frames")

    env.close()
    return result_mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5-dir", type=str, default="./data/libero_demos")
    parser.add_argument("--output-dir", type=str, default="./data/igca_masks")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--task-file", type=str, default=None,
                        help="Process a single HDF5 file (full path). Overrides suite scanning.")
    args = parser.parse_args()

    # Auto-detect dataset root
    dataset_root = args.dataset_root
    if dataset_root is None:
        for c in [
            "/DATA/disk0/yjb/.cache/huggingface/lerobot/physical-intelligence/libero",
            os.path.expanduser("~/.cache/huggingface/lerobot/physical-intelligence/libero"),
        ]:
            if os.path.isdir(c):
                dataset_root = c
                break
    print(f"Dataset root: {dataset_root}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load LeRobot episode metadata
    eps = []
    with open(os.path.join(dataset_root, "meta", "episodes.jsonl")) as f:
        for line in f:
            eps.append(json.loads(line))

    # Group episodes by task stem (normalized)
    from collections import defaultdict
    task_episodes = defaultdict(list)
    for e in eps:
        task_desc = e["tasks"][0]
        task_stem = task_desc.lower().replace(" ", "_")
        task_episodes[task_stem].append(e["episode_index"])

    # Process HDF5 files
    all_mapping = {}

    if args.task_file:
        # Single file mode
        task_stem = os.path.basename(args.task_file).replace("_demo.hdf5", "")
        print(f"  {task_stem}")
        try:
            result = process_task(args.task_file, task_episodes, dataset_root, args.output_dir, args.resolution)
            all_mapping.update(result)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Scan all suites
        suite_dirs = sorted(os.listdir(args.hdf5_dir))
        for suite_name in suite_dirs:
            suite_path = os.path.join(args.hdf5_dir, suite_name)
            if not os.path.isdir(suite_path):
                continue

            hdf5_files = sorted([f for f in os.listdir(suite_path) if f.endswith(".hdf5")])
            if not hdf5_files:
                continue

            print(f"\n[{suite_name}] {len(hdf5_files)} tasks")
            for hdf5_file in hdf5_files:
                task_stem = hdf5_file.replace("_demo.hdf5", "")
                print(f"  {task_stem}")
                try:
                    result = process_task(
                        os.path.join(suite_path, hdf5_file),
                        task_episodes,
                        dataset_root,
                        args.output_dir,
                        args.resolution,
                    )
                    all_mapping.update(result)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    # Save mapping
    mapping_path = os.path.join(args.output_dir, "mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(all_mapping, f, indent=2)
    print(f"\nSaved mapping to {mapping_path}")
    print(f"Total episodes with masks: {len(all_mapping)}")


if __name__ == "__main__":
    main()
