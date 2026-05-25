#!/usr/bin/env python3
"""Visualize pre-computed IGCA masks overlaid on RGB images.

Generates a video showing RGB + mask overlay for a given episode.

Usage:
    cd /DATA/disk0/yjb/projects/VLA/openpi
    .venv/bin/python scripts/visualize_masks.py \
        --episode 0 \
        --mask-dir ./data/igca_masks \
        --output ./data/igca_masks/vis_episode_000000.mp4
"""

import argparse
import os
import sys

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--mask-dir", type=str, default="./data/igca_masks")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    # Auto-detect dataset root
    dataset_root = args.dataset_root
    if dataset_root is None:
        candidates = [
            "/DATA/disk0/yjb/.cache/huggingface/lerobot/physical-intelligence/libero",
            os.path.expanduser("~/.cache/huggingface/lerobot/physical-intelligence/libero"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                dataset_root = c
                break

    # Load masks
    mask_path = os.path.join(args.mask_dir, f"episode_{args.episode:06d}.npz")
    masks = np.load(mask_path)["masks"]  # [T, 256, 256]
    print(f"Masks shape: {masks.shape}, unique: {np.unique(masks)}")

    # Load images from parquet
    import pyarrow.parquet as pq
    from PIL import Image
    import io

    parquet_path = None
    data_dir = os.path.join(dataset_root, "data")
    for chunk in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, chunk, f"episode_{args.episode:06d}.parquet")
        if os.path.exists(p):
            parquet_path = p
            break

    if parquet_path is None:
        print(f"Parquet file for episode {args.episode} not found")
        return

    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    print(f"Frames: {len(df)}")

    # Output dir
    vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vis")
    os.makedirs(vis_dir, exist_ok=True)

    if args.output is None:
        args.output = os.path.join(vis_dir, f"vis_episode_{args.episode:06d}.mp4")

    # We'll write frames to a temp dir, then use ffmpeg to make mp4
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="igca_vis_")

    num_frames = min(len(df), len(masks))
    for i in range(num_frames):
        # Decode image
        img_data = df["image"].iloc[i]
        if isinstance(img_data, dict) and "bytes" in img_data:
            img_bytes = img_data["bytes"]
            img = np.array(Image.open(io.BytesIO(img_bytes)))
        else:
            img = np.array(img_data)

        # Ensure 256x256x3
        if img.shape[0] == 3:  # CHW -> HWC
            img = img.transpose(1, 2, 0)
        if img.shape[:2] != (256, 256):
            img = cv2.resize(img, (256, 256))

        # Create overlay
        mask = masks[i]  # [256, 256]
        overlay = img.copy()
        # Red tint on masked regions
        overlay[mask == 1, 0] = np.clip(overlay[mask == 1, 0].astype(int) + 100, 0, 255)  # R
        overlay[mask == 1, 1] = (overlay[mask == 1, 1] * 0.5).astype(np.uint8)  # G
        overlay[mask == 1, 2] = (overlay[mask == 1, 2] * 0.5).astype(np.uint8)  # B

        # Contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        # Side by side: original | overlay
        combined = np.concatenate([img, overlay], axis=1)  # [256, 512, 3]
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

        # Add frame number
        cv2.putText(combined_bgr, f"frame {i}/{num_frames}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save frame as png
        frame_path = os.path.join(tmp_dir, f"frame_{i:06d}.png")
        cv2.imwrite(frame_path, combined_bgr)

        # Also save a few key frames as standalone images
        if i in (0, num_frames // 4, num_frames // 2, 3 * num_frames // 4, num_frames - 1):
            key_path = os.path.join(vis_dir, f"episode_{args.episode:06d}_frame_{i:04d}.png")
            cv2.imwrite(key_path, combined_bgr)
            print(f"  Key frame saved: {key_path}")

    # Use ffmpeg to create mp4 with h264
    import subprocess
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", str(args.fps),
        "-i", os.path.join(tmp_dir, "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", args.output
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Video saved to {args.output}")
    else:
        print(f"ffmpeg failed: {result.stderr[:200]}")
        print(f"Key frames are still available in {vis_dir}")

    # Cleanup temp
    import shutil
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
