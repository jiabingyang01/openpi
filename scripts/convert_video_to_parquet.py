#!/usr/bin/env python3
"""Convert ManipArena LeRobot v2.1 (video) to v2.0-style (image-in-parquet).

Reads mp4 video files, decodes frames, encodes as JPEG, and embeds them
directly into parquet files — matching the LIBERO format that LeRobot
loads without video decoding.

Usage:
    python scripts/convert_video_to_parquet.py \
        --input $SNAP/real/semantic_reasoning/press_button_in_order \
                $SNAP/real/execution_reasoning/put_blocks_to_color \
        --output /DATA/disk1/yjb/datasets/maniparena_converted/
"""

import argparse
import io
import json
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


def decode_video_frames(video_path: str) -> list[np.ndarray]:
    frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def resize_image(img: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.array(Image.fromarray(img).resize((width, height), Image.BILINEAR))


def _convert_episode(args_tuple):
    """Top-level function for multiprocessing (must be picklable)."""
    ep, input_dir, out_data_dir, image_keys, out_h, out_w, jpeg_quality, do_resize = args_tuple
    input_dir = Path(input_dir)
    out_data_dir = Path(out_data_dir)

    src_parquet = input_dir / f"data/chunk-000/episode_{ep:06d}.parquet"
    if not src_parquet.exists():
        return 0
    table = pq.read_table(src_parquet)
    n_frames = len(table)

    image_data = {}
    for img_key in image_keys:
        video_path = input_dir / f"videos/chunk-000/{img_key}/episode_{ep:06d}.mp4"
        if not video_path.exists():
            image_data[img_key] = [
                encode_jpeg(np.zeros((out_h, out_w, 3), dtype=np.uint8), jpeg_quality)
                for _ in range(n_frames)
            ]
            continue

        frames = decode_video_frames(str(video_path))
        while len(frames) < n_frames:
            frames.append(frames[-1] if frames else np.zeros((out_h, out_w, 3), dtype=np.uint8))
        frames = frames[:n_frames]

        encoded = []
        for f in frames:
            if do_resize:
                f = resize_image(f, out_h, out_w)
            encoded.append(encode_jpeg(f, jpeg_quality))
        image_data[img_key] = encoded

    columns = {}
    for col_name in table.column_names:
        columns[col_name] = table.column(col_name).to_pylist()
    for img_key in image_keys:
        columns[img_key] = [
            {"bytes": img_bytes, "path": f"frame_{i:06d}.jpg"}
            for i, img_bytes in enumerate(image_data[img_key])
        ]

    new_table = pa.table(columns)
    pq.write_table(new_table, out_data_dir / f"episode_{ep:06d}.parquet")
    return n_frames


def convert_task(input_dir: Path, output_dir: Path, jpeg_quality: int = 85,
                 resize: tuple[int, int] | None = None):
    meta_dir = input_dir / "meta"
    info = json.loads((meta_dir / "info.json").read_text())

    n_episodes = info["total_episodes"]
    image_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]

    if not image_keys:
        print(f"  No video features found, skipping {input_dir.name}")
        return

    orig_shape = info["features"][image_keys[0]]["shape"]
    out_h, out_w = resize if resize else (orig_shape[0], orig_shape[1])

    # Create output dirs
    output_dir.mkdir(parents=True, exist_ok=True)
    out_data_dir = output_dir / "data" / "chunk-000"
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir = output_dir / "meta"
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    # Copy meta files
    for mf in meta_dir.iterdir():
        shutil.copy2(mf, out_meta_dir / mf.name)

    # Update info.json
    new_features = {}
    for k, v in info["features"].items():
        if v["dtype"] == "video":
            new_features[k] = {
                "dtype": "image",
                "shape": [out_h, out_w, 3],
                "names": ["height", "width", "channel"],
            }
        else:
            new_features[k] = v

    new_info = {**info}
    new_info["features"] = new_features
    new_info["codebase_version"] = "v2.0"
    new_info.pop("video_path", None)
    (out_meta_dir / "info.json").write_text(json.dumps(new_info, indent=4))

    # Parallel conversion
    n_cpus = min(multiprocessing.cpu_count(), 32)
    print(f"  Using {n_cpus} workers")

    tasks = [
        (ep, str(input_dir), str(out_data_dir), image_keys, out_h, out_w, jpeg_quality, resize is not None)
        for ep in range(n_episodes)
    ]

    total_frames = 0
    with ProcessPoolExecutor(max_workers=n_cpus) as executor:
        futures = {executor.submit(_convert_episode, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=n_episodes, desc=f"  {input_dir.name}"):
            total_frames += future.result()

    print(f"  Done: {n_episodes} episodes, {total_frames} frames")


def main():
    parser = argparse.ArgumentParser(description="Convert ManipArena video to image-in-parquet")
    parser.add_argument("--input", nargs="+", required=True, help="Input task directories")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality (default 85)")
    parser.add_argument("--resize", type=int, nargs=2, default=None, help="Resize to H W")
    args = parser.parse_args()

    output_base = Path(args.output)

    for inp in args.input:
        inp = Path(inp)
        task_name = inp.name
        out_dir = output_base / task_name if len(args.input) > 1 else output_base
        print(f"Converting {task_name}...")
        convert_task(inp, out_dir, jpeg_quality=args.jpeg_quality,
                     resize=tuple(args.resize) if args.resize else None)

    print("\nAll done! Update your config's local_root to point to the converted directories.")


if __name__ == "__main__":
    main()
