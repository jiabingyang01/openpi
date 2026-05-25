#!/usr/bin/env python3
"""Download LIBERO HDF5 demo files for mask pre-computation.

Usage:
    cd /DATA/disk0/yjb/projects/VLA/openpi
    PYTHONPATH=third_party/libero:$PYTHONPATH \
    examples/libero/.venv/bin/python scripts/download_libero_demos.py \
        --download-dir ./data/libero_demos
"""

import argparse
import os
import sys
import zipfile
import urllib.request

from tqdm import tqdm

DATASET_LINKS = {
    "libero_spatial": "https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip",
    "libero_object": "https://utexas.box.com/shared/static/avkklgeq0e1dgzxz52x488whpu8mgspk.zip",
    "libero_goal": "https://utexas.box.com/shared/static/iv5e4dos8yy2b212pkzkpxu9wbdgjfeg.zip",
    "libero_100": "https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip",
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract(name, url, download_dir):
    zip_path = os.path.join(download_dir, f"{name}.zip")

    # Download
    print(f"\nDownloading {name}...")
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=name) as t:
        urllib.request.urlretrieve(url, filename=zip_path, reporthook=t.update_to)

    # Extract
    print(f"Extracting {name}...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(path=download_dir)

    # Remove zip
    os.remove(zip_path)
    print(f"{name} done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-dir", type=str, default="./data/libero_demos")
    parser.add_argument("--suites", type=str, nargs="+", default=list(DATASET_LINKS.keys()))
    args = parser.parse_args()

    os.makedirs(args.download_dir, exist_ok=True)

    for name in args.suites:
        if name not in DATASET_LINKS:
            print(f"[SKIP] Unknown suite: {name}")
            continue

        # Check if already downloaded
        # libero_100 extracts to libero_10/ and libero_90/
        if name == "libero_100":
            check_dirs = ["libero_10", "libero_90"]
        else:
            check_dirs = [name]

        all_exist = True
        for d in check_dirs:
            target = os.path.join(args.download_dir, d)
            if not os.path.isdir(target) or len([f for f in os.listdir(target) if f.endswith(".hdf5")]) < 10:
                all_exist = False
                break

        if all_exist:
            print(f"[SKIP] {name}: already exists")
            continue

        try:
            download_and_extract(name, DATASET_LINKS[name], args.download_dir)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            # Clean up partial zip
            zip_path = os.path.join(args.download_dir, f"{name}.zip")
            if os.path.exists(zip_path):
                print(f"  Zip file kept at {zip_path}, manually extract with: unzip -o {zip_path} -d {args.download_dir}")

    # Summary
    print("\n=== Summary ===")
    for d in ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]:
        target = os.path.join(args.download_dir, d)
        if os.path.isdir(target):
            count = len([f for f in os.listdir(target) if f.endswith(".hdf5")])
            print(f"  {d}: {count} hdf5 files")
        else:
            print(f"  {d}: NOT FOUND")


if __name__ == "__main__":
    main()
