"""Dump LIBERO 'agentview' camera params (pos / quat / fovy) for one or more
bddl scenes, so we can hardcode them into APSG's PinholeCamera.

Run inside the LIBERO conda env (the openpi venv won't have robosuite/mujoco).

    python scripts/dump_libero_cam.py
"""
from __future__ import annotations
import glob
import os
import sys

# libero is not pip-installed in this venv; add the source checkout to sys.path.
LIBERO_SRC = "/home/yjb/projects/VLA/LIBERO"
if LIBERO_SRC not in sys.path:
    sys.path.insert(0, LIBERO_SRC)

from libero.libero.envs import OffScreenRenderEnv

BDDL_ROOT = "/home/yjb/projects/VLA/LIBERO/libero/libero/bddl_files"

def dump_one(bddl_path: str):
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=224, camera_widths=224)
    env.reset()
    sim = env.sim
    mid = sim.model.camera_name2id("agentview")
    print(f"--- {os.path.relpath(bddl_path, BDDL_ROOT)} ---")
    print(f"  cam_pos  = {sim.model.cam_pos[mid].tolist()}")
    print(f"  cam_quat = {sim.model.cam_quat[mid].tolist()}  # wxyz")
    print(f"  fovy_deg = {float(sim.model.cam_fovy[mid])}")
    env.close()

def main():
    # Pick one task from each suite to compare.
    picks = []
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        files = sorted(glob.glob(os.path.join(BDDL_ROOT, suite, "*.bddl")))
        if files:
            picks.append(files[0])
    for p in picks:
        try:
            dump_one(p)
        except Exception as e:
            print(f"FAILED {p}: {e}")

if __name__ == "__main__":
    main()
