"""Parallel LIBERO eval client.

Each worker process:
  * Connects to its own policy server (host:port).
  * Runs a shard of (task_id, episode_idx) pairs assigned via --worker-id / --num-workers.
  * Writes a per-worker JSON with per-task success/episode counts so an external
    aggregator can compute per-subtask success rates across all workers.

Sharding is done at the (task, episode) level in round-robin fashion, so load is
balanced even when num_workers does not divide num_tasks evenly (e.g. 8 workers
over a 10-task suite).

Launch one of these per GPU in parallel; see run_parallel_eval.sh.
"""

import collections
import contextlib
import dataclasses
import io
import json
import logging
import math
import os
import pathlib
import sys
import time
from typing import Optional

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    # Server connection
    host: str = "0.0.0.0"
    port: int = 8000

    # Client config
    resize_size: int = 224
    replan_steps: int = 5

    # LIBERO suite
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    seed: int = 9

    # Sharding
    num_workers: int = 1
    worker_id: int = 0

    # Output
    local_log_dir: str = "./logs"
    video_out_path: str = "data/libero/videos"
    run_id: Optional[str] = None  # shared across workers; defaults to a timestamped id
    results_json: Optional[str] = None  # per-worker JSON result path
    save_video: bool = False  # disabled by default: videos from 8 workers = a lot of IO
    # Pilot study: dump per-step (action, proprio, was_vla_call) for Skip-VLA oracle analysis.
    pilot_dump: bool = False
    # Skip-VLA (Scheme B): at each replan boundary, learned gate decides whether
    # to skip VLA and extrapolate the chunk instead. Set --skipvla-ckpt to enable.
    skipvla_ckpt: Optional[str] = None
    skipvla_tau: float = 0.7       # gate threshold: skip iff p > τ
    skipvla_max_skips: int = 15    # K_max safety guard (doc §2.4)


# Per-suite episode step budget (same as the original main.py).
_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _setup_logger(args: Args, run_id: str):
    os.makedirs(args.local_log_dir, exist_ok=True)
    log_path = os.path.join(args.local_log_dir, f"{run_id}-worker{args.worker_id}.txt")
    log_file = open(log_path, "w")

    # Quiet the stdout stream used when this worker is mirrored to the terminal:
    # just "[wN] msg", no date, no level. File log stays verbose (timestamps) for debugging.
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(f"[w{args.worker_id}] %(message)s"))
    root.addHandler(console)

    logger = logging.getLogger(__name__)
    return logger, log_file, log_path


def _log(msg: str, logger, log_file):
    logger.info(msg)
    log_file.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    log_file.flush()


@contextlib.contextmanager
def _silence_stdout():
    """Swallow libero/robosuite's noisy `print` calls (e.g. '[Warning]: datasets path ...')."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _assigned_episodes(task_id: int, num_trials: int, worker_id: int, num_workers: int):
    """Return the episode indices for this task that belong to this worker."""
    return [
        ep for ep in range(num_trials)
        if (task_id * num_trials + ep) % num_workers == worker_id
    ]


def _run_episode(client, env, task_description, initial_state, max_steps, args, replay_images,
                 skipvla=None):
    """Run one episode.

    Returns (success, episode_stats) where episode_stats collects what's needed for
    VLA-style inference reporting:
      * vla_latencies_ms: list of client.infer() wall times, one per VLA call
      * control_steps: number of env.step() calls that executed the policy
        (i.e. excludes the num_steps_wait warmup dummy steps)
      * control_wall_s: wall time spanning those control_steps (for frequency = Hz)
      * episode_wall_s: total wall time from first real step to terminal step
    """
    env.reset()
    action_plan = collections.deque()
    env.set_init_state(initial_state)

    vla_latencies_ms = []
    mlp_latencies_ms = []     # Skip-VLA: time to generate a chunk via the predictor
    control_steps = 0
    control_wall_start = None
    action_chunk_len = None  # policy-returned chunk length (e.g. 50 for pi0); set on first infer

    # Pilot trajectory recording (zero overhead if args.pilot_dump is False).
    pilot_actions = []      # executed actions, list of (action_dim,) arrays
    pilot_proprio = []      # proprio at step, list of (proprio_dim,) arrays
    pilot_is_vla = []       # bool per step: was a fresh VLA infer() issued this step
    pilot_dump = getattr(args, "pilot_dump", False)

    # Skip-VLA Scheme B: per-episode skip accounting.
    if skipvla is not None:
        skipvla.reset()
    skipvla_n_skip = 0
    skipvla_n_call = 0
    prev_proprio_vec = None

    obs = None
    t = 0
    done = False
    while t < max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
        wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))

        if args.save_video:
            replay_images.append(img)

        if control_wall_start is None:
            control_wall_start = time.perf_counter()

        fresh_vla_call = False
        if not action_plan:
            proprio_vec = np.concatenate((
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ))

            use_skip = False
            if skipvla is not None:
                use_skip = skipvla.should_skip(proprio_vec, prev_proprio_vec)

            if use_skip:
                # Scheme B: gate approved a skip. MLP generates a full chunk (same length
                # as VLA's action_chunk_len, so it's a drop-in replacement for reporting).
                # We still only consume replan_steps before the next boundary (matches
                # openpi's deployment convention), so extras are discarded just like VLA's.
                K_gen = action_chunk_len if action_chunk_len is not None else args.replan_steps
                t0 = time.perf_counter()
                full_chunk = skipvla.predict_chunk(K=K_gen)
                mlp_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                action_plan.extend(full_chunk[: args.replan_steps])
                skipvla.on_skip()
                skipvla_n_skip += 1
            else:
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": proprio_vec,
                    "prompt": str(task_description),
                }
                t0 = time.perf_counter()
                action_chunk = client.infer(element)["actions"]
                vla_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                if action_chunk_len is None:
                    action_chunk_len = len(action_chunk)
                assert len(action_chunk) >= args.replan_steps, (
                    f"replan_steps={args.replan_steps} but policy returned only {len(action_chunk)} actions"
                )
                action_plan.extend(action_chunk[: args.replan_steps])
                if skipvla is not None:
                    skipvla.on_call()
                skipvla_n_call += 1
                fresh_vla_call = True

            prev_proprio_vec = proprio_vec.copy()

        action = action_plan.popleft()

        if skipvla is not None:
            skipvla.record(action)

        if pilot_dump:
            # Record what we see right before executing this action. Proprio matches what
            # the policy consumes for its "observation/state".
            pilot_actions.append(np.asarray(action, dtype=np.float32).copy())
            pilot_proprio.append(np.concatenate((
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )).astype(np.float32))
            pilot_is_vla.append(fresh_vla_call)

        obs, _, done, _ = env.step(action.tolist())
        control_steps += 1
        if done:
            break
        t += 1

    control_wall_s = (time.perf_counter() - control_wall_start) if control_wall_start is not None else 0.0
    stats = {
        "vla_latencies_ms": vla_latencies_ms,
        "mlp_latencies_ms": mlp_latencies_ms,
        "control_steps": control_steps,
        "control_wall_s": control_wall_s,
        "action_chunk_len": action_chunk_len,  # what the policy returned per infer()
        "skipvla_n_skip": skipvla_n_skip,
        "skipvla_n_call": skipvla_n_call,
    }
    if pilot_dump:
        stats["pilot_actions"] = np.stack(pilot_actions) if pilot_actions else np.zeros((0, 0), dtype=np.float32)
        stats["pilot_proprio"] = np.stack(pilot_proprio) if pilot_proprio else np.zeros((0, 0), dtype=np.float32)
        stats["pilot_is_vla"] = np.asarray(pilot_is_vla, dtype=bool)
    return done, stats


def eval_libero(args: Args) -> None:
    assert 0 <= args.worker_id < args.num_workers, (
        f"worker_id={args.worker_id} must be in [0, num_workers={args.num_workers})"
    )

    np.random.seed(args.seed)

    # Run id is shared across workers so all per-worker JSONs land in the same aggregation bucket.
    run_id = args.run_id or f"EVAL-{args.task_suite_name}-{time.strftime('%Y_%m_%d-%H_%M_%S')}"
    logger, log_file, log_path = _setup_logger(args, run_id)

    with _silence_stdout():
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks

    if args.task_suite_name not in _MAX_STEPS:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    max_steps = _MAX_STEPS[args.task_suite_name]

    video_root = None
    if args.save_video:
        video_root = pathlib.Path(args.video_out_path) / args.task_suite_name / run_id / f"worker{args.worker_id}"
        video_root.mkdir(parents=True, exist_ok=True)

    _log(f"start suite={args.task_suite_name} port={args.port} shard={args.worker_id}/{args.num_workers}", logger, log_file)

    # Optional Skip-VLA Scheme B gate (non-invasive: disabled unless --skipvla-ckpt set).
    skipvla = None
    if args.skipvla_ckpt:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from skipvla import SkipVLAController
        skipvla = SkipVLAController(
            args.skipvla_ckpt, tau=args.skipvla_tau, max_skips=args.skipvla_max_skips,
        )
        _log(f"SkipVLA gate loaded: ckpt={args.skipvla_ckpt}  "
             f"tau={args.skipvla_tau}  K_max={args.skipvla_max_skips}",
             logger, log_file)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Plan this worker's shard up front so we can skip tasks with no assigned episodes.
    shard = []
    for task_id in range(num_tasks):
        eps = _assigned_episodes(task_id, args.num_trials_per_task, args.worker_id, args.num_workers)
        if eps:
            shard.append((task_id, eps))

    assigned_total = sum(len(eps) for _, eps in shard)
    _log(f"handling {assigned_total} episodes across {len(shard)} tasks", logger, log_file)

    per_task = {}
    total_episodes = 0
    total_successes = 0

    # Pilot trace accumulator: list of per-episode records.
    # Only populated when args.pilot_dump is True.
    pilot_traces = [] if args.pilot_dump else None

    for ti, (task_id, episode_ids) in enumerate(shard):
        with _silence_stdout():
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_successes = 0
        task_episodes = 0
        task_start = time.time()
        # Timing accumulators for this task (fed into the JSON for the aggregator).
        task_vla_lat_ms = []          # every single VLA forward, ms
        task_control_steps = 0        # total env.step() calls (excluding the dummy-wait ramp)
        task_control_wall_s = 0.0     # wall time of those control steps
        task_episode_walls_s = []     # per-episode wall time (includes dummy-wait) for reference
        task_action_chunk_len = None  # policy-returned chunk length
        task_skipvla_skip = 0         # Scheme B: number of chunk boundaries we skipped
        task_skipvla_call = 0         # number of chunk boundaries where we called VLA
        task_mlp_lat_ms = []          # predictor forward times (only non-empty with Skip-VLA)

        for episode_idx in episode_ids:
            replay_images = []
            ep_t0 = time.perf_counter()
            stats = None
            try:
                with _silence_stdout():
                    success, stats = _run_episode(
                        client, env, task_description, initial_states[episode_idx],
                        max_steps, args, replay_images, skipvla=skipvla,
                    )
            except Exception as e:  # noqa: BLE001
                _log(f"task{task_id} ep{episode_idx} ERROR: {e}", logger, log_file)
                success = False

            ep_wall_s = time.perf_counter() - ep_t0
            task_episode_walls_s.append(ep_wall_s)
            if stats is not None:
                # Drop the first VLA call of each episode: it absorbs any
                # client-side websocket handshake / server-side torch.compile warmup.
                lats = stats["vla_latencies_ms"]
                task_vla_lat_ms.extend(lats[1:] if len(lats) > 1 else lats)
                task_control_steps += stats["control_steps"]
                task_control_wall_s += stats["control_wall_s"]
                if task_action_chunk_len is None and stats["action_chunk_len"] is not None:
                    task_action_chunk_len = stats["action_chunk_len"]
                task_skipvla_skip += stats.get("skipvla_n_skip", 0)
                task_skipvla_call += stats.get("skipvla_n_call", 0)
                task_mlp_lat_ms.extend(stats.get("mlp_latencies_ms", []))
                if pilot_traces is not None and "pilot_actions" in stats:
                    pilot_traces.append({
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "success": bool(success),
                        "actions": stats["pilot_actions"],
                        "proprio": stats["pilot_proprio"],
                        "is_vla": stats["pilot_is_vla"],
                    })

            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1

            if args.save_video and replay_images:
                suffix = "success" if success else "failure"
                seg = task_description.replace(" ", "_")
                imageio.mimwrite(
                    video_root / f"t{task_id}_e{episode_idx}_{seg}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            log_file.write(
                f"{time.strftime('%H:%M:%S')} task{task_id} ep{episode_idx} "
                f"success={success} task={task_successes}/{task_episodes} "
                f"overall={total_successes}/{total_episodes} "
                f"({total_successes / max(total_episodes, 1) * 100:.1f}%)\n"
            )
            log_file.flush()

        env.close()
        per_task[task_id] = {
            "task_description": task_description,
            "successes": task_successes,
            "episodes": task_episodes,
            # Raw lists so the aggregator can compute true median/p95/std across workers.
            # Integers keep JSON small; ~2 KB per task per worker.
            "vla_latencies_ms": [round(x, 2) for x in task_vla_lat_ms],
            "episode_walls_s": [round(x, 3) for x in task_episode_walls_s],
            "control_steps": task_control_steps,
            "control_wall_s": round(task_control_wall_s, 3),
            "action_chunk_len": task_action_chunk_len,  # e.g. 50 for pi0
            "replan_steps": args.replan_steps,            # e.g. 5 in openpi default
            "skipvla_skip": task_skipvla_skip,
            "skipvla_call": task_skipvla_call,
            "mlp_latencies_ms": [round(x, 3) for x in task_mlp_lat_ms],
        }
        elapsed = time.time() - task_start
        # Quick per-task timing for the console (aggregator recomputes cleaner numbers later).
        lat_str = "-"
        hz_str = "-"
        skip_str = ""
        if task_vla_lat_ms:
            lat_str = f"{sum(task_vla_lat_ms)/len(task_vla_lat_ms):.0f}ms"
        if task_control_wall_s > 0 and task_control_steps > 0:
            hz_str = f"{task_control_steps / task_control_wall_s:.1f}Hz"
        if skipvla is not None:
            total_bd = task_skipvla_skip + task_skipvla_call
            if total_bd > 0:
                skip_str = f"  skip={task_skipvla_skip}/{total_bd} ({task_skipvla_skip/total_bd*100:.0f}%)"
        _log(
            f"[{ti+1}/{len(shard)}] task{task_id}: {task_successes}/{task_episodes} "
            f"({task_successes / max(task_episodes, 1) * 100:.0f}%)  "
            f"overall {total_successes}/{total_episodes} ({total_successes / max(total_episodes, 1) * 100:.1f}%)  "
            f"{elapsed:.0f}s  lat={lat_str}  ctrl={hz_str}{skip_str}  -- {task_description}",
            logger, log_file,
        )

    done_msg = (f"DONE: {total_successes}/{total_episodes} "
                f"({total_successes / max(total_episodes, 1) * 100:.2f}%)")
    if skipvla is not None:
        done_msg += f"  skip_rate={skipvla.skip_rate()*100:.2f}% ({skipvla.n_skip}/{skipvla.n_skip+skipvla.n_call})"
    _log(done_msg, logger, log_file)

    result = {
        "run_id": run_id,
        "task_suite_name": args.task_suite_name,
        "num_trials_per_task": args.num_trials_per_task,
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "per_task": per_task,
    }

    results_json = args.results_json or os.path.join(
        args.local_log_dir, f"{run_id}-worker{args.worker_id}.json"
    )
    pathlib.Path(results_json).parent.mkdir(parents=True, exist_ok=True)
    with open(results_json, "w") as f:
        json.dump(result, f, indent=2)
    _log(f"results -> {results_json}", logger, log_file)

    if pilot_traces is not None:
        traces_path = os.path.join(args.local_log_dir, f"{run_id}-traces-worker{args.worker_id}.npz")
        # Save as a single npz: one ragged set of arrays keyed by episode index.
        # Use pickle for the per-episode dicts to keep it simple.
        import pickle
        with open(traces_path.replace(".npz", ".pkl"), "wb") as f:
            pickle.dump(pilot_traces, f, protocol=pickle.HIGHEST_PROTOCOL)
        _log(f"pilot traces ({len(pilot_traces)} episodes) -> {traces_path.replace('.npz', '.pkl')}",
             logger, log_file)
    log_file.close()


if __name__ == "__main__":
    eval_libero(tyro.cli(Args))
