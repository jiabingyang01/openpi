#!/usr/bin/env bash
# Run LIBERO eval across NUM_GPUS GPUs in parallel.
#
#   1. Launches NUM_GPUS policy servers, one per GPU (CUDA_VISIBLE_DEVICES=i) on port BASE_PORT+i.
#   2. Waits for each server port to accept connections.
#   3. Launches NUM_GPUS eval clients that each handle a sharded subset of episodes.
#   4. Waits for all clients to finish.
#   5. Stops the servers and aggregates per-worker JSONs into a per-subtask summary.
#
# Env vars you can override (most people only need MODEL and TASK_SUITE):
#   MODEL               (default pi0)               pi0 | pi05
#   TASK_SUITE          (default libero_spatial)    libero_spatial|libero_object|libero_goal|libero_10|libero_90
#   CKPT_ROOT           (default /DATA/disk1/yjb/.cache/openpi/openpi-assets/pytorch_checkpoints)
#   POLICY_CONFIG       (derived: ${MODEL}_libero)  overrides MODEL if set explicitly
#   POLICY_DIR          (derived: $CKPT_ROOT/${MODEL}_libero)
#   USE_DEFAULT         (default 0)                 if 1, skip policy:checkpoint
#   NUM_GPUS            (default 8)
#   BASE_PORT           (default 8000)
#   NUM_TRIALS          (default 50)                trials per subtask
#   REPLAN_STEPS        (default 5)
#   SEED                (default 9)
#   LOG_DIR             (default ./logs/parallel_<RUN_ID>)
#   RUN_ID              (default ${MODEL}-${TASK_SUITE}-<timestamp>)
#   OPENPI_DIR          (default: script's ../.. i.e. repo root)
#   SERVER_READY_TIMEOUT(default 600)
#   EXTRA_SERVER_ARGS   additional args appended to each serve_policy call
#   EXTRA_CLIENT_ARGS   additional args appended to each main_parallel call
#
# Examples:
#   # Default: pi0 on libero_spatial
#   bash examples/libero/run_parallel_eval.sh
#   # pi0.5 on libero_goal
#   MODEL=pi05 TASK_SUITE=libero_goal bash examples/libero/run_parallel_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =====================================================================
# USER CONFIG — edit these values directly, no need to export in shell.
# (Any shell env var of the same name still overrides, via ${VAR:-...}.)
# =====================================================================

# Which suite to eval: libero_spatial | libero_object | libero_goal | libero_10 | libero_90
TASK_SUITE="${TASK_SUITE:-libero_spatial}"

# Which VLA to eval: "pi0" or "pi05". The rest (config name, checkpoint dir) is derived.
MODEL="${MODEL:-pi0}"

# Checkpoint root. Each model's ckpt is expected at $CKPT_ROOT/${MODEL}_libero/
# with assets/, config.json, model.safetensors inside.
CKPT_ROOT="${CKPT_ROOT:-/DATA/disk1/yjb/.cache/openpi/openpi-assets/pytorch_checkpoints}"

# Derived. Override POLICY_CONFIG / POLICY_DIR explicitly if you have a custom ckpt.
POLICY_CONFIG="${POLICY_CONFIG:-${MODEL}_libero}"
POLICY_DIR="${POLICY_DIR:-${CKPT_ROOT}/${MODEL}_libero}"

# Set to 1 to skip policy:checkpoint and use openpi's default ckpt (rarely used).
USE_DEFAULT="${USE_DEFAULT:-0}"

# Parallelism
NUM_GPUS="${NUM_GPUS:-8}"
BASE_PORT="${BASE_PORT:-8000}"

# Eval params
NUM_TRIALS="${NUM_TRIALS:-50}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
SEED="${SEED:-8}"

# RUN_ID: identifies this run. Defaults to ${MODEL}-${TASK_SUITE}-<timestamp>, giving
# distinct log dirs for pi0 vs pi05 and across suites. Override for custom labels:
#   RUN_ID="pi0_base_ep30k"
#   RUN_ID="pi05_ft_${TASK_SUITE}_v1"
TS="$(date +%Y_%m_%d-%H_%M_%S)"
RUN_ID="${RUN_ID:-${MODEL}-${TASK_SUITE}}-${TS}"

# Where all logs/JSON go. Override LOG_DIR if you want a custom path.
LOG_DIR="${LOG_DIR:-./logs/parallel_${RUN_ID}}"

# Misc
OPENPI_DIR="${OPENPI_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-600}"

# Two separate python envs per examples/libero/README.md:
#   SERVER_PYTHON -> openpi top-level .venv (transformers, torch 2.x, openpi).
#   CLIENT_PYTHON -> examples/libero/.venv (py3.8, torch 1.11+cu113, robosuite, libero).
SERVER_PYTHON="${SERVER_PYTHON:-${OPENPI_DIR}/.venv/bin/python}"
CLIENT_PYTHON="${CLIENT_PYTHON:-${OPENPI_DIR}/examples/libero/.venv/bin/python}"

# This system is missing /usr/share/glvnd/egl_vendor.d/10_nvidia.json, so libEGL falls back
# to mesa which needs /dev/dri perms we don't have. Point it at NVIDIA's ICD via a local JSON.
EGL_ICD_JSON="${EGL_ICD_JSON:-${SCRIPT_DIR}/10_nvidia.json}"

# torch.compile / triton caches (~500MB per GPU after first compile).
# Kept outside LOG_DIR so they're reused across runs (subsequent runs start ~instantly).
# Delete this whole dir to force a clean recompile.
COMPILE_CACHE_ROOT="${COMPILE_CACHE_ROOT:-${OPENPI_DIR}/.cache/parallel_eval}"

# Mirror worker 0's client output to the terminal so you see live eval progress.
# Set to 0 to silence. All workers always log to files regardless.
MIRROR_WORKER_0="${MIRROR_WORKER_0:-1}"

# Extra args appended to serve_policy / main_parallel. Leave empty if unused.
# Example: EXTRA_CLIENT_ARGS="--save-video"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
EXTRA_CLIENT_ARGS="${EXTRA_CLIENT_ARGS:-}"

# =====================================================================
# End user config.
# =====================================================================

mkdir -p "$LOG_DIR"
mkdir -p "$COMPILE_CACHE_ROOT"

echo "[launcher] OPENPI_DIR=$OPENPI_DIR"
echo "[launcher] RUN_ID=$RUN_ID"
echo "[launcher] LOG_DIR=$LOG_DIR"
echo "[launcher] NUM_GPUS=$NUM_GPUS BASE_PORT=$BASE_PORT TASK_SUITE=$TASK_SUITE"

if [[ "$USE_DEFAULT" == "1" ]]; then
  POLICY_ARGS="--env LIBERO"
else
  if [[ -z "$POLICY_DIR" ]]; then
    echo "[launcher] ERROR: POLICY_DIR is required unless USE_DEFAULT=1" >&2
    exit 1
  fi
  if [[ ! -d "$POLICY_DIR" ]]; then
    echo "[launcher] ERROR: POLICY_DIR does not exist: $POLICY_DIR" >&2
    echo "[launcher]        expected layout: \$POLICY_DIR/{assets,config.json,model.safetensors}" >&2
    exit 1
  fi
  # if [[ ! -f "$POLICY_DIR/model.safetensors" ]]; then
  #   echo "[launcher] ERROR: model.safetensors missing under $POLICY_DIR" >&2
  #   exit 1
  # fi
  echo "[launcher] MODEL=$MODEL POLICY_CONFIG=$POLICY_CONFIG POLICY_DIR=$POLICY_DIR"
  POLICY_ARGS="--env LIBERO policy:checkpoint --policy.config ${POLICY_CONFIG} --policy.dir ${POLICY_DIR}"
fi

# Verify SERVER_PYTHON has server deps.
if [[ ! -x "$SERVER_PYTHON" ]]; then
  echo "[launcher] ERROR: SERVER_PYTHON not found: $SERVER_PYTHON" >&2
  echo "[launcher]        expected top-level openpi venv; see README.md" >&2
  exit 1
fi
if ! "$SERVER_PYTHON" -c "import transformers, torch, openpi" 2>/dev/null; then
  echo "[launcher] ERROR: $SERVER_PYTHON missing server deps (transformers/torch/openpi)." >&2
  "$SERVER_PYTHON" -c "import transformers, torch, openpi" >&2 || true
  exit 1
fi

# Verify CLIENT_PYTHON has client deps.
if [[ ! -x "$CLIENT_PYTHON" ]]; then
  echo "[launcher] ERROR: CLIENT_PYTHON not found: $CLIENT_PYTHON" >&2
  echo "[launcher]        expected examples/libero/.venv; per README run:" >&2
  echo "[launcher]          uv venv --python 3.8 examples/libero/.venv" >&2
  echo "[launcher]          source examples/libero/.venv/bin/activate" >&2
  echo "[launcher]          uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \\" >&2
  echo "[launcher]              --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match" >&2
  echo "[launcher]          uv pip install -e packages/openpi-client" >&2
  echo "[launcher]          uv pip install -e third_party/libero" >&2
  exit 1
fi
# Client imports libero.libero.envs (needs robosuite) and openpi_client.
CLIENT_LIBERO_PATH="${OPENPI_DIR}/third_party/libero"
if ! PYTHONPATH="${CLIENT_LIBERO_PATH}${PYTHONPATH:+:$PYTHONPATH}" \
     "$CLIENT_PYTHON" -c "import robosuite, libero.libero, openpi_client" 2>/dev/null; then
  echo "[launcher] ERROR: $CLIENT_PYTHON missing client deps (robosuite/libero/openpi_client)." >&2
  echo "[launcher]        your libero venv may be empty; check with:" >&2
  echo "[launcher]          source examples/libero/.venv/bin/activate && uv pip list" >&2
  PYTHONPATH="${CLIENT_LIBERO_PATH}${PYTHONPATH:+:$PYTHONPATH}" \
    "$CLIENT_PYTHON" -c "import robosuite, libero.libero, openpi_client" >&2 || true
  exit 1
fi

echo "[launcher] SERVER_PYTHON=$SERVER_PYTHON"
echo "[launcher] CLIENT_PYTHON=$CLIENT_PYTHON"
echo "[launcher] EGL_ICD_JSON=$EGL_ICD_JSON"

# libero's __init__.py prompts interactively on first import to create ~/.libero/config.yaml,
# which would deadlock 8 parallel client processes. Pre-create it with the package defaults.
LIBERO_CFG_DIR="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
LIBERO_CFG_FILE="$LIBERO_CFG_DIR/config.yaml"
if [[ ! -f "$LIBERO_CFG_FILE" ]]; then
  echo "[launcher] initializing libero config at $LIBERO_CFG_FILE"
  mkdir -p "$LIBERO_CFG_DIR"
  (cd "$OPENPI_DIR" && PYTHONPATH="${CLIENT_LIBERO_PATH}${PYTHONPATH:+:$PYTHONPATH}" \
    echo "n" | "$CLIENT_PYTHON" -c "import libero.libero" >/dev/null 2>&1) || true
  if [[ ! -f "$LIBERO_CFG_FILE" ]]; then
    echo "[launcher] ERROR: failed to create $LIBERO_CFG_FILE" >&2
    exit 1
  fi
fi

SERVER_PIDS=()
CLIENT_PIDS=()

cleanup() {
  local rc=$?
  echo "[launcher] cleanup (rc=$rc): stopping ${#SERVER_PIDS[@]} server(s)"
  for pid in "${SERVER_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  # give them a moment, then SIGKILL stragglers
  sleep 3
  for pid in "${SERVER_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
  exit "$rc"
}
trap cleanup EXIT INT TERM

wait_for_port() {
  # Check /proc/net/tcp for a LISTEN socket (state 0A) on the given port WITHOUT opening a
  # connection. Earlier versions did "exec 3<>/dev/tcp/127.0.0.1/$port" which triggered
  # the server's websockets handler to log a scary "InvalidMessage" Traceback every probe.
  local port=$1
  local port_hex
  port_hex=$(printf '%04X' "$port")
  local deadline=$(( SECONDS + SERVER_READY_TIMEOUT ))
  while (( SECONDS < deadline )); do
    if awk -v p=":$port_hex" '$2 ~ p"$" && $4 == "0A"{found=1; exit} END{exit !found}' \
         /proc/net/tcp /proc/net/tcp6 2>/dev/null; then
      return 0
    fi
    sleep 2
  done
  echo "[launcher] timed out waiting for port $port" >&2
  return 1
}

echo "[launcher] starting $NUM_GPUS policy servers..."
for ((i=0; i<NUM_GPUS; i++)); do
  port=$(( BASE_PORT + i ))
  server_log="$LOG_DIR/server-gpu${i}-port${port}.log"
  echo "[launcher]   server $i: CUDA_VISIBLE_DEVICES=$i port=$port log=$server_log"
  (
    cd "$OPENPI_DIR"
    # Per-GPU torch.compile / triton cache dirs. Without this, 8 servers race on the same
    # ~/.cache/torch/inductor/ path and crash with "could not convert string to float: ''".
    # Stored OUTSIDE LOG_DIR so they're reused across runs (first run compiles ~4GB, later runs
    # start instantly).
    CUDA_VISIBLE_DEVICES="$i" \
    TORCHINDUCTOR_CACHE_DIR="$COMPILE_CACHE_ROOT/inductor-gpu${i}" \
    TRITON_CACHE_DIR="$COMPILE_CACHE_ROOT/triton-gpu${i}" \
    exec "$SERVER_PYTHON" scripts/serve_policy.py --port "$port" ${EXTRA_SERVER_ARGS:-} $POLICY_ARGS
  ) > "$server_log" 2>&1 &
  SERVER_PIDS+=("$!")
done

echo "[launcher] waiting for servers to accept connections..."
for ((i=0; i<NUM_GPUS; i++)); do
  port=$(( BASE_PORT + i ))
  wait_for_port "$port"
  echo "[launcher]   port $port ready"
done

echo "[launcher] launching $NUM_GPUS eval clients..."
for ((i=0; i<NUM_GPUS; i++)); do
  port=$(( BASE_PORT + i ))
  client_log="$LOG_DIR/client-worker${i}.log"
  results_json="$LOG_DIR/${RUN_ID}-worker${i}.json"
  echo "[launcher]   worker $i -> port $port, results=$results_json"

  # Worker 0 gets mirrored to terminal so you see live progress.
  # Other workers log silently to files.
  # MUJOCO_EGL_DEVICE_ID=$i pins each client's EGL context to its own GPU (matches compose.yml).
  if [[ "$i" == "0" && "$MIRROR_WORKER_0" == "1" ]]; then
    (
      cd "$OPENPI_DIR"
      PYTHONPATH="${CLIENT_LIBERO_PATH}${PYTHONPATH:+:$PYTHONPATH}" \
      MUJOCO_EGL_DEVICE_ID="$i" __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_ICD_JSON" \
      exec "$CLIENT_PYTHON" -u examples/libero/main_parallel.py \
        --host 127.0.0.1 --port "$port" \
        --task-suite-name "$TASK_SUITE" \
        --num-trials-per-task "$NUM_TRIALS" \
        --replan-steps "$REPLAN_STEPS" \
        --seed "$SEED" \
        --num-workers "$NUM_GPUS" \
        --worker-id "$i" \
        --run-id "$RUN_ID" \
        --local-log-dir "$LOG_DIR" \
        --results-json "$results_json" \
        ${EXTRA_CLIENT_ARGS:-} 2>&1
    ) | tee "$client_log" &
    CLIENT_PIDS+=("$!")
  else
    (
      cd "$OPENPI_DIR"
      PYTHONPATH="${CLIENT_LIBERO_PATH}${PYTHONPATH:+:$PYTHONPATH}" \
      MUJOCO_EGL_DEVICE_ID="$i" __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_ICD_JSON" \
      exec "$CLIENT_PYTHON" -u examples/libero/main_parallel.py \
        --host 127.0.0.1 --port "$port" \
        --task-suite-name "$TASK_SUITE" \
        --num-trials-per-task "$NUM_TRIALS" \
        --replan-steps "$REPLAN_STEPS" \
        --seed "$SEED" \
        --num-workers "$NUM_GPUS" \
        --worker-id "$i" \
        --run-id "$RUN_ID" \
        --local-log-dir "$LOG_DIR" \
        --results-json "$results_json" \
        ${EXTRA_CLIENT_ARGS:-}
    ) > "$client_log" 2>&1 &
    CLIENT_PIDS+=("$!")
  fi
done

echo "[launcher] (worker 0 output mirrored to this terminal; workers 1-$((NUM_GPUS-1)) log to $LOG_DIR)"

echo "[launcher] waiting for clients..."
client_rc=0
for pid in "${CLIENT_PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "[launcher] client pid=$pid exited non-zero" >&2
    client_rc=1
  fi
done

echo "[launcher] aggregating results..."
combined_json="$LOG_DIR/${RUN_ID}-combined.json"
summary_log="$LOG_DIR/${RUN_ID}-summary.txt"
(
  cd "$OPENPI_DIR"
  "$SERVER_PYTHON" examples/libero/aggregate_results.py \
    --pattern "${LOG_DIR}/${RUN_ID}-worker*.json" \
    --out "$combined_json" \
    --log "$summary_log"
) | tee "$LOG_DIR/aggregate.stdout.log"

echo "[launcher] done. logs in $LOG_DIR"
exit "$client_rc"
