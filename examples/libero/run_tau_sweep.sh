#!/usr/bin/env bash
# Sequential τ sweep for Skip-VLA d040 gate. Each run is a full 500-episode eval.
#
# Usage:
#   bash examples/libero/run_tau_sweep.sh
#
# Picks up CKPT_ROOT / SERVER_PYTHON / CLIENT_PYTHON defaults from
# run_parallel_eval.sh unless overridden here.

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/../.."   # → repo root

CKPT_ROOT="${CKPT_ROOT:-/DATA/disk0/yjb/.cache/openpi/openpi-assets/pytorch_checkpoints}"
SKIPVLA_CKPT="${SKIPVLA_CKPT:-.cache/skipvla/spatial_P+G_d040.pt}"
BASE_PORT="${BASE_PORT:-9000}"
NUM_TRIALS="${NUM_TRIALS:-50}"
TAUS="${TAUS:-0.9 0.7 0.6 0.5}"

SWEEP_LOG="./logs/tau_sweep_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$SWEEP_LOG")"

echo "===== tau sweep start $(date) =====" | tee -a "$SWEEP_LOG"
echo "TAUS=$TAUS  NUM_TRIALS=$NUM_TRIALS  BASE_PORT=$BASE_PORT" | tee -a "$SWEEP_LOG"
echo "CKPT_ROOT=$CKPT_ROOT  SKIPVLA_CKPT=$SKIPVLA_CKPT" | tee -a "$SWEEP_LOG"
echo | tee -a "$SWEEP_LOG"

for tau in $TAUS; do
  tau_tag=$(echo "$tau" | tr -d '.')
  run_id="pi0-spatial-SKIPVLA-v2-d040-tau${tau_tag}"
  echo "====================== tau=$tau ======================" | tee -a "$SWEEP_LOG"
  echo "$(date): launching $run_id" | tee -a "$SWEEP_LOG"

  export CKPT_ROOT BASE_PORT NUM_TRIALS
  export RUN_ID="$run_id"
  export MIRROR_WORKER_0="${MIRROR_WORKER_0:-0}"
  export EXTRA_CLIENT_ARGS="--skipvla-ckpt ${SKIPVLA_CKPT} --skipvla-tau ${tau} --skipvla-max-skips 5"

  if bash examples/libero/run_parallel_eval.sh >> "$SWEEP_LOG" 2>&1; then
    echo "$(date): tau=$tau done" | tee -a "$SWEEP_LOG"
  else
    echo "$(date): tau=$tau FAILED (rc=$?) — continuing" | tee -a "$SWEEP_LOG"
  fi
  echo | tee -a "$SWEEP_LOG"
done

echo "===== tau sweep finished $(date) =====" | tee -a "$SWEEP_LOG"
echo "sweep log: $SWEEP_LOG"
