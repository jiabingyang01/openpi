#!/usr/bin/env bash
# 训练单个 RSS2026 config：预检数据路径 -> 计算 norm_stats -> 训练（pi05 从 pi05_base 微调）。
#
# 用法：
#   bash scripts/train_rss2026.sh <config_name> [exp_name]
# 例：
#   GPUS=0 bash scripts/train_rss2026.sh pi05_insert-mouse-battery
#   GPUS=0,1 BATCH_SIZE=64 bash scripts/train_rss2026.sh pi05_multitask
#   RESUME=1 GPUS=0 bash scripts/train_rss2026.sh pi05_insert-mouse-battery   # 断点续训
#
# 环境变量（可覆盖）：
#   GPUS          指定可见 GPU（CUDA_VISIBLE_DEVICES），默认 0
#   BATCH_SIZE    覆盖 config 的 batch_size（必须能被 GPU 数整除），默认沿用 config
#   FSDP_DEVICES  FSDP 分片设备数（必须整除 GPU 数），默认 1（纯数据并行）
#   MEM_FRACTION  XLA_PYTHON_CLIENT_MEM_FRACTION，默认 0.9
#   SKIP_NORM     =1 跳过 compute_norm_stats（已算过时）
#   RESUME        =1 用 --resume（否则 --overwrite）
#   WANDB         =0 关闭 wandb
#   EXTRA         透传给 train.py 的额外参数，如 EXTRA="--num_train_steps=80000"
set -euo pipefail

cd "$(dirname "$0")/.."   # 仓库根目录

CONFIG="${1:?用法: bash scripts/train_rss2026.sh <config_name> [exp_name]}"
EXP_NAME="${2:-$CONFIG}"
GPUS="${GPUS:-0}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.9}"

mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/${CONFIG}_${EXP_NAME}_${TS}.log"

# --- 预检：local_root 路径是否存在（避免训练中途才发现数据缺失）---
echo "[1/3] 预检数据路径 (config=$CONFIG) ..."
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run python - "$CONFIG" <<'PY'
import sys
import openpi.training.config as c
name = sys.argv[1]
cfg = c.get_config(name)
bc = getattr(cfg.data, "base_config", None)
lr = getattr(bc, "local_root", None) if bc is not None else None
import os, pathlib
roots = [lr] if isinstance(lr, str) else (list(lr) if lr else [])
if not roots:
    print(f"  [警告] 无法从 config 解析 local_root（可能未设置）。跳过路径预检。")
    sys.exit(0)
missing = [r for r in roots if not os.path.isdir(r)]
for r in roots:
    print(f"  - {'OK ' if os.path.isdir(r) else '缺失'} {r}")
if missing:
    print("\n  [错误] 上述数据目录不存在。请先下载数据，或修改 config.py 里的 local_root。")
    print("  下载：bash scripts/download_rss2026_data.sh")
    sys.exit(2)
PY

# --- 计算 norm_stats ---
if [[ "${SKIP_NORM:-0}" == "1" ]]; then
  echo "[2/3] 跳过 compute_norm_stats (SKIP_NORM=1)"
else
  echo "[2/3] 计算 norm_stats (config=$CONFIG) ..."
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py "$CONFIG"
fi

# --- 训练 ---
RESUME_FLAG="--overwrite"
[[ "${RESUME:-0}" == "1" ]] && RESUME_FLAG="--resume"

TRAIN_ARGS=( "$CONFIG" --exp_name="$EXP_NAME" "$RESUME_FLAG" --fsdp_devices="$FSDP_DEVICES" )
[[ -n "${BATCH_SIZE:-}" ]] && TRAIN_ARGS+=( --batch_size="$BATCH_SIZE" )
[[ "${WANDB:-1}" == "0" ]] && TRAIN_ARGS+=( --wandb_enabled=False )
# shellcheck disable=SC2206
[[ -n "${EXTRA:-}" ]] && TRAIN_ARGS+=( ${EXTRA} )

echo "[3/3] 开始训练: CUDA_VISIBLE_DEVICES=$GPUS  args=${TRAIN_ARGS[*]}"
echo "      日志: $LOG_FILE"
CUDA_VISIBLE_DEVICES="$GPUS" XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION" \
  uv run scripts/train.py "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
