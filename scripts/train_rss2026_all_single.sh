#!/usr/bin/env bash
# 一键并行训练 3 个 Single-Task config，每个占一张 GPU，后台运行、各自日志。
#
# 用法：
#   GPU_LIST="0 1 2" bash scripts/train_rss2026_all_single.sh
#   GPU_LIST="3 4 5" SKIP_NORM=1 bash scripts/train_rss2026_all_single.sh   # norm_stats 已算过
#
# 环境变量：
#   GPU_LIST   3 个 GPU 编号（每任务一张），默认 "0 1 2"
#   其余（BATCH_SIZE/FSDP_DEVICES/SKIP_NORM/WANDB/EXTRA）会透传给 train_rss2026.sh
#
# 训练在后台启动；查看进度：tail -f logs/<config>_*.log ；停止：kill <PID>。
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=( pi05_insert-mouse-battery pi05_seal-water-bottle-cap pi05_tower-of-hanoi-game )
read -r -a GPUS <<< "${GPU_LIST:-0 1 2}"

if [[ "${#GPUS[@]}" -lt "${#CONFIGS[@]}" ]]; then
  echo "需要 ${#CONFIGS[@]} 张 GPU，但 GPU_LIST 只给了 ${#GPUS[@]} 个：${GPUS[*]}"
  exit 1
fi

echo "空闲显存检查（开训前请确认这些卡可用）："
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv 2>/dev/null || true
echo "------------------------------------------------------------"

mkdir -p logs
PIDS=()
for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"; gpu="${GPUS[$i]}"
  ts="$(date +%Y%m%d_%H%M%S)"
  launchlog="logs/${cfg}_launch_${ts}.log"
  echo ">>> 启动 $cfg 于 GPU $gpu （后台），launcher 日志: $launchlog"
  GPUS="$gpu" nohup bash scripts/train_rss2026.sh "$cfg" "$cfg" >"$launchlog" 2>&1 &
  PIDS+=("$!")
  sleep 2   # 错开启动，避免同时抢 norm_stats / 下载缓存
done

echo "------------------------------------------------------------"
echo "已后台启动 ${#PIDS[@]} 个训练，PID: ${PIDS[*]}"
echo "查看进度: tail -f logs/pi05_*_*.log"
echo "停止全部: kill ${PIDS[*]}"
