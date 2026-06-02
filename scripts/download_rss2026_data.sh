#!/usr/bin/env bash
# 下载 RSS2026 Post-Training Challenge 数据集（HF: Posttraining-RFM-RSS2026/Challenge-phase1-dataset）
#
# 用法：
#   bash scripts/download_rss2026_data.sh              # 默认：3 个任务的 expert-data（跑基线最小集）
#   bash scripts/download_rss2026_data.sh expert       # 同上（显式）
#   bash scripts/download_rss2026_data.sh all          # 全部子集（expert + failure + success-and-hil）
#   SUBSETS="expert-data success-and-hil-data" bash scripts/download_rss2026_data.sh   # 自定义子集
#   TASKS="insert-mouse-battery" bash scripts/download_rss2026_data.sh                 # 只下某个任务
#
# 环境变量（可覆盖）：
#   DATA_ROOT  下载目标目录（默认 /DATA/disk0/yjb/datasets/rss2026/Challenge-phase1-dataset）
#   HF_HOME    HuggingFace 缓存目录（可选）
#   TASKS      任务列表（默认三任务）
#   SUBSETS    子集列表（由第 1 个参数 expert/all 决定，或显式覆盖）
set -euo pipefail

REPO="Posttraining-RFM-RSS2026/Challenge-phase1-dataset"
DATA_ROOT="${DATA_ROOT:-/DATA/disk0/yjb/datasets/rss2026/Challenge-phase1-dataset}"
TASKS="${TASKS:-insert-mouse-battery seal-water-bottle-cap tower-of-hanoi-game}"

MODE="${1:-expert}"
if [[ -z "${SUBSETS:-}" ]]; then
  case "$MODE" in
    expert) SUBSETS="expert-data" ;;
    all)    SUBSETS="expert-data failure-data success-and-hil-data" ;;
    *)      echo "未知模式: $MODE （应为 expert 或 all，或用 SUBSETS=... 覆盖）"; exit 1 ;;
  esac
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "找不到 huggingface-cli。请先安装：pip install -U 'huggingface_hub[cli]'"
  exit 1
fi

[[ -n "${HF_HOME:-}" ]] && export HF_HOME && echo "HF_HOME=$HF_HOME"
mkdir -p "$DATA_ROOT"
echo "目标目录 DATA_ROOT=$DATA_ROOT"
echo "任务: $TASKS"
echo "子集: $SUBSETS"
echo "------------------------------------------------------------"

for t in $TASKS; do
  for s in $SUBSETS; do
    echo ">>> 下载 $t/$s ..."
    huggingface-cli download "$REPO" --repo-type dataset \
      --include "$t/$s/*" \
      --local-dir "$DATA_ROOT"
  done
done

echo "------------------------------------------------------------"
echo "下载完成。已下内容："
du -sh "$DATA_ROOT"/*/ 2>/dev/null || true
echo ""
echo "下一步：确认 config.py 里各 config 的 local_root 指向这些路径，例如："
echo "  $DATA_ROOT/insert-mouse-battery/expert-data"
echo "然后：bash scripts/train_rss2026.sh pi05_insert-mouse-battery"
