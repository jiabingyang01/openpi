#!/bin/bash
# 8 GPU 并行预计算 LIBERO segmentation masks
# 收集所有 HDF5 文件，每 8 个一批分配到 8 张 GPU 并行处理
#
# Usage:
#   cd /DATA/disk0/yjb/projects/VLA/openpi
#   bash scripts/run_precompute_masks.sh

set -euo pipefail
cd "$(dirname "$0")/.."

HDF5_DIR="./data/libero_demos"
OUTPUT_DIR="./data/igca_masks"
PYTHON="examples/libero/.venv/bin/python"
SCRIPT="scripts/precompute_libero_masks.py"
NUM_GPUS=8

# Collect all HDF5 files from all suites
ALL_FILES=()
for suite in libero_spatial libero_object libero_goal libero_10; do
    suite_dir="$HDF5_DIR/$suite"
    if [ -d "$suite_dir" ]; then
        for f in "$suite_dir"/*.hdf5; do
            [ -f "$f" ] && ALL_FILES+=("$f")
        done
    fi
done

total=${#ALL_FILES[@]}
echo "Total HDF5 files: $total"
echo "Using $NUM_GPUS GPUs"
echo ""

# Process in batches of NUM_GPUS
for ((batch_start=0; batch_start<total; batch_start+=NUM_GPUS)); do
    pids=()
    batch_end=$((batch_start + NUM_GPUS))
    if [ $batch_end -gt $total ]; then
        batch_end=$total
    fi

    echo "--- Batch $((batch_start/NUM_GPUS + 1)): files $batch_start-$((batch_end-1)) ---"

    for ((i=batch_start; i<batch_end; i++)); do
        gpu=$((i - batch_start))
        task_name=$(basename "${ALL_FILES[$i]}" _demo.hdf5)
        echo "  GPU $gpu: $task_name"

        CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=third_party/libero:${PYTHONPATH:-} \
        $PYTHON $SCRIPT \
            --hdf5-dir "$HDF5_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --task-file "${ALL_FILES[$i]}" &
        pids+=($!)
    done

    # Wait for this batch to finish
    wait "${pids[@]}" 2>/dev/null || true
    echo "  Batch done!"
    echo ""
done

echo "=== All done! ==="
ls "$OUTPUT_DIR"/episode_*.npz 2>/dev/null | wc -l | xargs -I{} echo "Total mask files: {}"
