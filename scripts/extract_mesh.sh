#!/usr/bin/env bash
set -euo pipefail

SCENE_PATH=${1:-datasets/mill19/building}
MODEL_PATH=${2:-output/mill19_building_topogs}
GPU_NUM=${GPU_NUM:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
IMAGES=${IMAGES:-train/rgbs}
RESOLUTION=${RESOLUTION:-4}
VOXEL_SIZE=${VOXEL_SIZE:-0.001}
MAX_DEPTH=${MAX_DEPTH:-5}

if [ "${GPU_NUM}" -gt 1 ]; then
  torchrun --standalone --nnodes=1 --nproc-per-node="${GPU_NUM}" render_mesh.py \
    --bsz "${BATCH_SIZE}" -s "${SCENE_PATH}" --resolution "${RESOLUTION}" \
    --model_path "${MODEL_PATH}" --images "${IMAGES}" \
    --voxel_size "${VOXEL_SIZE}" --max_depth "${MAX_DEPTH}" --use_depth_filter
else
  python render_mesh.py --bsz "${BATCH_SIZE}" -s "${SCENE_PATH}" --resolution "${RESOLUTION}" \
    --model_path "${MODEL_PATH}" --images "${IMAGES}" \
    --voxel_size "${VOXEL_SIZE}" --max_depth "${MAX_DEPTH}" --use_depth_filter
fi
