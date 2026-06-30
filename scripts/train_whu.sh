#!/usr/bin/env bash
set -euo pipefail

SCENE_PATH=${1:-datasets/WHU/area1}
OUTPUT_PATH=${2:-output/whu_area1_topogs}
GPU_NUM=${GPU_NUM:-4}
BATCH_SIZE=${BATCH_SIZE:-8}

torchrun --standalone --nnodes=1 --nproc-per-node="${GPU_NUM}" train.py \
  --bsz "${BATCH_SIZE}" \
  -s "${SCENE_PATH}" \
  --resolution 1 \
  --model_path "${OUTPUT_PATH}" \
  --iterations 100000 \
  --images train/rgbs \
  --single_view_weight_from_iter 10000 \
  --depth_l1_weight_final 0.01 \
  --depth_l1_weight_init 0.5 \
  --dpt_loss_from_iter 10000 \
  --dpt_end_iter 30000 \
  --scale_loss_from_iter 0 \
  --multi_view_weight_from_iter 30000 \
  --default_voxel_size 0.001
