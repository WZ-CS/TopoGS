#!/usr/bin/env bash
set -euo pipefail

SCENE_PATH=${1:-datasets/MatrixCity/aerial/small_city/aerial/train/block_all}
OUTPUT_PATH=${2:-output/matrixcity_topogs}
GPU_NUM=${GPU_NUM:-8}
BATCH_SIZE=${BATCH_SIZE:-8}

torchrun --standalone --nnodes=1 --nproc-per-node="${GPU_NUM}" train.py \
  --bsz "${BATCH_SIZE}" \
  -s "${SCENE_PATH}" \
  --resolution 1 \
  --model_path "${OUTPUT_PATH}" \
  --iterations 150000 \
  --images images \
  --single_view_weight_from_iter 20000 \
  --scale_loss_from_iter 0 \
  --depth_l1_weight_final 0.01 \
  --depth_l1_weight_init 0.5 \
  --dpt_loss_from_iter 20000 \
  --multi_view_weight_from_iter 50000 \
  --multi_view_max_angle 15 \
  --multi_view_min_dis 0.01 \
  --multi_view_max_dis 25 \
  --dpt_end_iter 50000 \
  --default_voxel_size 0.0005 \
  --multi_view_patch_size 11
