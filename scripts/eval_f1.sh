#!/usr/bin/env bash
set -euo pipefail

PRED_MESH=${1:-output/mill19_building_topogs/possion_mesh/tsdf_fusion.ply}
GT_MESH=${2:-datasets/mill19/building/tsdf_fusion.ply}
DTAU=${DTAU:-0.5}

python eval_f1.py --ply_path_pred "${PRED_MESH}" --ply_path_gt "${GT_MESH}" --dtau "${DTAU}"
