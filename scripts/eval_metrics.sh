#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-output/mill19_building_topogs}

python metrics.py --model_paths "${MODEL_PATH}"
