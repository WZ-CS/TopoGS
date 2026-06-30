# TopoGS Open Source Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current TopoGS working directory into a clean official GitHub release for `WZ-CS/TopoGS`.

**Architecture:** The release keeps the TopoGS training/rendering/evaluation code and vendored CityGS-X CUDA extension sources, while excluding datasets, outputs, checkpoints, third-party Depth-Anything code, local binaries, private-path scripts, backup implementations, caches, and generated analysis artifacts. Documentation and scripts become the public API for reproduction.

**Tech Stack:** Python 3.8, PyTorch 2.0/CUDA 11.8, conda, bash scripts, custom CUDA extensions under `submodule_cityx`, Git/GitHub.

---

## File Structure Map

- Modify `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/.gitignore`: replace the current local ignore file with release-safe ignore rules.
- Modify `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/README.md`: replace CityGS-X README with TopoGS release documentation.
- Create `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/LICENSE.md`: research/non-commercial license notice.
- Modify `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/environment.yml`: rename environment and remove unnecessary notebook/cloud/dev packages where practical without risking runtime dependencies.
- Create `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/scripts/*.sh`: standard training, rendering, metrics, mesh, and F1 scripts.
- Keep core implementation files: `train.py`, `train_internal.py`, `densification.py`, `render.py`, `render_mesh.py`, `metrics.py`, `eval_f1.py`, `multi_view_precess.py`, `arguments/`, `gaussian_renderer/`, `scene/`, `utils/`, `tools/`, `lpipsPyTorch/`.
- Keep vendored extension source directories: `submodule_cityx/diff-gaussian-rasterization`, `submodule_cityx/simple-knn`.
- Remove from the release tree or leave untracked and ignored: `Depth-Anything-V2/`, `cuda_11.8.0_520.61.05_linux.run`, `__pycache__/`, generated figures, `.npy` intermediates, private scripts, backup Python files, build outputs, egg-info, local logs, datasets, outputs, checkpoints.

## Task 1: Release Inventory And Safety Baseline

**Files:**
- Inspect: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS`
- Modify: none

- [ ] **Step 1: Capture current untracked file inventory**

Run:

```bash
git status --short
find . -type f -size +20M -print0 | xargs -0 ls -lh 2>/dev/null || true
find . -type d \( -name build -o -name __pycache__ -o -name "*.egg-info" \) -print | sort
```

Expected: output includes the 4GB CUDA installer, the Depth-Anything checkpoint, build directories, pycache directories, and egg-info directories. Do not add these files to git.

- [ ] **Step 2: Confirm local commit identity**

Run:

```bash
git config user.name
git config user.email
```

Expected:

```text
Shiqiang Gong
87632303+gongshiqiang02@users.noreply.github.com
```

- [ ] **Step 3: Commit nothing in this task**

Run:

```bash
git status --short
```

Expected: the only committed history remains the design spec commit plus untracked working files.

## Task 2: Clean Ignore Rules

**Files:**
- Modify: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/.gitignore`

- [ ] **Step 1: Replace `.gitignore` with release rules**

Edit `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/.gitignore` to exactly:

```gitignore
# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/
.pytest_cache/
.mypy_cache/

# Build artifacts
build/
dist/
*.o
*.out
*.log
core*

# Local editors and OS files
.DS_Store
.vscode/
.idea/

# Datasets, outputs, checkpoints, and logs
data/
dataset/
datasets/
datasets_*/
output/
outputs/
results/
exp/
experiments/
logs/
tensorboard_3d/
screenshots/
checkpoints/
*.pth
*.pt
*.ckpt

# Large external dependencies and local installers
Depth-Anything-V2/
cuda_*.run
*.whl
*.zip
*.tar
*.tar.gz

# Generated analysis artifacts
*.npy
*.npz
figure*_matrices.*
saca_active_ratio.*
sim_*.*

# Local scratch scripts
run.sh
train.sh
train2.sh
```

- [ ] **Step 2: Verify ignore behavior for excluded large files**

Run:

```bash
git status --ignored --short Depth-Anything-V2 cuda_11.8.0_520.61.05_linux.run __pycache__ submodule_cityx/diff-gaussian-rasterization/build submodule_cityx/simple-knn/build
```

Expected: excluded files and directories appear with `!!`.

- [ ] **Step 3: Commit ignore rules**

Run:

```bash
git add .gitignore
git commit -m "chore: add release ignore rules"
```

Expected: one commit containing only `.gitignore`.

## Task 3: Remove Generated Extension Artifacts

**Files:**
- Delete: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/submodule_cityx/diff-gaussian-rasterization/build`
- Delete: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/submodule_cityx/diff-gaussian-rasterization/diff_gaussian_rasterization.egg-info`
- Delete: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/submodule_cityx/simple-knn/build`
- Delete: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/submodule_cityx/simple-knn/simple_knn.egg-info`

- [ ] **Step 1: Remove generated extension outputs**

Run:

```bash
rm -rf submodule_cityx/diff-gaussian-rasterization/build
rm -rf submodule_cityx/diff-gaussian-rasterization/diff_gaussian_rasterization.egg-info
rm -rf submodule_cityx/simple-knn/build
rm -rf submodule_cityx/simple-knn/simple_knn.egg-info
```

Expected: no command output.

- [ ] **Step 2: Confirm only source files remain under extension directories**

Run:

```bash
find submodule_cityx -type d \( -name build -o -name "*.egg-info" \) -print
find submodule_cityx -type f -size +5M -print
```

Expected: both commands print nothing.

- [ ] **Step 3: Commit nothing in this task**

Run:

```bash
git status --short submodule_cityx
```

Expected: no tracked deletion output yet because the extension directories are still untracked at this stage.

## Task 4: Create Public Scripts

**Files:**
- Create: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/scripts/train_mill19.sh`
- Create: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/scripts/train_urbanscene3d.sh`
- Create: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/scripts/train_tanks_temples.sh`
- Create: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/scripts/train_matrixcity.sh`
- Create: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/scripts/train_whu.sh`
- Create: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/scripts/render.sh`
- Create: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/scripts/eval_metrics.sh`
- Create: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/scripts/eval_f1.sh`
- Create: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/scripts/extract_mesh.sh`

- [ ] **Step 1: Create `scripts/train_mill19.sh`**

Create the file with:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCENE_PATH=${1:-datasets/mill19/building}
OUTPUT_PATH=${2:-output/mill19_building_topogs}
GPU_NUM=${GPU_NUM:-4}
BATCH_SIZE=${BATCH_SIZE:-8}

torchrun --standalone --nnodes=1 --nproc-per-node="${GPU_NUM}" train.py \
  --bsz "${BATCH_SIZE}" \
  -s "${SCENE_PATH}" \
  --resolution 4 \
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
```

- [ ] **Step 2: Create `scripts/train_urbanscene3d.sh`**

Create the file with:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCENE_PATH=${1:-datasets/urbanscene3d/residence}
OUTPUT_PATH=${2:-output/urbanscene3d_residence_topogs}
GPU_NUM=${GPU_NUM:-4}
BATCH_SIZE=${BATCH_SIZE:-8}

torchrun --standalone --nnodes=1 --nproc-per-node="${GPU_NUM}" train.py \
  --bsz "${BATCH_SIZE}" \
  -s "${SCENE_PATH}" \
  --resolution 4 \
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
```

- [ ] **Step 3: Create `scripts/train_tanks_temples.sh`**

Create the file with:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCENE_PATH=${1:-datasets/tanks_temples/train}
OUTPUT_PATH=${2:-output/tanks_temples_train_topogs}
GPU_NUM=${GPU_NUM:-4}
BATCH_SIZE=${BATCH_SIZE:-8}

torchrun --standalone --nnodes=1 --nproc-per-node="${GPU_NUM}" train.py \
  --bsz "${BATCH_SIZE}" \
  -s "${SCENE_PATH}" \
  --resolution 4 \
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
```

- [ ] **Step 4: Create `scripts/train_matrixcity.sh`**

Create the file with:

```bash
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
```

- [ ] **Step 5: Create `scripts/train_whu.sh`**

Create the file with:

```bash
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
```

- [ ] **Step 6: Create evaluation and utility scripts**

Create `scripts/render.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCENE_PATH=${1:-datasets/mill19/building}
MODEL_PATH=${2:-output/mill19_building_topogs}
GPU_NUM=${GPU_NUM:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
IMAGES=${IMAGES:-train/rgbs}
RESOLUTION=${RESOLUTION:-4}

if [ "${GPU_NUM}" -gt 1 ]; then
  torchrun --standalone --nnodes=1 --nproc-per-node="${GPU_NUM}" render.py \
    --bsz "${BATCH_SIZE}" -s "${SCENE_PATH}" --resolution "${RESOLUTION}" \
    --model_path "${MODEL_PATH}" --images "${IMAGES}" --skip_train
else
  python render.py --bsz "${BATCH_SIZE}" -s "${SCENE_PATH}" --resolution "${RESOLUTION}" \
    --model_path "${MODEL_PATH}" --images "${IMAGES}" --skip_train
fi
```

Create `scripts/eval_metrics.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-output/mill19_building_topogs}

python metrics.py --model_paths "${MODEL_PATH}"
```

Create `scripts/eval_f1.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

PRED_MESH=${1:-output/mill19_building_topogs/possion_mesh/tsdf_fusion.ply}
GT_MESH=${2:-datasets/mill19/building/tsdf_fusion.ply}
DTAU=${DTAU:-0.5}

python eval_f1.py --ply_path_pred "${PRED_MESH}" --ply_path_gt "${GT_MESH}" --dtau "${DTAU}"
```

Create `scripts/extract_mesh.sh`:

```bash
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
```

- [ ] **Step 7: Make scripts executable and validate shell syntax**

Run:

```bash
chmod +x scripts/*.sh
bash -n scripts/*.sh
```

Expected: no output from `bash -n`.

- [ ] **Step 8: Commit scripts**

Run:

```bash
git add scripts
git commit -m "docs: add release reproduction scripts"
```

Expected: one commit containing only files under `scripts/`.

## Task 5: Replace Public Documentation

**Files:**
- Modify: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/README.md`
- Create: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/LICENSE.md`

- [ ] **Step 1: Replace `README.md`**

Replace the file with release documentation containing these exact sections in this order:

```markdown
<div align="center">

# TopoGS: Topology-Aware Anchor Feature Aggregation for Large-Scale 3D Gaussian Splatting

Wei Zhang, Shiqiang Gong, Shengkai Yu, Zeyu Wang, Qi Wang

Northwestern Polytechnical University

</div>

This repository contains the official implementation of **TopoGS**, a topology-aware anchor feature aggregation framework for large-scale 3D Gaussian Splatting.

TopoGS builds on an octree-anchor 3DGS backbone and introduces two lightweight modules:

- **Hierarchical Anchor Coupling (HAC)** establishes cross-level gradient pathways through per-level parent-self-child context triplets.
- **Structure-Aware Containment Aggregation (SACA)** uses octree containment to softly weight topologically connected cross-level anchors.

## Updates

- 2026-06-30: Initial training, rendering, and evaluation code release.

## Todo List

- [x] Release training, rendering, and evaluation code.
- [ ] Release pretrained checkpoints.
- [ ] Add arXiv and project-page links when available.

## Installation

The code was developed with Python 3.8, PyTorch 2.0, and CUDA 11.8. Similar CUDA-enabled Linux environments should work, but may require small dependency adjustments.

```bash
git clone https://github.com/WZ-CS/TopoGS.git
cd TopoGS

conda env create -f environment.yml
conda activate topogs

pip install submodule_cityx/diff-gaussian-rasterization
pip install submodule_cityx/simple-knn
```

## Depth Preparation

TopoGS can use monocular depth priors for real-world datasets. We do not vendor Depth-Anything-V2 in this repository. Install it externally and download its checkpoint from the official project:

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2
```

Download the Depth-Anything-V2-Large checkpoint following the official instructions, then generate grayscale depth maps:

```bash
python run.py --encoder vitl --pred-only --grayscale \
  --img-path /path/to/images \
  --outdir /path/to/depths
```

Create depth scale parameters:

```bash
python utils/make_depth_scale.py \
  --base_dir /path/to/scene \
  --depths_dir /path/to/depths
```

For multi-view filtering:

```bash
python multi_view_precess.py \
  -s /path/to/scene \
  --resolution 4 \
  --model_path /path/to/scene/train/mask \
  --images train/rgbs \
  --pixel_thred 1
```

## Data

Datasets are not included in this repository. We evaluate on public benchmarks used in the paper:

- Mill19
- UrbanScene3D
- Tanks & Temples
- MatrixCity
- WHU

Prepare each scene in a COLMAP-compatible layout. For Mill19, UrbanScene3D, Tanks & Temples, and WHU-style real-world scenes, the expected layout is:

```text
datasets/<scene_name>/
├── train/
│   ├── rgbs/
│   ├── depths/
│   └── mask/
├── val/
│   └── rgbs/
└── sparse/
    └── 0/
```

For MatrixCity aerial scenes, the expected layout is:

```text
datasets/MatrixCity/aerial/small_city/aerial/
├── train/block_all/
│   ├── images/
│   ├── depth/
│   ├── mask/
│   └── sparse/0/
└── test/block_all_test/
    ├── images/
    └── sparse/0/
```

## Training

Single-scene training example:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
  --bsz 8 \
  -s datasets/mill19/building \
  --resolution 4 \
  --model_path output/mill19_building_topogs \
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
```

Benchmark script templates are provided under `scripts/`:

```bash
bash scripts/train_mill19.sh datasets/mill19/building output/mill19_building_topogs
bash scripts/train_urbanscene3d.sh datasets/urbanscene3d/residence output/urbanscene3d_residence_topogs
bash scripts/train_tanks_temples.sh datasets/tanks_temples/train output/tanks_temples_train_topogs
bash scripts/train_matrixcity.sh datasets/MatrixCity/aerial/small_city/aerial/train/block_all output/matrixcity_topogs
bash scripts/train_whu.sh datasets/WHU/area1 output/whu_area1_topogs
```

## Rendering And Evaluation

Render held-out views:

```bash
bash scripts/render.sh datasets/mill19/building output/mill19_building_topogs
```

Compute image metrics:

```bash
bash scripts/eval_metrics.sh output/mill19_building_topogs
```

Extract a mesh:

```bash
bash scripts/extract_mesh.sh datasets/mill19/building output/mill19_building_topogs
```

Compute F1 score for reconstructed meshes:

```bash
bash scripts/eval_f1.sh output/mill19_building_topogs/possion_mesh/tsdf_fusion.ply datasets/mill19/building/tsdf_fusion.ply
```

## Acknowledgements

This project builds on and benefits from the following works and codebases:

- 3D Gaussian Splatting by GraphDECO/Inria
- Scaffold-GS
- Octree-GS
- CityGS-X
- CityGaussian and CityGaussianV2
- PGSR
- Geo-GS
- Depth-Anything-V2

Please also follow the licenses and terms of the upstream projects and datasets.

## Citation

If you find TopoGS useful, please cite:

```bibtex
@article{zhang2026topogs,
  title={TopoGS: Topology-Aware Anchor Feature Aggregation for Large-Scale 3D Gaussian Splatting},
  author={Zhang, Wei and Gong, Shiqiang and Yu, Shengkai and Wang, Zeyu and Wang, Qi},
  journal={IEEE Transactions on Multimedia},
  year={2026}
}
```
```

- [ ] **Step 2: Create `LICENSE.md`**

Create the file with:

```markdown
# TopoGS License

This software is provided for non-commercial research and evaluation use only.

Copyright (c) 2026, TopoGS authors.

Permission is granted to use, copy, modify, and distribute this software and its documentation for non-commercial research and evaluation purposes, provided that the above copyright notice and this permission notice appear in all copies or substantial portions of the software.

Commercial use, including but not limited to use in commercial products, services, or internal commercial workflows, is not permitted without prior written permission from the authors.

This repository contains code derived from or inspired by several upstream research codebases, including 3D Gaussian Splatting, CityGS-X, Scaffold-GS, Octree-GS, PGSR, and related projects. Files that retain upstream copyright or license headers remain subject to those notices. Users are responsible for complying with the licenses and terms of all upstream projects and datasets.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

- [ ] **Step 3: Verify README no longer presents CityGS-X as the project**

Run:

```bash
rg -n "This repo contains official implementations of CityGS-X|ICCV 2025|gyy456/CityGS-X|Yuanyuan Gao" README.md
```

Expected: no matches.

- [ ] **Step 4: Commit documentation**

Run:

```bash
git add README.md LICENSE.md
git commit -m "docs: add TopoGS release documentation"
```

Expected: one commit containing only README and license changes.

## Task 6: Remove Private And Obsolete Release Files

**Files:**
- Delete from working tree or leave ignored: private scripts and obsolete files listed below.
- Keep tracked release files only.

- [ ] **Step 1: Remove private-path scripts and obsolete top-level scripts**

Run:

```bash
rm -f eval_f1.sh
rm -f gather_matrix_image.sh
rm -f gsq_train_matrixcity.sh
rm -f 'gsq_train_whu_area1&5.sh'
rm -f gsq_train_whu_area4.sh
rm -f gsq_train_whu_area5.sh
rm -f mesh.sh
rm -f train_matrix_city.sh
rm -f train_matrix_city_zw.sh
rm -f train_mill19.sh
rm -f zw_train_building_2.sh
rm -f zw_train_residence_2.sh
rm -f zw_train_rubble_2.sh
rm -f zw_train_sciart.sh
```

Expected: no command output.

- [ ] **Step 2: Remove backup and temporary Python files**

Run:

```bash
rm -f 'train_internal-基于1-3增加tcr版但没实验.py'
rm -f 'train_internal无改动版.py'
rm -f 'gaussian_renderer/__init__-基于1-3增加tcr版但没实验.py'
rm -f 'gaussian_renderer/__init__-改进1-3效果最好.py'
rm -f 'scene/gaussian_model-基于1-3增加tcr版但没实验.py'
rm -f 'scene/gaussian_model-改进1-3效果最好.py'
rm -f 'scene/gaussian_model-改进1，第一版非常慢.py'
rm -f 'scene/gaussian_model-改进2，基本没效果.py'
rm -f 'scene/gaussian_model_有bug备份版.py'
rm -f 'scene/gaussian_model无改动.py'
rm -f analyze.py
rm -f analyze_statistic.py
rm -f figure_2.py
rm -f run_diff_k.py
rm -f test.py
rm -f verify_stability.py
rm -f weight.py
```

Expected: no command output.

- [ ] **Step 3: Remove generated figures and intermediate data**

Run:

```bash
rm -f figure1_matrices.pdf figure1_matrices.png
rm -f figure3_active_ratio.pdf figure3_active_ratio.png
rm -f saca_active_ratio.pdf saca_active_ratio.png
rm -f saca_active_ratio_building.npy
rm -f sim_citygs.npy sim_topogs_post.npy sim_topogs_pre.npy
rm -f metrix.txt
```

Expected: no command output.

- [ ] **Step 4: Leave external/deep large directories ignored**

Run:

```bash
git status --ignored --short Depth-Anything-V2 cuda_11.8.0_520.61.05_linux.run
```

Expected: both paths are shown as ignored with `!!`; do not delete them unless the user explicitly wants local disk cleanup.

- [ ] **Step 5: Confirm removed files are absent from candidate release set**

Run:

```bash
find . -maxdepth 2 -type f | sort | rg 'gsq_|zw_|无改动|改进|基于|figure[0-9]|saca_active_ratio|sim_|metrix|analyze|cuda_11|eval_f1.sh|mesh.sh|train_matrix_city|train_mill19' || true
```

Expected: no matches except files under `.git/` if any appear from commit messages; ignore `.git/` matches.

- [ ] **Step 6: Commit nothing in this task**

Run:

```bash
git status --short
```

Expected: deleted files were untracked, so they should not appear as tracked deletions.

## Task 7: Curate Analysis Tools

**Files:**
- Create: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/tools/analysis/`
- Move: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/plot_figure1_ab_matrices.py`
- Move: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/dump_level_features.py`
- Move: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/dump_saca_active_ratio.py`
- Move: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/saca_diagnose.py`

- [ ] **Step 1: Move analysis tools into `tools/analysis`**

Run:

```bash
mkdir -p tools/analysis
mv plot_figure1_ab_matrices.py tools/analysis/plot_figure1_ab_matrices.py
mv dump_level_features.py tools/analysis/dump_level_features.py
mv dump_saca_active_ratio.py tools/analysis/dump_saca_active_ratio.py
mv saca_diagnose.py tools/analysis/saca_diagnose.py
```

Expected: no command output.

- [ ] **Step 2: Check for hard-coded private paths in retained analysis tools**

Run:

```bash
rg -n "/mnt|/scratch|/pscratch|/home|zhangwei|gsq|datasets_local|tianji_data" tools/analysis || true
```

Expected: no matches. If matches appear, replace them with argparse options or relative example paths before committing.

- [ ] **Step 3: Commit analysis tool curation**

Run:

```bash
git add tools/analysis
git commit -m "chore: curate TopoGS analysis tools"
```

Expected: one commit containing files under `tools/analysis/`.

## Task 8: Environment And Non-Algorithm Cleanup

**Files:**
- Modify: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/environment.yml`
- Modify: `/Users/gsq/Desktop/Works/3DV/TopoGS/TopoGS/train.py`

- [ ] **Step 1: Rename conda environment**

Edit the first line of `environment.yml`:

```yaml
name: topogs
```

Expected: the environment name is `topogs`, matching README.

- [ ] **Step 2: Remove unused debug import from `train.py`**

In `train.py`, delete this line:

```python
import debugpy
```

Expected: no behavior change because all debugpy usage is commented out.

- [ ] **Step 3: Check for syntax after the small cleanup**

Run:

```bash
python -m py_compile train.py
```

Expected: no output.

- [ ] **Step 4: Commit environment cleanup**

Run:

```bash
git add environment.yml train.py
git commit -m "chore: clean release environment metadata"
```

Expected: one commit containing only `environment.yml` and `train.py`.

## Task 9: Stage Core Release Files

**Files:**
- Add: release implementation files and directories.
- Do not add: ignored large/external/local files.

- [ ] **Step 1: Add intended release files explicitly**

Run:

```bash
git add \
  arguments \
  gaussian_renderer \
  lpipsPyTorch \
  scene \
  submodule_cityx \
  tools \
  utils \
  convert.py \
  densification.py \
  eval_f1.py \
  gsq_metrics.py \
  merge.py \
  metrics.py \
  multi_view_precess.py \
  normaldepth.py \
  render.py \
  render_mesh.py \
  train_internal.py
```

Expected: files are staged; ignored build artifacts remain unstaged.

- [ ] **Step 2: Review staged files before commit**

Run:

```bash
git diff --cached --name-only | sort
git diff --cached --name-only | rg 'Depth-Anything|cuda_11|__pycache__|build/|egg-info|\.npy$|\.pth$|\.pdf$|\.png$|gsq_train|zw_train|无改动|改进|基于' || true
```

Expected: first command lists only intended release files. Second command prints no matches except `gsq_metrics.py`; `gsq_metrics.py` is allowed if it is retained as an evaluation helper.

- [ ] **Step 3: Commit core release files**

Run:

```bash
git commit -m "feat: add TopoGS release code"
```

Expected: one commit containing the core implementation and vendored extension source.

## Task 10: Release Verification

**Files:**
- Inspect: full tracked set
- Modify: files only if verification finds release blockers

- [ ] **Step 1: Check tracked file size**

Run:

```bash
git ls-files -z | xargs -0 ls -lh | awk '$5 ~ /G/ || ($5 ~ /M/ && $5+0 > 20) {print}'
```

Expected: no output.

- [ ] **Step 2: Check excluded directories are not tracked**

Run:

```bash
git ls-files | rg '(^Depth-Anything-V2/|cuda_11|__pycache__|/build/|egg-info|^output/|^datasets/|\.pth$|\.npy$|\.npz$)' || true
```

Expected: no output.

- [ ] **Step 3: Check private paths and stale project identity**

Run:

```bash
git ls-files -z | xargs -0 rg -n "/mnt|/scratch|/pscratch|/home|zhangwei|datasets_local|tianji_data|This repo contains official implementations of CityGS-X|gyy456/CityGS-X|Yuanyuan Gao" || true
```

Expected: no output. If `CityGS-X` appears only in acknowledgement text or method comparison labels, it is acceptable; inspect each match and keep only those legitimate references.

- [ ] **Step 4: Run shell script syntax checks**

Run:

```bash
bash -n scripts/*.sh
```

Expected: no output.

- [ ] **Step 5: Run Python compile checks**

Run:

```bash
python -m py_compile \
  train.py train_internal.py densification.py render.py render_mesh.py metrics.py gsq_metrics.py \
  eval_f1.py multi_view_precess.py normaldepth.py convert.py merge.py \
  tools/analysis/*.py
```

Expected: no output. If imports require unavailable compiled CUDA modules at compile time, `py_compile` should still pass because it checks syntax without importing modules.

- [ ] **Step 6: Check README command references exist**

Run:

```bash
for path in \
  scripts/train_mill19.sh \
  scripts/train_urbanscene3d.sh \
  scripts/train_tanks_temples.sh \
  scripts/train_matrixcity.sh \
  scripts/train_whu.sh \
  scripts/render.sh \
  scripts/eval_metrics.sh \
  scripts/eval_f1.sh \
  scripts/extract_mesh.sh \
  submodule_cityx/diff-gaussian-rasterization/setup.py \
  submodule_cityx/simple-knn/setup.py; do
  test -e "$path" || { echo "missing $path"; exit 1; }
done
```

Expected: no output.

- [ ] **Step 7: Commit verification fixes if any were needed**

If verification required edits, run:

```bash
git add .
git commit -m "chore: fix release verification issues"
```

Expected: a commit only if files changed. If no files changed, skip this step.

## Task 11: Configure Remote And Push

**Files:**
- Modify: local git remote configuration only

- [ ] **Step 1: Inspect existing remotes**

Run:

```bash
git remote -v
```

Expected: either no remotes or a remote that needs to point to `WZ-CS/TopoGS`.

- [ ] **Step 2: Add or update origin**

If no `origin` exists, run:

```bash
git remote add origin https://github.com/WZ-CS/TopoGS.git
```

If `origin` exists but points elsewhere, run:

```bash
git remote set-url origin https://github.com/WZ-CS/TopoGS.git
```

Expected: `git remote -v` shows `https://github.com/WZ-CS/TopoGS.git`.

- [ ] **Step 3: Work around malformed lowercase proxy variables only for git commands**

Run:

```bash
env -u http_proxy -u https_proxy -u all_proxy git ls-remote origin
```

Expected: command lists the remote `main` branch. If authentication is required, git prompts through the local credential helper.

- [ ] **Step 4: Push release branch**

Run:

```bash
env -u http_proxy -u https_proxy -u all_proxy git push -u origin main
```

Expected: push succeeds to `WZ-CS/TopoGS`.

- [ ] **Step 5: Final status check**

Run:

```bash
git status --short --branch
```

Expected: branch tracks `origin/main`; ignored local large/external files may remain in the working directory but should not appear as untracked files.

## Self-Review

- Spec coverage: tasks cover clean release scope, README, license, Depth-Anything exclusion, vendored CUDA extension source cleanup, standardized scripts, checkpoints omission, dataset docs, identity, verification, remote, and push.
- Placeholder scan: script defaults use concrete example paths and environment variables, not unresolved plan placeholders. README intentionally documents example paths.
- Type and path consistency: README references `metrics.py --model_paths` because current metrics code accepts `--model_paths`; scripts use repository-root execution so Python entry-point paths remain valid.
