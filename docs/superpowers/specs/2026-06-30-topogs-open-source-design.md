# TopoGS Open Source Release Design

## Goal

Prepare an official, clean GitHub release of TopoGS at `WZ-CS/TopoGS`, suitable for reproducing the paper's training, rendering, and evaluation workflows without publishing large outputs, datasets, or private experiment artifacts.

## Confirmed Decisions

- Release style: clean official release.
- README language: English.
- License: research/non-commercial license compatible with the inherited 3DGS/CityGS-X code lineage.
- Commit identity: `Shiqiang Gong <87632303+gongshiqiang02@users.noreply.github.com>`.
- Paper links: do not add arXiv, DOI, or project-page links yet; add paper metadata and citation text only.
- Checkpoints: do not publish pretrained checkpoints in this release; list checkpoint release as a TODO.
- Datasets: do not publish datasets; document public benchmark preparation for Mill19, UrbanScene3D, Tanks & Temples, MatrixCity, and WHU.
- Depth-Anything-V2: do not vendor the third-party repository or checkpoint; document external setup and usage.
- `submodule_cityx`: keep `diff-gaussian-rasterization` and `simple-knn` source code in the repository, following CityGS-X, but remove build artifacts and egg-info.
- Scripts: replace personal and private-path scripts with standardized templates under `scripts/`.
- Code cleaning: allow non-algorithm cleanup needed for release quality and path consistency; do not change TopoGS algorithm behavior.

## Release Scope

The release should keep the core TopoGS implementation and reproducibility surface:

- Core Python entry points: training, rendering, mesh extraction, metrics, depth preprocessing, and F1 evaluation.
- Core packages: `arguments/`, `gaussian_renderer/`, `scene/`, `utils/`, `lpipsPyTorch/`, `tools/`.
- CUDA extension sources under `submodule_cityx/diff-gaussian-rasterization` and `submodule_cityx/simple-knn`.
- Standardized scripts under `scripts/`.
- Documentation: `README.md`, `LICENSE.md`, environment setup, dataset layout, training/evaluation examples, acknowledgements, and citation.

The release should exclude:

- Datasets, outputs, checkpoints, pretrained weights, logs, and local results.
- `Depth-Anything-V2/` and its checkpoint.
- CUDA driver installer files.
- Python caches, build folders, compiled binaries, egg-info, and local editor metadata.
- Personal/private-path scripts such as `gsq_train_*`, `zw_train_*`, and per-user variants.
- Experimental backup files with Chinese suffixes and obsolete duplicate implementations.
- Temporary analysis files, generated figures, and intermediate `.npy` data that are not required to run TopoGS.

## Repository Structure

Target structure:

```text
TopoGS/
├── README.md
├── LICENSE.md
├── environment.yml
├── train.py
├── train_internal.py
├── render.py
├── render_mesh.py
├── metrics.py
├── eval_f1.py
├── multi_view_precess.py
├── arguments/
├── gaussian_renderer/
├── scene/
├── utils/
├── tools/
├── lpipsPyTorch/
├── submodule_cityx/
│   ├── diff-gaussian-rasterization/
│   └── simple-knn/
└── scripts/
    ├── train_mill19.sh
    ├── train_urbanscene3d.sh
    ├── train_tanks_temples.sh
    ├── train_matrixcity.sh
    ├── train_whu.sh
    ├── render.sh
    ├── eval_metrics.sh
    ├── eval_f1.sh
    └── extract_mesh.sh
```

## README Design

The README should present TopoGS as "Topology-Aware Anchor Feature Aggregation for Large-Scale 3D Gaussian Splatting" by Wei Zhang, Shiqiang Gong, Shengkai Yu, Zeyu Wang, and Qi Wang.

Required sections:

- Overview: explain HAC and SACA in concise release-oriented prose.
- News/TODO: include code release and future checkpoint release.
- Installation: conda environment, CUDA/PyTorch expectations, install CUDA extensions from `submodule_cityx`.
- Depth preparation: link to Depth-Anything-V2 and explain external checkpoint download and depth-map generation.
- Data: public benchmark links and expected directory structures for Mill19, UrbanScene3D, Tanks & Temples, MatrixCity, and WHU.
- Training: single-scene and benchmark script examples.
- Rendering and metrics: commands for `render.py`, `metrics.py`, `render_mesh.py`, and `eval_f1.py`.
- Acknowledgement: cite CityGS-X, 3DGS/GraphDECO, Scaffold-GS, Octree-GS, PGSR, CityGaussian/CityGaussianV2, Geo-GS, Depth-Anything-V2, and other relevant inherited components.
- Citation: include a BibTeX entry for the TopoGS paper without arXiv/DOI fields for now.

## Error Handling And Safety

- If a file has unclear ownership or may be needed for reproducibility, inspect imports and README references before deleting it.
- If a script contains private paths but useful hyperparameters, rewrite it into a placeholder-based script rather than preserving the private path.
- If a third-party license is missing from vendored CUDA extension sources, note the upstream attribution in README and keep inherited license headers intact.
- Do not commit large files over 20 MB unless explicitly approved.
- Do not push until local checks confirm no datasets, outputs, checkpoints, or private absolute paths remain in the tracked set.

## Verification

Before pushing:

- Confirm `git status --short` contains only intended release files.
- Confirm no tracked file exceeds 20 MB.
- Search for private paths and personal prefixes: `/mnt`, `/scratch`, `/pscratch`, `/home`, `zhangwei`, `gsq`, `datasets_local`, `tianji_data`.
- Search for stale CityGS-X identity in README and release scripts.
- Confirm `Depth-Anything-V2/`, CUDA installer, `output/`, `datasets/`, `__pycache__/`, `build/`, `.egg-info/`, and generated checkpoint files are not tracked.
- Run Python syntax checks on release Python files where feasible.
- Validate README command paths correspond to actual retained files.

## Out Of Scope

- Re-running full training or reproducing paper metrics.
- Publishing datasets, model checkpoints, or generated output folders.
- Rewriting the TopoGS algorithm.
- Creating an arXiv/project-page badge before those public links exist.
