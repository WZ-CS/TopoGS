"""
Measure SACA Active Ratios on a Trained Model
==============================================

Loads a trained model's ply file and computes the per-level statistics of:
  - active_parent: fraction of level-L anchors that have at least one child at L+1
  - active_child:  fraction of level-L anchors that have a valid parent at L-1

Also reports ratios restricted to typical camera visibility (optional, if the
trained model has saved visibility stats).

Usage:
    python saca_diagnose.py \\
        --model_path output/building_改进3_SACA \\
        --iteration 100000

Dependencies: pip install plyfile torch numpy
"""

import torch
import numpy as np
import argparse
import os
import glob


def load_anchors_from_ply(model_path, iteration):
    """Load anchor positions, levels, and init_pos from saved ply."""
    from plyfile import PlyData

    ply_dir = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    ply_candidates = sorted(glob.glob(os.path.join(ply_dir, "*.ply")))
    if not ply_candidates:
        raise FileNotFoundError(f"No ply file in {ply_dir}")

    all_pos, all_levels = [], []
    voxel_size = None
    for p in ply_candidates:
        plydata = PlyData.read(p)
        pos = np.stack([
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ], axis=1).astype(np.float32)
        levels = np.asarray(plydata.elements[0]["level"]).astype(np.int32)
        if voxel_size is None:
            voxel_size = float(plydata.elements[0]["info"][0])
        all_pos.append(pos)
        all_levels.append(levels)

    pos = np.concatenate(all_pos)
    levels = np.concatenate(all_levels)

    # Load init_pos
    init_pos_path = os.path.join(ply_dir, "additional_attributes.npz")
    if os.path.exists(init_pos_path):
        init_pos = np.load(init_pos_path)["init_pos"].astype(np.float32)
    else:
        # Fallback: reconstruct from min
        init_pos = pos.min(axis=0).astype(np.float32)
        print(f"  WARNING: init_pos not found, using min as fallback: {init_pos}")

    return pos, levels, voxel_size, init_pos


def compute_saca_masks(pos, levels, voxel_size, init_pos, fork=2):
    """
    Returns (is_active_parent, is_active_child) boolean arrays over all anchors.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_t = torch.tensor(pos, device=device)
    levels_t = torch.tensor(levels, device=device, dtype=torch.long)
    init_pos_t = torch.tensor(init_pos, device=device)

    N = pos_t.shape[0]
    max_level = int(levels_t.max().item())

    is_active_parent = torch.zeros(N, dtype=torch.bool, device=device)
    is_active_child = torch.zeros(N, dtype=torch.bool, device=device)

    P1, P2, P3 = 73856093, 19349669, 83492791

    for L in range(1, max_level + 1):
        cur_mask = (levels_t == L)
        par_mask = (levels_t == L - 1)
        if cur_mask.sum() == 0 or par_mask.sum() == 0:
            continue

        cur_idx_global = torch.where(cur_mask)[0]
        par_idx_global = torch.where(par_mask)[0]

        cur_pos = pos_t[cur_mask]
        par_pos = pos_t[par_mask]

        par_voxel_size = voxel_size / (float(fork) ** (L - 1))
        EPS = 1e-4
        cur_par_grid = torch.floor((cur_pos - init_pos_t) / par_voxel_size + EPS).long()
        par_grid = torch.floor((par_pos - init_pos_t) / par_voxel_size + EPS).long()

        cur_keys = (cur_par_grid[:, 0] * P1) ^ (cur_par_grid[:, 1] * P2) ^ (cur_par_grid[:, 2] * P3)
        par_keys = (par_grid[:, 0] * P1) ^ (par_grid[:, 1] * P2) ^ (par_grid[:, 2] * P3)

        sorted_par, sort_idx = torch.sort(par_keys)
        search_pos = torch.searchsorted(sorted_par, cur_keys).clamp(0, par_keys.shape[0] - 1)
        matched = (sorted_par[search_pos] == cur_keys)

        if matched.any():
            matched_cur_local = torch.where(matched)[0]
            matched_par_local = sort_idx[search_pos[matched]]
            is_active_child[cur_idx_global[matched_cur_local]] = True
            is_active_parent[par_idx_global[matched_par_local]] = True

    return is_active_parent.cpu().numpy(), is_active_child.cpu().numpy()


def report(pos, levels, is_active_parent, is_active_child, scene_name):
    max_level = int(levels.max())

    print(f"\n{'=' * 70}")
    print(f"Scene: {scene_name}")
    print(f"  Total anchors: {len(levels):,}")
    print(f"  Levels: {max_level + 1}")
    print(f"{'=' * 70}")

    # Overall
    total_par = (levels < max_level).sum()
    total_chi = (levels > 0).sum()
    overall_par = is_active_parent.sum() / max(total_par, 1) * 100
    overall_chi = is_active_child.sum() / max(total_chi, 1) * 100

    print(f"\nOVERALL:")
    print(f"  Active parent ratio (level < max): {overall_par:.2f}%  "
          f"({is_active_parent.sum():,} / {total_par:,})")
    print(f"  Active child ratio  (level > 0):   {overall_chi:.2f}%  "
          f"({is_active_child.sum():,} / {total_chi:,})")

    # Per level
    print(f"\nPER-LEVEL BREAKDOWN:")
    print(f"  {'Level':<8}{'Anchors':<12}{'Active Par%':<14}{'Active Chi%':<14}{'Note':<20}")
    print(f"  {'-' * 68}")
    for L in range(max_level + 1):
        mask = (levels == L)
        n_L = mask.sum()
        if n_L == 0:
            continue
        par_pct = is_active_parent[mask].sum() / n_L * 100 if L < max_level else None
        chi_pct = is_active_child[mask].sum() / n_L * 100 if L > 0 else None
        par_str = f"{par_pct:.1f}%" if par_pct is not None else "N/A"
        chi_str = f"{chi_pct:.1f}%" if chi_pct is not None else "N/A"

        note = ""
        if par_pct is not None and par_pct < 30:
            note = "parent filter aggressive"
        if chi_pct is not None and chi_pct < 30 and not note:
            note = "child filter aggressive"

        print(f"  L{L:<7}{n_L:<12,}{par_str:<14}{chi_str:<14}{note:<20}")

    # Diagnosis
    print(f"\nDIAGNOSIS:")
    if overall_par < 20 and overall_chi < 20:
        print("  SACA filter is VERY AGGRESSIVE (<20% active).")
        print("  Expected behavior: may help large scenes, hurt small scenes.")
    elif overall_par < 50 and overall_chi < 50:
        print("  SACA filter is MODERATELY AGGRESSIVE (20-50% active).")
        print("  Expected behavior: some improvement across scenes.")
    else:
        print("  SACA filter is MILD (>50% active).")
        print("  Expected behavior: small but consistent improvement.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=100000)
    parser.add_argument("--scene_name", type=str, default="")
    parser.add_argument("--fork", type=int, default=2, help="octree fork (usually 2)")
    args = parser.parse_args()

    if not args.scene_name:
        args.scene_name = os.path.basename(args.model_path.rstrip('/'))

    print(f"Loading from {args.model_path} (iter {args.iteration})...")
    pos, levels, voxel_size, init_pos = load_anchors_from_ply(
        args.model_path, args.iteration
    )
    print(f"  Loaded {len(levels):,} anchors")
    print(f"  Voxel size: {voxel_size}")
    print(f"  init_pos: {init_pos}")

    print(f"\nComputing SACA masks...")
    is_par, is_chi = compute_saca_masks(pos, levels, voxel_size, init_pos, args.fork)

    report(pos, levels, is_par, is_chi, args.scene_name)


if __name__ == "__main__":
    main()
