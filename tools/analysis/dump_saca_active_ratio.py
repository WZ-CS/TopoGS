# dump_saca_active_ratio.py
import os, sys, argparse
import torch
import numpy as np
from plyfile import PlyData


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--iteration', type=int, default=99997)
    p.add_argument('--out', required=True, help='output .npy')
    return p.parse_args()


def rebuild_saca_masks(anchor_pos, level, voxel_size, fork, init_pos):
    """
    Replica of GaussianModel.rebuild_saca_masks, standalone.
    
    Args:
        anchor_pos: (N, 3) float tensor
        level: (N,) int tensor
        voxel_size: float (level-0 voxel size)
        fork: int (octree branching factor, usually 2)
        init_pos: (3,) tensor (octree origin)
    
    Returns:
        is_active_parent: (N,) bool
        is_active_child:  (N,) bool
    """
    N = anchor_pos.shape[0]
    device = anchor_pos.device
    max_level = int(level.max().item())
    
    is_active_parent = torch.zeros(N, dtype=torch.bool, device=device)
    is_active_child  = torch.zeros(N, dtype=torch.bool, device=device)
    
    P1, P2, P3 = 73856093, 19349669, 83492791
    
    for L in range(1, max_level + 1):
        cur_mask = (level == L)
        par_mask = (level == L - 1)
        if cur_mask.sum() == 0 or par_mask.sum() == 0:
            continue
        
        cur_idx_global = torch.where(cur_mask)[0]
        par_idx_global = torch.where(par_mask)[0]
        
        cur_pos = anchor_pos[cur_mask]
        par_pos = anchor_pos[par_mask]
        
        # parent voxel size at level L-1
        par_voxel_size = voxel_size / (float(fork) ** (L - 1))
        
        cur_par_grid = torch.round((cur_pos - init_pos) / par_voxel_size).long()
        par_grid     = torch.round((par_pos - init_pos) / par_voxel_size).long()
        
        cur_keys = (cur_par_grid[:, 0] * P1) ^ (cur_par_grid[:, 1] * P2) ^ (cur_par_grid[:, 2] * P3)
        par_keys = (par_grid[:, 0]     * P1) ^ (par_grid[:, 1]     * P2) ^ (par_grid[:, 2]     * P3)
        
        sorted_par, sort_idx = torch.sort(par_keys)
        search_pos = torch.searchsorted(sorted_par, cur_keys).clamp(0, par_keys.shape[0] - 1)
        matched = (sorted_par[search_pos] == cur_keys)
        
        if matched.any():
            matched_cur_local = torch.where(matched)[0]
            matched_par_local = sort_idx[search_pos[matched]]
            is_active_child[cur_idx_global[matched_cur_local]] = True
            is_active_parent[par_idx_global[matched_par_local]] = True
    
    return is_active_parent, is_active_child


def main():
    args = parse_args()
    
    # ============================================================
    # 1. Load ply: anchor positions + levels
    # ============================================================
    ply_path = os.path.join(
        args.model_path, 'point_cloud',
        f'iteration_{args.iteration}', 'point_cloud.ply'
    )
    print(f'Reading: {ply_path}')
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    
    anchor_pos = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
    level = np.asarray(v['level']).astype(np.int64)
    voxel_size = float(v['info'][0])      # info[0] = voxel_size
    
    # ============================================================
    # 2. Load init_pos from additional_attributes.npz
    # ============================================================
    npz_path = os.path.join(
        args.model_path, 'point_cloud',
        f'iteration_{args.iteration}', 'additional_attributes.npz'
    )
    add = np.load(npz_path)
    init_pos = add['init_pos'].astype(np.float32)
    
    print(f'N anchors: {anchor_pos.shape[0]}')
    print(f'voxel_size: {voxel_size}')
    print(f'init_pos: {init_pos}')
    print(f'level: min={level.min()}, max={level.max()}')
    
    # ============================================================
    # 3. SACA mask compute
    # ============================================================
    fork = 2  # 你的 cfg 默认
    
    anchor_pos_t = torch.from_numpy(anchor_pos).cuda()
    level_t      = torch.from_numpy(level).cuda()
    init_pos_t   = torch.from_numpy(init_pos).cuda()
    
    is_act_par, is_act_chi = rebuild_saca_masks(
        anchor_pos_t, level_t, voxel_size, fork, init_pos_t
    )
    
    # ============================================================
    # 4. Per-level active ratio
    # ============================================================
    unique_levels = sorted(np.unique(level).tolist())
    
    print(f'\n{"Level":<8}{"#anchors":<12}{"active_par":<14}{"active_chi":<14}')
    print('-' * 50)
    
    par_ratios, chi_ratios, counts = [], [], []
    for lv in unique_levels:
        mask = (level_t == lv)
        n = int(mask.sum().item())
        counts.append(n)
        
        n_par = int(is_act_par[mask].sum().item())
        n_chi = int(is_act_chi[mask].sum().item())
        
        par_r = n_par / n if n > 0 else 0
        chi_r = n_chi / n if n > 0 else 0
        par_ratios.append(par_r)
        chi_ratios.append(chi_r)
        
        print(f'L{lv:<7}{n:<12}{par_r*100:>6.2f} %     {chi_r*100:>6.2f} %')
    
    # Combined active ratio = OR(parent, child)
    combined_ratios = []
    for lv in unique_levels:
        mask = (level_t == lv)
        n = int(mask.sum().item())
        if n == 0:
            combined_ratios.append(0)
            continue
        combined = is_act_par[mask] | is_act_chi[mask]
        combined_ratios.append(combined.sum().item() / n)
    
    # ============================================================
    # 5. Save
    # ============================================================
    np.save(args.out, {
        'levels': unique_levels,
        'counts': counts,
        'active_parent_ratio': par_ratios,
        'active_child_ratio':  chi_ratios,
        'active_combined_ratio': combined_ratios,
    }, allow_pickle=True)
    print(f'\nSaved {args.out}')


if __name__ == '__main__':
    main()