# dump_level_sim_v2.py —— 完全跳过 Scene, 只读 ply
import os, sys, glob, argparse
import torch
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--iteration', type=int, default=99997)
    p.add_argument('--dump_out', required=True)
    p.add_argument('--use_hac', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    # 读 cfg_args 拿训练 config
    from argparse import Namespace
    with open(os.path.join(args.model_path, 'cfg_args')) as f:
        cfg = eval(f.read())
    print(f'feat_dim={getattr(cfg,"feat_dim",32)}, '
          f'n_offsets={getattr(cfg,"n_offsets",10)}, '
          f'fork={getattr(cfg,"fork",2)}')

    # 构造 model (不依赖 Scene)
    from gaussian_renderer import GaussianModel
    gaussians = GaussianModel(
        feat_dim=getattr(cfg, 'feat_dim', 32),
        n_offsets=getattr(cfg, 'n_offsets', 10),
        fork=getattr(cfg, 'fork', 2),
        use_feat_bank=getattr(cfg, 'use_feat_bank', False),
        appearance_dim=getattr(cfg, 'appearance_dim', 32),
        add_opacity_dist=getattr(cfg, 'add_opacity_dist', False),
        add_cov_dist=getattr(cfg, 'add_cov_dist', False),
        add_color_dist=getattr(cfg, 'add_color_dist', False),
        add_level=getattr(cfg, 'add_level', False),
        visible_threshold=getattr(cfg, 'visible_threshold', -1),
        dist2level=getattr(cfg, 'dist2level', 'round'),
        base_layer=getattr(cfg, 'base_layer', 10),
        progressive=getattr(cfg, 'progressive', True),
        extend=getattr(cfg, 'extend', 1.1),
    )

    # === 直接读 ply, 跳过 distributed_load_ply 分支 ===
    # 不调 gaussians.load_ply(), 自己手动解析 ply 提取 _anchor_feat 和 _level
    ply_path = os.path.join(args.model_path, 'point_cloud',
                            f'iteration_{args.iteration}', 'point_cloud.ply')
    print(f'Reading ply: {ply_path}')

    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    names = v.data.dtype.names
    print(f'PLY properties ({len(names)}):')
    for n in names:
        print(f'  {n}')

    # 提取 anchor feature (feat_0, feat_1, ..., feat_{D-1})
    feat_cols = sorted([n for n in names if n.startswith('feat_') or n.startswith('f_anchor')],
                       key=lambda x: int(x.split('_')[-1]))
    if not feat_cols:
        feat_cols = sorted([n for n in names if 'anchor_feat' in n],
                           key=lambda x: int(x.split('_')[-1]))
    print(f'\nFeature columns ({len(feat_cols)}): {feat_cols[:5]} ... {feat_cols[-3:]}')
    feat = np.stack([v[c] for c in feat_cols], axis=1)  # (N, D)
    print(f'Feature shape: {feat.shape}')

    # 提取 level
    if 'level' in names:
        level = np.array(v['level']).astype(int)
    else:
        # 看看叫什么
        cand = [n for n in names if 'level' in n.lower()]
        print(f'No "level" property. Candidates: {cand}')
        sys.exit(1)
    print(f'Level: min={level.min()}, max={level.max()}, shape={level.shape}')

    # HAC: 如果是 TopoGS post,要 load 进 gaussians 再调
    if args.use_hac:
        # 先把 feat 和 level 灌进 gaussians
        gaussians._anchor_feat = torch.from_numpy(feat).float().cuda()
        gaussians._level = torch.from_numpy(level).long().unsqueeze(-1).cuda()
        # 调 _apply_cross_level_context
        try:
            from gaussian_renderer import _apply_cross_level_context
            with torch.no_grad():
                feat_t = gaussians._anchor_feat
                lvl_t  = gaussians._level
                feat_aug = _apply_cross_level_context(gaussians, feat_t, lvl_t)
            feat = feat_aug.cpu().numpy()
            print('[HAC] applied delta_l')
        except Exception as e:
            print(f'[HAC] apply failed: {e}, using raw feat')

    # === 计算 per-level cosine matrix ===
    unique_levels = np.sort(np.unique(level))
    L = len(unique_levels)
    print(f'\nFound {L} levels: {unique_levels.tolist()}')

    means, counts = [], []
    for lv in unique_levels:
        m = (level == lv)
        cnt = int(m.sum())
        counts.append(cnt)
        means.append(feat[m].mean(axis=0) if cnt > 0 else np.zeros(feat.shape[1]))
    M = np.stack(means)
    print(f'Anchors per level: {counts}')

    M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    sim = M_norm @ M_norm.T

    print(f'\nCosine matrix ({L}x{L}):')
    np.set_printoptions(precision=3, suppress=True)
    print(sim)
    off = sim[~np.eye(L, dtype=bool)]
    print(f'\nOff-diagonal: mean={off.mean():.4f}, min={off.min():.4f}, max={off.max():.4f}')

    np.save(args.dump_out, {
        'sim': sim, 'levels': unique_levels.tolist(),
        'counts': counts, 'use_hac': args.use_hac,
    }, allow_pickle=True)
    print(f'\nSaved {args.dump_out}')


if __name__ == '__main__':
    main()