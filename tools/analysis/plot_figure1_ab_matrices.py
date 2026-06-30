# plot_figure1_ab_matrices.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'mathtext.fontset': 'cm',
})

# Load 真实数据
data_a = np.load('sim_citygs.npy', allow_pickle=True).item()
data_b = np.load('sim_topogs_pre.npy', allow_pickle=True).item()

sim_a = data_a['sim']
sim_b = data_b['sim']
levels = data_a['levels']
L = len(levels)
level_labels = [f'L{i}' for i in levels]

# off-diag stats
off_a = sim_a[~np.eye(L, dtype=bool)]
off_b = sim_b[~np.eye(L, dtype=bool)]
print(f'(a) CityGS-X off-diag mean: {off_a.mean():.3f}')
print(f'(b) TopoGS   off-diag mean: {off_b.mean():.3f}')

# Custom colormap: 蓝(neg) -> 白(0) -> 红(pos)
# 用 diverging,因为 TopoGS 有负值
cmap = plt.cm.RdBu_r

# 共享色标
vmin, vmax = -1.0, 1.0

fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0),
                          gridspec_kw={'wspace': 0.18})

for idx, (ax, sim, title) in enumerate(zip(
        axes, [sim_a, sim_b],
        ['(a) CityGS-X', '(b) TopoGS (Ours)'])):

    im = ax.imshow(sim, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

    for i in range(L):
        for j in range(L):
            v = sim[i, j]
            # 深背景白字, 浅背景黑字
            txt_color = 'white' if abs(v) > 0.6 else '#1a1a1a'
            ax.text(j, i, f'{v:.2f}',
                    ha='center', va='center',
                    fontsize=7.0, color=txt_color)

    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.set_xticklabels(level_labels, fontsize=8.5)
    ax.set_yticklabels(level_labels, fontsize=8.5)
    ax.set_title(title, fontsize=11.5, fontweight='bold', pad=8)
    ax.tick_params(length=0)

    # off-diag mean 写在子图下方
    off = sim[~np.eye(L, dtype=bool)]
    ax.text(0.5, -0.15,
            f'off-diag mean = {off.mean():.3f}',
            transform=ax.transAxes,
            ha='center', va='top', fontsize=9.5,
            color='#444', style='italic')

    for spine in ax.spines.values():
        spine.set_edgecolor('#888')
        spine.set_linewidth(0.8)

# 共享 colorbar
cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02,
                    aspect=18, fraction=0.04)
cbar.set_label('Cosine similarity', fontsize=10)
cbar.ax.tick_params(labelsize=8.5)

# 顶部小注
fig.text(0.5, 1.01,
         'Cross-level mean anchor feature similarity (L0 = coarsest, L7 = finest)',
         ha='center', va='bottom', fontsize=10, color='#444')

plt.savefig('figure1_matrices.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('figure1_matrices.pdf', bbox_inches='tight', facecolor='white')
print('Saved figure1_matrices.png / .pdf')