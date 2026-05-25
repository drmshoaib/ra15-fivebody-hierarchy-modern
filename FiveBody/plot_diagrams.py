"""
Five-body hierarchical stability analyser — diagram generator.

Produces four PNG figures saved alongside this script:
  body_layout.png       C2-symmetric five-body configuration
  hierarchy_states.png  Schematic of the four named hierarchy states
  transition_graph.png  Directed graph of all 12 hierarchy transitions
  ic_velocities.png     Initial y-velocities vs A for the sample IC sweep

Run from the FiveBody/ directory:
    python plot_diagrams.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})

# ── helpers ───────────────────────────────────────────────────────────────────

def calc_U(A, B, m0, m1, m4):
    pos = np.array([[A, 0], [B, 0], [-B, 0], [-A, 0], [0.0, 0.0]])
    M   = [m0, m1, m1, m0, m4]
    U = 0.0
    for i in range(5):
        for j in range(i + 1, 5):
            U += M[i] * M[j] / np.linalg.norm(pos[i] - pos[j])
    return U

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — C2-symmetric body layout
# ─────────────────────────────────────────────────────────────────────────────
A_diag, B_diag = 0.9, 0.32

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_aspect('equal')
ax.set_xlim(-1.35, 1.35)
ax.set_ylim(-0.80, 0.80)
ax.axhline(0, color='#cccccc', lw=0.8, zorder=0)
ax.axvline(0, color='#cccccc', lw=0.8, zorder=0)

body_pos = {
    0: ( A_diag,  0.0, 'tab:blue'),
    1: ( B_diag,  0.0, 'tab:red'),
    2: (-B_diag,  0.0, 'tab:red'),
    3: (-A_diag,  0.0, 'tab:blue'),
    4: (    0.0,  0.0, '#222222'),
}
body_labels = {
    0: r'$\mathbf{0}$',
    1: r'$\mathbf{1}$',
    2: r'$\mathbf{2}$',
    3: r'$\mathbf{3}$',
    4: r'$\mathbf{4}$',
}
body_sizes = {0: 110, 1: 110, 2: 110, 3: 110, 4: 160}

for n, (x, y, col) in body_pos.items():
    ax.scatter(x, y, s=body_sizes[n], color=col, zorder=5, linewidths=0.5,
               edgecolors='white')
    va_off = 0.12 if n != 4 else -0.15
    ax.text(x, y + va_off, body_labels[n], ha='center', va='bottom',
            fontsize=10, color=col, fontweight='bold')

# Velocity arrows (illustrative magnitudes)
vy0_diag, vy1_diag = 0.32, 0.42
arrow_kw = dict(arrowstyle='->', lw=1.8)
for body, x, vy, col, sign in [
    (0,  A_diag, vy0_diag, 'tab:blue',  1),
    (1,  B_diag, vy1_diag, 'tab:red',   1),
    (2, -B_diag, vy1_diag, 'tab:red',  -1),
    (3, -A_diag, vy0_diag, 'tab:blue', -1),
]:
    ax.annotate('', xy=(x, sign * vy * 0.55), xytext=(x, 0.0),
                arrowprops=dict(**arrow_kw, color=col))

ax.text(A_diag + 0.07,  vy0_diag * 0.28, r'$\dot{y}_0$', fontsize=10, color='tab:blue')
ax.text(B_diag + 0.07,  vy1_diag * 0.28, r'$\dot{y}_1$', fontsize=10, color='tab:red')
ax.text(-B_diag - 0.07, -vy1_diag * 0.28, r'$-\dot{y}_1$', fontsize=9,
        color='tab:red', ha='right')
ax.text(-A_diag - 0.07, -vy0_diag * 0.28, r'$-\dot{y}_0$', fontsize=9,
        color='tab:blue', ha='right')

# C2 rotation arc
theta = np.linspace(0.18, np.pi - 0.18, 120)
r_arc = 1.22
ax.plot(r_arc * np.cos(theta), r_arc * np.sin(theta), 'k--', lw=1.0, alpha=0.35)
ax.annotate('', xy=(-r_arc + 0.01, 0.18), xytext=(-r_arc + 0.01, -0.18),
            arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.0))
ax.text(-1.32, 0.0, r'C$_2$', fontsize=11, va='center',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#888888', alpha=0.9))

# Distance annotations
for (xfrom, xto, yline, lbl) in [
    (0.0, A_diag, -0.32, r'$A$'),
    (0.0, B_diag, -0.50, r'$B$'),
]:
    ax.annotate('', xy=(xto, yline), xytext=(xfrom, yline),
                arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.0))
    ax.text(xto / 2.0, yline - 0.08, lbl, ha='center', fontsize=11, color='#333333')

# Legend
patches = [
    mpatches.Patch(color='tab:blue', label=r'Bodies 0 & 3 — mass $m_0$'),
    mpatches.Patch(color='tab:red',  label=r'Bodies 1 & 2 — mass $m_1$'),
    mpatches.Patch(color='#222222',  label=r'Body 4 (centre) — mass $m_4$'),
]
ax.legend(handles=patches, loc='upper right', fontsize=9, framealpha=0.9)

ax.set_title(r'C$_2$-Symmetric Five-Body Initial Configuration', fontsize=13)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_xticks([-A_diag, -B_diag, 0, B_diag, A_diag])
ax.set_xticklabels([r'$-A$', r'$-B$', r'$0$', r'$+B$', r'$+A$'])
ax.set_yticks([])

fig.tight_layout()
fig.savefig('body_layout.png', bbox_inches='tight')
plt.close(fig)
print('body_layout.png  written')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Four hierarchy states (2×2 schematic)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Named Hierarchy States', fontsize=14, y=0.99)

# Each entry: title, description, body positions (after orbital evolution),
# body colours, and which pair forms the "inner" bond (drawn in gold).
states = [
    dict(
        title='H12 — outer binary: bodies 0 & 1',
        desc=(r'Both outer ($r_1>b_1$, $r_2>b_2$), bodies 0 & 1'
              '\ncloser to each other than 0 & 2'),
        pos={0: (0.80, 0.35), 1: (0.45, 0.15),
             2: (-0.45, -0.15), 3: (-0.80, -0.35), 4: (0, 0)},
        bond=(0, 1),
    ),
    dict(
        title='H13 — outer binary: bodies 0 & 2',
        desc=(r'Both outer, bodies 0 & 2'
              '\ncloser to each other than 0 & 1'),
        pos={0: (0.80, 0.30), 1: (0.55, -0.50),
             2: (-0.55, 0.50), 3: (-0.80, -0.30), 4: (0, 0)},
        bond=(0, 2),
    ),
    dict(
        title='H14 — inner: body 0 near centre',
        desc=(r'$r_1 < b_4$ (body 0 innermost),'
              '\n$r_2 > b_3$ (body 1 far outer)'),
        pos={0: (0.18, 0.08), 1: (0.90, 0.45),
             2: (-0.90, -0.45), 3: (-0.18, -0.08), 4: (0, 0)},
        bond=(0, 4),
    ),
    dict(
        title='H23 — inner: body 1 near centre',
        desc=(r'$r_2 < b_2$ (body 1 innermost),'
              '\n$r_1 > b_1$ (body 0 far outer)'),
        pos={0: (0.90, 0.45), 1: (0.16, 0.07),
             2: (-0.16, -0.07), 3: (-0.90, -0.45), 4: (0, 0)},
        bond=(1, 4),
    ),
]

body_colors = {0: 'tab:blue', 1: 'tab:red', 2: 'tab:red',
               3: 'tab:blue', 4: '#222222'}

for ax, sd in zip(axes.flat, states):
    ax.set_aspect('equal')
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.85, 0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axhline(0, color='#dddddd', lw=0.6, zorder=0)
    ax.axvline(0, color='#dddddd', lw=0.6, zorder=0)

    pos = sd['pos']

    # C2 symmetry lines (faint dashes between pairs)
    for (a, b) in [(0, 3), (1, 2)]:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                'k--', lw=0.7, alpha=0.20, zorder=1)

    # Inner-bond highlight
    b0, b1 = sd['bond']
    ax.plot([pos[b0][0], pos[b1][0]], [pos[b0][1], pos[b1][1]],
            color='gold', lw=4, alpha=0.75, solid_capstyle='round', zorder=2)

    # Bodies
    for n in range(5):
        x, y = pos[n]
        sz = 180 if n == 4 else 110
        ax.scatter(x, y, s=sz, color=body_colors[n], zorder=5,
                   edgecolors='white', linewidths=0.5)
        offset = 0.13 if y >= 0 else -0.16
        ax.text(x, y + offset, str(n), ha='center', va='bottom',
                fontsize=9, color=body_colors[n], fontweight='bold')

    ax.set_title(sd['title'], fontsize=10, pad=6)
    ax.text(0.5, 0.03, sd['desc'], transform=ax.transAxes,
            ha='center', va='bottom', fontsize=8.5, color='#444444',
            linespacing=1.5)

# Shared legend
patches = [
    mpatches.Patch(color='tab:blue', label='Bodies 0 & 3'),
    mpatches.Patch(color='tab:red',  label='Bodies 1 & 2'),
    mpatches.Patch(color='#222222',  label='Body 4 (centre)'),
    mpatches.Patch(color='gold',     label='Inner bonded pair'),
]
fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, 0.01), framealpha=0.9)

fig.tight_layout(rect=[0, 0.06, 1, 0.98])
fig.savefig('hierarchy_states.png', bbox_inches='tight')
plt.close(fig)
print('hierarchy_states.png  written')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Hierarchy-change transition graph
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect('equal')
ax.set_xlim(-1.7, 1.7)
ax.set_ylim(-1.7, 1.7)
ax.axis('off')
ax.set_title(
    'Hierarchy-Change Transition Graph\n'
    '12 directed transitions; numbers = hc[ ] index',
    fontsize=12, pad=10)

# Node layout — diamond
nodes = {'H12': (0.0, 1.15), 'H13': (1.15, 0.0),
         'H14': (0.0, -1.15), 'H23': (-1.15, 0.0)}
node_col = {'H12': '#4c72b0', 'H13': '#dd8452',
            'H14': '#55a868', 'H23': '#c44e52'}
node_r = 0.27

for name, (nx, ny) in nodes.items():
    c = plt.Circle((nx, ny), node_r, color=node_col[name], zorder=5)
    ax.add_patch(c)
    ax.text(nx, ny, name, ha='center', va='center', color='white',
            fontsize=12, fontweight='bold', zorder=6)

# 12 directed transitions (from, to, hc_index)
transitions = [
    ('H12', 'H13',  0), ('H12', 'H14',  1), ('H12', 'H23',  2),
    ('H13', 'H12',  3), ('H13', 'H14',  4), ('H13', 'H23',  5),
    ('H14', 'H12',  6), ('H14', 'H13',  7), ('H14', 'H23',  8),
    ('H23', 'H12',  9), ('H23', 'H13', 10), ('H23', 'H14', 11),
]

# Group paired transitions so opposite arrows are offset symmetrically
pairs = {}  # (nodeA, nodeB) -> list of (src, dst, idx)
for src, dst, idx in transitions:
    key = tuple(sorted([src, dst]))
    pairs.setdefault(key, []).append((src, dst, idx))

for (na, nb), pair_list in pairs.items():
    x0, y0 = nodes[na]
    x1, y1 = nodes[nb]
    dx, dy = x1 - x0, y1 - y0
    dist = np.hypot(dx, dy)
    ux, uy = dx / dist, dy / dist
    perp = np.array([-uy, ux])

    for k, (src, dst, idx) in enumerate(pair_list):
        sign = 1 if k == 0 else -1
        off = perp * 0.09 * sign
        sx = nodes[src][0] + ux * node_r + off[0]
        sy = nodes[src][1] + uy * node_r + off[1]
        ex = nodes[dst][0] - ux * node_r + off[0]
        ey = nodes[dst][1] - uy * node_r + off[1]

        col = node_col[src]
        rad = 0.18 * sign
        ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(
                        arrowstyle='->', color=col, lw=1.6,
                        connectionstyle=f'arc3,rad={rad}'))

        # Label at midpoint of the arc (approximate)
        mx = (sx + ex) / 2.0 + off[0] * 0.6
        my = (sy + ey) / 2.0 + off[1] * 0.6
        ax.text(mx, my, str(idx), ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.18', fc='white',
                          ec=col, alpha=0.92, lw=0.8))

fig.tight_layout()
fig.savefig('transition_graph.png', bbox_inches='tight')
plt.close(fig)
print('transition_graph.png  written')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Initial y-velocities across the sample A-sweep
# ─────────────────────────────────────────────────────────────────────────────
E0   = -0.07
B_sw = 0.25
m0sw = m1sw = m4sw = 1.0
C0sw = 0.02

A_arr  = np.linspace(0.35, 1.45, 300)
vy0_c  = np.full_like(A_arr, np.nan)
vy1_c  = np.full_like(A_arr, np.nan)

for i, A in enumerate(A_arr):
    c    = np.sqrt(C0sw / abs(E0))
    U    = calc_U(A, B_sw, m0sw, m1sw, m4sw)
    disc = -c**2 + 4.0 * (A**2 * m0sw + B_sw**2 * m1sw) * (E0 + U)
    if disc < 0.0:
        continue
    sq = np.sqrt(m0sw * m1sw * disc)
    vy0_c[i] = (A * c * m0sw - B_sw * sq) / (2 * m0sw**2 * A**2 + 2 * m1sw * m0sw * B_sw**2)
    vy1_c[i] = (B_sw * c * m1sw + A * sq) / (2 * m0sw * m1sw * A**2 + 2 * m1sw**2 * B_sw**2)

# Sample IC points
sample_A   = np.arange(0.40, 1.36, 0.05)
svy0, svy1 = [], []
for A in sample_A:
    c    = np.sqrt(C0sw / abs(E0))
    U    = calc_U(A, B_sw, m0sw, m1sw, m4sw)
    disc = -c**2 + 4.0 * (A**2 * m0sw + B_sw**2 * m1sw) * (E0 + U)
    if disc < 0.0:
        svy0.append(np.nan); svy1.append(np.nan); continue
    sq = np.sqrt(m0sw * m1sw * disc)
    svy0.append((A * c * m0sw - B_sw * sq) / (2 * m0sw**2 * A**2 + 2 * m1sw * m0sw * B_sw**2))
    svy1.append((B_sw * c * m1sw + A * sq) / (2 * m0sw * m1sw * A**2 + 2 * m1sw**2 * B_sw**2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
fig.suptitle(
    r'Sample IC sweep — $B=0.25$, $m_0=m_1=m_4=1$, $C_0=0.02$, $E_0=-0.07$',
    fontsize=11)

# Left: vy0
ax = axes[0]
ax.plot(A_arr, vy0_c,  color='tab:blue', lw=2.0, label=r'$\dot{y}_0$ (body 0)')
ax.plot(A_arr, -vy0_c, color='tab:blue', lw=1.5, ls='--', alpha=0.55,
        label=r'$\dot{y}_3 = -\dot{y}_0$ (body 3)')
ax.scatter(sample_A, svy0,  s=40, color='tab:blue',  zorder=5, label='sample ICs')
ax.scatter(sample_A, [-v for v in svy0], s=25, color='tab:blue',
           marker='D', zorder=5, alpha=0.6)
ax.axhline(0, color='k', lw=0.6)
ax.set_xlabel(r'$A$  (x-position of body 0)')
ax.set_ylabel(r'initial $y$-velocity')
ax.set_title(r'Bodies 0 & 3')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

# Right: vy1
ax = axes[1]
ax.plot(A_arr, vy1_c,  color='tab:red',  lw=2.0, label=r'$\dot{y}_1$ (body 1)')
ax.plot(A_arr, -vy1_c, color='tab:red',  lw=1.5, ls='--', alpha=0.55,
        label=r'$\dot{y}_2 = -\dot{y}_1$ (body 2)')
ax.scatter(sample_A, svy1,  s=40, color='tab:red',   zorder=5, label='sample ICs')
ax.scatter(sample_A, [-v for v in svy1], s=25, color='tab:red',
           marker='D', zorder=5, alpha=0.6)
ax.axhline(0, color='k', lw=0.6)
ax.set_xlabel(r'$A$  (x-position of body 0)')
ax.set_ylabel(r'initial $y$-velocity')
ax.set_title(r'Bodies 1 & 2')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig('ic_velocities.png', bbox_inches='tight')
plt.close(fig)
print('ic_velocities.png  written')

print('\nAll figures written to the current directory.')
