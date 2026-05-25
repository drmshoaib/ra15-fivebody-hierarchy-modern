"""
Five-body hierarchical stability analyser — phase portraits by hierarchy state.

Produces two PNG figures:
  phase_portraits.png        x-y phase space of bodies 0 & 1, coloured by
                             hierarchy state, for four representative ICs
  hierarchy_trajectories.png (r1, r2) trajectories overlaid on the
                             classification region map

Run from the FiveBody/ directory:
    python plot_phase_portraits.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 150,
})

# ── Physical constants ────────────────────────────────────────────────────────
E0   = -0.07
B_P  =  0.25
C0   =  0.02
C    = np.sqrt(C0 / abs(E0))        # angular momentum ≈ 0.5345
B1   = 0.35  / abs(E0)              # ≈ 5.000
B2   = 0.122 / abs(E0)              # ≈ 1.743
B3, B4 = B1, B2

# State colour palette
STATE_VALS   = [-1, 12, 13, 14, 23]
STATE_COLORS = {-1: '#bbbbbb', 12: '#4c72b0', 13: '#dd8452',
                 14: '#55a868', 23: '#c44e52'}
STATE_LABELS = {-1: 'Undetermined', 12: 'H12', 13: 'H13',
                 14: 'H14', 23: 'H23'}

# ── Equal-mass potential (C2 layout, x-axis only) ─────────────────────────────
def U_xaxis(A, B):
    dAB = np.abs(A - B)
    safe = np.where(dAB < 1e-12, np.inf, dAB)
    return 2.0/safe + 2.0/(A+B) + 2.5/A + 2.5/B

# ── IC setup ─────────────────────────────────────────────────────────────────
def setup_ic(A, B=B_P):
    U    = U_xaxis(A, B)
    disc = -C**2 + 4.0*(A**2 + B**2)*(E0 + U)
    if disc < 0.0:
        return None
    sq    = np.sqrt(disc)
    denom = 2.0*(A**2 + B**2)
    return np.array([A, 0., B, 0., 0., (A*C - B*sq)/denom, 0., (B*C + A*sq)/denom])

# ── C2-reduced equations of motion (equal masses) ────────────────────────────
def odes(s):
    x0,y0,x1,y1,vx0,vy0,vx1,vy1 = s
    r0 = np.array([x0,y0]); r1 = np.array([x1,y1])
    d   = r1 - r0;   sm = r1 + r0
    n0  = np.hypot(x0,y0); n1 = np.hypot(x1,y1)
    nd  = np.linalg.norm(d); ns = np.linalg.norm(sm)
    if min(n0, n1, nd, ns) < 1e-8:
        return np.zeros(8)
    F0 =  d/nd**3 - sm/ns**3 - 5*r0/(4*n0**3)
    F1 = -d/nd**3 - sm/ns**3 - 5*r1/(4*n1**3)
    return np.array([vx0,vy0,vx1,vy1, F0[0],F0[1], F1[0],F1[1]])

# ── RK4 integrator ────────────────────────────────────────────────────────────
def integrate(s0, T, dt=0.003, stride=3):
    n_steps = int(T / dt)
    n_out   = n_steps // stride + 1
    t_out   = np.empty(n_out)
    y_out   = np.empty((8, n_out))
    s = np.array(s0, dtype=float)
    t_out[0] = 0.0; y_out[:,0] = s
    k = 0
    for step in range(n_steps):
        k1 = odes(s)
        k2 = odes(s + 0.5*dt*k1)
        k3 = odes(s + 0.5*dt*k2)
        k4 = odes(s +     dt*k3)
        s  = s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        if (step+1) % stride == 0:
            k += 1
            if k < n_out:
                t_out[k] = (step+1)*dt
                y_out[:,k] = s
    return t_out[:k+1], y_out[:,:k+1]

# ── Hierarchy classifier (exact match of main.cpp identifyHierarchy) ──────────
def classify_vec(x0, y0, x1, y1):
    """Vectorised: returns integer array of hierarchy states."""
    r1  = np.hypot(x0, y0)
    r2  = np.hypot(x1, y1)
    r12 = np.hypot(x0 - x1, y0 - y1)
    r13 = np.hypot(x0 + x1, y0 + y1)   # |r0 − r2| since r2 = −r1

    state = np.full(len(x0), -1, dtype=int)

    m23   = (r2 < B2) & (r1 > B1)
    m14   = (r1 < B4) & (r2 > B3)
    mouter = ((r1 > B1) & (r2 > B2)) | ((r1 > B4) & (r2 > B3))
    mouter &= ~m23 & ~m14

    state[m23]  = 23
    state[m14]  = 14
    state[mouter & (r12 < r13)] = 12
    state[mouter & (r12 >= r13)] = 13
    return state

# ── Four representative initial conditions ───────────────────────────────────
# IC1: body 0 starts inside b4 — H14 appears when body 1 travels far out
# IC2: starts in H23 (body 0 at r > b1, body 1 at r < b2)
# IC3: starts in outer zone (both bodies beyond thresholds) → H12 ↔ H13
# IC4: larger-amplitude sample IC with transition dynamics

ic_configs = [
    dict(A=1.30, B=0.25, T=50, dt=0.004, stride=4,
         title=r'IC1: $A=1.30$, $B=0.25$ — H14 excursions'),
    dict(A=5.50, B=0.50, T=25, dt=0.003, stride=3,
         title=r'IC2: $A=5.50$, $B=0.50$ — H23 → outer'),
    dict(A=5.50, B=2.50, T=30, dt=0.003, stride=3,
         title=r'IC3: $A=5.50$, $B=2.50$ — H12 ↔ H13'),
    dict(A=3.00, B=0.25, T=50, dt=0.004, stride=4,
         title=r'IC4: $A=3.00$, $B=0.25$ — mixed transitions'),
]

# Integrate and classify all ICs
datasets = []
for cfg in ic_configs:
    ic = setup_ic(cfg['A'], cfg['B'])
    if ic is None:
        print(f"  IC A={cfg['A']}, B={cfg['B']} inaccessible — skipping")
        datasets.append(None); continue
    t, Y = integrate(ic, cfg['T'], dt=cfg['dt'], stride=cfg['stride'])
    states = classify_vec(Y[0], Y[1], Y[2], Y[3])
    state_counts = {v: int(np.sum(states == v)) for v in STATE_VALS}
    print(f"  IC A={cfg['A']:4.2f} B={cfg['B']:4.2f} T={cfg['T']:3d}: "
          + "  ".join(f"{STATE_LABELS[v]}={state_counts[v]}" for v in STATE_VALS))
    datasets.append(dict(t=t, Y=Y, states=states, cfg=cfg))

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Phase portraits coloured by hierarchy state
# Layout: 4 rows (one per IC) × 4 columns
#   col 0: (x0, vx0)   col 1: (y0, vy0)
#   col 2: (x1, vx1)   col 3: (y1, vy1)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(15, 14))
fig.suptitle('Phase Portraits by Hierarchy State\n'
             r'($m_0=m_1=m_4=1$, $C_0=0.02$, $E_0=-0.07$)',
             fontsize=13, y=0.995)

col_keys  = [(0, 4, r'$x_0$', r'$\dot{x}_0$'),
             (1, 5, r'$y_0$', r'$\dot{y}_0$'),
             (2, 6, r'$x_1$', r'$\dot{x}_1$'),
             (3, 7, r'$y_1$', r'$\dot{y}_1$')]

for row, ds in enumerate(datasets):
    for col, (xi, vi, xl, yl) in enumerate(col_keys):
        ax = axes[row, col]
        ax.set_xlabel(xl, labelpad=1)
        ax.set_ylabel(yl, labelpad=1)
        ax.tick_params(labelsize=8)

        if ds is None:
            ax.text(0.5, 0.5, 'inaccessible IC', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
            continue

        Y, states = ds['Y'], ds['states']
        xv = Y[xi]; vv = Y[vi]

        # Plot each state as a separate scatter layer (no alpha blending issues)
        for sv in STATE_VALS:
            mask = states == sv
            if not np.any(mask):
                continue
            ax.scatter(xv[mask], vv[mask], s=1.5,
                       color=STATE_COLORS[sv], linewidths=0,
                       rasterized=True, zorder=STATE_VALS.index(sv)+1)

        # Mark initial point
        ax.scatter(xv[0], vv[0], s=25, color='black', marker='o',
                   zorder=10, linewidths=0.5, edgecolors='white')

        if col == 0:
            ax.set_title(ds['cfg']['title'], fontsize=9, pad=3)
        ax.grid(True, alpha=0.15, lw=0.4)

# Shared legend at bottom
legend_handles = [
    mpatches.Patch(color=STATE_COLORS[sv], label=STATE_LABELS[sv])
    for sv in STATE_VALS
] + [Line2D([0],[0], marker='o', color='w', markerfacecolor='black',
            markersize=6, label='IC start')]
fig.legend(handles=legend_handles, loc='lower center', ncol=6,
           fontsize=10, bbox_to_anchor=(0.5, 0.003), framealpha=0.95,
           markerscale=2)

fig.tight_layout(rect=[0, 0.04, 1, 0.995])
fig.savefig('phase_portraits.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('phase_portraits.png  written')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — (r1, r2) trajectories overlaid on the hierarchy region map
# ─────────────────────────────────────────────────────────────────────────────
r_max = 9.0
N     = 600
r_arr = np.linspace(0, r_max, N)
R1g, R2g = np.meshgrid(r_arr, r_arr)

Zg  = np.zeros_like(R1g, dtype=int)
m23 = (R1g > B1) & (R2g < B2)
m14 = (R1g < B4) & (R2g > B3)
mou = ((R1g > B1) & (R2g > B2)) | ((R1g > B4) & (R2g > B3))
mou &= ~m23 & ~m14
Zg[m23] = 1; Zg[m14] = 2; Zg[mou] = 3

cmap_bg  = ListedColormap(['#e8e8e8', '#d0e4f0', '#fce0d0', '#d4eed4'])
norm_bg  = BoundaryNorm([-0.5,0.5,1.5,2.5,3.5], cmap_bg.N)

fig, axes2 = plt.subplots(2, 2, figsize=(12, 11))
fig.suptitle(r'$(r_1, r_2)$ Trajectories Coloured by Hierarchy State', fontsize=13)

for ax, ds in zip(axes2.flat, datasets):
    # Background region map
    ax.pcolormesh(R1g, R2g, Zg, cmap=cmap_bg, norm=norm_bg,
                  shading='auto', rasterized=True, alpha=0.7)

    # Boundary lines
    for b in [B1, B2]:
        ax.axvline(b, color='#555555', lw=0.9, ls='--', alpha=0.5)
        ax.axhline(b, color='#555555', lw=0.9, ls='--', alpha=0.5)

    # Region labels (small, in corners)
    kw = dict(fontsize=8, fontweight='bold', alpha=0.75)
    ax.text(r_max*0.97, B2*0.5, 'H23', ha='right', color='#1a5276', **kw)
    ax.text(B4*0.5, r_max*0.97, 'H14', ha='center', va='top',
            color='#922b21', **kw)
    ax.text(r_max*0.97, r_max*0.97, 'outer\n(H12/H13)', ha='right', va='top',
            color='#1e8449', **kw)
    ax.text((B1+B2)/2, (B1+B2)/2, 'undef.', ha='center',
            color='#555555', **kw)

    if ds is None:
        ax.set_title('inaccessible IC', fontsize=10)
        ax.set_xlim(0, r_max); ax.set_ylim(0, r_max)
        continue

    Y, states = ds['Y'], ds['states']
    r1_t = np.hypot(Y[0], Y[1])
    r2_t = np.hypot(Y[2], Y[3])

    # Thin the data for faster rendering while keeping state boundaries sharp
    step = max(1, len(r1_t) // 3000)
    idx  = np.arange(0, len(r1_t), step)

    for sv in STATE_VALS:
        mask = states[idx] == sv
        if not np.any(mask):
            continue
        ax.scatter(r1_t[idx[mask]], r2_t[idx[mask]],
                   s=2.5, color=STATE_COLORS[sv], linewidths=0,
                   rasterized=True, zorder=4)

    # Mark start
    ax.scatter(r1_t[0], r2_t[0], s=50, color='black', marker='*', zorder=10)

    ax.set_xlim(0, r_max); ax.set_ylim(0, r_max)
    ax.set_xlabel(r'$r_1 = |\mathbf{r}_0|$', labelpad=2)
    ax.set_ylabel(r'$r_2 = |\mathbf{r}_1|$', labelpad=2)
    ax.set_title(ds['cfg']['title'], fontsize=10)
    ax.set_aspect('equal')

    # Boundary labels on axes
    ax.set_xticks([0, B2, B1, r_max])
    ax.set_xticklabels(['0', fr'$b_2$\n{B2:.2f}', fr'$b_1$\n{B1:.2f}', ''])
    ax.set_yticks([0, B2, B1, r_max])
    ax.set_yticklabels(['0', f'{B2:.2f}', f'{B1:.2f}', ''])
    ax.tick_params(labelsize=8)

# Shared legend
legend_handles2 = [
    mpatches.Patch(color=STATE_COLORS[sv], label=STATE_LABELS[sv])
    for sv in STATE_VALS
] + [Line2D([0],[0], marker='*', color='w', markerfacecolor='black',
            markersize=9, label='IC start')]
fig.legend(handles=legend_handles2, loc='lower center', ncol=6, fontsize=10,
           bbox_to_anchor=(0.5, 0.005), framealpha=0.95, markerscale=1.5)

fig.tight_layout(rect=[0, 0.045, 1, 0.995])
fig.savefig('hierarchy_trajectories.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('hierarchy_trajectories.png  written')

print('\nAll figures written.')
