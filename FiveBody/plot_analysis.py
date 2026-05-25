"""
Five-body hierarchical stability analyser — additional analytical diagrams.

Produces four PNG figures saved alongside this script:
  sample_orbit.png       Orbit trajectories in the x-y plane (3 ICs)
  hierarchy_regions.png  Hierarchy classification map in (r1, r2) space
  accessible_ics.png     Discriminant map in (A, B) parameter space
  energy_momentum.png    Energy and angular momentum conservation

Run from the FiveBody/ directory:
    python plot_analysis.py
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
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})

# ── Physical constants (must match main.cpp) ──────────────────────────────────
E0   = -0.07
B_P  =  0.25      # fixed B for the sample sweep
C0   =  0.02
C    = np.sqrt(C0 / abs(E0))    # angular momentum ≈ 0.5345

# Boundary thresholds: read from b_default.txt (0.35 0.122 0.35 0.122)
# scaled by 1/|E0| inside the code
B1 = 0.35  / abs(E0)   # ≈ 5.000
B2 = 0.122 / abs(E0)   # ≈ 1.743
B3, B4 = B1, B2

# ── Equal-mass potential on the x-axis (vectorised) ──────────────────────────
# U = 2/|A-B| + 2/(A+B) + 5/(2A) + 5/(2B)   [m0=m1=m4=1, G=1]
def U_xaxis(A, B):
    dAB = np.abs(A - B)
    safe = np.where(dAB < 1e-12, np.inf, dAB)
    return 2.0/safe + 2.0/(A+B) + 2.5/A + 2.5/B

# ── Potential from general 4-dof state (body 4 at origin, C2 symmetry) ───────
def U_state(x0, y0, x1, y1):
    pos = np.array([[x0,y0],[x1,y1],[-x1,-y1],[-x0,-y0],[0.,0.]])
    U = 0.0
    for i in range(5):
        for j in range(i+1, 5):
            r = np.hypot(pos[i,0]-pos[j,0], pos[i,1]-pos[j,1])
            if r > 1e-15:
                U += 1.0 / r
    return U

# ── IC setup (all bodies on x-axis, velocities only in y) ─────────────────────
def setup_ic(A, B=B_P):
    U    = U_xaxis(A, B)
    disc = -C**2 + 4.0*(A**2 + B**2)*(E0 + U)
    if disc < 0.0:
        return None
    sq    = np.sqrt(disc)
    denom = 2.0*(A**2 + B**2)
    vy0   = (A*C - B*sq) / denom
    vy1   = (B*C + A*sq) / denom
    return np.array([A, 0., B, 0., 0., vy0, 0., vy1])

# ── C2-reduced equations of motion (equal masses m0=m1=m4=1) ─────────────────
#   F0 = (r1-r0)/|r1-r0|^3  -  (r1+r0)/|r1+r0|^3  -  5*r0/(4*|r0|^3)
#   F1 = (r0-r1)/|r0-r1|^3  -  (r0+r1)/|r0+r1|^3  -  5*r1/(4*|r1|^3)
def odes(s):
    x0,y0,x1,y1,vx0,vy0,vx1,vy1 = s
    r0 = np.array([x0,y0]); r1 = np.array([x1,y1])
    d   = r1 - r0;   sm = r1 + r0
    n0  = np.hypot(x0,y0); n1 = np.hypot(x1,y1)
    nd  = np.linalg.norm(d); ns = np.linalg.norm(sm)
    if min(n0, n1, nd, ns) < 1e-8:
        return np.array([vx0,vy0,vx1,vy1,0.,0.,0.,0.])
    F0 = d/nd**3  - sm/ns**3 - 5*r0/(4*n0**3)
    F1 = -d/nd**3 - sm/ns**3 - 5*r1/(4*n1**3)
    return np.array([vx0,vy0,vx1,vy1, F0[0],F0[1], F1[0],F1[1]])

# ── Fourth-order Runge-Kutta integrator ───────────────────────────────────────
def integrate(s0, T, dt=0.002, stride=1):
    """Integrate using RK4; return (t_array, state_array shape (8,n))."""
    n_steps = int(T / dt)
    n_out   = n_steps // stride + 1
    t_out   = np.empty(n_out)
    y_out   = np.empty((8, n_out))
    s = np.array(s0, dtype=float)
    t_out[0] = 0.0; y_out[:,0] = s
    out_idx = 1
    for k in range(n_steps):
        k1 = odes(s)
        k2 = odes(s + 0.5*dt*k1)
        k3 = odes(s + 0.5*dt*k2)
        k4 = odes(s +     dt*k3)
        s  = s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        if (k+1) % stride == 0 and out_idx < n_out:
            t_out[out_idx] = (k+1)*dt
            y_out[:,out_idx] = s
            out_idx += 1
    return t_out[:out_idx], y_out[:,:out_idx]

# ── Energy and angular momentum from 4-dof state ─────────────────────────────
def energy(s):
    x0,y0,x1,y1,vx0,vy0,vx1,vy1 = s
    # v3=-v0, v2=-v1, v4=0  →  K = 2*(0.5*v0^2) + 2*(0.5*v1^2) = v0^2+v1^2
    K = vx0**2 + vy0**2 + vx1**2 + vy1**2
    return K - U_state(x0, y0, x1, y1)

def angmom(s):
    x0,y0,x1,y1,vx0,vy0,vx1,vy1 = s
    # L = m*(x*vy-y*vx) summed over all bodies; bodies 2,3 mirror 1,0
    return 2*(x0*vy0 - y0*vx0 + x1*vy1 - y1*vx1)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Sample orbit trajectories
# ─────────────────────────────────────────────────────────────────────────────
print('Integrating sample orbits ...')
A_cases = [0.50, 0.90, 1.20]
T_orb   = 6.0
DT_ORB  = 0.002    # step size — small enough for smooth curves

fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
fig.suptitle(
    fr'Sample Orbits — $B={B_P}$, $m_0=m_1=m_4=1$, $C_0={C0}$, $E_0={E0}$,'
    fr'  $T={T_orb}$',
    fontsize=12)

for ax, A in zip(axes, A_cases):
    ic = setup_ic(A)
    ax.set_title(fr'$A = {A}$')
    ax.set_xlabel(r'$x$')
    if ax is axes[0]:
        ax.set_ylabel(r'$y$')

    if ic is None:
        ax.text(0.5, 0.5, 'IC inaccessible\n(discriminant < 0)',
                transform=ax.transAxes, ha='center', va='center', color='gray')
        continue

    t_s, Y = integrate(ic, T_orb, dt=DT_ORB, stride=1)
    x0t, y0t = Y[0], Y[1]
    x1t, y1t = Y[2], Y[3]

    # Fade the trail from light (early) to full (late) using line segments
    n = len(t_s)
    seg = max(1, n // 200)          # thin the points to ~200 drawing segments
    idx = np.arange(0, n, seg)

    for i in range(len(idx)-1):
        a = 0.15 + 0.85 * idx[i] / n
        sl = slice(idx[i], idx[i+1]+1)
        ax.plot( x0t[sl],  y0t[sl], color='tab:blue', lw=1.0, alpha=a)
        ax.plot(-x0t[sl], -y0t[sl], color='tab:blue', lw=0.7, alpha=a*0.5, ls='--')
        ax.plot( x1t[sl],  y1t[sl], color='tab:red',  lw=1.0, alpha=a)
        ax.plot(-x1t[sl], -y1t[sl], color='tab:red',  lw=0.7, alpha=a*0.5, ls='--')

    # Start (circle) and end (diamond) markers
    for xs, ys, col in [(x0t,y0t,'tab:blue'),(-x0t,-y0t,'tab:blue'),
                         (x1t,y1t,'tab:red'), (-x1t,-y1t,'tab:red')]:
        ax.scatter(xs[0],  ys[0],  s=28, color=col, edgecolors='white',
                   linewidths=0.5, zorder=5)
        ax.scatter(xs[-1], ys[-1], s=35, color=col, marker='D', zorder=6)

    ax.scatter(0, 0, s=60, color='#111111', marker='*', zorder=7)
    ax.set_aspect('equal')
    ax.axhline(0, color='#dddddd', lw=0.5, zorder=0)
    ax.axvline(0, color='#dddddd', lw=0.5, zorder=0)

legend_handles = [
    mpatches.Patch(color='tab:blue', label='Bodies 0 & 3  (solid / dashed)'),
    mpatches.Patch(color='tab:red',  label='Bodies 1 & 2  (solid / dashed)'),
    Line2D([0],[0], marker='*', color='w', markerfacecolor='#111111',
           markersize=10, label='Body 4 (centre)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',
           markersize=6, label='start'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor='gray',
           markersize=6, label='end'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, 0.01))
fig.tight_layout(rect=[0, 0.09, 1, 0.95])
fig.savefig('sample_orbit.png', bbox_inches='tight')
plt.close(fig)
print('  sample_orbit.png  written')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Hierarchy classification regions in (r1, r2) space
# ─────────────────────────────────────────────────────────────────────────────
r_max = 7.5
N_grid = 700
r_arr = np.linspace(0, r_max, N_grid)
R1, R2 = np.meshgrid(r_arr, r_arr)

Z = np.zeros_like(R1, dtype=int)   # 0 = undetermined

m23   = (R1 > B1) & (R2 < B2)                                          # H23
m14   = (R1 < B4) & (R2 > B3)                                          # H14
mouter = ((R1 > B1) & (R2 > B2)) | ((R1 > B4) & (R2 > B3))            # outer
mouter &= ~m23 & ~m14

Z[m23]    = 1
Z[m14]    = 2
Z[mouter] = 3

cmap4 = ListedColormap(['#e0e0e0', '#aec6e8', '#f4a582', '#b8e0b8'])
norm4 = BoundaryNorm([-0.5,0.5,1.5,2.5,3.5], cmap4.N)

fig, ax = plt.subplots(figsize=(7, 6.5))
ax.pcolormesh(R1, R2, Z, cmap=cmap4, norm=norm4, shading='auto', rasterized=True)

# Boundary lines
for val, ori in [(B1,'v'),(B2,'v'),(B1,'h'),(B2,'h')]:
    if ori == 'v':
        ax.axvline(val, color='#333333', lw=1.1, ls='--', alpha=0.65)
    else:
        ax.axhline(val, color='#333333', lw=1.1, ls='--', alpha=0.65)

ax.text(B1+0.12, 0.35, fr'$b_1 = {B1:.2f}$', fontsize=9, color='#333333',
        rotation=90, va='bottom')
ax.text(0.12, B2+0.15, fr'$b_2 = {B2:.3f}$', fontsize=9, color='#333333')

# Region labels
ax.text(6.4, 0.85, 'H23', ha='center', va='center', fontsize=14,
        fontweight='bold', color='#1a5276')
ax.text(0.85, 6.4, 'H14', ha='center', va='center', fontsize=14,
        fontweight='bold', color='#922b21')
ax.text(6.2, 5.5, 'H12 / H13\n(outer)', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#1e8449')
ax.text(2.9, 2.5, 'undetermined', ha='center', va='center',
        fontsize=11, color='#555555')

# Sample IC starting radii (r1=A, r2=B=0.25)
A_samp = np.arange(0.40, 1.36, 0.05)
ax.scatter(A_samp, np.full_like(A_samp, B_P), s=30, color='#7f00ff', zorder=5)
ax.plot([A_samp[0], A_samp[-1]], [B_P, B_P], color='#7f00ff',
        lw=0.9, ls=':', alpha=0.7)
ax.text(A_samp[-1]+0.1, B_P, fr'sample ICs ($B={B_P}$)',
        va='center', fontsize=9, color='#7f00ff')

legend_els = [
    mpatches.Patch(facecolor='#e0e0e0', label='Undetermined  (state = −1)'),
    mpatches.Patch(facecolor='#aec6e8', label='H23  (body 1 near centre)'),
    mpatches.Patch(facecolor='#f4a582', label='H14  (body 0 near centre)'),
    mpatches.Patch(facecolor='#b8e0b8', label='H12 or H13  (outer binary)'),
]
ax.legend(handles=legend_els, loc='upper left', fontsize=9, framealpha=0.92)

ax.set_xlabel(r'$r_1 = |\mathbf{r}_0|$  — distance of body 0 from centre')
ax.set_ylabel(r'$r_2 = |\mathbf{r}_1|$  — distance of body 1 from centre')
ax.set_title(r'Hierarchy Classification Regions in $(r_1,\,r_2)$ Space',
             fontsize=12)
ax.set_xlim(0, r_max); ax.set_ylim(0, r_max)
fig.tight_layout()
fig.savefig('hierarchy_regions.png', bbox_inches='tight')
plt.close(fig)
print('  hierarchy_regions.png  written')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Accessible IC region in (A, B) parameter space
# ─────────────────────────────────────────────────────────────────────────────
A_rng = np.linspace(0.10, 2.50, 350)
B_rng = np.linspace(0.10, 1.50, 350)
AA, BB = np.meshgrid(A_rng, B_rng)

dAB_g  = np.abs(AA - BB)
safe_g = np.where(dAB_g < 1e-6, np.inf, dAB_g)
U_g    = 2/safe_g + 2/(AA+BB) + 2.5/AA + 2.5/BB
DISC   = -C**2 + 4.0*(AA**2 + BB**2)*(E0 + U_g)
DISC[dAB_g < 0.012] = np.nan   # mask the A≈B singular band

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(AA, BB, DISC, levels=[-1e4, 0], colors=['#f4a582'], alpha=0.75)
ax.contourf(AA, BB, DISC, levels=[0, 1e6], colors=['#b8e0b8'], alpha=0.75)
ax.contour(AA, BB, DISC, levels=[0], colors=['#333333'], linewidths=1.8)

# A = B diagonal
diag = np.linspace(0.1, 1.5, 200)
ax.plot(diag, diag, 'k--', lw=1.0, alpha=0.35, label=r'$A = B$ (singular)')

# Sample sweep
ax.axhline(B_P, color='#7f00ff', lw=1.6, ls='--', alpha=0.85)
A_sw = np.arange(0.40, 1.36, 0.05)
ax.scatter(A_sw, np.full_like(A_sw, B_P), s=30, color='#7f00ff', zorder=5)
ax.text(2.35, B_P+0.04, fr'$B={B_P}$ sweep', fontsize=9, color='#7f00ff',
        ha='right', va='bottom')

# Annotate regions
ax.text(1.8, 0.7, r'$\Delta \geq 0$' '\naccessible', ha='center',
        fontsize=11, color='#1e8449', fontweight='bold')
ax.text(0.4, 1.3, r'$\Delta < 0$' '\ninaccessible', ha='center',
        fontsize=11, color='#922b21', fontweight='bold')

legend_els = [
    mpatches.Patch(facecolor='#b8e0b8', label=r'$\Delta \geq 0$ — IC accessible at $E_0$'),
    mpatches.Patch(facecolor='#f4a582', label=r'$\Delta < 0$ — IC inaccessible at $E_0$'),
    Line2D([0],[0], color='#333333', lw=1.8, label=r'$\Delta = 0$ boundary'),
    Line2D([0],[0], color='#7f00ff', lw=1.6, ls='--', label=fr'sample sweep $B={B_P}$'),
    Line2D([0],[0], color='k',       lw=1.0, ls='--', alpha=0.4, label=r'$A = B$ (singular)'),
]
ax.legend(handles=legend_els, fontsize=9, loc='lower right', framealpha=0.92)
ax.set_xlabel(r'$A$ — x-position of body 0')
ax.set_ylabel(r'$B$ — x-position of body 1')
ax.set_title(r'Accessible IC Region — discriminant $\Delta$ in $(A,\,B)$ space',
             fontsize=12)
ax.set_xlim(0.10, 2.50); ax.set_ylim(0.10, 1.50)
fig.tight_layout()
fig.savefig('accessible_ics.png', bbox_inches='tight')
plt.close(fig)
print('  accessible_ics.png  written')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Energy and angular momentum conservation
# ─────────────────────────────────────────────────────────────────────────────
print('Integrating conservation test ...')
A_em  = 0.90
T_em  = 20.0
DT_EM = 0.001

ic_em = setup_ic(A_em)
E0_em = energy(ic_em)
L0_em = angmom(ic_em)
print(f'  A={A_em}: E0={E0_em:.6f}  L0={L0_em:.6f}  (expected c={C:.6f})')

t_em, Y_em = integrate(ic_em, T_em, dt=DT_EM, stride=5)   # store every 5th step

E_arr = np.array([energy(Y_em[:,k]) for k in range(Y_em.shape[1])])
L_arr = np.array([angmom(Y_em[:,k]) for k in range(Y_em.shape[1])])

dE = np.abs((E_arr - E0_em) / E0_em)
dL = np.abs((L_arr - L0_em) / abs(L0_em))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
fig.suptitle(
    fr'Conservation Properties — RK4 integration, $A={A_em}$, $B={B_P}$, $T={T_em}$',
    fontsize=12)

ax1.semilogy(t_em, dE, color='tab:blue', lw=1.2, label=r'$|(E-E_0)/E_0|$')
ax1.axhline(1e-5, color='tab:red', ls='--', lw=1.4, alpha=0.8,
            label=r'C++ abort threshold ($10^{-5}$)')
ax1.set_ylabel(r'Relative energy error')
ax1.legend(fontsize=10)
ax1.grid(True, which='both', alpha=0.2)
ax1.set_ylim(bottom=1e-16)

ax2.semilogy(t_em, dL, color='tab:orange', lw=1.2,
             label=r'$|(L-L_0)/L_0|$')
ax2.set_ylabel(r'Relative angular momentum error')
ax2.set_xlabel(r'$t$')
ax2.legend(fontsize=10)
ax2.grid(True, which='both', alpha=0.2)
ax2.set_ylim(bottom=1e-16)

# Shade the region below the abort threshold
ax1.axhspan(0, 1e-5, alpha=0.06, color='green')
ax1.text(T_em*0.02, 5e-6, 'safe zone', fontsize=8, color='green', alpha=0.7)

fig.tight_layout()
fig.savefig('energy_momentum.png', bbox_inches='tight')
plt.close(fig)
print('  energy_momentum.png  written')

print('\nAll figures written.')
