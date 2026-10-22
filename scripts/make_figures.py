"""Figures for the paper, from saved data only (re-runs a few traces with return_log). Writes figs/*.pdf and data/table_*.tex"""
import sys, pathlib, csv, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from run_tracking import run
from run_openloop import run as run_ol
plt.rcParams.update({'font.size': 7, 'axes.linewidth': .6, 'lines.linewidth': 1.0, 'figure.dpi': 200, 'axes.spines.top': False, 'axes.spines.right': False})
C = {'pid': '#c2410c', 'nonlinear': '#0f766e'}; deg = np.degrees
# ---- fig_feas: open-loop cases
m, A_fb = run_ol('free_kin', kth=6.0, outdir='data', tag='openloop_free_kin_fb')
fig, ax = plt.subplots(1, 2, figsize=(3.4, 2.3))
cs = [('locked', '#6b7280', 'K1 locked'), ('free_eq', '#b45309', 'K2 free, equal'), ('free_kin', '#7c3aed', 'K3 consistent'), ('free_kin_fb', '#0f766e', r'K3 + $k_\theta$=6'), ('rear_only', '#be123c', 'K4 rear only'), ('free_open', '#0369a1', 'K5 free diff.')]
for c, col, lab in cs:
    d = pd.read_csv(f'data/openloop_{c}.csv', comment='#'); ax[0].plot(d.t, deg(d.theta), color=col, label=lab); ax[1].plot(d.t, deg(np.unwrap(d.psi_c)), color=col)
ax[0].plot(d.t, deg(d.theta_des), 'k--', lw=.7, label=r'$\theta_d$'); ax[0].set(xlabel='t [s]', ylabel=r'pivot angle $\theta$ [deg]'); ax[1].set(xlabel='t [s]', ylabel=r'chassis heading $\psi_c$ [deg]')
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=4.8, frameon=False, ncol=4, loc='upper center'); fig.tight_layout(pad=.4, rect=(0, 0, 1, .86)); fig.savefig('figs/fig_feas.pdf'); plt.close(fig)
# ---- tracks (wide) + errors
fig, ax = plt.subplots(2, 3, figsize=(7.0, 3.3), gridspec_kw={'height_ratios': [2, 1]})
for j, kind in enumerate(('straight', 'circle', 'eight')):
    for ctrl in ('pid', 'nonlinear'):
        mm, A, ref = run(kind, ctrl, ret=True); ax[0, j].plot(A['xF'], A['yF'], color=C[ctrl], label=ctrl.upper() if ctrl == 'pid' else 'Nonlinear'); ax[1, j].plot(A['t'], 1000*np.hypot(A['ex'], A['ey']), color=C[ctrl])
    ax[0, j].plot(ref['x'], ref['y'], 'k--', lw=.6, label='reference'); ax[0, j].set_title(kind, fontsize=7); (ax[0, j].set_aspect('equal', 'datalim') if kind != 'straight' else ax[0, j].set_ylim(-.1, .3)); ax[0, j].set(xlabel='x [m]', ylabel='y [m]'); ax[1, j].set(xlabel='t [s]', ylabel='position error [mm]')
ax[0, 0].legend(fontsize=5.5, frameon=False); fig.tight_layout(pad=.4); fig.savefig('figs/fig_tracks.pdf'); plt.close(fig)
# ---- wheels / pivot on the figure-eight (nonlinear)
mm, A, ref = run('eight', 'nonlinear', ret=True); t = A['t']
fig, ax = plt.subplots(3, 1, figsize=(3.4, 3.2), sharex=True)
ax[0].plot(t, deg(A['theta']), color='#0f766e', label=r'$\theta$'); ax[0].plot(t, deg(A['thd']), 'k--', lw=.6, label=r'$\theta_d$'); ax[0].set_ylabel('pivot [deg]'); ax[0].legend(fontsize=5, frameon=False, ncol=2)
for i, n in enumerate(('fL', 'fR', 'rL', 'rR')): ax[1].plot(t, A['om'][:, i], lw=.7, label=n)
ax[1].set_ylabel('wheel [rad/s]'); ax[1].legend(fontsize=5, frameon=False, ncol=4)
for i, n in enumerate(('fL', 'fR', 'rL', 'rR')): ax[2].plot(t, A['tau'][:, i], lw=.7)
ax[2].axhline(6.1, color='k', lw=.4, ls=':'); ax[2].axhline(-6.1, color='k', lw=.4, ls=':'); ax[2].set(ylabel='torque [N m]', xlabel='t [s]'); fig.tight_layout(pad=.4); fig.savefig('figs/fig_wheels.pdf'); plt.close(fig)
# ---- sensitivity (robust.csv)
R = pd.read_csv('data/robust.csv'); fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.7))
nom = {c: float(pd.read_csv('data/track_summary.csv').query("kind=='circle' and ctrl==@c").xtrack_rmse.iloc[0]) for c in C}
def panel(a, grp, lab, xs):
    for c in C:
        d = R[(R.group == grp) & (R.ctrl == c)]; y = 100*(d.xtrack_rmse.values - nom[c])/nom[c]; f = d.fail.values.astype(bool); a.plot(range(len(y)), np.where(f, np.nan, y), 'o-', ms=2.5, color=C[c]); a.plot(np.where(f)[0], [150 + (6 if c == 'pid' else 0)]*int(f.sum()), 'x', ms=4, color=C[c]); a.set_ylim(-40, 165)
    a.set_xticks(range(len(xs))); a.set_xticklabels(xs, fontsize=5, rotation=30); a.set_title(lab, fontsize=6); a.axhline(0, color='k', lw=.4)
panel(ax[0], 'pivot_fric', 'pivot friction [N m]', ['0', '0.5', '2', '8', '32']); panel(ax[1], 'grip_mu', r'grip $\mu$', ['0.8', '0.5', '0.3', '0.2']); panel(ax[2], 'motor_mism', 'motor mismatch', ['5%', '10%', '20%', 'R 50%', 'FL 50%'])
for c in C:
    d = R[(R.group == 'sensor_noise') & (R.ctrl == c)]; d = d.assign(lv=d.level.str[:2]).groupby('lv').xtrack_rmse.mean(); ax[3].plot(range(3), 100*(d.values - nom[c])/nom[c], 'o-', ms=2.5, color=C[c])
ax[3].set_xticks(range(3)); ax[3].set_xticklabels(['L1', 'L2', 'L3'], fontsize=5); ax[3].set_title('sensor noise', fontsize=6); ax[3].axhline(0, color='k', lw=.4)
ax[0].set_ylabel(r'$\Delta$ RMSE [%] (x = failed)'); fig.tight_layout(pad=.4); fig.savefig('figs/fig_sens.pdf'); plt.close(fig)
# ---- failure map
X = pd.read_csv('data/extra_failmap.csv'); fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.6), sharey=True)
for a, c in zip(ax, ('pid', 'nonlinear')):
    d = X[X.ctrl == c]; P = d.pivot(index='mu', columns='fric', values='xtrack_max'); a.imshow(np.log10(P.values.clip(1e-2, 3)), origin='lower', aspect='auto', cmap='RdYlGn_r', vmin=-2, vmax=0.5)
    for i, mu in enumerate(P.index):
        for j, fr in enumerate(P.columns): a.text(j, i, 'F' if P.values[i, j] > .3 else '', ha='center', va='center', fontsize=5)
    a.set_xticks(range(len(P.columns))); a.set_xticklabels([f'{v:g}' for v in P.columns], fontsize=5); a.set_yticks(range(len(P.index))); a.set_yticklabels(P.index); a.set_title(c.upper() if c == 'pid' else 'Nonlinear', fontsize=6); a.set_xlabel('pivot friction [N m]')
ax[0].set_ylabel(r'$\mu$'); fig.tight_layout(pad=.4); fig.savefig('figs/fig_fail.pdf'); plt.close(fig)
# ---- LHS box
L = pd.read_csv('data/extra_lhs.csv'); fig, ax = plt.subplots(figsize=(1.6, 1.6)); ax.boxplot([1000*L[L.ctrl == c].xtrack_rmse for c in C], tick_labels=['PID', 'Nonlinear'], widths=.5); ax.set_ylabel('cross-track RMS [mm]'); fig.tight_layout(pad=.4); fig.savefig('figs/fig_lhs.pdf'); plt.close(fig)
# ---- trail map
T = pd.read_csv('data/extra_trailmap.csv'); P = T.pivot(index='mu', columns='e', values='err_deg'); fig, ax = plt.subplots(figsize=(3.4, 1.4))
im = ax.imshow(np.log10(P.values.clip(.1, 100)), origin='lower', aspect='auto', cmap='viridis_r')
for i in range(P.shape[0]):
    for j in range(P.shape[1]): ax.text(j, i, f'{P.values[i, j]:.1f}', ha='center', va='center', fontsize=5, color='w' if P.values[i, j] < 10 else 'k')
ax.set_xticks(range(P.shape[1])); ax.set_xticklabels([f'{v:+.2f}' for v in P.columns], fontsize=5); ax.set_yticks(range(P.shape[0])); ax.set_yticklabels(P.index); ax.set(xlabel='trail e [m]', ylabel=r'$\mu$'); fig.tight_layout(pad=.4); fig.savefig('figs/fig_map.pdf'); plt.close(fig)
print('figures done')
