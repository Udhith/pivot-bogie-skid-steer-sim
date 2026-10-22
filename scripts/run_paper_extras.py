"""Extra exhibits for the paper. Writes data/extra_*.csv.
 X1 failure map: pivot friction x mu (circle R=2, both controllers)
 X2 combined disturbances: Latin hypercube N=20 (scipy qmc, seed 0), both controllers, same draws
 X3 ablation: alignment feed-forward (theta_d_dot) on/off, both controllers, straight/circle/eight
 X4 trail map: open-loop free differential, trail e x mu"""
import sys, pathlib, csv, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_tracking import run
from run_openloop import run as run_ol
from scipy.stats import qmc
def save(name, rows):
    with open(f'data/extra_{name}.csv', 'w', newline='') as f: w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
fail = lambda m: bool(m['jackknife'] or m['xtrack_max'] > 0.30)
R = []
for ctrl in ('pid', 'nonlinear'):
    for mu in (0.2, 0.4, 0.8):
        for fr in (0.0, 2.0, 5.0, 8.0, 10.0, 12.0, 16.0):
            m = run('circle', ctrl, mu=mu, pivot_coulomb=fr); R.append(dict(ctrl=ctrl, mu=mu, fric=fr, xtrack_max=m['xtrack_max'], align=m['align_rmse_deg'], fail=fail(m)))
            print('X1', ctrl, mu, fr, round(m['xtrack_max'], 3), fail(m), flush=True)
save('failmap', R)
lhs = qmc.LatinHypercube(d=7, seed=0).random(20)
lo = np.array([0.3, 0.0, 100, 0.3, 0.0, 0.0, 0.0]); hi = np.array([0.8, 8.0, 150, 0.9, 0.2, 0.02, 0.0175*2])   # mu, fric, mass, com_a, motor spread, pos noise [m], angle noise [rad]
R = []
for i, u in enumerate(lhs):
    x = lo + u*(hi-lo); g = 1 + x[4]*(2*np.random.default_rng(100+i).random(4)-1)
    for ctrl in ('pid', 'nonlinear'):
        m = run('circle', ctrl, mu=float(x[0]), pivot_coulomb=float(x[1]), mass=float(x[2]), chassis_com_a=float(x[3]), motor_gain=g, noise=dict(pos=float(x[5]), head=float(x[6]), theta=float(x[6])), seed=i)
        R.append(dict(draw=i, ctrl=ctrl, mu=x[0], fric=x[1], mass=x[2], com_a=x[3], mism=x[4], npos=x[5], nang=x[6], xtrack_rmse=m['xtrack_rmse'], xtrack_max=m['xtrack_max'], align=m['align_rmse_deg'], fail=fail(m)))
        print('X2', i, ctrl, round(m['xtrack_max'], 3), fail(m), flush=True)
save('lhs', R)
R = []
for kind in ('straight', 'circle', 'eight'):
    for ctrl in ('pid', 'nonlinear'):
        for ff in (True, False):
            m = run(kind, ctrl, ff=ff); R.append(dict(kind=kind, ctrl=ctrl, ff=ff, pos_rmse=m['pos_rmse'], xtrack_max=m['xtrack_max'], align_rmse_deg=m['align_rmse_deg'], energy_J=m['energy_J']))
            print('X3', kind, ctrl, ff, round(m['align_rmse_deg'], 3), flush=True)
save('ablation', R)
R = []
for mu in (0.3, 0.5, 0.8):
    for e in (-0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3):
        m, _ = run_ol('free_open', e=e, mu=mu, m_chassis=113.0); R.append(dict(mu=mu, e=e, err_deg=m['theta_err_deg'], yaw=m['yawrate_mean'], jackknife=m['jackknife']))
        print('X4', mu, e, round(m['theta_err_deg'], 2), flush=True)
save('trailmap', R)
