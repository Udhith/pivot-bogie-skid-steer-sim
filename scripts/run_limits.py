"""Boundary probes beyond the nominal envelope (circle R=2). Writes data/limits.csv."""
import sys, pathlib, csv, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_tracking import run
R = []
def go(g, lv, c, **kw):
    m = run('circle', c, **kw); fail = bool(m['jackknife'] or m['xtrack_max'] > 0.30)
    R.append(dict(group=g, level=lv, ctrl=c, xtrack_rmse=m['xtrack_rmse'], xtrack_max=m['xtrack_max'], align_rmse_deg=m['align_rmse_deg'], theta_max_deg=m['theta_max_deg'], slip_rms=m['slip_rms'], fail=fail))
    print(f"{g:10s} {str(lv):10s} {c:9s} xtr_rms={m['xtrack_rmse']:.3f} xtr_max={m['xtrack_max']:.3f} align={m['align_rmse_deg']:.2f} thmax={m['theta_max_deg']:.1f} slip={m['slip_rms']:.3f} FAIL={fail}", flush=True)
for c in ('pid', 'nonlinear'):
    for f in (12, 16, 24): go('pivot_fric', f, c, pivot_coulomb=float(f))
    for f in (8, 16): go('fric+kth12', f, c, pivot_coulomb=float(f), kth=12.0)
    for v in (2.0, 3.0): go('speed', v, c, v=v, T=20.0)
    go('noise_big', '100mm/5deg', c, noise=dict(pos=0.1, head=np.radians(5), theta=np.radians(5)), seed=0)
    go('tau_low', '3 Nm', c, tau_max=3.0); go('tau_low', '1.5 Nm', c, tau_max=1.5)
with open('data/limits.csv', 'w', newline='') as f: w = csv.DictWriter(f, R[0].keys()); w.writeheader(); w.writerows(R)
