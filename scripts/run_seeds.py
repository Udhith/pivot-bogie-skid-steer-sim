"""Repeat key robustness cases over 10 noise seeds; report mean/std/worst. Writes data/seeds.csv."""
import sys, pathlib, csv, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_tracking import run
cases = {'noise_L2 (20mm,1deg)': dict(noise=dict(pos=0.02, head=np.radians(1), theta=np.radians(1))),
         'noise_L3 (50mm,2.9deg)': dict(noise=dict(pos=0.05, head=np.radians(2.86), theta=np.radians(2.86)))}
rows = []
print(f"{'case':24s} {'ctrl':9s} {'xtr_rms mean+-sd':>18s} {'xtr_max worst':>13s} {'align mean+-sd':>16s} fails")
for name, kw in cases.items():
    for ctrl in ('pid', 'nonlinear'):
        R = [run('circle', ctrl, seed=s, **kw) for s in range(10)]
        a = lambda k: np.array([r[k] for r in R]); f = int(sum(r['jackknife'] or r['xtrack_max'] > 0.3 for r in R))
        rows.append(dict(case=name, ctrl=ctrl, xtr_rms_mean=a('xtrack_rmse').mean(), xtr_rms_sd=a('xtrack_rmse').std(), xtr_max_worst=a('xtrack_max').max(), align_mean=a('align_rmse_deg').mean(), align_sd=a('align_rmse_deg').std(), fails=f))
        print(f"{name:24s} {ctrl:9s} {a('xtrack_rmse').mean():8.4f}+-{a('xtrack_rmse').std():.4f} {a('xtrack_max').max():13.3f} {a('align_rmse_deg').mean():8.2f}+-{a('align_rmse_deg').std():.2f} {f}/10", flush=True)
with open('data/seeds.csv', 'w', newline='') as fh: w = csv.DictWriter(fh, rows[0].keys()); w.writeheader(); w.writerows(rows)
