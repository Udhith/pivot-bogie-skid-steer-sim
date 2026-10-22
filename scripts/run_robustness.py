"""One-at-a-time robustness sweeps on the circle (R=2 m, 0.5 m/s, 125 kg, e=0, kth=6) for both path loops.
Appends rows to data/robust.csv as it goes. FAIL = jackknife or max cross-track > 0.30 m or NaN.
Usage: python scripts/run_robustness.py"""
import sys, pathlib, csv, os, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_tracking import run
OUT = 'data/robust.csv'; rows = []
def go(group, level, ctrl, **kw):
    try: m = run('circle', ctrl, **kw)
    except Exception as ex: m = dict(pos_rmse=np.nan, xtrack_rmse=np.nan, xtrack_max=np.nan, align_rmse_deg=np.nan, slip_rms=np.nan, energy_J=np.nan, jackknife=True, theta_max_deg=np.nan, err=str(ex)[:60])
    fail = bool(m['jackknife'] or not np.isfinite(m['xtrack_max']) or m['xtrack_max'] > 0.30)
    r = dict(group=group, level=level, ctrl=ctrl, xtrack_rmse=m['xtrack_rmse'], xtrack_max=m['xtrack_max'], align_rmse_deg=m['align_rmse_deg'], slip_rms=m['slip_rms'], energy_J=m['energy_J'], theta_max_deg=m['theta_max_deg'], jackknife=m['jackknife'], fail=fail)
    rows.append(r); new = not os.path.exists(OUT)
    with open(OUT, 'a', newline='') as f:
        w = csv.DictWriter(f, r.keys());
        if new: w.writeheader()
        w.writerow(r)
    print(f"{group:12s} {str(level):14s} {ctrl:9s} xtr_rms={r['xtrack_rmse']:.3f} xtr_max={r['xtrack_max']:.3f} align={r['align_rmse_deg']:.2f} slip={r['slip_rms']:.3f} E={r['energy_J']:.0f} FAIL={fail}", flush=True)
if os.path.exists(OUT): os.remove(OUT)
rng = np.random.default_rng(1)
for ctrl in ('pid', 'nonlinear'):
    for c in (0.0, 0.5, 2.0, 8.0, 32.0): go('pivot_fric', f'{c} Nm', ctrl, pivot_coulomb=c)
    for mu in (0.8, 0.5, 0.3, 0.2): go('grip_mu', mu, ctrl, mu=mu)
    for M in (100, 150): go('mass', f'{M} kg', ctrl, mass=M)
    for a in (0.3, 0.9): go('com_a', f'{a} m', ctrl, chassis_com_a=a)
    for s in (0.05, 0.10, 0.20):
        g = 1 + s*(2*np.random.default_rng(7).random(4) - 1); go('motor_mism', f'+-{int(s*100)}%', ctrl, motor_gain=g)
    go('motor_mism', 'rear_R=50%', ctrl, motor_gain=[1, 1, 1, 0.5]); go('motor_mism', 'front_L=50%', ctrl, motor_gain=[0.5, 1, 1, 1])
    for lv, (p, h, t) in enumerate(((0.005, 0.0035, 0.0035), (0.02, 0.0175, 0.0175), (0.05, 0.05, 0.05)), 1):
        for sd in range(3): go('sensor_noise', f'L{lv}(pos{p*1000:.0f}mm,ang{np.degrees(h):.2f}deg) s{sd}', ctrl, noise=dict(pos=p, head=h, theta=t), seed=sd)
    for v in (1.0, 1.5): go('speed', f'{v} m/s', ctrl, v=v, T=25.0)
    for R in (1.5, 1.0, 0.7): go('radius', f'R={R} m', ctrl, R=R, T=30.0)
