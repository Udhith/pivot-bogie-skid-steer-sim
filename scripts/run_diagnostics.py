"""Diagnostics on top of the open-loop cases; writes data/diag_*.csv.
 D1 skid-steer (pivot locked) vs motor torque limit and friction
 D2 pivot-angle feedback gain kth on the kinematically consistent rear command
 D3 trail offset e with the rear differential free (equal torque)"""
import sys, pathlib, csv
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_openloop import run
def dump(name, rows):
    with open(f'data/diag_{name}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
def fmt(m): return f"th_end={m['theta_end_deg']:6.1f} err={m['theta_err_deg']:6.2f} yaw={m['yawrate_mean']:6.3f} slipR={m['slip_rear_rms']:.3f} E={m['energy_J']:6.0f} jack={m['jackknife']}"
print("D1 locked pivot (4WD skid-steer), commanded yaw 0.3 rad/s:"); R = []
for tau in (15, 30, 60):
    for mu in (0.8, 0.4):
        m, _ = run('locked', mu=mu, tau_max=tau); m['tau_max'] = tau; R.append(m); print(f"  tau_max={tau:3d} mu={mu}: yaw={m['yawrate_mean']:.3f} slipF={m['slip_front_rms']:.3f} slipR={m['slip_rear_rms']:.3f} E={m['energy_J']:.0f}")
dump('locked', R)
print("D2 free_kin + theta feedback:"); R = []
for k in (0, 1, 2, 3, 6, 10):
    m, _ = run('free_kin', kth=k); m['kth'] = k; R.append(m); print(f"  kth={k:2d}: {fmt(m)}")
dump('kth', R)
print("D3 free_open (rear differential free) vs trail e:"); R = []
for e in (-0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3):
    m, _ = run('free_open', e=e); R.append(m); print(f"  e={e:+.2f}: {fmt(m)}")
dump('trail', R)
