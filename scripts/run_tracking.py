"""Closed-loop tracking: kinds {straight,circle,eight} x controllers {pid,nonlinear}. Writes data/track_<kind>_<ctrl>.csv and data/track_summary.csv.
Usage: python scripts/run_tracking.py [--offset 0.15 --v 0.5 --R 2.0 --mass 125 --e 0.0 --kth 6]"""
import sys, pathlib, argparse, csv, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sim.build_model import Bogie
from sim import parameters as P, trajectory_generator as TG, controllers as C, metrics as M
def run(kind, ctrl, T=None, v=0.5, R=2.0, offset=0.15, mass=125.0, e=0.0, kth=6.0, mu=None, gains=None, dt=0.002, log_every=10, save=None, motor_gain=None, noise=None, seed=0, ramp=2.0, ret=False, ff=True, **over):
    prm = dict(P.load()[0]); prm['m_chassis'] = mass - prm['m_bogie']; prm['pivot_trail_e'] = e; prm.update(over)
    T = T or {'straight': 20.0, 'circle': 30.0, 'eight': 60.0}[kind]
    bg = Bogie(prm=prm, mu=mu, dt=dt); bg.reset(); l = prm['lever_l']
    if motor_gain is not None: bg.motor_gain = np.asarray(motor_gain, float)
    rng = np.random.default_rng(seed); nz = noise or dict(pos=0.0, head=0.0, theta=0.0)   # noise std: pos [m], head [rad], theta [rad]
    ref = TG.make(kind, v=v, R=R, T=T, dt=dt, x0=l, y0=offset, ramp=ramp)          # reference starts offset (initial lateral error)
    ct = C.Controller(prm, kind=ctrl, ktheta=kth, dt=dt, ff=ff, **(gains or {}))
    keys = 't xF yF psi theta thd ex ey epsi'.split(); L = {k: [] for k in keys}; L.update(om=[], omref=[], tau=[], lat=[]); lat_rms = []
    for k in range(int(T/dt)):
        pos = bg.pos_F(); pose = (pos[0], pos[1], bg.psi_c()); refk = TG.at(ref, k)
        meas = (pos[0]+nz['pos']*rng.standard_normal(), pos[1]+nz['pos']*rng.standard_normal(), pose[2]+nz['head']*rng.standard_normal())
        omref, d = ct.wheel_refs(meas, bg.theta()+nz['theta']*rng.standard_normal(), refk); bg.step(omref)
        if k % log_every == 0:
            L['t'].append(k*dt); L['xF'].append(pos[0]); L['yF'].append(pos[1]); L['psi'].append(pose[2]); L['theta'].append(bg.theta()); L['thd'].append(d['thd'])
            tex, tey, tep = ct.errors(pose, refk); L['ex'].append(tex); L['ey'].append(tey); L['epsi'].append(tep)
            L['om'].append(bg.wheel_speeds()); L['omref'].append(omref); L['tau'].append(bg.ctrl.copy()); L['lat'].append(bg.wheel_lateral_slip())
    A = {k: np.array(v_) for k, v_ in L.items()}
    for k in ('om', 'omref', 'tau', 'lat'): pass
    met = M.compute(A, ref, dt*log_every)
    if save:
        with open(save, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['t','xF','yF','psi','theta','thd','ex','ey','epsi','xr','yr']); 
            for i in range(len(A['t'])): kk = i*log_every; w.writerow([A[c][i] for c in keys] + [ref['x'][kk], ref['y'][kk]])
    if ret: return met, A, ref
    return met
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--offset', type=float, default=0.15); ap.add_argument('--v', type=float, default=0.5); ap.add_argument('--R', type=float, default=2.0)
    ap.add_argument('--mass', type=float, default=125.0); ap.add_argument('--e', type=float, default=0.0); ap.add_argument('--kth', type=float, default=6.0); a = ap.parse_args()
    rows = []; print(f"{'kind':9s} {'ctrl':9s} {'posRMSE':>8s} {'xtrRMSE':>8s} {'xtrMax':>7s} {'hdgRMSE':>8s} {'alignRMS':>8s} {'slip':>6s} {'E_J':>7s} jack")
    for kind in ('straight', 'circle', 'eight'):
        for ctrl in ('pid', 'nonlinear'):
            m = run(kind, ctrl, v=a.v, R=a.R, offset=a.offset, mass=a.mass, e=a.e, kth=a.kth, save=f'data/track_{kind}_{ctrl}.csv'); m.update(kind=kind, ctrl=ctrl); rows.append(m)
            print(f"{kind:9s} {ctrl:9s} {m['pos_rmse']:8.3f} {m['xtrack_rmse']:8.3f} {m['xtrack_max']:7.3f} {m['head_rmse_deg']:8.2f} {m['align_rmse_deg']:8.2f} {m['slip_rms']:6.3f} {m['energy_J']:7.0f} {m['jackknife']}")
    with open('data/track_summary.csv', 'w', newline='') as f: w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
