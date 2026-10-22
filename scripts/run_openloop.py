"""Open-loop steering-feasibility cases on the 4-wheel-drive pivot bogie (MuJoCo, M3).
Command: v_F ramps 0->vF in 2 s; from t=3 s omega_c ramps 0->wc over 2 s; held to T. NO feedback on theta.
Cases (same chassis command, different rear/pivot treatment):
  C1 locked    : pivot locked at 0, rear wheels follow the chassis like a skid-steer axle
  C2 free_eq   : pivot free, rear wheels equal speed (bogie not steered)
  C3 free_kin  : pivot free, rear wheel commands kinematically consistent (feed-forward of theta_d, omega_b)
  C4 rear_only : old concept: front wheels free (zero torque), rear differential steers with omega_b = omega_c
Usage: python scripts/run_openloop.py [--vF 0.8 --wc 0.3 --T 12]"""
import sys, pathlib, argparse, csv, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sim.build_model import Bogie
smooth = lambda x: 3*x**2 - 2*x**3

def profile(t, vF, wc):
    v = vF*smooth(np.clip(t/2.0, 0, 1)); w = wc*smooth(np.clip((t-3.0)/2.0, 0, 1)); return v, w

def commands(case, t, bg, vF, wc, dt, kth=0.0):
    p = bg.prm; r, wr, wf, l = p['wheel_radius'], p['track_w'], p['front_track'], p['lever_l']
    v, w = profile(t, vF, wc)
    front = [(v - wf/2*w)/r, (v + wf/2*w)/r]           # L, R (left = +y, ccw yaw positive)
    if case == 'locked':   rear = [(v - wr/2*w)/r, (v + wr/2*w)/r]
    elif case in ('free_eq', 'free_open'): rear = [v/r, v/r]
    elif case == 'free_kin':
        th = lambda tt: -np.arctan(l*profile(tt, vF, wc)[1]/max(profile(tt, vF, wc)[0], 1e-3)) if profile(tt, vF, wc)[0] > 0.05 else 0.0
        thd = th(t); thd_dot = (th(t+1e-3) - th(t-1e-3))/2e-3
        vb = v/np.cos(thd); wb = thd_dot + w + kth*(thd - bg.theta())   # kth: optional theta feedback (1/s)
        rear = [(vb - wr/2*wb)/r, (vb + wr/2*wb)/r]
    elif case == 'rear_only':
        front = [np.nan, np.nan]; rear = [(v - wr/2*w)/r, (v + wr/2*w)/r]
    return np.array(front + rear)

def run(case, vF=0.8, wc=0.3, T=12.0, dt=0.002, mu=None, e=None, outdir=None, tag=None, kth=0.0, **over):
    prm = None
    if e is not None or over:
        from sim import parameters as P
        prm = dict(P.load()[0]); prm.update(over)
        if e is not None: prm['pivot_trail_e'] = e
    bg = Bogie(prm=prm, mu=mu, dt=dt, locked=(case == 'locked')); bg.reset()
    l = bg.prm['lever_l']; rows = []; energy = 0.0
    for k in range(int(T/dt)):
        t = k*dt; bg.step(commands(case, t, bg, vF, wc, dt, kth), rear_open_diff=(case == 'free_open'))
        om = bg.wheel_speeds(); energy += float(np.sum(np.abs(bg.ctrl*om)))*dt
        if k % 10 == 0:
            v, w = profile(t, vF, wc); thd = -np.arctan(l*w/v) if v > 0.05 else 0.0
            rows.append([t, *bg.pos_F(), bg.psi_c(), bg.theta(), thd, *om, *bg.ctrl, *bg.wheel_lateral_slip()])
    A = np.array(rows); th_max = np.radians(bg.prm['theta_max_deg'])
    win = A[:, 0] >= 8.0                                      # steady turning window
    dpsi = np.gradient(np.unwrap(A[:, 3]), A[:, 0])
    ls = A[win, -4:]
    m = dict(case=case, e=bg.prm['pivot_trail_e'], mu=bg.prm['mu'] if mu is None else mu,
             theta_end_deg=np.degrees(A[-1, 4]), theta_des_deg=np.degrees(A[-1, 5]),
             theta_err_deg=np.degrees(np.mean(np.abs(A[win, 4] - A[win, 5]))),
             yawrate_mean=float(np.mean(dpsi[win])), yawrate_cmd=wc,
             slip_front_rms=float(np.sqrt(np.mean(ls[:, :2]**2))), slip_rear_rms=float(np.sqrt(np.mean(ls[:, 2:]**2))),
             energy_J=energy, jackknife=bool(np.max(np.abs(A[:, 4])) > 0.98*th_max and case != 'locked'))
    if outdir:
        outdir = pathlib.Path(outdir); outdir.mkdir(exist_ok=True, parents=True)
        hdr = ['t','xF','yF','psi_c','theta','theta_des','w_fL','w_fR','w_rL','w_rR','tau_fL','tau_fR','tau_rL','tau_rR','lat_fL','lat_fR','lat_rL','lat_rR']
        with open(outdir/f"{tag or case}.csv", 'w', newline='') as f:
            f.write(f"# case={case} vF={vF} wc={wc} T={T} dt={dt} mu={m['mu']} e={m['e']}\n"); w_ = csv.writer(f); w_.writerow(hdr); w_.writerows(A)
    return m, A

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--vF', type=float, default=0.8); ap.add_argument('--wc', type=float, default=0.3)
    ap.add_argument('--T', type=float, default=12.0); a = ap.parse_args()
    res = []
    for case in ('locked', 'free_eq', 'free_kin', 'rear_only'):
        m, _ = run(case, a.vF, a.wc, a.T, outdir='data', tag=f'openloop_{case}'); res.append(m)
    keys = list(res[0].keys())
    with open('data/openloop_summary.csv', 'w', newline='') as f:
        w_ = csv.DictWriter(f, keys); w_.writeheader(); w_.writerows(res)
    print(f"{'case':10s} {'th_end':>7s} {'th_des':>7s} {'|th_err|':>8s} {'yaw_rate':>8s} {'slipF':>6s} {'slipR':>6s} {'E[J]':>7s} jackknife")
    for m in res:
        print(f"{m['case']:10s} {m['theta_end_deg']:7.1f} {m['theta_des_deg']:7.1f} {m['theta_err_deg']:8.2f} {m['yawrate_mean']:8.3f} {m['slip_front_rms']:6.3f} {m['slip_rear_rms']:6.3f} {m['energy_J']:7.1f} {m['jackknife']}")
    print(f"(commanded yaw rate {a.wc} rad/s; slip = rms lateral wheel-centre speed in m/s over t>=8 s)")
