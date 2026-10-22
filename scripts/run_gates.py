"""Gate checks. G3: M0 kinematic model vs MuJoCo (same commands). G4: smallest trackable circle radius.
CL-locked: closed-loop tracking with the pivot locked (4-wheel skid-steer). Writes data/gates_*.csv."""
import sys, pathlib, csv, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_openloop import run as run_ol, profile
from run_tracking import run as run_trk
from sim import robot_model as M0, parameters as P, trajectory_generator as TG, controllers as C, metrics as MT
from sim.build_model import Bogie
prm = dict(P.load()[0]); prm['m_chassis'] = 125.0 - prm['m_bogie']; l = prm['lever_l']

print("G3: M0 (ideal kinematics) vs MuJoCo, open-loop v_F=0.8 wc=0.3, theta feedback kth=6, T=12 s")
rowsG3 = []
for kth in (6.0, 10.0):
    m, A = run_ol('free_kin', kth=kth, m_chassis=prm['m_chassis'])
    thd = lambda t: -np.arctan(l*profile(t, 0.8, 0.3)[1]/max(profile(t, 0.8, 0.3)[0], 1e-3)) if profile(t, 0.8, 0.3)[0] > 0.05 else 0.0
    thd_dot = lambda t: (thd(t+1e-3) - thd(t-1e-3))/2e-3
    v_fn = lambda t, s: profile(t, 0.8, 0.3)[0]/np.cos(thd(t))
    wb_fn = lambda t, s: thd_dot(t) + profile(t, 0.8, 0.3)[1] + kth*(thd(t) - s[3])
    ts, Y = M0.simulate([l, 0, 0, 0], v_fn, wb_fn, l, 12.0, dt=0.002)
    idx = np.searchsorted(ts, A[:, 0]); idx = np.clip(idx, 0, len(ts)-1)
    pe = np.hypot(Y[0, idx]-A[:, 1], Y[1, idx]-A[:, 2]); he = np.abs(Y[2, idx]-A[:, 3]); te = np.abs(Y[3, idx]-A[:, 4])
    path_len = float(np.sum(np.hypot(np.diff(A[:, 1]), np.diff(A[:, 2]))))
    r = dict(kth=kth, pos_rmse_mm=1000*np.sqrt(np.mean(pe**2)), pos_end_mm=1000*pe[-1], head_rmse_deg=np.degrees(np.sqrt(np.mean(he**2))), theta_rmse_deg=np.degrees(np.sqrt(np.mean(te**2))), path_len_m=path_len)
    rowsG3.append(r); print({k: round(float(v), 3) for k, v in r.items()})
with open('data/gates_g3.csv', 'w', newline='') as f: w = csv.DictWriter(f, rowsG3[0].keys()); w.writeheader(); w.writerows(rowsG3)

print("\nG4: smallest circle radius tracked (v=0.5 m/s, 125 kg, e=0). Kinematic limit R_F,min = l/tan(theta_lim); controller limit 60 deg")
print(f"  l/tan(60deg) = {l/np.tan(np.radians(60)):.3f} m ; l/tan(70deg) = {l/np.tan(np.radians(70)):.3f} m")
rowsG4 = []
for R in (1.5, 1.0, 0.8, 0.7, 0.6, 0.5):
    for ctrl in ('pid', 'nonlinear'):
        m = run_trk('circle', ctrl, R=R, v=0.5, T=40.0, offset=0.0)
        fail = bool(m['jackknife'] or m['xtrack_max'] > 0.30); rowsG4.append(dict(R=R, ctrl=ctrl, xtrack_max=m['xtrack_max'], theta_max_deg=m['theta_max_deg'], align_rmse_deg=m['align_rmse_deg'], fail=fail))
        print(f"  R={R:.1f} {ctrl:9s} xtr_max={m['xtrack_max']:.3f} theta_max={m['theta_max_deg']:.1f} align={m['align_rmse_deg']:.2f} FAIL={fail}", flush=True)
with open('data/gates_g4.csv', 'w', newline='') as f: w = csv.DictWriter(f, rowsG4[0].keys()); w.writeheader(); w.writerows(rowsG4)

print("\nCL-locked: closed loop with pivot LOCKED (rigid rear axle), same path loops")
rowsL = []
for kind, R in (('straight', 2.0), ('circle', 4.0), ('circle', 2.0)):
    for ctrl in ('pid', 'nonlinear'):
        dt = 0.002; T = 20.0 if kind == 'straight' else 40.0
        bg = Bogie(prm=prm, dt=dt, locked=True); bg.reset(); ref = TG.make(kind, v=0.5, R=R, T=T, dt=dt, x0=l, y0=0.0)
        ct = C.Controller(prm, kind=ctrl, dt=dt); r_, wr_, wf = prm['wheel_radius'], prm['track_w'], prm['front_track']; ex_log = []; lat = []; en = 0.0; yaw_ref = []; yaw = []
        for k in range(int(T/dt)):
            pos = bg.pos_F(); pose = (pos[0], pos[1], bg.psi_c()); refk = TG.at(ref, k)
            ex, ey, ep = ct.errors(pose, refk); vF, wc = ct.path_loop(ex, ey, ep, refk[3], refk[4])
            om = np.array([(vF-wf/2*wc)/r_, (vF+wf/2*wc)/r_, (vF-wr_/2*wc)/r_, (vF+wr_/2*wc)/r_]); bg.step(om)
            if k % 10 == 0: ex_log.append(np.hypot(ex, ey)); lat.append(bg.wheel_lateral_slip()); en += float(np.sum(np.abs(bg.ctrl*bg.wheel_speeds())))*dt*10
        e_ = np.array(ex_log); t_ = np.arange(len(e_))*dt*10; mk = t_ >= 5
        rowsL.append(dict(kind=kind, R=R, ctrl=ctrl, pos_rmse=float(np.sqrt(np.mean(e_[mk]**2))), pos_max=float(e_[mk].max()), slip_rms=float(np.sqrt(np.mean(np.array(lat)[mk]**2))), energy_J=en))
        print(f"  {kind:8s} R={R:.1f} {ctrl:9s} pos_rmse={rowsL[-1]['pos_rmse']:.3f} pos_max={rowsL[-1]['pos_max']:.3f} slip={rowsL[-1]['slip_rms']:.3f} E={en:.0f}", flush=True)
with open('data/gates_locked.csv', 'w', newline='') as f: w = csv.DictWriter(f, rowsL[0].keys()); w.writeheader(); w.writerows(rowsL)
