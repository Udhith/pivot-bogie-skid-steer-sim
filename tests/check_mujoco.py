"""Day-3 checks on the MuJoCo model: V1 static loads, V2 straight roll, V6 mirror symmetry, V4 dt convergence."""
import sys, pathlib, numpy as np, mujoco
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sim.build_model import Bogie
ok = True
def report(name, passed, msg):
    global ok; ok &= bool(passed); print(("PASS " if passed else "FAIL ") + name + ": " + msg)

# V1 static loads: expected front/rear split from moment balance about the rear (x=0) wheel line
b = Bogie(); m, d = b.m, b.d
for _ in range(int(3/m.opt.timestep)): mujoco.mj_step(m, d)
g = 9.81; front = rear = 0.0; f6 = np.zeros(6)
fl_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ('front_L_g', 'front_R_g')}
for i in range(d.ncon):
    mujoco.mj_contactForce(m, d, i, f6); c = d.contact[i]
    if fl_ids & {c.geom1, c.geom2}: front += f6[0]
    else: rear += f6[0]
bodies = range(1, m.nbody)
Mtot = sum(m.body_mass[i] for i in bodies)
exp_front = sum(m.body_mass[i]*d.xipos[i][0] for i in bodies)*g/b.prm['lever_l']
report("V1 total load", abs(front+rear-Mtot*g)/(Mtot*g) < 0.01, f"{front+rear:.1f} N vs weight {Mtot*g:.1f} N")
report("V1 front/rear split", abs(front-exp_front)/exp_front < 0.03, f"front {front:.1f} N vs expected {exp_front:.1f} N")
report("V1 no drift at rest", abs(b.psi_c()) < 1e-3 and abs(b.theta()) < 1e-3 and np.linalg.norm(b.pos_F()-[b.prm['lever_l'],0]) < 5e-3,
       f"psi={b.psi_c():.2e} theta={b.theta():.2e} F={b.pos_F()}")

# V2 straight roll: v = r*omega at steady state (high mu)
b = Bogie(mu=1.5); b.reset(); wr = 10.0
for _ in range(int(4/b.m.opt.timestep)): b.step([np.nan, np.nan, wr, wr])
v, vexp = b.bogie_vx(), b.prm['wheel_radius']*wr
report("V2 straight roll", abs(v-vexp)/vexp < 0.03, f"v={v:.4f} vs r*w={vexp:.4f} m/s, wheel speeds {b.wheel_speeds()[2:]}")

# V6 mirror symmetry (swap wL,wR); V4 timestep convergence on the same test
def run(wl, wr, dt, T=3.0):
    b = Bogie(dt=dt); b.reset()
    for _ in range(int(T/dt)): b.step([np.nan, np.nan, wl, wr])
    return np.array([*b.pos_F(), b.psi_c(), b.theta()])
A, B = run(8, 12, 0.002), run(12, 8, 0.002)
mir = np.array([A[0]-B[0], A[1]+B[1], A[2]+B[2], A[3]+B[3]])
report("V6 mirror symmetry", np.max(np.abs(mir)) < 1e-3, f"max asymmetry {np.max(np.abs(mir)):.2e}  (A={np.round(A,4)})")
C = run(8, 12, 0.001)
rel = np.abs(A-C)/(np.abs(C)+1e-3)
report("V4 dt convergence", np.max(rel) < 0.05, f"dt 0.002 vs 0.001 -> {np.round(A,4)} vs {np.round(C,4)}")
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
