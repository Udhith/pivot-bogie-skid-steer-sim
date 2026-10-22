"""Live 3D MuJoCo viewer of the closed-loop robot.
Usage: python scripts/view_sim.py [--kind circle|eight|straight] [--ctrl nonlinear|pid] [--v 0.5 --R 2.0] [--speed 1.0] [--nogui]
--nogui runs the same loop without a window (for checking the script only)."""
import sys, time, pathlib, argparse, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sim.build_model import Bogie
from sim import parameters as P, trajectory_generator as TG, controllers as C
ap = argparse.ArgumentParser()
ap.add_argument('--kind', default='circle'); ap.add_argument('--ctrl', default='nonlinear')
ap.add_argument('--v', type=float, default=0.5); ap.add_argument('--R', type=float, default=2.0)
ap.add_argument('--offset', type=float, default=0.15); ap.add_argument('--kth', type=float, default=6.0)
ap.add_argument('--T', type=float, default=30.0); ap.add_argument('--speed', type=float, default=1.0)
ap.add_argument('--notrail', action='store_true'); ap.add_argument('--nogui', action='store_true'); a = ap.parse_args()
prm = dict(P.load()[0]); dt = 0.002
bg = Bogie(prm=prm, dt=dt); bg.reset(); l = prm['lever_l']
ref = TG.make(a.kind, v=a.v, R=a.R, T=a.T, dt=dt, x0=l, y0=a.offset, ramp=2.0)
ct = C.Controller(prm, kind=a.ctrl, ktheta=a.kth, dt=dt)
def tick(k):
    pos = bg.pos_F(); refk = TG.at(ref, k)
    omref, d = ct.wheel_refs((pos[0], pos[1], bg.psi_c()), bg.theta(), refk); bg.step(omref)
N = int(a.T/dt)
if a.nogui:
    for k in range(N): tick(k)
    print('ok, final theta [deg]', np.degrees(bg.theta())); sys.exit()
import mujoco, mujoco.viewer
Z = 0.02
def draw_poly(scn, pts, rgba, width):
    """Append a polyline of capsules to a user scene."""
    for p, q in zip(pts[:-1], pts[1:]):
        if scn.ngeom >= scn.maxgeom: return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.array([p[0], p[1], Z]), np.array([q[0], q[1], Z]))
        scn.ngeom += 1
def redraw(scn, refpts, trail):
    scn.ngeom = 0
    draw_poly(scn, refpts, (0.2, 0.5, 1.0, 0.6), 0.012)      # blue = planned path
    if len(trail) > 1: draw_poly(scn, trail, (1.0, 0.15, 0.1, 1.0), 0.02)   # red = path actually driven (front midpoint)
refpts = np.c_[ref['x'][::50], ref['y'][::50]]; trail = []
with mujoco.viewer.launch_passive(bg.m, bg.d) as v:
    v.cam.distance = 6.0; v.cam.elevation = -35; v.cam.azimuth = 90; v.cam.lookat[:] = [1.0, 1.0, 0.0]
    t0 = time.time(); k = 0
    while v.is_running() and k < N:
        while k*dt/a.speed < time.time()-t0 and k < N:
            tick(k); k += 1
            if k % 25 == 0 and not a.notrail: p = bg.pos_F(); trail.append((p[0], p[1]))
        if not a.notrail:
            with v.lock(): redraw(v.user_scn, refpts, trail)
        v.sync(); time.sleep(0.005)
    while v.is_running(): v.sync(); time.sleep(0.05)
