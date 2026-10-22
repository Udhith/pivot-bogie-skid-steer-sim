"""Builds the MuJoCo model (M3) from params.yaml. Frame: x forward (toward front axle), y left, z up.
Pivot P is the chassis-frame origin; bogie hinges about +z at P, so theta = psi_b - psi_c."""
import numpy as np, mujoco
from . import parameters as P

def build_xml(prm=None, mu=None, dt=0.002, locked=False):
    p = dict(P.load()[0]) if prm is None else dict(prm)
    mu = p['mu'] if mu is None else mu
    r, w, l = p['wheel_radius'], p['track_w'], p['lever_l']
    fw, mw = p['front_track'], p['m_wheel']
    hw = p['tyre_halfwidth']; th = np.radians(p['theta_max_deg'])
    if locked: th = np.radians(0.001)   # pivot locked at 0 -> 4WD skid-steer baseline
    e = p.get('pivot_trail_e', 0.0)
    mc = p['m_chassis'] - 2*mw          # chassis frame mass (front wheels counted in m_chassis)
    mb = p['m_bogie'] - 2*mw            # bogie frame mass (driven wheels counted in m_bogie)
    a, h = p['chassis_com_a'], p['chassis_com_h'] - r
    def box_inertia(m, sx, sy, sz):
        return m/3*np.array([sy**2+sz**2, sx**2+sz**2, sx**2+sy**2])
    Ic = box_inertia(mc, 1.0, 0.7, 0.3); Ib = box_inertia(mb, 0.1, w, 0.3)
    wheel = lambda name, y, x=0: f'''<body name="{name}" pos="{x} {y} 0">
        <joint name="{name}_j" type="hinge" axis="0 1 0" damping="{p.get('wheel_damping',0.01)}"/>
        <geom name="{name}_g" type="cylinder" size="{r} {hw}" euler="90 0 0" mass="{mw}"
              friction="0.1 0.005 0.0001" contype="1" conaffinity="1" condim="3"/></body>'''
    return f'''<mujoco model="pivot_bogie">
  <option timestep="{dt}" integrator="implicitfast" cone="{p.get('cone','elliptic')}" impratio="{p.get('impratio',10)}" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="ground" type="plane" size="30 30 0.1" friction="{mu} 0.005 0.0001" contype="1" conaffinity="1" condim="3"/>
    {p.get('extra_xml','')}
    <body name="chassis" pos="0 0 {r+0.0005}">
      <freejoint name="root"/>
      <inertial pos="{a} 0 {h}" mass="{mc}" diaginertia="{Ic[0]} {Ic[1]} {Ic[2]}"/>
      <geom type="box" pos="{a} 0 {h}" size="0.5 0.35 0.02" contype="0" conaffinity="0" rgba="0.3 0.3 0.3 0.5"/>
      {wheel('front_L', fw/2, l)}
      {wheel('front_R', -fw/2, l)}
      <body name="bogie" pos="0 0 0">
        <joint name="pivot" type="hinge" axis="0 0 1" damping="{p['pivot_viscous']}"
               frictionloss="{p['pivot_coulomb']}" limited="true" range="{-np.degrees(th)} {np.degrees(th)}"/>
        <inertial pos="0 0 0.05" mass="{mb}" diaginertia="{Ib[0]} {Ib[1]} {Ib[2]}"/>
        <geom type="box" size="0.05 {w/2} 0.01" pos="0 0 0.1" contype="0" conaffinity="0" rgba="0.8 0.4 0.1 0.6"/>
        {wheel('wheel_L', w/2, -e)}
        {wheel('wheel_R', -w/2, -e)}
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="tau_fL" joint="front_L_j" ctrllimited="true" ctrlrange="-{p['tau_max']} {p['tau_max']}"/>
    <motor name="tau_fR" joint="front_R_j" ctrllimited="true" ctrlrange="-{p['tau_max']} {p['tau_max']}"/>
    <motor name="tau_L" joint="wheel_L_j" ctrllimited="true" ctrlrange="-{p['tau_max']} {p['tau_max']}"/>
    <motor name="tau_R" joint="wheel_R_j" ctrllimited="true" ctrlrange="-{p['tau_max']} {p['tau_max']}"/>
  </actuator>
</mujoco>'''

WHEELS = ('front_L', 'front_R', 'wheel_L', 'wheel_R')   # order of 4-vectors: fL, fR, rL, rR

class Bogie:
    """Wrapper: state readout and external PI wheel-speed loops. Actuator order: tau_fL, tau_fR, tau_L, tau_R."""
    def __init__(self, prm=None, mu=None, dt=0.002, Kp=3.0, Ki=10.0, locked=False):
        self.prm = dict(P.load()[0]) if prm is None else dict(prm)
        self.xml = build_xml(self.prm, mu, dt, locked)
        self.m = mujoco.MjModel.from_xml_string(self.xml); self.d = mujoco.MjData(self.m)
        n = lambda t, s: mujoco.mj_name2id(self.m, t, s)
        J, B = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_BODY
        self.qp = self.m.jnt_qposadr[n(J, 'pivot')]
        self.vj = [self.m.jnt_dofadr[n(J, w + '_j')] for w in WHEELS]
        self.bid = {k: n(B, k) for k in ('chassis', 'bogie') + WHEELS}
        self.Kp, self.Ki, self.I = Kp, Ki, np.zeros(4); self.tau_max = self.prm['tau_max']
        self.ctrl = np.zeros(4); self.motor_gain = np.ones(4); mujoco.mj_forward(self.m, self.d)
    def reset(self, theta0=0.0):
        mujoco.mj_resetData(self.m, self.d); self.d.qpos[self.qp] = theta0
        self.I[:] = 0; self.Ic = 0.0; mujoco.mj_forward(self.m, self.d)
    def psi_c(self):
        R = self.d.xmat[self.bid['chassis']].reshape(3, 3); return np.arctan2(R[1, 0], R[0, 0])
    def theta(self): return self.d.qpos[self.qp]
    def pos_F(self):
        c = self.bid['chassis']; R = self.d.xmat[c].reshape(3, 3)
        return (self.d.xpos[c] + R @ np.array([self.prm['lever_l'], 0, 0]))[:2]
    def wheel_speeds(self): return np.array([self.d.qvel[a] for a in self.vj])
    def wheel_lateral_slip(self):
        """Sideways (axle-direction) velocity of each wheel centre, m/s: world velocity projected on the wheel's axle axis."""
        out = []
        for w in WHEELS:
            b = self.bid[w]; res = np.zeros(6)
            mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, b, res, 0)
            axle = self.d.xmat[b].reshape(3, 3)[:, 1]          # body-frame y axis = axle direction
            out.append(float(res[3:6] @ axle))
        return np.array(out)
    def bogie_vx(self):
        res = np.zeros(6); mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.bid['bogie'], res, 0)
        R = self.d.xmat[self.bid['bogie']].reshape(3, 3); return float(R[:2, 0] @ res[3:5])
    def step(self, w_ref, rear_open_diff=False):
        """w_ref: 4 wheel-speed references [fL,fR,rL,rR] (rad/s); NaN = zero torque (free wheel)."""
        w_ref = np.asarray(w_ref, float); free = np.isnan(w_ref)
        e = np.where(free, 0.0, w_ref - self.wheel_speeds()); self.I += e*self.m.opt.timestep
        self.I = np.clip(self.I, -self.tau_max/self.Ki, self.tau_max/self.Ki); self.I[free] = 0
        self.ctrl = np.clip(self.Kp*e + self.Ki*self.I, -self.tau_max, self.tau_max)
        if rear_open_diff:   # equal torque on both rear wheels from one common-mode speed loop (speed DIFFERENCE is free)
            wr = np.asarray(w_ref, float)[2:].mean(); ec = wr - self.wheel_speeds()[2:].mean()
            self.Ic = getattr(self, 'Ic', 0.0) + ec*self.m.opt.timestep
            self.Ic = float(np.clip(self.Ic, -self.tau_max/self.Ki, self.tau_max/self.Ki))
            self.ctrl[2:] = np.clip(self.Kp*ec + self.Ki*self.Ic, -self.tau_max, self.tau_max)
        self.d.ctrl[:] = self.ctrl*self.motor_gain; mujoco.mj_step(self.m, self.d)
