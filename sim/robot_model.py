"""M0: ideal no-slip kinematics of the pivot-steered vehicle (paper eq. 1-6).
State s=[xF,yF,psi_c,theta]; inputs v (bogie speed), omega_b (bogie yaw rate)."""
import numpy as np
from scipy.integrate import solve_ivp

def wheels_to_body(wL, wR, r, w):
    return 0.5*r*(wR+wL), r*(wR-wL)/w

def body_to_wheels(v, wb, r, w):
    return (v-0.5*w*wb)/r, (v+0.5*w*wb)/r   # (wL, wR)

def rhs(t, s, v_fn, wb_fn, l):
    psi, th = s[2], s[3]
    v, wb = v_fn(t, s), wb_fn(t, s)
    vF = v*np.cos(th)
    return [vF*np.cos(psi), vF*np.sin(psi), -(v/l)*np.sin(th), wb + (v/l)*np.sin(th)]

def simulate(s0, v_fn, wb_fn, l, T, dt=0.005):
    ts = np.arange(0, T, dt)
    sol = solve_ivp(rhs, (0, T), s0, t_eval=ts, args=(v_fn, wb_fn, l), rtol=1e-9, atol=1e-11)
    return sol.t, sol.y

def min_turn_radius(l, theta_max):
    return abs(l)/np.tan(theta_max)
