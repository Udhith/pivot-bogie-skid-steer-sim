"""Reference paths for the front-axle midpoint F, built from a curvature profile kappa(t) and speed v_r(t):
psi' = kappa*v, x' = v cos psi, y' = v sin psi (integrated at fine step). Speed ramps 0 -> v over 'ramp' s.
Kinds: straight, circle (radius R), eight (two smooth tangent lobes of radius R, sign-switching curvature)."""
import numpy as np
smooth = lambda x: 3*x**2 - 2*x**3
def make(kind, v=0.5, R=2.0, T=30.0, dt=0.002, x0=0.0, y0=0.0, ramp=2.0, kappa_ramp=3.0):
    n = int(T/dt)+1; t = np.arange(n)*dt
    vr = v*smooth(np.clip(t/ramp, 0, 1))
    if kind == 'straight': kap = np.zeros(n)
    elif kind == 'circle': kap = (1.0/R)*smooth(np.clip((t-1.0)/kappa_ramp, 0, 1))
    elif kind == 'eight':
        Tp = 4*np.pi*R/v; kap = (1.0/R)*np.tanh(4.0*np.sin(2*np.pi*(t-1.0)/Tp))*smooth(np.clip((t-1.0)/kappa_ramp, 0, 1))
    else: raise ValueError(kind)
    wr = kap*vr; psi = np.concatenate([[0], np.cumsum(wr[:-1]*dt)])
    x = x0 + np.concatenate([[0], np.cumsum(vr[:-1]*np.cos(psi[:-1])*dt)])
    y = y0 + np.concatenate([[0], np.cumsum(vr[:-1]*np.sin(psi[:-1])*dt)])
    return dict(t=t, x=x, y=y, psi=psi, v=vr, w=wr, kappa=kap, dt=dt)
def at(ref, k): k = min(k, len(ref['t'])-1); return ref['x'][k], ref['y'][k], ref['psi'][k], ref['v'][k], ref['w'][k]
