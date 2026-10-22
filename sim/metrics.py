import numpy as np
def compute(L, ref, dt_log, settle=5.0):
    """L: dict of logged arrays (t,xF,yF,psi,theta,thd,ex,ey,epsi,om,omref,tau,lat). Metrics after 'settle' s unless noted."""
    t = L['t']; m = t >= settle
    e = np.hypot(L['ex'], L['ey'])
    # cross-track: distance to nearest point on dense reference polyline
    P = np.column_stack([ref['x'][::5], ref['y'][::5]]); Q = np.column_stack([L['xF'], L['yF']])
    ct = np.array([np.min(np.hypot(P[:, 0]-q[0], P[:, 1]-q[1])) for q in Q[m]])
    n = max(1, int(0.2*m.sum())); idx = np.where(m)[0][-n:]
    out = dict(pos_rmse=float(np.sqrt(np.mean(e[m]**2))), pos_max=float(np.max(e[m])), xtrack_rmse=float(np.sqrt(np.mean(ct**2))), xtrack_max=float(np.max(ct)),
               head_rmse_deg=float(np.degrees(np.sqrt(np.mean(L['epsi'][m]**2)))), ss_err=float(np.mean(e[idx])),
               align_rmse_deg=float(np.degrees(np.sqrt(np.mean((L['theta'][m]-L['thd'][m])**2)))), align_max_deg=float(np.degrees(np.max(np.abs(L['theta'][m]-L['thd'][m])))),
               wheel_rmse=float(np.sqrt(np.mean((L['om'][m]-L['omref'][m])**2))), effort=float(np.sum(L['tau']**2)*dt_log),
               energy_J=float(np.sum(np.abs(L['tau']*L['om']))*dt_log), slip_rms=float(np.sqrt(np.mean(L['lat'][m]**2))),
               jackknife=bool(np.max(np.abs(L['theta'])) > np.radians(69.0)), theta_max_deg=float(np.degrees(np.max(np.abs(L['theta'])))))
    return out
