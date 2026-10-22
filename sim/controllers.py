"""Path loops (PID, nonlinear) for the front-axle midpoint F + shared pivot-angle alignment loop + 4-wheel allocation.
Path loop output: (vF_d, wc_d). Alignment: theta_d = -atan(l*wc_d/vF_d); wb = wc_d + theta_d_dot + k_theta*(theta_d - theta);
vb = vF_d/cos(theta_d). Wheel refs [fL, fR, rL, rR] in rad/s (left = +y)."""
import numpy as np
wrap = lambda a: (a + np.pi) % (2*np.pi) - np.pi
class Controller:
    def __init__(self, prm, kind='nonlinear', kx=1.0, ky=2.0, kpsi=1.5, kI=0.0, ktheta=6.0, vmin=0.05, theta_lim_deg=60.0, tau_d=0.05, dt=0.002, ff=True):
        self.p, self.kind = prm, kind; self.kx, self.ky, self.kpsi, self.kI, self.kth = kx, ky, kpsi, kI, ktheta
        self.vmin, self.thlim, self.tau_d, self.dt = vmin, np.radians(theta_lim_deg), tau_d, dt
        self.ff = ff; self.I = 0.0; self.thd_prev = 0.0; self.thd_dot = 0.0; self.first = True
    def errors(self, pose, ref):
        x, y, psi = pose; xr, yr, pr = ref[:3]
        c, s = np.cos(psi), np.sin(psi)
        return c*(xr-x)+s*(yr-y), -s*(xr-x)+c*(yr-y), wrap(pr-psi)
    def path_loop(self, ex, ey, epsi, vr, wr):
        if self.kind == 'pid':
            self.I += ey*self.dt
            vF = vr + self.kx*ex; wc = wr + vr*(self.ky*ey + self.kI*self.I) + self.kpsi*epsi
        else:
            vF = vr*np.cos(epsi) + self.kx*ex; wc = wr + vr*(self.ky*ey + self.kpsi*np.sin(epsi))
        return max(vF, self.vmin), wc
    def wheel_refs(self, pose, theta, ref):
        p = self.p; r, wr_, wf, l = p['wheel_radius'], p['track_w'], p['front_track'], p['lever_l']
        ex, ey, epsi = self.errors(pose, ref); vF, wc = self.path_loop(ex, ey, epsi, ref[3], ref[4])
        thd = float(np.clip(-np.arctan(l*wc/vF), -self.thlim, self.thlim))
        if self.first: self.thd_prev, self.first = thd, False
        raw = (thd - self.thd_prev)/self.dt; a = self.dt/(self.tau_d + self.dt)
        self.thd_dot += a*(raw - self.thd_dot); self.thd_prev = thd         # low-passed derivative
        wb = wc + (self.thd_dot if self.ff else 0.0) + self.kth*(thd - theta); vb = vF/np.cos(thd)
        front = [(vF - wf/2*wc)/r, (vF + wf/2*wc)/r]; rear = [(vb - wr_/2*wb)/r, (vb + wr_/2*wb)/r]
        om = np.array(front + rear); lim = p['omega_max']; m = np.max(np.abs(om))
        if m > lim: om *= lim/m
        return om, dict(ex=ex, ey=ey, epsi=epsi, vF=vF, wc=wc, thd=thd, wb=wb)
