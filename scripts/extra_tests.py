"""Extra simulation tests (Tests 1-8). Usage: python scripts/extra_tests.py [test names...]; writes data/ext_<test>.csv.
Tests: dt, encoder, delay, slope, steps, payload, reverse, stiction, startstop, modeA"""
import sys, pathlib, csv, collections, time, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sim.build_model import Bogie
from sim import parameters as P, trajectory_generator as TG, controllers as C, metrics as M
wrap = C.wrap

class RevController(C.Controller):
    """Path loop valid for v_r < 0: w = w_r + v_r k_y e_y + k_psi |v_r| sin(e_psi); vF kept <= -vmin."""
    def path_loop(self, ex, ey, epsi, vr, wr):
        vF = vr*np.cos(epsi) + self.kx*ex; wc = wr + vr*self.ky*ey + self.kpsi*abs(vr)*np.sin(epsi)
        return min(vF, -self.vmin), wc
    def wheel_refs(self, pose, theta, ref):
        p = self.p; r, wr_, wf, l = p['wheel_radius'], p['track_w'], p['front_track'], p['lever_l']
        ex, ey, epsi = self.errors(pose, ref); vF, wc = self.path_loop(ex, ey, epsi, ref[3], ref[4])
        thd = float(np.clip(-np.arctan(l*wc/vF), -self.thlim, self.thlim))
        if self.first: self.thd_prev, self.first = thd, False
        raw = (thd - self.thd_prev)/self.dt; a = self.dt/(self.tau_d + self.dt)
        self.thd_dot += a*(raw - self.thd_dot); self.thd_prev = thd
        wb = wc + self.thd_dot + self.kth*(thd - theta); vb = vF/np.cos(thd)
        om = np.array([(vF - wf/2*wc)/r, (vF + wf/2*wc)/r, (vb - wr_/2*wb)/r, (vb + wr_/2*wb)/r]); lim = p['omega_max']; m = np.max(np.abs(om))
        if m > lim: om *= lim/m
        return om, dict(thd=thd, wb=wb)


class StandstillController(C.Controller):
    """Standstill mode: when the reference speed is below v_stop all wheel references are zero and the path integrator is reset;
    at low speed theta_d blends from the reference curvature (v_F small) to -atan(l*w_c/v_F), which removes the 1/v_F singularity."""
    def __init__(self, *a, v_stop=0.02, v_blend=(0.05, 0.15), **k): super().__init__(*a, **k); self.v_stop, self.v_blend = v_stop, v_blend
    def wheel_refs(self, pose, theta, ref):
        p = self.p; r, wr_, wf, l = p['wheel_radius'], p['track_w'], p['front_track'], p['lever_l']
        ex, ey, epsi = self.errors(pose, ref); kr = ref[4]/ref[3] if abs(ref[3]) > 1e-6 else 0.0
        if abs(ref[3]) < self.v_stop:
            self.I = 0.0; thd = float(np.clip(-np.arctan(l*kr), -self.thlim, self.thlim)); self.thd_prev, self.thd_dot, self.first = thd, 0.0, False
            return np.zeros(4), dict(ex=ex, ey=ey, epsi=epsi, vF=0.0, wc=0.0, thd=thd, wb=0.0)
        vF, wc = self.path_loop(ex, ey, epsi, ref[3], ref[4]); w = float(np.clip((vF - self.v_blend[0])/(self.v_blend[1]-self.v_blend[0]), 0, 1))
        kc = w*wc/vF + (1-w)*kr; thd = float(np.clip(-np.arctan(l*kc), -self.thlim, self.thlim))
        if self.first: self.thd_prev, self.first = thd, False
        raw = (thd - self.thd_prev)/self.dt; a = self.dt/(self.tau_d + self.dt)
        self.thd_dot += a*(raw - self.thd_dot); self.thd_prev = thd
        wb = wc + self.thd_dot + self.kth*(thd - theta); vb = vF/np.cos(thd)
        om = np.array([(vF - wf/2*wc)/r, (vF + wf/2*wc)/r, (vb - wr_/2*wb)/r, (vb + wr_/2*wb)/r]); lim = p['omega_max']; mm = np.max(np.abs(om))
        if mm > lim: om *= lim/mm
        return om, dict(ex=ex, ey=ey, epsi=epsi, vF=vF, wc=wc, thd=thd, wb=wb)

def profile_ref(kind, vfun, R=2.0, T=30.0, dt=0.002, x0=0.0, y0=0.0, kappa_ramp=3.0):
    """Reference with an arbitrary speed profile vfun(t); circle or straight curvature."""
    n = int(T/dt)+1; t = np.arange(n)*dt; sm = lambda x: 3*x**2-2*x**3
    vr = np.array([vfun(x) for x in t]); kap = np.zeros(n) if kind == 'straight' else (1.0/R)*sm(np.clip((t-1.0)/kappa_ramp, 0, 1))
    wr = kap*vr; psi = np.concatenate([[0], np.cumsum(wr[:-1]*dt)])
    x = x0+np.concatenate([[0], np.cumsum(vr[:-1]*np.cos(psi[:-1])*dt)]); y = y0+np.concatenate([[0], np.cumsum(vr[:-1]*np.sin(psi[:-1])*dt)])
    return dict(t=t, x=x, y=y, psi=psi, v=vr, w=wr, kappa=kap, dt=dt)

def set_payload(bg, prm, a, total):
    """Chassis CoM at distance a ahead of pivot and chassis mass so that robot total = total (kg)."""
    c = bg.bid['chassis']; mc0 = bg.m0_chassis; mc = mc0 + (total - bg.total0)
    bg.m.body_mass[c] = mc; bg.m.body_inertia[c] = bg.I0_chassis*mc/mc0; bg.m.body_ipos[c] = [a, 0, bg.ipos0[2]]

def sim(kind='circle', ctrl='nonlinear', T=30.0, v=0.5, R=2.0, offset=0.15, mass=125.0, kth=6.0, mu=None, dt=0.002, th_bits=None, th_delay=0.0, cmd_delay=0.0,
        slope_deg=0.0, slope_dir_deg=-90.0, payload=None, reverse=False, stiction=None, vfun=None, modeA=False, standstill=False, kz=6.0, ramp=2.0, **over):
    prm = dict(P.load()[0]); prm['m_chassis'] = mass - prm['m_bogie']; prm.update(over)
    if stiction: prm['pivot_coulomb'] = stiction[0]
    bg = Bogie(prm=prm, mu=mu, dt=dt); bg.reset(); l = prm['lever_l']
    c = bg.bid['chassis']; bg.m0_chassis = float(bg.m.body_mass[c]); bg.I0_chassis = bg.m.body_inertia[c].copy(); bg.ipos0 = bg.m.body_ipos[c].copy(); bg.total0 = mass
    if slope_deg:
        g = 9.81; a = np.radians(slope_deg); ph = np.radians(slope_dir_deg); bg.m.opt.gravity[:] = [g*np.sin(a)*np.cos(ph), g*np.sin(a)*np.sin(ph), -g*np.cos(a)]
    pdof = bg.m.jnt_dofadr[[i for i in range(bg.m.njnt) if bg.m.joint(i).name == 'pivot'][0]]
    if reverse: ref = TG.make(kind, v=-abs(v), R=R, T=T, dt=dt, x0=l, y0=offset, ramp=ramp)
    elif vfun: ref = profile_ref(kind, vfun, R=R, T=T, dt=dt, x0=l, y0=offset)
    else: ref = TG.make(kind, v=v, R=R, T=T, dt=dt, x0=l, y0=offset, ramp=ramp)
    CT = RevController if reverse else (StandstillController if standstill else C.Controller)
    ct = CT(prm, kind=ctrl, ktheta=kth, dt=dt); q = None if th_bits is None else 2*np.pi/2**th_bits
    nth, ncm = int(round(th_delay/dt)), int(round(cmd_delay/dt)); thbuf = collections.deque([0.0]*(nth+1), maxlen=nth+1); cmdbuf = collections.deque([np.zeros(4)]*(ncm+1), maxlen=ncm+1)
    keys = 't xF yF psi theta thd ex ey epsi'.split(); L = {k: [] for k in keys}; L.update(om=[], omref=[], tau=[], lat=[]); tan_prev = None; tan_dot = 0.0
    N = int(T/dt)
    for k in range(N):
        t = k*dt
        if payload:
            for (t0, t1, a0, a1, m0, m1) in payload:
                if t0 <= t <= t1 or (t > t1 and t - dt <= t1):
                    f = np.clip((t-t0)/max(t1-t0, 1e-9), 0, 1); set_payload(bg, prm, a0+(a1-a0)*f, m0+(m1-m0)*f)
        if stiction and stiction[1] > stiction[0]:
            w = bg.d.qvel[pdof]; bg.d.qfrc_applied[pdof] = -(stiction[1]-stiction[0])*np.exp(-(w/0.05)**2)*np.tanh(w/0.005)
        pos = bg.pos_F(); pose = (pos[0], pos[1], bg.psi_c()); refk = TG.at(ref, k)
        th = bg.theta(); th_m = th if q is None else round(th/q)*q; thbuf.append(th_m); th_use = thbuf[0]
        if modeA:
            ex, ey, ep = ct.errors(pose, refk); vFd, wc = ct.path_loop(ex, ey, ep, refk[3], refk[4]); tn = -l*wc/vFd
            if tan_prev is None: tan_prev = tn
            a_ = dt/(0.05+dt); tan_dot += a_*((tn-tan_prev)/dt - tan_dot); tan_prev = tn
            th_s = float(np.clip(th_use, -1.3, 1.3)); z = np.tan(th_s) - tn; vb = vFd/np.cos(th_s)
            wb = np.cos(th_s)**2*(tan_dot - kz*z - (vFd/(l*ct.ky))*np.sin(ep)) - (vb/l)*np.sin(th_s)
            rr, ww = prm['wheel_radius'], prm['track_w']; om = np.array([np.nan, np.nan, (vb-ww/2*wb)/rr, (vb+ww/2*wb)/rr]); d = dict(thd=float(np.arctan(tn)))
            lim = prm['omega_max']; mm = np.nanmax(np.abs(om))
            if mm > lim: om = om*lim/mm
        else:
            om, d = ct.wheel_refs(pose, th_use, refk)
        cmdbuf.append(om); bg.step(cmdbuf[0])
        if k % 10 == 0:
            L['t'].append(t); L['xF'].append(pos[0]); L['yF'].append(pos[1]); L['psi'].append(pose[2]); L['theta'].append(bg.theta()); L['thd'].append(d['thd'])
            ex, ey, ep = ct.errors(pose, refk); L['ex'].append(ex); L['ey'].append(ey); L['epsi'].append(ep)
            wsp = bg.wheel_speeds(); L['om'].append(wsp); L['omref'].append(np.where(np.isnan(cmdbuf[0]), wsp, cmdbuf[0])); L['tau'].append(bg.ctrl.copy()); L['lat'].append(bg.wheel_lateral_slip())
        if abs(bg.theta()) > 1.3 and k > 500 and False: break
    A = {k: np.array(x) for k, x in L.items()}
    m = M.compute(A, ref, dt*10); tm = prm['tau_max']; m['sat_frac'] = float((np.abs(A['tau']) >= 0.999*tm).any(axis=1).mean()); m['peak_tau'] = float(np.abs(A['tau']).max()); m['stall'] = bool(m['pos_max'] > 0.5)
    idx = np.minimum(np.arange(len(A['t']))*10, len(ref['v'])-1); mv = (A['t'] >= 5) & (np.abs(ref['v'][idx]) > 0.05); dth = A['theta']-A['thd']
    m['align_mov_rms'] = float(np.degrees(np.sqrt(np.mean(dth[mv]**2)))) if mv.any() else float('nan'); m['align_mov_max'] = float(np.degrees(np.max(np.abs(dth[mv])))) if mv.any() else float('nan')
    m['fail'] = bool(m['jackknife'] or m['stall'] or not np.isfinite(m['xtrack_max']) or m['xtrack_max'] > 0.30); m['xF_end'] = float(A['xF'][-1]); return m

def row(test, tag, m, **kw):
    r = dict(test=test, case=tag, **kw); r.update({k: m[k] for k in ('pos_rmse', 'pos_max', 'xtrack_rmse', 'xtrack_max', 'align_rmse_deg', 'align_max_deg', 'slip_rms', 'theta_max_deg', 'sat_frac', 'peak_tau', 'align_mov_rms', 'align_mov_max', 'jackknife', 'stall', 'fail')}); return r

def run_test(name):
    rows = []; add = lambda tag, m, **kw: (rows.append(row(name, tag, m, **kw)), print(f"{name:9s} {tag:34s} xtr_rms={m['xtrack_rmse']:.4f} xtr_max={m['xtrack_max']:.3f} pos_max={m['pos_max']:.3f} sat={m['sat_frac']:.3f} align_rms={m['align_rmse_deg']:.2f} mov_rms={m['align_mov_rms']:.2f} mov_max={m['align_mov_max']:.1f} thmax={m['theta_max_deg']:.1f} FAIL={m['fail']}", flush=True))
    if name == 'dt':
        for kind in ('circle', 'eight'):
            for ctrl in ('pid', 'nonlinear'):
                for dt in (0.001, 0.002, 0.004):
                    for cone, imp in (('elliptic', 10), ('pyramidal', 10), ('elliptic', 1)):
                        add(f'{kind}/{ctrl}/dt={dt}/{cone}/imp={imp}', sim(kind, ctrl, T=30 if kind == 'circle' else 60, dt=dt, cone=cone, impratio=imp), kind=kind, ctrl=ctrl, dt=dt, cone=cone, impratio=imp)
    if name == 'encoder':
        for bits in (6, 8, 10, 12, 14, None):
            for kind in ('circle', 'eight'): add(f'{kind}/bits={bits}', sim(kind, T=30 if kind == 'circle' else 60, th_bits=bits), kind=kind, bits=bits)
    if name == 'delay':
        for d in (0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3):
            add(f'theta-delay {int(d*1000)} ms', sim(th_delay=d), kind='circle', mode='theta', delay_ms=int(d*1000))
            add(f'command-delay {int(d*1000)} ms', sim(cmd_delay=d), kind='circle', mode='command', delay_ms=int(d*1000))
    if name == 'slope':
        for sl in (0, 1, 2, 4, 6, 7, 8):
            for ctrl in ('pid', 'nonlinear'): add(f'cross-slope {sl} deg/{ctrl}', sim(ctrl=ctrl, slope_deg=sl), slope_deg=sl, ctrl=ctrl)
    if name == 'slope2':
        for sl in (6, 8, 10):
            for tm in (6.1, 9.0, 12.0): add(f'slope {sl} deg, tau_max {tm}', sim(slope_deg=sl, tau_max=tm), slope_deg=sl, tau_max=tm)
    if name == 'steps':
        l = P.load()[0]['lever_l']
        for h in (0, 0.01, 0.02, 0.04, 0.06, 0.08):
            xml = f'<geom type="box" pos="{l+2.0} 2.15 {h/2}" size="0.55 0.3 {max(h,1e-4)/2}" friction="0.8 0.005 0.0001" contype="1" conaffinity="1" condim="3"/>' if h > 0 else ''
            for ctrl in ('pid', 'nonlinear'): add(f'step {int(h*1000)} mm/{ctrl}', sim(ctrl=ctrl, extra_xml=xml), step_mm=int(h*1000), ctrl=ctrl)
    if name == 'steps2':
        l = P.load()[0]['lever_l']
        for h in (0.04, 0.06, 0.08, 0.10):
            xml = f'<geom type="box" pos="{l+2.0} 2.15 {h/2}" size="0.55 0.3 {h/2}" friction="0.8 0.005 0.0001" contype="1" conaffinity="1" condim="3"/>'
            for tm in (6.1, 9.0, 12.0): add(f'step {int(h*1000)} mm, tau_max {tm}', sim(extra_xml=xml, tau_max=tm), step_mm=int(h*1000), tau_max=tm)
    if name == 'payload':
        for a0, a1 in ((0.57, 0.57), (0.57, 0.40), (0.57, 0.80), (0.30, 0.85)):
            for dm in (0, -50):
                pl = [(8.0, 18.0, a0, a1, 125.0, 125.0+dm)]
                for ctrl in ('pid', 'nonlinear'): add(f'a {a0}->{a1}, dm {dm}/{ctrl}', sim(ctrl=ctrl, T=30, payload=pl), a0=a0, a1=a1, dmass=dm, ctrl=ctrl)
        for mt in (100, 150):
            pl = [(8.0, 18.0, 0.30, 0.85, mt, mt)]
            add(f'a 0.3->0.85 at {mt} kg', sim(mass=mt, T=30, payload=pl), a0=0.3, a1=0.85, dmass=0, ctrl='nonlinear', mass=mt)
    if name == 'reverse':
        for kind in ('straight', 'circle', 'eight'):
            add(f'reverse {kind}', sim(kind, 'nonlinear', T={'straight': 20, 'circle': 30, 'eight': 60}[kind], reverse=True), kind=kind, ctrl='nonlinear-reverse')
        for kth in (0.0, 0.3, 1.0): add(f'reverse circle kth={kth}', sim('circle', 'nonlinear', reverse=True, kth=kth), kind='circle', ctrl='nonlinear', kth=kth)
    if name == 'stiction':
        for fc in (0.5, 2.0, 5.0):
            for ratio in (1, 2, 4):
                for mu in (0.8, 0.4): add(f'Fc={fc} Fs/Fc={ratio} mu={mu}', sim(stiction=(fc, fc*ratio), mu=mu), Fc=fc, ratio=ratio, mu=mu)
    if name == 'startstop':
        def vf(t, v=0.5, tr=2.0, ts=10.0, tstop=0.5, th=4.0):
            if t < ts: return v*(3*np.clip(t/tr, 0, 1)**2-2*np.clip(t/tr, 0, 1)**3)
            if t < ts+tstop: return v*(1-(t-ts)/tstop)
            if t < ts+tstop+th: return 0.0
            x = np.clip((t-ts-tstop-th)/tr, 0, 1); return v*(3*x**2-2*x**3)
        for mt in (100, 125, 150):
            for ctrl in ('pid', 'nonlinear'):
                for ts in (0.5, 0.1): 
                    add(f'stop in {ts} s, {mt} kg/{ctrl}', sim(ctrl=ctrl, mass=mt, T=26, vfun=lambda t, ts=ts: vf(t, tstop=ts)), mass=mt, ctrl=ctrl, stop_s=ts)
    if name == 'startstop2':
        def vf(t, v=0.5, tr=2.0, ts=10.0, tstop=0.5, th=4.0):
            sm = lambda x: 3*x**2-2*x**3
            if t < ts: return v*sm(np.clip(t/tr, 0, 1))
            if t < ts+tstop: return v*(1-(t-ts)/tstop)
            if t < ts+tstop+th: return 0.0
            return v*sm(np.clip((t-ts-tstop-th)/tr, 0, 1))
        for mt in (100, 125, 150):
            for ctrl in ('pid', 'nonlinear'):
                for ts in (0.5, 0.1): add(f'standstill stop {ts} s, {mt} kg/{ctrl}', sim(ctrl=ctrl, mass=mt, T=26, standstill=True, vfun=lambda t, ts=ts: vf(t, tstop=ts)), mass=mt, ctrl=ctrl, stop_s=ts)
        for kind in ('circle', 'eight'):
            for ctrl in ('pid', 'nonlinear'): add(f'standstill nominal {kind}/{ctrl}', sim(kind, ctrl, T=30 if kind == 'circle' else 60, standstill=True), kind=kind, ctrl=ctrl, nominal=1)
        add('standstill circle R=0.8', sim('circle', 'nonlinear', R=0.8, standstill=True), kind='circle', R=0.8)
    if name == 'modeA':
        for kind in ('circle', 'eight', 'straight'):
            for R in ((2.0, 3.0, 4.0) if kind == 'circle' else (2.0,)):
                for kz in (2.0, 6.0): add(f'ModeA {kind} R={R} kz={kz}', sim(kind, 'nonlinear', R=R, T={'straight': 20, 'circle': 30, 'eight': 60}[kind], modeA=True, kz=kz), kind=kind, R=R, kz=kz)
    if rows:
        with open(f'data/ext_{name}.csv', 'w', newline='') as f: w = csv.DictWriter(f, sorted({k for r in rows for k in r}, key=lambda k: (k not in ('test', 'case'), k))); w.writeheader(); w.writerows(rows)

if __name__ == '__main__':
    names = sys.argv[1:] or 'dt encoder delay slope slope2 steps payload reverse stiction startstop modeA'.split()
    for n in names: t0 = time.time(); run_test(n); print(f'-- {n} done in {time.time()-t0:.0f}s', flush=True)
