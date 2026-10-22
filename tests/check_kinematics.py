"""Symbolic check that eq.(3)-(4) follow from the geometry, plus numeric divergence-rate test."""
import sympy as sp, numpy as np, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sim import robot_model as rm, parameters as P

t = sp.symbols('t'); l, v = sp.symbols('l v', positive=True)
psi_c = sp.Function('psi_c')(t); psi_b = sp.Function('psi_b')(t)
xP = sp.Function('xP')(t); yP = sp.Function('yP')(t)
xF = xP + l*sp.cos(psi_c); yF = yP + l*sp.sin(psi_c)
nc = sp.Matrix([-sp.sin(psi_c), sp.cos(psi_c)])
vF = sp.Matrix([sp.diff(xF, t), sp.diff(yF, t)])
vF = vF.subs({sp.diff(xP, t): v*sp.cos(psi_b), sp.diff(yP, t): v*sp.sin(psi_b)})
lat = sp.simplify(vF.dot(nc))                  # lateral velocity of F
sol = sp.solve(sp.Eq(lat, 0), sp.diff(psi_c, t))[0]
expected = -(v/l)*sp.sin(psi_b - psi_c)
assert sp.simplify(sol - expected) == 0, sol
print("SYMBOLIC OK: psi_c_dot = -(v/l) sin(theta)")

prm, _ = P.load(); l_ = prm['lever_l']; v_ = 0.5
for sign, name in ((+1, 'forward'), (-1, 'reverse')):
    tt, y = rm.simulate([0, 0, 0, 1e-4], lambda *_: sign*v_, lambda *_: 0.0, l_, 3.0)
    th = np.abs(y[3]); k = np.polyfit(tt[tt > 0.5], np.log(th[tt > 0.5]), 1)[0]
    print(f"{name}: fitted log-slope {k:+.4f} 1/s, predicted v/l = {sign*v_/l_:+.4f}")
    assert abs(k - sign*v_/l_) < 0.02
print("NUMERIC OK: divergence rate matches v/l (forward grows, reverse decays)")
