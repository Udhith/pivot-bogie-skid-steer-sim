"""Sweep total robot mass (100/125/150 kg) x trail e (0, 0.2 m): feedback case and free-differential case.
Total mass = m_chassis + m_bogie (m_bogie fixed at its params value). Writes data/diag_mass_trail.csv."""
import sys, pathlib, csv
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_openloop import run
from sim import parameters as P
mb = P.load()[0]['m_bogie']; R = []
print(f"{'case':10s} {'M_tot':>5s} {'e':>5s} {'err_deg':>8s} {'yaw':>6s} {'slipR':>6s} {'E_J':>6s} jack")
for case, kw in (('free_kin', dict(kth=6.0)), ('free_open', {})):
    for M in (100, 125, 150):
        for e in (0.0, 0.2):
            m, _ = run(case, e=e, m_chassis=M-mb, **kw); m.update(M_total=M, kth=kw.get('kth', 0)); R.append(m)
            print(f"{case:10s} {M:5d} {e:5.2f} {m['theta_err_deg']:8.2f} {m['yawrate_mean']:6.3f} {m['slip_rear_rms']:6.3f} {m['energy_J']:6.0f} {m['jackknife']}")
with open('data/diag_mass_trail.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, R[0].keys()); w.writeheader(); w.writerows(R)
