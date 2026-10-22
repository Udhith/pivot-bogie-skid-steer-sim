"""Speed boundary on the R=2 m circle (125 kg): effect of start-up ramp at 2 m/s, then steady-state limit with a 12 s ramp. Writes data/speed_limit.csv."""
import sys, pathlib, csv
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_tracking import run
R = []
def go(tag, c, **kw):
    m = run('circle', c, **kw); f = bool(m['jackknife'] or m['xtrack_max'] > 0.30)
    R.append(dict(test=tag, ctrl=c, v=kw['v'], ramp=kw.get('ramp', 2.0), offset=kw.get('offset', 0.15), xtrack_max=m['xtrack_max'], align_rmse_deg=m['align_rmse_deg'], slip_rms=m['slip_rms'], fail=f)); print(R[-1], flush=True)
for c in ('pid', 'nonlinear'):
    for ramp in (2.0, 6.0): go('ramp', c, v=2.0, T=40.0, ramp=ramp, offset=0.15)
    for v in (2.5, 3.0, 3.5): go('limit', c, v=v, T=45.0, ramp=12.0, offset=0.0)
with open('data/speed_limit.csv', 'w', newline='') as f: w = csv.DictWriter(f, R[0].keys()); w.writeheader(); w.writerows(R)
