"""Derive geometric parameters from the STEP-solid summary (cad/solids.json, mm, CAD frame:
X length, Y width, Z up). Heuristics (checked by eye against the CAD): wheels are the
near-square-in-XZ solids with diameter >= 255 mm; the slewing bearing is the ~121 mm disc
nearest the origin. Writes values with src: cad into params.yaml (other keys untouched)."""
import json, sys, re, numpy as np
S=json.load(open('cad/solids.json'))
sz=lambda s:[h-l for l,h in zip(s['lo'],s['hi'])]
ctr=lambda s:[(l+h)/2 for l,h in zip(s['lo'],s['hi'])]
W=[s for s in S if sz(s)[0]>255 and sz(s)[2]>255 and abs(sz(s)[0]-sz(s)[2])<4]
assert len(W)==4,len(W)
bog=[w for w in W if ctr(w)[0]>-300]; fix=[w for w in W if ctr(w)[0]<=-300]
assert len(bog)==2 and len(fix)==2
B=[s for s in S if abs(sz(s)[0]-121.2)<1 and abs(sz(s)[1]-121.2)<1 and sz(s)[2]<25]
assert len(B)==1; px,py,pz=ctr(B[0])
r=np.mean([sz(w)[0]/2 for w in W])/1000
rs=[sz(w)[0]/2000 for w in W]
track_b=abs(ctr(bog[0])[1]-ctr(bog[1])[1])/1000
track_f=abs(ctr(fix[0])[1]-ctr(fix[1])[1])/1000
xb=np.mean([ctr(w)[0] for w in bog]); xf=np.mean([ctr(w)[0] for w in fix])
ell=(px-xf)/1000                 # pivot to fixed-axle distance (pivot is at the bogie end)
e=(px-xb)/1000                   # signed: + means bogie axle lies toward the fixed (front) axle
zb=np.mean([ctr(w)[2] for w in bog]); zf=np.mean([ctr(w)[2] for w in fix])
out=dict(wheel_radius=float(round(r,4)),track_w=float(round(track_b,4)),front_track=float(round(track_f,4)),
         lever_l=float(round(ell,4)),pivot_trail_e=float(round(e,4))+0.0,pivot_offset_e=float(round(e,4))+0.0)
print("wheel radii (m):",[round(x,4) for x in rs])
print("pivot (mm):",round(px,1),round(py,1),round(pz,1),"| bogie axle x",round(xb,1),"fixed axle x",round(xf,1))
print("bogie axle z",round(zb,1),"fixed axle z",round(zf,1),"-> level mismatch (mm):",round(zf-zb,1))
print("bearing height above bogie axle (mm):",round(pz-zb,1),"; above fixed axle:",round(pz-zf,1))
print(out)
if '--write' in sys.argv:
    t=open('params.yaml').read()
    for k,v in out.items():
        t=re.sub(rf'^({k}:\s*)\{{value: [^,]+,\s*src: [^}}]+\}}',rf'\g<1>{{value: {v}, src: cad}}',t,flags=re.M)
    open('params.yaml','w').write(t); print("params.yaml updated")
