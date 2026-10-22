import sys, json
from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.TopoDS import TopoDS
r=STEPControl_Reader(); r.ReadFile(sys.argv[1]); r.TransferRoots()
shape=r.OneShape()
ex=TopExp_Explorer(shape,TopAbs_SOLID); out=[]; i=0
while ex.More():
    s=TopoDS.Solid(ex.Current()); b=Bnd_Box(); BRepBndLib.Add_s(s,b)
    g=GProp_GProps(); BRepGProp.VolumeProperties_s(s,g); c=g.CentreOfMass()
    x0,y0,z0=b.CornerMin().X(),b.CornerMin().Y(),b.CornerMin().Z(); x1,y1,z1=b.CornerMax().X(),b.CornerMax().Y(),b.CornerMax().Z()
    out.append(dict(i=i,vol=g.Mass(),c=[c.X(),c.Y(),c.Z()],lo=[x0,y0,z0],hi=[x1,y1,z1]))
    i+=1; ex.Next()
json.dump(out,open(sys.argv[2],'w'))
print(len(out),"solids")
for o in out: print(o['i'],"vol=%.3e"%o['vol'],"size=",[round(h-l,1) for l,h in zip(o['lo'],o['hi'])],"c=",[round(v,1) for v in o['c']])
