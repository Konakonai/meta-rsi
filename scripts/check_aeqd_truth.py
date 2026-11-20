"""Check that the AEQD ground truth of local tasks matches ellipsoidal measures.

Places 1,000 random polygons and lines (about 20 km across) at latitudes 0-64 deg with the same
AEQD placement as the task builder, then compares planar area, length and point distance in the
reference AEQD projection (centred on the bounding box) with pyproj.Geod on WGS84.

Run: nix-shell --run "python -m scripts.check_aeqd_truth"
"""
import numpy as np, shapely, geopandas as gpd
from pyproj import Geod
from benchmarks.gsb.geo import place, local_crs
G=Geod(ellps="WGS84"); rng=np.random.default_rng(0)
worst={"area":0,"length":0,"dist":0}
for lat in [0,30,45,60,64]:
  for k in range(200):
    lon=rng.uniform(-170,170)
    # random polygon within a 20 km box, off-centre like real data
    off=rng.uniform(-5000,5000,2)
    pts=rng.uniform(-10000,10000,(12,2))+off
    poly=shapely.convex_hull(shapely.MultiPoint(pts))
    line=shapely.LineString(pts[:6])
    g=gpd.GeoDataFrame(geometry=place([poly,line],lat,lon),crs=4326)
    b=g.total_bounds; crs=local_crs((b[1]+b[3])/2,(b[0]+b[2])/2)
    q=g.to_crs(crs)
    a_e,_=G.geometry_area_perimeter(g.geometry[0]); l_e=G.geometry_length(g.geometry[1])
    worst["area"]=max(worst["area"],abs(q.geometry[0].area/abs(a_e)-1))
    worst["length"]=max(worst["length"],abs(q.geometry[1].length/l_e-1))
    p=g.geometry[1].coords; x=np.array(p)
    d_e=G.inv(x[0,0],x[0,1],x[-1,0],x[-1,1])[2]
    qa=np.array(q.geometry[1].coords); d_p=np.hypot(*(qa[0]-qa[-1]))
    worst["dist"]=max(worst["dist"],abs(d_p/d_e-1))
print({k:f"{v*100:.5f}%" for k,v in worst.items()})
