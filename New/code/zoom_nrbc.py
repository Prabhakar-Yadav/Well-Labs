# Diagnostic: zoom into the NRBC canal to see what the corridor is sampling.
import numpy as np, rasterio, cv2, shapefile
from rasterio.warp import transform as wt
from rasterio.windows import Window

OP='New/NRBC_D10_Survey_1_ortho.tif'
SHP='New/NRBC_Canal/NRBC_D10.shp'
HALF_W_M=5.0; CROP_M=120
feats=[(sr.record[0],np.array(sr.shape.points,float)) for sr in shapefile.Reader(SHP).iterShapeRecords()]
# use the two main lines
mains=[f for f in feats if f[0] in ('NRBC canal','D-10pa')] or feats[:2]

tiles=[]
with rasterio.open(OP) as src:
    H,W=src.height,src.width; tr=src.transform; crs=src.crs; px_m=abs(tr.a)
    hw=int(HALF_W_M/px_m); cs=int(CROP_M/px_m)
    for fname,pts in mains:
        xu,yu=wt('EPSG:4326',crs,pts[:,0].tolist(),pts[:,1].tolist()); xu,yu=np.array(xu),np.array(yu)
        chain=np.concatenate([[0],np.cumsum(np.hypot(np.diff(xu),np.diff(yu)))]); total=chain[-1]
        for frac in (0.2,0.5,0.8):
            cm=frac*total; px=np.interp(cm,chain,xu); py=np.interp(cm,chain,yu)
            r,c=rasterio.transform.rowcol(tr,float(px),float(py))
            r0=min(max(0,r-cs//2),H-cs); c0=min(max(0,c-cs//2),W-cs)
            d=src.read([1,2,3,4],window=Window(c0,r0,cs,cs))
            if d.shape[1]<cs or d.shape[2]<cs: continue
            rgb=np.transpose(d[:3],(1,2,0)).astype(np.float32)
            for ch in range(3):
                lo,hi=np.percentile(rgb[:,:,ch],[2,98]); rgb[:,:,ch]=np.clip((rgb[:,:,ch]-lo)/(hi-lo+1e-6)*255,0,255)
            rgb=rgb.astype(np.uint8)
            # overlay line points + corridor in this crop
            sc=np.arange(max(0,cm-CROP_M),min(total,cm+CROP_M),1.0)
            sx=np.interp(sc,chain,xu); sy=np.interp(sc,chain,yu)
            rr,cc=rasterio.transform.rowcol(tr,sx.tolist(),sy.tolist()); rr,cc=np.array(rr),np.array(cc)
            loc=np.array([(cc[i]-c0,rr[i]-r0) for i in range(len(rr)) if 0<=cc[i]-c0<cs and 0<=rr[i]-r0<cs],np.int32)
            ov=rgb.copy()
            if len(loc)>1:
                corr=np.zeros((cs,cs),np.uint8); cv2.polylines(corr,[loc],False,255,2*hw)
                ov[corr>0]=(0.5*ov[corr>0]+0.5*np.array([255,0,0])).astype(np.uint8)  # corridor red
                cv2.polylines(ov,[loc],False,(255,255,0),2)                            # centerline yellow
            big=cv2.resize(np.hstack([rgb,ov]),(700,350))
            cv2.putText(big,f'{fname} @ {cm:.0f}m',(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
            tiles.append(big)
grid=np.vstack(tiles)
cv2.imwrite('New/results/_nrbc_zoom_check.png',cv2.cvtColor(grid,cv2.COLOR_RGB2BGR))
print('saved _nrbc_zoom_check.png with',len(tiles),'crops')
