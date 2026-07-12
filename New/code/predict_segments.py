# Step 3: predict blockage class (4-class) + blockage% for each aligned TLBC segment.
# Corridor = +/- half the canal width around the (aligned) line. Fills 'predicted' and
# 'blockpct'; leaves 'actual' for manual check.  python predict_segments.py
import glob, numpy as np, rasterio, cv2, shapefile
from rasterio.windows import Window
SEG='gis/TLBC_alignment/tlbc_segments'
ORTHOS=['_keep_locally/TLBC_D95_ATTANUR_2_ortho.tif','_keep_locally/TLBC_D95_ATTANUR_3_ortho.tif',
        '_keep_locally/TLBC_D95_ATTANUR_1_ortho.tif','_keep_locally/TLBC_D95_ATTANUR_4_ortho.tif',
        '_keep_locally/TLBC_D95_SHAKAPUR_ortho.tif']
def classify(pct):                       # 4-class by blockage %
    if pct<=5:  return 'clear'
    if pct<=30: return 'partial'
    if pct<=80: return 'moderate'
    return 'full'
def veg_frac(rgb,mask):
    cp=mask>0
    if cp.sum()<20: return None
    r,g,b=[rgb[:,:,i].astype(np.float32) for i in range(3)]
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV); h,s,v=[hsv[:,:,i].astype(np.float32) for i in range(3)]
    exg=2*g-r-b; vari=(g-r)/(g+r-b+1e-6); green=(h>=35)&(h<=85)&(s>40)&(v>50)
    veg=((exg>25)|((vari>0.1)&green))&cp
    n=max(1,int(cp.sum())); return float(veg.sum()/n)

r=shapefile.Reader(SEG+'.shp'); flds=[x[0] for x in r.fields[1:]]; fields_def=list(r.fields[1:])
segs=[]
for sr in r.shapeRecords():
    segs.append(dict(rec=dict(zip(flds,sr.record)), pts=np.array(sr.shape.points,float), veg=None))
r.close()
print('segments:',len(segs),flush=True)
for op in ORTHOS:
    if not glob.glob(op): print('missing',op); continue
    todo=[s for s in segs if s['veg'] is None]
    if not todo: break
    with rasterio.open(op) as src:
        H,W=src.height,src.width; tr=src.transform; px_m=abs(tr.a); b=src.bounds
        n0=sum(1 for s in segs if s['veg'] is not None)
        for sg in todo:
            pts=sg['pts']; cx,cy=pts[:,0].mean(),pts[:,1].mean()
            if not (b.left<=cx<=b.right and b.bottom<=cy<=b.top): continue
            hw=max(2,int((0.5+sg['rec']['width']/2)/px_m))   # corridor = 0.5 m + half-width
            rr,cc=rasterio.transform.rowcol(tr,pts[:,0].tolist(),pts[:,1].tolist()); rr,cc=np.array(rr),np.array(cc)
            ins=(rr>=0)&(rr<H)&(cc>=0)&(cc<W); rr,cc=rr[ins],cc[ins]
            if len(rr)<2: continue
            r0=max(0,rr.min()-hw); r1=min(H,rr.max()+hw); a0=max(0,cc.min()-hw); a1=min(W,cc.max()+hw)
            if r1-r0<4 or a1-a0<4: continue
            d=src.read([1,2,3,4],window=Window(a0,r0,a1-a0,r1-r0)); valid=d[3]==255
            if valid.mean()<0.3: continue
            rgb=np.transpose(d[:3],(1,2,0)); mk=np.zeros(rgb.shape[:2],np.uint8)
            loc=np.array([(cc[i]-a0,rr[i]-r0) for i in range(len(rr))],np.int32)
            cv2.polylines(mk,[loc],False,255,2*hw); mk=((mk>0)&valid).astype(np.uint8)
            v=veg_frac(rgb,mk)
            if v is not None: sg['veg']=v
        print(f'{op.split("/")[-1]}: sampled {sum(1 for s in segs if s["veg"] is not None)-n0} more',flush=True)

# rewrite shapefile: same geometry (aligned) + filled predicted/blockpct, keep actual
w=shapefile.Writer(SEG,shapeType=shapefile.POLYLINE)
for fd in fields_def: w.field(*fd)
import collections; dist=collections.Counter(); nulls=0
for sg in segs:
    d=sg['rec']; pts=sg['pts']
    if sg['veg'] is None: pred=''; pct=0; nulls+=1
    else: pct=int(round(sg['veg']*100)); pred=classify(pct); dist[pred]+=1
    w.line([[[float(x),float(y)] for x,y in pts]])
    w.record(d['id'],d['canal'],d['ctype'],d['width'],d['chainage'],pred,d.get('actual','') or '',pct)
w.close()
import os
if os.path.exists(SEG+'.prj')==False: open(SEG+'.prj','w').write(open('gis/TLBC_alignment/tlbc_segments.prj').read())
print('predicted distribution:',dict(dist),'| not-sampled:',nulls)
print('done -> updated',SEG+'.shp')
