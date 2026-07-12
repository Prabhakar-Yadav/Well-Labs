# Fill silt gaps along the traced canal using CHM = DSM - DTM, where the old canal
# mask missed the channel but DSM/DTM data exists. Keeps existing silt; fills blanks
# inside a narrow channel buffer (+/- CH_M). Class thresholds learned from existing silt.
import numpy as np, rasterio, cv2, shapefile
from rasterio import Affine
from rasterio.warp import transform as wt, reproject, Resampling
from rasterio.features import rasterize
from shapely.geometry import LineString
from shapely.ops import unary_union
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CH_M=5.0; TARGET=2600; SILT='outputs/08_silt'; DD='_keep_locally/TLBC_DSM_DTM'
TILES={'ATTANUR_2':'TLBC_D95_ATTANUR_2_ortho.tif','ATTANUR_3':'_keep_locally/TLBC_D95_ATTANUR_3_ortho.tif'}
COL={1:(0,190,0),2:(255,165,0),3:(220,20,20)}
lines=[np.array(s.points,float) for s in shapefile.Reader('TLBC_D95_Canal/TLBC_D95.shp').shapes()]

def reproj(p,h2,w2,tr,crs):
    with rasterio.open(p) as d:
        Hd,Wd=d.height,d.width; dd=max(1,max(Hd,Wd)//TARGET); hh,ww=Hd//dd,Wd//dd
        arr=d.read(1,out_shape=(hh,ww)).astype(np.float32); st=d.transform*Affine.scale(Wd/ww,Hd/hh)
        out=np.full((h2,w2),-32767.0,np.float32)
        reproject(arr,out,src_transform=st,src_crs=d.crs,dst_transform=tr,dst_crs=crs,src_nodata=-32767,dst_nodata=-32767,resampling=Resampling.bilinear)
        return out
def chan_mask(h2,w2,tr,crs,buf):
    g=[]
    for pts in lines:
        xu,yu=wt('EPSG:4326',crs,pts[:,0].tolist(),pts[:,1].tolist())
        if len(xu)>=2: g.append(LineString(list(zip(xu,yu))).buffer(buf))
    return rasterize([(unary_union(g),1)],out_shape=(h2,w2),transform=tr,fill=0,dtype='uint8').astype(bool)

for tag,op in TILES.items():
    with rasterio.open(f'{SILT}/{tag}_silt_refined.tif') as s:   # off-canal already removed
        H,W=s.height,s.width; ds=max(1,max(H,W)//TARGET); h2,w2=H//ds,W//ds
        silt=s.read(1,out_shape=(h2,w2),resampling=Resampling.nearest); tr=s.transform*Affine.scale(W/w2,H/h2); crs=s.crs
    dsm=reproj(f'{DD}/TLBC_D95_{tag}_dsm.tif',h2,w2,tr,crs); dtm=reproj(f'{DD}/TLBC_D95_{tag}_dtm.tif',h2,w2,tr,crs)
    valid=(dsm!=-32767)&(dtm!=-32767); chm=np.where(valid,dsm-dtm,np.nan)
    def med(c):
        v=chm[(silt==c)&valid]; return float(np.nanmedian(v)) if v.size>20 else None
    m1,m2,m3=med(1),med(2),med(3)
    t12=(m1+m2)/2 if (m1 is not None and m2 is not None) else 0.2
    t23=(m2+m3)/2 if (m2 is not None and m3 is not None) else 0.5
    chan=chan_mask(h2,w2,tr,crs,CH_M)
    fill=chan&(silt==0)&valid&(chm>-1)&(chm<5)   # sane depth range
    cls=np.where(chm<t12,1,np.where(chm<t23,2,3)).astype(np.uint8)
    filled=silt.copy(); filled[fill]=cls[fill]
    print(f'{tag}: t12={t12:.2f} t23={t23:.2f} | existing {int((silt>0).sum())} px + filled {int(fill.sum())} channel px',flush=True)
    with rasterio.open(op) as o:
        Ho,Wo=o.height,o.width; vs=max(1,max(Ho,Wo)//2200); vh,vw=Ho//vs,Wo//vs
        rgb=np.transpose(o.read([1,2,3],out_shape=(3,vh,vw)),(1,2,0)).astype(np.float32)
        for c in range(3):
            p=rgb[:,:,c]; lo,hi=np.percentile(p,[2,98]); rgb[:,:,c]=np.clip((p-lo)/(hi-lo+1e-6)*255,0,255)
        canvas=rgb.astype(np.uint8); dtr=o.transform*Affine.scale(Wo/vw,Ho/vh)
    sr=np.zeros((vh,vw),np.uint8)
    reproject(filled,sr,src_transform=tr,src_crs=crs,dst_transform=dtr,dst_crs=crs,resampling=Resampling.nearest)
    for c2,col in COL.items():
        m=cv2.dilate((sr==c2).astype(np.uint8),np.ones((3,3),np.uint8),iterations=2).astype(bool); canvas[m]=col
    LAB={1:f'Low (<{t12:.1f}m)',2:f'Medium ({t12:.1f}-{t23:.1f}m)',3:f'High (>{t23:.1f}m)'}
    fig,ax=plt.subplots(figsize=(9,int(9*vh/vw)+1)); ax.imshow(canvas)
    ax.set_title(f'{tag} — Silt Detection (gaps filled from DSM-DTM along canal)',fontsize=12); ax.axis('off')
    ax.legend(handles=[mpatches.Patch(color=np.array(COL[k])/255,label=LAB[k]) for k in COL],loc='lower right',fontsize=9)
    plt.tight_layout(); plt.savefig(f'{SILT}/{tag}_silt_filled.png',dpi=150,bbox_inches='tight'); plt.close()
    print(f'   -> {SILT}/{tag}_silt_filled.png',flush=True)
print('done')
