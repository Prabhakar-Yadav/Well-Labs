# Clip EXISTING silt rasters to a corridor around the traced canal line, removing
# off-canal detections. Does NOT recompute silt (no re-validation needed).
# Writes  *_silt_refined.tif  and  *_silt_refined_map.png  (before/after).  python refine_silt.py
import numpy as np, rasterio, cv2, shapefile
from rasterio import Affine
from rasterio.warp import transform as wt, reproject, Resampling
from rasterio.features import rasterize
from shapely.geometry import LineString
from shapely.ops import unary_union
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CORRIDOR_M=25.0; VIS=2000; SILT='outputs/08_silt'
TILES={'ATTANUR_2':'TLBC_D95_ATTANUR_2_ortho.tif','ATTANUR_3':'_keep_locally/TLBC_D95_ATTANUR_3_ortho.tif'}
COL={1:(0,190,0),2:(255,165,0),3:(220,20,20)}; LAB={1:'Low (<0.5m)',2:'Medium (0.5-1m)',3:'High (>1m)'}
lines=[np.array(s.points,float) for s in shapefile.Reader('TLBC_D95_Canal/TLBC_D95.shp').shapes()]

def corridor(crs,transform,shape):
    geoms=[]
    for pts in lines:
        xu,yu=wt('EPSG:4326',crs,pts[:,0].tolist(),pts[:,1].tolist())
        if len(xu)>=2: geoms.append(LineString(list(zip(xu,yu))).buffer(CORRIDOR_M))
    return rasterize([(unary_union(geoms),1)],out_shape=shape,transform=transform,fill=0,dtype='uint8').astype(bool)

def overlay(canvas,silt_grid,src_tr,crs,dst_tr,vh,vw):
    sr=np.zeros((vh,vw),np.uint8)
    reproject(silt_grid,sr,src_transform=src_tr,src_crs=crs,dst_transform=dst_tr,dst_crs=crs,resampling=Resampling.nearest)
    out=canvas.copy()
    for cls,col in COL.items():
        m=cv2.dilate((sr==cls).astype(np.uint8),np.ones((3,3),np.uint8),iterations=2).astype(bool)
        out[m]=col
    return out

for tag,op in TILES.items():
    with rasterio.open(f'{SILT}/{tag}_silt.tif') as s:
        silt=s.read(1); tr=s.transform; crs=s.crs; prof=s.profile
    m=corridor(crs,tr,silt.shape)
    before=int((silt>0).sum()); refined=np.where(m,silt,0).astype(silt.dtype); after=int((refined>0).sum())
    prof.update(compress='deflate')
    with rasterio.open(f'{SILT}/{tag}_silt_refined.tif','w',**prof) as d: d.write(refined,1)
    print(f'{tag}: on-canal-corridor kept {after} of {before} silt px  (removed {before-after} off-canal = {100*(before-after)/max(1,before):.0f}%)',flush=True)
    # before/after map on the ortho
    with rasterio.open(op) as o:
        H,W=o.height,o.width; vs=max(1,max(H,W)//VIS); vh,vw=H//vs,W//vs
        rgb=np.transpose(o.read([1,2,3],out_shape=(3,vh,vw)),(1,2,0)).astype(np.float32)
        for c in range(3):
            p=rgb[:,:,c]; lo,hi=np.percentile(p,[2,98]); rgb[:,:,c]=np.clip((p-lo)/(hi-lo+1e-6)*255,0,255)
        canvas=rgb.astype(np.uint8); dst_tr=o.transform*Affine.scale(W/vw,H/vh)
    left=overlay(canvas,silt,tr,crs,dst_tr,vh,vw)
    right=overlay(canvas,refined,tr,crs,dst_tr,vh,vw)
    fig,ax=plt.subplots(1,2,figsize=(16,int(9*vh/vw)+1))
    ax[0].imshow(left);  ax[0].set_title(f'{tag} — BEFORE (silt everywhere the old mask fired)',fontsize=11); ax[0].axis('off')
    ax[1].imshow(right); ax[1].set_title(f'{tag} — AFTER (clipped to canal corridor, {CORRIDOR_M:.0f} m)',fontsize=11); ax[1].axis('off')
    ax[1].legend(handles=[mpatches.Patch(color=np.array(COL[k])/255,label=LAB[k]) for k in COL],loc='lower right',fontsize=9)
    plt.tight_layout(); plt.savefig(f'{SILT}/{tag}_silt_refined_map.png',dpi=140,bbox_inches='tight'); plt.close()
    print(f'   map -> {SILT}/{tag}_silt_refined_map.png',flush=True)
print('DONE — refined rasters + before/after maps written. Silt VALUES unchanged (validation still valid).')
