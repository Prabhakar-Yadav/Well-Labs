# Build training patches+masks from hand-traced canal centerlines (densified).
import os, sys, glob, random, numpy as np, rasterio, cv2, shapefile
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

OUT='New/train_lines'; os.makedirs(f'{OUT}/img',exist_ok=True); os.makedirs(f'{OUT}/msk',exist_ok=True)
for d in ('img','msk'):
    for f in glob.glob(f'{OUT}/{d}/*.png'): os.remove(f)
PS=512; SHP='TLBC_D95_Canal/TLBC_D95.shp'
HALF_W={'TLBC':3.0,'Distributary-95':2.0}; DEFAULT_HW=2.5
STEP_PX=224; CAP=160; NEG_FRAC=0.30
random.seed(1)
ORTHOS={t:(glob.glob(f'_keep_locally/TLBC_D95_{t}_ortho.tif')+glob.glob(f'TLBC_D95_{t}_ortho.tif'))[0]
        for t in ['ATTANUR_2','ATTANUR_3','SHAKAPUR']}      # only tiles with canal inside
r=shapefile.Reader(SHP)
feat=[(sr.record[0],np.array(sr.shape.points,float)) for sr in r.iterShapeRecords()]

n_pos=n_neg=0
for tag,opath in ORTHOS.items():
    with rasterio.open(opath) as s:
        H,W=s.height,s.width; tr=s.transform; px_m=abs(tr.a)
        segs=[]
        for name,pts in feat:
            xu,yu=warp_transform('EPSG:4326',s.crs,pts[:,0].tolist(),pts[:,1].tolist())
            rows,cols=rasterio.transform.rowcol(tr,xu,yu)
            hw=max(2,int(HALF_W.get(name,DEFAULT_HW)/px_m))
            segs.append((np.column_stack([cols,rows]).astype(float),hw))   # (x,y)
        # densify along each segment
        centres=[]
        for pix,hw in segs:
            for i in range(len(pix)-1):
                p0,p1=pix[i],pix[i+1]; d=np.hypot(*(p1-p0))
                if d<1: continue
                for k in range(max(1,int(d/STEP_PX))):
                    x,y=p0+(p1-p0)*(k/max(1,int(d/STEP_PX)))
                    if PS//2<=x<W-PS//2 and PS//2<=y<H-PS//2: centres.append((int(y),int(x)))
        random.shuffle(centres); centres=centres[:CAP]
        saved=0
        for cy,cx in centres:
            r0=min(max(0,cy-PS//2),H-PS); c0=min(max(0,cx-PS//2),W-PS)
            d=s.read([1,2,3,4],window=Window(c0,r0,PS,PS))
            if d.shape[1]<PS or d.shape[2]<PS or (d[3]==255).mean()<0.4: continue
            mask=np.zeros((PS,PS),np.uint8)
            for pix,hw in segs:
                local=[(int(x-c0),int(y-r0)) for x,y in pix if -hw<=x-c0<PS+hw and -hw<=y-r0<PS+hw]
                if len(local)>1: cv2.polylines(mask,[np.array(local,np.int32)],False,255,max(3,2*hw))
            fr=(mask>0).mean()
            if fr<0.01 or fr>0.7: continue
            nm=f'{tag}_{cy}_{cx}'
            cv2.imwrite(f'{OUT}/img/{nm}.png',cv2.cvtColor(np.transpose(d[:3],(1,2,0)),cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'{OUT}/msk/{nm}.png',mask); saved+=1; n_pos+=1
        # negatives: random valid patches with NO canal
        want_neg=int(saved*NEG_FRAC); got=0; tries=0
        while got<want_neg and tries<want_neg*8:
            tries+=1
            cy=random.randint(PS,H-PS); cx=random.randint(PS,W-PS)
            r0,c0=cy-PS//2,cx-PS//2
            d=s.read([1,2,3,4],window=Window(c0,r0,PS,PS))
            if (d[3]==255).mean()<0.7: continue
            empty=True
            for pix,hw in segs:
                if any(c0-hw<=x<c0+PS+hw and r0-hw<=y<r0+PS+hw for x,y in pix): empty=False; break
            if not empty: continue
            nm=f'{tag}_neg_{cy}_{cx}'
            cv2.imwrite(f'{OUT}/img/{nm}.png',cv2.cvtColor(np.transpose(d[:3],(1,2,0)),cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'{OUT}/msk/{nm}.png',np.zeros((PS,PS),np.uint8)); got+=1; n_neg+=1
    print(f'{tag}: pos={saved} neg={got}')
print(f'TOTAL: {n_pos} canal + {n_neg} negative = {n_pos+n_neg} patches')
