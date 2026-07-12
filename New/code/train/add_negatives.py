# Add many non-canal (farmland) negative patches so the model learns what ISN'T a canal.
import os, glob, random, numpy as np, rasterio, cv2
from rasterio.windows import Window
OUT='New/train_lines'; PS=512; random.seed(7)
# remove old negatives only (keep the 22 positives)
for d in ('img','msk'):
    for f in glob.glob(f'{OUT}/{d}/*_neg_*.png'): os.remove(f)
# ATTANUR_1 & 4 have NO canal inside -> every valid patch is a clean negative
SRC={'ATTANUR_1':70,'ATTANUR_4':50}
tot=0
for tag,want in SRC.items():
    fs=glob.glob(f'_keep_locally/TLBC_D95_{tag}_ortho.tif')+glob.glob(f'TLBC_D95_{tag}_ortho.tif')
    if not fs: continue
    with rasterio.open(fs[0]) as s:
        H,W=s.height,s.width; got=0; tries=0
        while got<want and tries<want*10:
            tries+=1
            cy=random.randint(0,H-PS); cx=random.randint(0,W-PS)
            d=s.read([1,2,3,4],window=Window(cx,cy,PS,PS))
            if (d[3]==255).mean()<0.85: continue
            nm=f'{tag}_neg_{cy}_{cx}'
            cv2.imwrite(f'{OUT}/img/{nm}.png',cv2.cvtColor(np.transpose(d[:3],(1,2,0)),cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'{OUT}/msk/{nm}.png',np.zeros((PS,PS),np.uint8)); got+=1; tot+=1
        print(f'{tag}: +{got} negatives')
pos=len(glob.glob(f'{OUT}/img/*.png'))-tot
print(f'TOTAL now: {pos} canal + {tot} negatives')
