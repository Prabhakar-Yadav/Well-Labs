# Compare original vs line-fine-tuned model on a canal crop of ATTANUR_3.
import glob, numpy as np, rasterio, cv2, torch, shapefile
import segmentation_models_pytorch as smp
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
MEAN=torch.tensor([0.485,0.456,0.406]).view(3,1,1); STD=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
TILE,STEP,THR=512,384,0.4
OPATH=glob.glob('_keep_locally/TLBC_D95_ATTANUR_3_ortho.tif')[0]

def load(p):
    m=smp.Unet('resnet50',encoder_weights=None,in_channels=3,classes=1).to(DEVICE)
    m.load_state_dict(torch.load(p,map_location=DEVICE,weights_only=True)); m.eval(); return m

def tta(model,t):
    ps=[]
    for k in range(4):
        x=torch.rot90(t,k,[2,3])
        with torch.no_grad(), torch.amp.autocast('cuda',enabled=(DEVICE=='cuda')):
            p=torch.sigmoid(model(x))
        ps.append(torch.rot90(p,-k,[2,3]))
    return torch.stack(ps).mean(0)

def infer(model, rgb, valid):
    H,W=rgb.shape[:2]; out=np.zeros((H,W),np.float32); cnt=np.zeros((H,W),np.float32)
    for r in range(0,max(1,H-TILE+1),STEP):
        for c in range(0,max(1,W-TILE+1),STEP):
            patch=rgb[r:r+TILE,c:c+TILE]
            if patch.shape[0]<TILE or patch.shape[1]<TILE: continue
            if valid[r:r+TILE,c:c+TILE].mean()<0.2: continue
            t=torch.from_numpy(patch/255.).permute(2,0,1).float(); t=((t-MEAN)/STD).unsqueeze(0).to(DEVICE)
            p=tta(model,t).cpu().numpy()[0,0]
            out[r:r+TILE,c:c+TILE]+=p; cnt[r:r+TILE,c:c+TILE]+=1
    return np.where(cnt>0,out/cnt,0)

# pick crop around canal in ATTANUR_3
r=shapefile.Reader('TLBC_D95_Canal/TLBC_D95.shp')
with rasterio.open(OPATH) as s:
    H,W=s.height,s.width; tr=s.transform
    pts_all=[]
    for sr in r.iterShapeRecords():
        p=np.array(sr.shape.points)
        xu,yu=warp_transform('EPSG:4326',s.crs,p[:,0].tolist(),p[:,1].tolist())
        rows,cols=rasterio.transform.rowcol(tr,xu,yu)
        for rr,cc in zip(rows,cols):
            if 0<=rr<H and 0<=cc<W: pts_all.append((rr,cc))
    pts_all=np.array(pts_all)
    # centre the crop on an actual canal+valid location (a positive patch spot)
    import os as _os
    posf=[f for f in glob.glob('New/train_lines/img/ATTANUR_3_*.png') if '_neg_' not in f]
    nm=_os.path.basename(posf[0])[:-4].split('_'); cy,cx=int(nm[-2]),int(nm[-1])
    CS=4096
    r0=min(max(0,cy-CS//2),H-CS); c0=min(max(0,cx-CS//2),W-CS)
    d=s.read([1,2,3,4],window=Window(c0,r0,CS,CS))
    rgb=np.transpose(d[:3],(1,2,0)); valid=(d[3]==255)
    gtline=[(int(cc-c0),int(rr-r0)) for rr,cc in pts_all if r0<=rr<r0+CS and c0<=cc<c0+CS]

m0=load('outputs/05b_binary/best_binary_unet_v5.pth')
m1=load('outputs/05b_binary/best_binary_unet_v5_lines.pth')
p0=infer(m0,rgb,valid); p1=infer(m1,rgb,valid)

# visualise (downsample for display)
def disp(rgb):
    v=cv2.resize(rgb,(1000,1000)).astype(np.float32)
    for c in range(3):
        lo,hi=np.percentile(v[:,:,c],[2,98]); v[:,:,c]=np.clip((v[:,:,c]-lo)/(hi-lo+1e-6)*255,0,255)
    return v.astype(np.uint8)
base=disp(rgb); sc=1000/CS
gt=base.copy()
if len(gtline)>1: cv2.polylines(gt,[(np.array(gtline)*sc).astype(np.int32)],False,(0,255,255),2)
def ov(p,col):
    o=base.copy(); pm=cv2.resize((p>THR).astype(np.uint8),(1000,1000)); o[pm>0]=col;
    return cv2.addWeighted(base,0.5,o,0.5,0)
panel=np.hstack([gt, ov(p0,(255,0,0)), ov(p1,(0,0,255))])
cv2.putText(panel,'GT line (yellow)',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
cv2.putText(panel,'ORIGINAL model',(1010,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
cv2.putText(panel,'FINE-TUNED model',(2010,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
cv2.imwrite('New/train_lines/_compare.png', cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
print('canal px: original=%d finetuned=%d'%((p0>THR).sum(),(p1>THR).sum()))
print('saved _compare.png')
