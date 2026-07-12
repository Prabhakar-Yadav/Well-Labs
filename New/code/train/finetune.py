# Fine-tune canal model on patches built from the hand-traced lines.
import os, glob, numpy as np, cv2, torch, torch.nn as nn
import segmentation_models_pytorch as smp

DATA='New/train_lines'
SRC='outputs/05b_binary/best_binary_unet_v5.pth'
DST='outputs/05b_binary/best_binary_unet_v5_lines.pth'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS,BATCH,LR=25,4,5e-5
MEAN=torch.tensor([0.485,0.456,0.406]).view(3,1,1); STD=torch.tensor([0.229,0.224,0.225]).view(3,1,1)

X,Y=[],[]
for pf in sorted(glob.glob(f'{DATA}/img/*.png')):
    img=cv2.cvtColor(cv2.imread(pf),cv2.COLOR_BGR2RGB)
    m=cv2.imread(os.path.join(DATA,'msk',os.path.basename(pf)),0)
    X.append(img); Y.append((m>127).astype(np.float32))
n_pos=sum(1 for y in Y if y.sum()>0)
print(f'{len(X)} patches ({n_pos} canal, {len(X)-n_pos} negative) | device={DEVICE}')

def aug(im,mk):
    if np.random.rand()<.5: im,mk=im[:,::-1],mk[:,::-1]
    if np.random.rand()<.5: im,mk=im[::-1],mk[::-1]
    k=np.random.randint(4); im,mk=np.rot90(im,k),np.rot90(mk,k)
    if np.random.rand()<.4: im=np.clip(im.astype(np.float32)*np.random.uniform(.8,1.2),0,255).astype(np.uint8)
    return np.ascontiguousarray(im),np.ascontiguousarray(mk)
pos_idx=[i for i,y in enumerate(Y) if y.sum()>0]; neg_idx=[i for i,y in enumerate(Y) if y.sum()==0]
def batch(bs):
    half=max(1,bs//2)
    idx=list(np.random.choice(pos_idx,half))+list(np.random.choice(neg_idx,bs-half)); ims=[];mks=[]
    for i in idx:
        a,b=aug(X[i],Y[i])
        t=torch.from_numpy(a/255.).permute(2,0,1).float(); t=(t-MEAN)/STD
        ims.append(t); mks.append(torch.from_numpy(b.copy()).float())
    return torch.stack(ims).to(DEVICE),torch.stack(mks).unsqueeze(1).to(DEVICE)

model=smp.Unet('resnet50',encoder_weights=None,in_channels=3,classes=1).to(DEVICE)
model.load_state_dict(torch.load(SRC,map_location=DEVICE,weights_only=True))
opt=torch.optim.Adam(model.parameters(),lr=LR); bce=nn.BCEWithLogitsLoss()
sc=torch.amp.GradScaler('cuda',enabled=(DEVICE=='cuda'))
def dice(l,t):
    p=torch.sigmoid(l).flatten(); t=t.flatten(); return 1-(2*(p*t).sum()+1)/(p.sum()+t.sum()+1)
steps=max(10,len(X)//BATCH); model.train()
for ep in range(EPOCHS):
    tot=0
    for _ in range(steps):
        im,mk=batch(BATCH); opt.zero_grad()
        with torch.amp.autocast('cuda',enabled=(DEVICE=='cuda')):
            out=model(im); loss=bce(out,mk)+dice(out,mk)
        sc.scale(loss).backward(); sc.step(opt); sc.update(); tot+=loss.item()
    if ep%10==0 or ep==EPOCHS-1: print(f'  epoch {ep+1}/{EPOCHS} loss={tot/steps:.4f}',flush=True)
torch.save(model.state_dict(),DST)
print('saved fine-tuned model ->',DST)
