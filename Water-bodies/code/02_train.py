# One model, Sentinel-2 only. UNet + ResNet34, 6-band input. Reports the SAME
# metrics (P/R/F1/IoU + ponds-found) on both TRAIN and TEST every few epochs.
#   python Water-bodies/code/02_train.py [epochs]
import sys, os, numpy as np, torch, torch.nn as nn
import segmentation_models_pytorch as smp
from scipy import ndimage
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
BATCH, LR, STEPS = 16, 3e-4, 50
OUT = 'Water-bodies/results/model'; os.makedirs(OUT, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0); np.random.seed(0)
d = np.load('Water-bodies/data/pond_dataset/dataset.npz', allow_pickle=True)
X, Y, SP = d['X'].astype('float32') / 10000.0, d['Y'], d['split']
tr, te = SP == 'train', SP == 'test'
mu = X[tr].mean((0, 2, 3), keepdims=True); sd = X[tr].std((0, 2, 3), keepdims=True) + 1e-6
Xn = (X - mu) / sd
Xtr, Ytr, Xte, Yte = Xn[tr], Y[tr], Xn[te], Y[te]
print(f'train {len(Xtr)} test {len(Xte)} | pond px train {100*Ytr.mean():.2f}% test {100*Yte.mean():.2f}%', flush=True)
def batch(n):
    idx = np.random.randint(0, len(Xtr), n); xs, ys = [], []
    for i in idx:
        x, y = Xtr[i], Ytr[i]; k = np.random.randint(4)
        x = np.rot90(x, k, (1, 2)).copy(); y = np.rot90(y, k, (0, 1)).copy()
        if np.random.rand() < 0.5: x = x[:, :, ::-1].copy(); y = y[:, ::-1].copy()
        xs.append(x); ys.append(y.astype('float32'))
    return torch.from_numpy(np.stack(xs)).float().to(dev), torch.from_numpy(np.stack(ys)).float().unsqueeze(1).to(dev)
model = smp.Unet('resnet34', encoder_weights='imagenet', in_channels=6, classes=1).to(dev)
dice = smp.losses.DiceLoss(mode='binary'); bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0], device=dev))
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
@torch.no_grad()
def ev(Xs, Ys):
    model.eval(); pr = []
    for i in range(0, len(Xs), 32):
        xb = torch.from_numpy(Xs[i:i+32]).float().to(dev); pr.append(torch.sigmoid(model(xb))[:, 0].cpu().numpy())
    prob = np.concatenate(pr); best = None
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        p = prob > t; g = Ys > 0; tp = (p & g).sum(); fp = (p & ~g).sum(); fn = (~p & g).sum()
        P = tp/max(tp+fp, 1); R = tp/max(tp+fn, 1); F = 2*P*R/max(P+R, 1e-9); I = tp/max(tp+fp+fn, 1)
        fo = to = 0
        for k in range(len(Ys)):
            lb, n = ndimage.label(g[k])
            for j in range(1, n+1):
                to += 1; fo += 1 if p[k][lb == j].any() else 0
        if best is None or F > best[3]: best = (t, P, R, F, I, f'{fo}/{to}')
    return best
bf = -1
for ep in range(1, EPOCHS+1):
    model.train(); tot = 0
    for _ in range(STEPS):
        x, y = batch(BATCH); lo = model(x); l = dice(lo, y) + bce(lo, y)
        opt.zero_grad(); l.backward(); opt.step(); tot += l.item()
    sch.step()
    if ep % 5 == 0 or ep == 1:
        (tt, Pt, Rt, Ft, It, fot) = ev(Xtr, Ytr); (tv, Pv, Rv, Fv, Iv, fov) = ev(Xte, Yte)
        s = ''
        if Fv > bf: bf = Fv; torch.save(model.state_dict(), f'{OUT}/pond_model.pt'); np.savez(f'{OUT}/norm.npz', mu=mu, sd=sd); s = ' *'
        print(f'ep{ep:3d} L{tot/STEPS:.3f} | TRAIN F1={Ft:.3f} IoU={It:.3f} ponds={fot} | TEST F1={Fv:.3f} IoU={Iv:.3f} P={Pv:.3f} R={Rv:.3f} ponds={fov}{s}', flush=True)
print(f'BEST test F1={bf:.3f} -> {OUT}/pond_model.pt', flush=True)
