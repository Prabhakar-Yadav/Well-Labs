# Run the trained pond model across each full Sentinel-2 tile -> detected ponds
# (shapefile + marked overview). Windowed reads from JP2s (no big stacks).
#   python Water-bodies/code/03_inference.py [thr]
import sys, os, glob, numpy as np, torch, torch.nn as nn, rasterio, shapefile
import segmentation_models_pytorch as smp
from scipy import ndimage
from rasterio.enums import Resampling
from rasterio.features import shapes as rio_shapes
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
MIN_A = 800.0
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
SDIR = 'Water-bodies/data/Sentinel-2'
TILES = [('June_T43QGU', 'S2A_*T43QGU*'), ('Dec_T43QGU', 'S2C_*T43QGU*'), ('Dec_T43PGT', 'S2C_*T43PGT*')]
nz = np.load('Water-bodies/results/model/norm.npz'); MU, SD = nz['mu'], nz['sd']
model = smp.Unet('resnet34', encoder_weights=None, in_channels=6, classes=1)
model.load_state_dict(torch.load('Water-bodies/results/model/pond_model.pt', map_location='cpu', weights_only=False))
model = model.to(dev).eval()
def perim(ring):
    x = np.array([p[0] for p in ring]); y = np.array([p[1] for p in ring])
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))
for tname, pat in TILES:
    SAFE = glob.glob(f'{SDIR}/{pat}.SAFE')[0]
    G10 = glob.glob(f'{SAFE}/GRANULE/*/IMG_DATA/R10m')[0]; G20 = glob.glob(f'{SAFE}/GRANULE/*/IMG_DATA/R20m')[0]
    PFX = os.path.basename(glob.glob(f'{G10}/*_B02_10m.jp2')[0]).replace('_B02_10m.jp2', '')
    with rasterio.open(f'{G10}/{PFX}_B02_10m.jp2') as s:
        H, W = s.height, s.width; tr = s.transform; crs = s.crs
    b10 = [rasterio.open(f'{G10}/{PFX}_{c}_10m.jp2') for c in ['B02', 'B03', 'B04', 'B08']]
    b20 = [rasterio.open(f'{G20}/{PFX}_{c}_20m.jp2') for c in ['B11', 'B12']]
    with rasterio.open(f'{G20}/{PFX}_SCL_20m.jp2') as s:
        scl = s.read(1, out_shape=(H, W), resampling=Resampling.nearest)
    cloud = np.isin(scl, [0, 1, 2, 3, 8, 9, 10]); del scl
    prob = np.zeros((H, W), 'float32'); WIN = 1024
    for r0 in range(0, H, WIN):
        for c0 in range(0, W, WIN):
            h = min(WIN, H - r0); w = min(WIN, W - c0)
            win = rasterio.windows.Window(c0, r0, w, h)
            bands = [b.read(1, window=win).astype('float32') for b in b10]
            for b in b20:
                bands.append(b.read(1, window=rasterio.windows.Window(c0 // 2, r0 // 2, (w + 1) // 2, (h + 1) // 2),
                                    out_shape=(h, w), resampling=Resampling.bilinear).astype('float32'))
            arr = np.stack(bands) / 10000.0
            if arr.mean() < 0.01:
                continue
            arr = (arr - MU[0]) / SD[0]
            x = torch.from_numpy(arr).float().unsqueeze(0).to(dev)
            ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
            x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode='reflect')
            with torch.no_grad():
                p = torch.sigmoid(model(x))[0, 0, :h, :w].cpu().numpy()
            prob[r0:r0 + h, c0:c0 + w] = p
    for b in b10 + b20:
        b.close()
    prob[cloud] = 0
    mask = (prob >= THR).astype('uint8')
    OUT = f'Water-bodies/results/predictions/{tname}'; os.makedirs(OUT, exist_ok=True)
    w = shapefile.Writer(f'{OUT}/ponds', shapeType=shapefile.POLYGON)
    w.field('id', 'N', 8); w.field('area_m2', 'F', 16, 1); w.field('area_ha', 'F', 12, 4); w.field('circum_m', 'F', 12, 1)
    dets = []
    for geom, _ in rio_shapes(mask, mask=mask.astype(bool), transform=tr):
        ring = geom['coordinates'][0]; x = np.array([p[0] for p in ring]); y = np.array([p[1] for p in ring])
        a = abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])) / 2.0
        if a < MIN_A:
            continue
        dets.append((a, perim(ring), np.array(ring)))
    dets.sort(key=lambda z: -z[0])
    for i, (a, pr, ring) in enumerate(dets, 1):
        w.poly([[list(pt) for pt in ring]]); w.record(i, round(a, 1), round(a / 1e4, 4), round(pr, 1))
    w.close(); open(f'{OUT}/ponds.prj', 'w').write(crs.to_wkt())
    # overview
    DS = 10
    with rasterio.open(f'{G10}/{PFX}_B04_10m.jp2') as s: R_ = s.read(1, out_shape=(H//DS, W//DS)).astype('float32')
    with rasterio.open(f'{G10}/{PFX}_B03_10m.jp2') as s: G_ = s.read(1, out_shape=(H//DS, W//DS)).astype('float32')
    with rasterio.open(f'{G10}/{PFX}_B02_10m.jp2') as s: B_ = s.read(1, out_shape=(H//DS, W//DS)).astype('float32')
    rgb = np.dstack([R_, G_, B_]); lo, hi = np.percentile(rgb[rgb.sum(2) > 0], [2, 98]); rgb = np.clip((rgb-lo)/(hi-lo), 0, 1)
    a_, e_, c_, f_ = tr.a*DS, tr.e*DS, tr.c, tr.f
    plt.figure(figsize=(14, 14)); plt.imshow(rgb); plt.axis('off')
    plt.title(f'{tname} — {len(dets)} detected ponds', fontsize=13)
    for a, pr, ring in dets:
        px = (ring[:, 0]-c_)/a_; py = (ring[:, 1]-f_)/e_; plt.plot(np.append(px, px[0]), np.append(py, py[0]), '-', color='cyan', lw=0.7)
    plt.tight_layout(); plt.savefig(f'{OUT}/overview.png', dpi=110, bbox_inches='tight'); plt.close()
    print(f'{tname}: {len(dets)} ponds -> {OUT}/ponds.shp', flush=True)
print('DONE', flush=True)
