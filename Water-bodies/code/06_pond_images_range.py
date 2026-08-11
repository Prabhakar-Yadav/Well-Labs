# Generate 100 (or N) marked pond images starting from a given SIZE RANK.
# Ponds are ranked LARGEST -> SMALLEST (rank 1 = largest).
# Each image: pond outline + area + circumference. Plus a numbered overview + CSV.
#   python Water-bodies/code/06_pond_images_range.py <TILE> <START_RANK> [COUNT]
# e.g.  python Water-bodies/code/06_pond_images_range.py Dec_T43PGT 500 100
import sys, os, glob, csv, numpy as np, rasterio, cv2, shapefile
from rasterio.windows import from_bounds
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

TILE = sys.argv[1] if len(sys.argv) > 1 else 'Dec_T43PGT'
START = int(sys.argv[2]) if len(sys.argv) > 2 else 500        # size rank to start from
COUNT = int(sys.argv[3]) if len(sys.argv) > 3 else 100
FT = cv2.FONT_HERSHEY_SIMPLEX
PATS = {'June_T43QGU': 'S2A_*T43QGU*', 'Dec_T43QGU': 'S2C_*T43QGU*', 'Dec_T43PGT': 'S2C_*T43PGT*'}
if TILE not in PATS:
    sys.exit(f'TILE must be one of: {list(PATS)}')

SAFE = glob.glob(f'Water-bodies/data/Sentinel-2/{PATS[TILE]}.SAFE')[0]
G = glob.glob(f'{SAFE}/GRANULE/*/IMG_DATA/R10m')[0]
PFX = os.path.basename(glob.glob(f'{G}/*_B02_10m.jp2')[0]).replace('_B02_10m.jp2', '')
OUT = f'Water-bodies/results/predictions/{TILE}/rank_{START}_{START+COUNT-1}'
os.makedirs(f'{OUT}/images', exist_ok=True)

r = shapefile.Reader(f'Water-bodies/results/predictions/{TILE}/ponds.shp')
shs = r.shapes(); recs = r.records()
fl = [f[0] for f in r.fields[1:]]; ai = fl.index('area_m2'); ci = fl.index('circum_m')
dets = [(float(rec[ai]), float(rec[ci]), np.array(sh.points)) for sh, rec in zip(shs, recs)]
dets.sort(key=lambda z: -z[0])                                # largest first
sub = dets[START - 1:START - 1 + COUNT]
print(f'{TILE}: {len(dets)} ponds total; imaging rank {START}..{START+len(sub)-1}', flush=True)

src = rasterio.open(f'{G}/{PFX}_B04_10m.jp2'); gs = rasterio.open(f'{G}/{PFX}_B03_10m.jp2'); bs = rasterio.open(f'{G}/{PFX}_B02_10m.jp2')
with open(f'{OUT}/index.csv', 'w', newline='') as fh:
    cw = csv.writer(fh); cw.writerow(['id', 'rank', 'area_m2', 'area_ha', 'circumference_m'])
    for i, (a, c, ring) in enumerate(sub, 1):
        cw.writerow([i, START - 1 + i, round(a, 1), round(a / 1e4, 4), round(c, 1)])
for i, (a, c, ring) in enumerate(sub, 1):
    cx, cy = ring[:, 0].mean(), ring[:, 1].mean(); half = 350
    win = from_bounds(cx - half, cy - half, cx + half, cy + half, src.transform); out = 320
    ch = [x.read(1, window=win, out_shape=(out, out), boundless=True, fill_value=0) for x in (src, gs, bs)]
    rgb = np.dstack(ch).astype('float32')
    if (rgb.sum(2) > 0).sum() < 10:
        continue
    lo, hi = np.percentile(rgb[rgb.sum(2) > 0], [2, 98])
    rgb = np.ascontiguousarray(np.clip((rgb - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype('uint8'))
    wtr = rasterio.transform.from_bounds(cx - half, cy - half, cx + half, cy + half, out, out); inv = ~wtr
    pts = np.array([[*(inv * (x, y))] for x, y in ring], np.int32); cv2.polylines(rgb, [pts], True, (0, 255, 255), 2)
    rgb = cv2.copyMakeBorder(rgb, 52, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(rgb, f'{TILE}  pond #{i} (rank {START-1+i})', (6, 19), FT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(rgb, f'Area {a:.0f} m2 ({a/1e4:.2f} ha)  Circum {c:.0f} m', (6, 40), FT, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(f'{OUT}/images/pond_{i:03d}.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
src.close(); gs.close(); bs.close()

with rasterio.open(f'{G}/{PFX}_B02_10m.jp2') as s:
    H, W = s.height, s.width; tr = s.transform
DS = 10
with rasterio.open(f'{G}/{PFX}_B04_10m.jp2') as s: R_ = s.read(1, out_shape=(H // DS, W // DS)).astype('float32')
with rasterio.open(f'{G}/{PFX}_B03_10m.jp2') as s: G_ = s.read(1, out_shape=(H // DS, W // DS)).astype('float32')
with rasterio.open(f'{G}/{PFX}_B02_10m.jp2') as s: B_ = s.read(1, out_shape=(H // DS, W // DS)).astype('float32')
rgb = np.dstack([R_, G_, B_]); lo, hi = np.percentile(rgb[rgb.sum(2) > 0], [2, 98]); rgb = np.clip((rgb - lo) / (hi - lo), 0, 1)
a_, e_, c_, f_ = tr.a * DS, tr.e * DS, tr.c, tr.f
fig, (ax, axl) = plt.subplots(1, 2, figsize=(20, 13), gridspec_kw={'width_ratios': [3, 1]})
ax.imshow(rgb); ax.axis('off'); ax.set_title(f'{TILE} — ponds rank {START}..{START+len(sub)-1}', fontsize=13)
for i, (a, c, ring) in enumerate(sub, 1):
    px = (ring[:, 0] - c_) / a_; py = (ring[:, 1] - f_) / e_
    ax.plot(np.append(px, px[0]), np.append(py, py[0]), '-', color='red', lw=1.0)
    ax.text(px.mean(), py.mean(), str(i), color='yellow', fontsize=6, ha='center', va='center')
axl.axis('off'); L = [f'{"#":>3} {"area(ha)":>8} {"circ(m)":>7}']
for i, (a, c, ring) in enumerate(sub, 1):
    L.append(f'{i:>3} {a/1e4:>8.2f} {c:>7.0f}')
axl.text(0, 1, '\n'.join(L), family='monospace', fontsize=6.5, va='top'); axl.set_title('area + circumference', fontsize=11)
plt.tight_layout(); plt.savefig(f'{OUT}/overview.png', dpi=120, bbox_inches='tight'); plt.close()
print(f'DONE -> {OUT}/  ({len(sub)} images + overview.png + index.csv)', flush=True)
