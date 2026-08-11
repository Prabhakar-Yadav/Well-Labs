# Top 100 SMALLEST detected ponds per tile as marked images (area + circumference
# on each) + a numbered whole-tile overview with legend. We target small ponds.
#   python Water-bodies/code/05_small_pond_images.py [N]
import sys, os, glob, csv, numpy as np, rasterio, cv2, shapefile
from rasterio.windows import from_bounds
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
FT = cv2.FONT_HERSHEY_SIMPLEX
SDIR = 'Water-bodies/data/Sentinel-2'
TILES = [('June_T43QGU', 'S2A_*T43QGU*'), ('Dec_T43QGU', 'S2C_*T43QGU*'), ('Dec_T43PGT', 'S2C_*T43PGT*')]
for tname, pat in TILES:
    shp = f'Water-bodies/results/predictions/{tname}/ponds.shp'
    if not os.path.exists(shp): continue
    SAFE = glob.glob(f'{SDIR}/{pat}.SAFE')[0]
    G10 = glob.glob(f'{SAFE}/GRANULE/*/IMG_DATA/R10m')[0]
    PFX = os.path.basename(glob.glob(f'{G10}/*_B02_10m.jp2')[0]).replace('_B02_10m.jp2', '')
    OUT = f'Water-bodies/results/predictions/{tname}/small_ponds'
    os.makedirs(f'{OUT}/images', exist_ok=True)
    r = shapefile.Reader(shp); shs = r.shapes(); recs = r.records()
    fl = [f[0] for f in r.fields[1:]]; ai = fl.index('area_m2'); ci = fl.index('circum_m')
    dets = []
    for sh, rec in zip(shs, recs):
        ring = np.array(sh.points)
        dets.append((float(rec[ai]), float(rec[ci]), ring, ring[:, 0].mean(), ring[:, 1].mean()))
    dets.sort(key=lambda z: z[0])           # smallest first
    dets = dets[:N]
    src = rasterio.open(f'{G10}/{PFX}_B04_10m.jp2'); gs = rasterio.open(f'{G10}/{PFX}_B03_10m.jp2'); bs = rasterio.open(f'{G10}/{PFX}_B02_10m.jp2')
    with open(f'{OUT}/index.csv', 'w', newline='') as fh:
        cw = csv.writer(fh); cw.writerow(['id', 'area_m2', 'area_ha', 'circumference_m'])
        for i, (a, c, ring, cx, cy) in enumerate(dets, 1):
            cw.writerow([i, round(a, 1), round(a/1e4, 4), round(c, 1)])
    for i, (a, c, ring, cx, cy) in enumerate(dets, 1):
        half = 350                          # fixed 700 m frame around small pond
        win = from_bounds(cx-half, cy-half, cx+half, cy+half, src.transform); out = 300
        try:
            ch = [x.read(1, window=win, out_shape=(out, out), boundless=True, fill_value=0) for x in (src, gs, bs)]
        except Exception: continue
        rgb = np.dstack(ch).astype('float32')
        if (rgb.sum(2) > 0).sum() < 10: continue
        lo, hi = np.percentile(rgb[rgb.sum(2) > 0], [2, 98])
        rgb = np.ascontiguousarray(np.clip((rgb-lo)/(hi-lo+1e-6)*255, 0, 255).astype('uint8'))
        wtr = rasterio.transform.from_bounds(cx-half, cy-half, cx+half, cy+half, out, out); inv = ~wtr
        pts = np.array([[*(inv*(x, y))] for x, y in ring], np.int32)
        cv2.polylines(rgb, [pts], True, (0, 255, 255), 2)
        rgb = cv2.copyMakeBorder(rgb, 52, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(rgb, f'{tname}  small pond #{i}', (6, 19), FT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(rgb, f'Area {a:.0f} m2 ({a/1e4:.3f} ha)  Circum {c:.0f} m', (6, 40), FT, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(f'{OUT}/images/small_{i:03d}.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    src.close(); gs.close(); bs.close()
    # overview
    with rasterio.open(f'{G10}/{PFX}_B02_10m.jp2') as s: H, W = s.height, s.width; tr = s.transform
    DS = 10
    with rasterio.open(f'{G10}/{PFX}_B04_10m.jp2') as s: R_ = s.read(1, out_shape=(H//DS, W//DS)).astype('float32')
    with rasterio.open(f'{G10}/{PFX}_B03_10m.jp2') as s: G_ = s.read(1, out_shape=(H//DS, W//DS)).astype('float32')
    with rasterio.open(f'{G10}/{PFX}_B02_10m.jp2') as s: B_ = s.read(1, out_shape=(H//DS, W//DS)).astype('float32')
    rgb = np.dstack([R_, G_, B_]); lo, hi = np.percentile(rgb[rgb.sum(2) > 0], [2, 98]); rgb = np.clip((rgb-lo)/(hi-lo), 0, 1)
    a_, e_, c_, f_ = tr.a*DS, tr.e*DS, tr.c, tr.f
    fig, (ax, axl) = plt.subplots(1, 2, figsize=(20, 13), gridspec_kw={'width_ratios': [3, 1]})
    ax.imshow(rgb); ax.axis('off'); ax.set_title(f'{tname} — 100 smallest detected ponds', fontsize=13)
    for i, (a, c, ring, cx, cy) in enumerate(dets, 1):
        px = (ring[:, 0]-c_)/a_; py = (ring[:, 1]-f_)/e_
        ax.plot(np.append(px, px[0]), np.append(py, py[0]), '-', color='red', lw=1.0)
        ax.text(px.mean(), py.mean(), str(i), color='yellow', fontsize=6, ha='center', va='center')
    axl.axis('off'); lines = [f'{"#":>3} {"area(m2)":>8} {"circ(m)":>7}']
    for i, (a, c, ring, cx, cy) in enumerate(dets, 1): lines.append(f'{i:>3} {a:>8.0f} {c:>7.0f}')
    axl.text(0, 1, '\n'.join(lines), family='monospace', fontsize=6.5, va='top'); axl.set_title('area + circumference', fontsize=11)
    plt.tight_layout(); plt.savefig(f'{OUT}/overview_small_ponds.png', dpi=120, bbox_inches='tight'); plt.close()
    print(f'{tname}: {len(dets)} small ponds -> {OUT}/', flush=True)
print('DONE', flush=True)
