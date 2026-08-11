# Build a matched IMAGE + POND-SHAPEFILE pair from a free open Sentinel-2 scene.
# The ponds are extracted (MNDWI) from the SAME image, so they coincide exactly
# -> open both in QGIS, they overlay; use directly as training labels.
#   python Water-bodies/code/07_make_training_pair.py <lon0> <lat0> <lon1> <lat1> <name>
import sys, os, json, urllib.request, numpy as np, rasterio, shapefile
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'; os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
from rasterio.features import shapes as rio_shapes
from scipy import ndimage

lon0, lat0, lon1, lat1 = map(float, sys.argv[1:5])
NAME = sys.argv[5] if len(sys.argv) > 5 else 'site'
OUT = f'Water-bodies/data/training_pair/{NAME}'
os.makedirs(OUT, exist_ok=True)

# find the lowest-cloud recent scene covering the bbox
body = json.dumps({'collections': ['sentinel-2-l2a'], 'bbox': [lon0, lat0, lon1, lat1],
                   'query': {'eo:cloud_cover': {'lt': 10}},
                   'sortby': [{'field': 'properties.eo:cloud_cover', 'direction': 'asc'}], 'limit': 1}).encode()
req = urllib.request.Request('https://earth-search.aws.element84.com/v1/search', data=body,
                             headers={'Content-Type': 'application/json'})
feat = json.load(urllib.request.urlopen(req, timeout=60))['features'][0]
base = os.path.dirname(feat['assets']['green']['href'])
print(f'{NAME}: scene {feat["id"]} cloud {feat["properties"]["eo:cloud_cover"]:.1f}%', flush=True)

def read10(b):
    with rasterio.open(f'{base}/{b}.tif') as s:
        ub = transform_bounds('EPSG:4326', s.crs, lon0, lat0, lon1, lat1)
        win = from_bounds(*ub, s.transform).round_offsets().round_lengths()
        a = s.read(1, window=win).astype('float32'); tr = s.window_transform(win); crs = s.crs
    return a, tr, crs, win

B04, tr, crs, win = read10('B04'); B03, _, _, _ = read10('B03'); B02, _, _, _ = read10('B02')
H, W = B04.shape
with rasterio.open(f'{base}/B11.tif') as s:
    ub = transform_bounds('EPSG:4326', s.crs, lon0, lat0, lon1, lat1)
    win20 = from_bounds(*ub, s.transform)
    B11 = s.read(1, out_shape=(H, W), window=win20, resampling=Resampling.bilinear).astype('float32')
print(f'  clip {H}x{W} ({H*10/1000:.0f}x{W*10/1000:.0f} km)', flush=True)

mndwi = (B03 - B11) / (B03 + B11 + 1e-6)
water = ndimage.binary_opening(mndwi > 0.0, iterations=1)
rgb = np.stack([B04, B03, B02]).astype('uint16')
meta = dict(driver='GTiff', height=H, width=W, count=3, dtype='uint16', crs=crs, transform=tr, compress='deflate')
with rasterio.open(f'{OUT}/{NAME}_image.tif', 'w', **meta) as d:
    d.write(rgb)

w = shapefile.Writer(f'{OUT}/{NAME}_ponds', shapeType=shapefile.POLYGON)
w.field('id', 'N', 8); w.field('area_m2', 'F', 16, 1); w.field('area_ha', 'F', 12, 4); w.field('circum_m', 'F', 12, 1)
n = 0; areas = []
for geom, _ in rio_shapes(water.astype('uint8'), mask=water, transform=tr):
    ring = geom['coordinates'][0]; x = np.array([p[0] for p in ring]); y = np.array([p[1] for p in ring])
    a = abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])) / 2; per = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    if a < 800:
        continue
    circ = 4 * np.pi * a / max(per * per, 1e-9)
    if circ < 0.08:
        continue
    w.poly([[list(p) for p in ring]]); w.record(n + 1, round(a, 1), round(a / 1e4, 4), round(per, 1)); n += 1; areas.append(a)
w.close(); open(f'{OUT}/{NAME}_ponds.prj', 'w').write(crs.to_wkt())
areas = np.array(areas)
print(f'  extracted {n} ponds | median {np.median(areas):.0f} m2 | <1ha {int((areas<1e4).sum())} -> {OUT}/', flush=True)
