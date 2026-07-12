import json, ast
p='07_blockage_detection.ipynb'; nb=json.load(open(p,encoding='utf-8'))

setup = """canal_data = {}

# --- Corridor filter: keep only canal within ~35 m of the hand-traced lines ---
import shapefile as _shp
from rasterio.warp import transform as _warp_xform
_CANAL_FEATS = [__import__('numpy').array(sr.shape.points, float)
                for sr in _shp.Reader('TLBC_D95_Canal/TLBC_D95.shp').iterShapeRecords()]
def _build_corridor(fpath, shape_ds, ds, buf_m=35.0):
    h_ds, w_ds = shape_ds
    with rasterio.open(fpath) as _s:
        tr=_s.transform; px_m=abs(tr.a); crs=_s.crs
    corr=np.zeros((h_ds,w_ds),np.uint8); buf_px=max(3,int(buf_m/(px_m*ds)))
    for pts in _CANAL_FEATS:
        xu,yu=_warp_xform('EPSG:4326',crs,pts[:,0].tolist(),pts[:,1].tolist())
        rows,cols=rasterio.transform.rowcol(tr,xu,yu)
        loc=[(int(c/ds),int(r/ds)) for r,c in zip(rows,cols)]
        loc=[(x,y) for x,y in loc if -buf_px<=x<w_ds+buf_px and -buf_px<=y<h_ds+buf_px]
        if len(loc)>1: cv2.polylines(corr,[np.array(loc,np.int32)],False,1,2*buf_px)
    return corr
"""

inject = """    canal_only, fic_candidate, rejected, counts = classify_canal_components(canal_raw, px_m_ds)

    _corr = _build_corridor(fpath, canal_only.shape, DS)
    if _corr.sum() > 0:
        _b = int(canal_only.sum()); canal_only = (canal_only * _corr).astype(np.uint8)
        print(f'  Corridor filter (<=35m from traced line): {_b} -> {int(canal_only.sum())} px')"""

done=0
for c in nb['cells']:
    if c['cell_type']!='code': continue
    s=''.join(c['source'])
    if s.splitlines()[0].startswith('# Cell 7 — Reload masks'):
        assert 'canal_data = {}' in s and '_build_corridor' not in s
        s=s.replace('canal_data = {}', setup, 1)
        s=s.replace('    canal_only, fic_candidate, rejected, counts = classify_canal_components(canal_raw, px_m_ds)', inject, 1)
        c['source']=[s]; done+=1
json.dump(nb,open(p,'w',encoding='utf-8'),indent=1)
for c in nb['cells']:
    if c['cell_type']=='code': ast.parse(''.join(c['source']))
print('corridor filter injected into Cell 7:', done, '| all cells parse OK')
