# Concise HTML report for the single-track Sentinel-2 pond model.
import base64, io, os
from PIL import Image
def b64(p, mw=900):
    if not os.path.exists(p): return None
    im = Image.open(p).convert('RGB')
    if im.width > mw: im = im.resize((mw, int(im.height*mw/im.width)))
    b = io.BytesIO(); im.save(b, 'JPEG', quality=82); return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()
def fig(p, c, mw=900):
    d = b64(p, mw); return f'<figure><img src="{d}"/><figcaption>{c}</figcaption></figure>' if d else ''
P = 'Water-bodies/results'
html = f'''<!doctype html><html><head><meta charset="utf-8"><title>Farm-Pond Detection — Sentinel-2</title>
<style>body{{font-family:system-ui,Arial;max-width:960px;margin:24px auto;padding:0 18px;color:#1a2027;line-height:1.55}}
h1{{border-bottom:3px solid #1565c0;padding-bottom:8px}}h2{{color:#1565c0;border-bottom:1px solid #d0d7de;padding-bottom:4px;margin-top:30px}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}th,td{{border:1px solid #cfd8dc;padding:7px 10px;font-size:14px}}th{{background:#e3f2fd}}
img{{max-width:100%;border:1px solid #b0bec5;border-radius:6px}}figcaption{{font-size:13px;color:#546e7a;margin-top:6px}}figure{{text-align:center}}
.good{{color:#2e7d32;font-weight:600}}.note{{background:#fff8e1;border-left:4px solid #f9a825;padding:10px 14px;font-size:14px}}</style></head><body>
<h1>Farm-Pond Detection on Sentinel-2 (single model)</h1>
<p>One model detects farm ponds from Sentinel-2 imagery. The 180 hand-drawn pond
labels were split 80/20 and overlaid on all 3 Sentinel-2 tiles (a pond appears
in every tile that covers it — giving more samples, and both dry and wet views).</p>

<h2>Data</h2>
<table><tr><th>Item</th><th>Detail</th></tr>
<tr><td>Labels</td><td>180 farm-pond polygons (Farmpond_D95.shp), split 80/20 by pond → 144 train / 36 test</td></tr>
<tr><td>Imagery</td><td>3 Sentinel-2 tiles: June T43QGU (dry), Dec T43QGU (wet), Dec T43PGT (wet)</td></tr>
<tr><td>Bands</td><td>6-band: B2,B3,B4,B8,B11,B12 @ 10 m</td></tr>
<tr><td>Dataset</td><td>2,142 patches (64×64); pond pixels 1.9%; a pond + all its tile-appearances stay on one side of the split (no leakage)</td></tr></table>

<h2>Model</h2>
<p>U-Net + ResNet-34 (ImageNet-pretrained), 6-band input, Dice + weighted BCE loss. One model, Sentinel-2 only.</p>

<h2>Results — same metrics on train &amp; test</h2>
<table><tr><th>Metric</th><th>TRAIN</th><th>TEST (held-out ponds)</th></tr>
<tr><td>F1</td><td>0.809</td><td class="good">0.739</td></tr>
<tr><td>IoU</td><td>0.680</td><td>0.586</td></tr>
<tr><td>Precision</td><td>—</td><td>0.660</td></tr>
<tr><td>Recall</td><td>—</td><td>0.840</td></tr>
<tr><td>Ponds found (object-level)</td><td>89%</td><td class="good">85% (1155/1352)</td></tr></table>
<p>Small train→test gap (0.809 → 0.739) = good generalisation, not overfitting.</p>

<h2>Inference on the 3 tiles</h2>
<table><tr><th>Tile</th><th>Detected ponds</th></tr>
<tr><td>June T43QGU (dry)</td><td>3,561</td></tr>
<tr><td>Dec T43QGU (wet)</td><td>18,641</td></tr>
<tr><td>Dec T43PGT (wet)</td><td>13,957</td></tr></table>
<div class="note">On the full 110×110 km tiles the model also flags many other small water / wet
features that resemble ponds (higher counts on the wet December tiles). The
per-pond accuracy above (test F1 0.739, 85% found) is the reliable figure;
full-tile counts include such look-alikes. Outputs: <code>results/predictions/&lt;tile&gt;/ponds.shp</code> (with area + circumference).</div>

<h2>Example</h2>
{fig(f'{P}/predictions/Dec_T43PGT/overview.png', 'Detected ponds on Dec T43PGT (cyan).', 900)}
</body></html>'''
open(f'{P}/report.html', 'w', encoding='utf-8').write(html)
print(f'report -> {P}/report.html ({len(html)//1024} KB)')
