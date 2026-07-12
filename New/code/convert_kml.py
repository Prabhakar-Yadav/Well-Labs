# Convert the D-10 KML to a shapefile, keeping ONLY canal/distributary/lateral
# lines (drops FIC, DPO, and point markers).  Run from the Wells-Lab root.
import xml.etree.ElementTree as ET, shapefile, os
KML = "New/D-10 Laterals, DPO & FIC's .kml"
OUT = 'New/NRBC_Canal/NRBC_D10'
ns = {'k': 'http://www.opengis.net/kml/2.2'}
root = ET.parse(KML).getroot()
os.makedirs('New/NRBC_Canal', exist_ok=True)
w = shapefile.Writer(OUT, shapeType=shapefile.POLYLINE); w.field('Name', 'C', 80)
kept = []
for p in root.findall('.//k:Placemark', ns):
    ls = p.find('.//k:LineString', ns)
    if ls is None:                       # skip point markers
        continue
    nm = (p.find('k:name', ns).text or '?').strip()
    if 'fic' in nm.lower() or 'dpo' in nm.lower():   # drop FIC + DPO
        continue
    pts = [[float(t.split(',')[0]), float(t.split(',')[1])]
           for t in ls.find('k:coordinates', ns).text.split()]
    if len(pts) < 2:
        continue
    w.line([pts]); w.record(nm); kept.append(nm)
w.close()
open(OUT + '.prj', 'w').write(
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')
print(f'Wrote {len(kept)} lines -> {OUT}.shp :', kept)
