# Canal Blockage — Production Package (for web/app integration)

Click-to-view boxes for two canal areas. Each reach is a
**box on the map**; clicking it should open its **zoomed image**.

```
production/
├── nrbc/          # NRBC D-10 canal, 557 hand-drawn segments (~35 m each), 570 images
│   │                 (13 segments sit in the overlap between the two orthophoto
│   │                 tiles and so have two images — same segment, two photos)
│   ├── boxes.geojson      ← MAIN integration file (see below)
│   ├── index.json         ← same data as a flat array (no geometry)
│   ├── index.csv          ← same, spreadsheet form
│   ├── images/            ← 0001.jpg, 0002.jpg, …  (one per box)
│   ├── overview_*.png     ← reference map with the box numbers drawn on it
│   └── blocks.shp/.shx/.dbf/.prj   ← same boxes as a GIS shapefile
├── nrbc_old_v1_3class/    # superseded: earlier 3-class (clear/partial/blocked),
│   │                        50 m-reach version, kept only for comparison
└── attanur/       # TLBC D-95 Attanur canal, 191 field-verified stretches
    └── (identical structure to nrbc/, see field-name note below)
```

Both packages are built from field-verified GIS segment files, not raw model output:
`gis/NRBC_alignment/nrbc_segments_handdrawn.shp` (NRBC) and
`gis/TLBC_alignment/tlbc_segments.shp` (TLBC/Attanur).

## How to integrate (the only file you need: `boxes.geojson`)
It is a standard **GeoJSON FeatureCollection in WGS84 (EPSG:4326)**. Every feature is
one blockage box (a polygon). The image path is in `properties.image`, **relative to the
area folder** (e.g. `images/0001.jpg`).

Each feature's `properties`:
| field | meaning |
|---|---|
| `id` | box number (matches the image filename and the overview map) |
| `status` | `clear`, `partial`, `moderate`, or `full` |
| `blockage_pct` | vegetation/blockage percent (0–100) |
| `struct_damage` (nrbc) / `damage` (attanur) | `Yes` / `No` — flags exposed rubble, cracked concrete, or eroded banks; **field names differ between the two packages**, not yet unified |
| `canal`, `type` | canal name; main / distributary / lateral |
| `chainage_m` | metres along that canal/lateral (per-line, not offset to a shared origin) |
| `lon`, `lat` | box centre (for a marker, if you prefer points to polygons) |
| `image` | relative path to the zoomed image, e.g. `images/0001.jpg` |

### Minimal Leaflet example (click box → open image)
```html
<div id="map" style="height:100vh"></div>
<script>
const map = L.map('map').setView([16.2, 77.3], 13);   // adjust to area
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

fetch('nrbc/boxes.geojson').then(r => r.json()).then(gj => {
  L.geoJSON(gj, {
    style: f => ({ color: {full:'red',moderate:'darkorange',partial:'orange',clear:'green'}[f.properties.status] || 'gray', weight: 2 }),
    onEachFeature: (f, layer) => layer.on('click', () => {
      const p = f.properties;
      // open a new page / popup showing the zoomed image
      window.open('nrbc/' + p.image, '_blank');
      // or in-page: layer.bindPopup(`<b>#${p.id} ${p.status}</b><br><img src="nrbc/${p.image}" width="380">`).openPopup();
    })
  }).addTo(map);
});
</script>
```
Mapbox / Google Maps / OpenLayers consume the same GeoJSON — load it as a source, style
by `properties.status`, and on a feature click read `properties.image`.

## Notes
- **Every reach has a box** — clear, partial, moderate, or full (green / orange / dark-orange / red).
- **Both NRBC and Attanur** statuses are **field-verified** (checked segment-by-segment in QGIS), not raw model output.
- Structural damage is a separate flag from the blockage status, shown as `Yes`/`No` on every box
  (field name `struct_damage` for nrbc, `damage` for attanur).
- Coordinates are lon/lat (WGS84). The `blocks.shp` is the same data for desktop GIS.
