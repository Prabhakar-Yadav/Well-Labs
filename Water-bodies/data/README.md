# Data layout

One folder per water body / site under `Water-bodies/data/<SITE>/`. Drop files
in using these exact names and the pipeline scripts will pick them up with no
config changes — just pass `<SITE>` as the argument.

```
Water-bodies/data/<SITE>/
├── sentinel2_stack.tif   # required for stage 1 & 3. 4-band GeoTIFF, band order:
│                         # [Blue(B2), Green(B3), Red(B4), NIR(B8)], 10m, same CRS
├── sentinel1_vv.tif      # optional for stage 1 (improves cloud-cover robustness). dB units
├── sentinel1_vh.tif      # optional, same grid as VV
├── dem.tif               # required for stage 2. Copernicus/SRTM 30m or better, same CRS
└── depth_points.csv      # optional for stage 3 calibration: columns lon,lat,depth_m
                           # (sonar/dGPS soundings, if/when available)
```

Notes:
- All rasters for a site should share the same CRS (a projected, metres-based
  CRS is best — reproject before dropping in if needed). The scripts don't
  reproject for you.
- `sentinel2_stack.tif` can be built from individual SAFE-product band JP2s
  with `gdal_merge.py -separate -o sentinel2_stack.tif B02.jp2 B03.jp2 B04.jp2 B08.jp2`.
- Nothing needs to exist until you have it — each stage script checks for its
  required inputs and tells you what's missing rather than failing silently.
