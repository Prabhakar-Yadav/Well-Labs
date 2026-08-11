# Water Bodies — clean project structure

Two tracks, deliberately separate (different sensors, cannot be mixed):
**(A)** farm-pond detection on high-res TLBC drone imagery, and
**(B)** water-body detection on Sentinel-2 satellite tiles.

```
Water-bodies/
├── data/
│   ├── Farmponds/          # the 180 hand-drawn pond labels (Farmpond_D95.shp)
│   ├── TLBC_pond/          # drone pond training dataset (dataset.npz)
│   ├── external/           # ESWD internet water data (Sentinel-2, for track B)
│   └── Sentinel-2/         # raw S2 tiles: June T43QGU, Dec T43QGU, Dec T43PGT
│
├── results/
│   ├── pond_maps/          # ⭐ MARKED POND PICTURES — 108 ponds, each with
│   │   ├── images/         #    outline + AREA + CIRCUMFERENCE (pond_0001.jpg ...)
│   │   ├── pond_index.csv  #    id, area_m2, area_ha, circumference_m, lon, lat
│   │   └── overview.png    #    montage of all 108
│   │
│   ├── TLBC_model/         # (A) trained pond detector (tlbc_pond_resnet34_best.pt)
│   ├── tlbc_predictions.png#     detector output on unseen flights
│   │
│   ├── sentinel_predictions/   # ⭐ MODEL'S OWN DETECTIONS on the 3 S2 tiles
│   │   └── <tile>/             #   June_T43QGU / Dec_T43QGU / Dec_T43PGT
│   │       ├── overview_numbered.png  # whole tile: water bodies marked+numbered+legend
│   │       ├── images/                # det_0001.jpg ... one per detection (area+circ)
│   │       └── all_detections.csv     # EVERY detection: area + circumference
│   │
│   ├── D95_fulltile/           # (B) June T43QGU raw water_bodies.shp + check.png
│   ├── D95_dec_QGU_fulltile/   # (B) Dec  T43QGU raw water_bodies.shp
│   ├── D95_dec_PGT_fulltile/   # (B) Dec  T43PGT raw water_bodies.shp
│   └── D95_model/          #     B5 water model used for the S2 inference
│
├── code/                   # numbered pipeline scripts (10,16,22,23,24 are current)
├── RESULTS.md              # full written record of methods + numbers
└── README.md               # this file
```

## The two tracks

**(A) Farm-pond detection — TLBC drone (the trustworthy one)**
Ponds are hundreds of px at 0.5 m/px, clearly visible. Trained on ATTANUR_2/3
flights, tested on SHAKAPUR + ATTANUR_1/4 (different flights). **F1 0.654,
70% of ponds found.** Build: `22_make_tlbc_dataset.py` → `23_train_tlbc.py`.

**(B) Water-body detection — Sentinel-2 (already inferred, 3 tiles)**
Trained on ESWD internet water. Detects reservoirs/rivers well; too coarse for
the small ponds. Each `*_fulltile/water_bodies.shp` = model detections with area.
Run: `16_run_full_tile.py`.

## The pond deliverable you asked for
`results/pond_maps/` — one drone picture per pond, outline drawn, **area (m²/ha)
and circumference (m)** written on it, plus `pond_index.csv`. Built by
`24_pond_maps_tlbc.py`. 108 ponds, total 31.2 ha, median 1,690 m² / 164 m.
Where the outline sits on a green field, that pond was dry when the drone flew.
