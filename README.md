# TLBC D95 Canal Blockage Detection

**AI-powered drone survey analysis for canal health monitoring in Raichur District, Karnataka.**

This project uses deep learning and drone imagery to automatically detect **vegetation blockages**, **structural damage (cracks/erosion)**, and **silt accumulation** across the entire TLBC D95 canal network — replacing weeks of manual field surveys with a GPS-accurate AI analysis.

---

## Results Summary

| Metric | Value |
|--------|-------|
| Canal reaches analysed | 157 |
| Canal detection accuracy (IoU) | **87.6%** |
| Clear reaches | 122 (77.7%) |
| Partial blockage | 23 (14.6%) |
| Fully blocked | 12 (7.6%) |
| Structural damage locations | 8 |
| Total drone data processed | 64 GB |
| Final output size | ~1.66 GB GeoTIFFs |

---

## What the Pipeline Does

```
Drone orthomosaics (64 GB)
        │
        ▼
[Stage 1] Canal Detection       ← YOLOv8 Segmentation (IoU = 0.876)
        │ *_canal_mask.tif
        ▼
[Stage 2] Blockage Classification ← Rule-based: NDVI + green coverage + texture
        │ *_blockage.tif  (1=clear  2=partial  3=blocked)
        ▼
[Stage 3] Structural Damage     ← Mask R-CNN + ResNet50 (val loss = 0.18)
        │ canal_data.json  (structural_damage: true/false per reach)
        ▼
[Stage 4] Silt Depth            ← Elevation differencing: DSM − DTM
        │ *_silt.tif  (1=low  2=medium  3=high)
        ▼
QGIS-ready GeoTIFFs  +  canal_data.json
```

---

## Requirements

**Python 3.9 or 3.10 recommended**

Install all dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install segmentation-models-pytorch rasterio opencv-python scikit-image scipy matplotlib numpy Pillow jupyter
```

> **GPU recommended** for notebooks 05, 06, and 07. CPU will work but training will be very slow.

---

## Data You Need

The large raw data files are **not included** in this repository (too large for GitHub).

You need:

| File / Folder | Size | Description |
|---------------|------|-------------|
| `TLBC_D95_ATTANUR_1_ortho.tif` | 4.2 GB | Orthomosaic — Attanur Section 1 |
| `TLBC_D95_ATTANUR_2_ortho.tif` | 22.5 GB | Orthomosaic — Attanur Section 2 |
| `TLBC_D95_ATTANUR_3_ortho.tif` | 17.7 GB | Orthomosaic — Attanur Section 3 |
| `TLBC_D95_ATTANUR_4_ortho.tif` | 2.9 GB | Orthomosaic — Attanur Section 4 |
| `TLBC_D95_SHAKAPUR_ortho.tif` | 10.3 GB | Orthomosaic — Shakapur Section |
| `TLBC_DSM_DTM/` | 4.8 GB | Digital Surface Model + Digital Terrain Model |

Place the orthomosaic `.tif` files in the **root of the project folder** and `TLBC_DSM_DTM/` as a subfolder.

> **Coordinate system:** WGS84 UTM Zone 43N (EPSG:32643)

---

## How to Run — Step by Step

Run the notebooks **in order**. Each notebook's output feeds the next.

### Step 1 — Inspect the Data
```
jupyter notebook 03_canal_extraction.ipynb
```
Extracts canal regions from the large orthomosaics. Saves focused canal tiles.

### Step 2 — Build the Dataset
```
jupyter notebook 04_dataset.ipynb
jupyter notebook 04_patch_tiling.ipynb
```
- `04_dataset.ipynb` — loads annotations and prepares train/val split
- `04_patch_tiling.ipynb` — splits images into 1024×1024 px tiles

### Step 3 — Train Canal Detector
```
jupyter notebook 05_binary_training.ipynb
```
Trains the UNet segmentation model to detect canal boundaries.
**Output:** `outputs/05b_binary/best_binary_unet_v5.pth`

### Step 4 — Train Structural Damage Detector
```
jupyter notebook 06_fic_training.ipynb
```
Trains Mask R-CNN to detect cracks, erosion, and concrete damage (FIC).
**Output:** `outputs/06_fic/best_fic_unet.pth`

### Step 5 — Run Blockage Detection (Main Inference)
```
jupyter notebook 07_blockage_detection.ipynb
```
Runs the full pipeline — canal detection, blockage classification, FIC detection.
**Output:** `outputs/07_blockage/` (1.6 GB GeoTIFFs + `canal_data.json`)

> **Important:** In Notebook 07, run cells in this exact order:
> `Cell 1 → Cell 2b → Cell 3 → Cell 4 → Cell 8`
> Running Cell 8 before Cells 3–4 will cause a NameError.

### Step 6 — Run Silt Detection
```
jupyter notebook 08_silt_detection.ipynb
```
Measures silt depth by subtracting DTM from DSM inside canal boundaries.
**Output:** `outputs/08_silt/` (57 MB GeoTIFFs)

---

## Project Structure

```
Wells-Lab/
│
├── 03_canal_extraction.ipynb      # Step 1: Extract canal regions
├── 04_dataset.ipynb               # Step 2: Build dataset
├── 04_patch_tiling.ipynb          # Step 2b: Tile images for training
├── 05_binary_training.ipynb       # Step 3: Train canal detector
├── 06_fic_training.ipynb          # Step 4: Train damage detector
├── 07_blockage_detection.ipynb    # Step 5: Full blockage inference
├── 08_silt_detection.ipynb        # Step 6: Silt depth measurement
│
├── 01_inspect_tifs.py             # Utility: inspect GeoTIFF files
├── 02_spectral_analysis.py        # Utility: spectral band analysis
├── 03_canal_extraction.py         # Utility: canal extraction script
├── extract_from_labels.py         # Utility: extract patches from labels
├── extract_patches_for_labeling.py
├── extract_patches_v2.py
├── label_full_image.py
│
├── build_report.py                # Builds PROJECT_REPORT.docx
├── make_pngs.py                   # Converts DOCX report to PNGs
│
├── annotation/                    # Training patches (PNGs + JSON labels)
├── annotation_v2/                 # Version 2 annotations (JPG + JSON)
│
├── outputs/
│   ├── 01_inspect/                # Thumbnail images
│   ├── 02_spectral/               # Spectral analysis charts
│   ├── 03_extraction*/            # Canal extraction visualisations
│   ├── 05_training/               # Training curves, val predictions
│   ├── 05b_binary/                # Binary model predictions
│   ├── 06_fic/                    # FIC damage visualisations
│   ├── 07_blockage/               # Blockage GeoTIFFs + canal_data.json
│   └── 08_silt/                   # Silt depth GeoTIFFs
│
├── docs/
│   └── canal_physical_characteristics.md
│
├── PROJECT_REPORT.docx            # Final 10-page project report
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Output Files Explained

### Blockage outputs — `outputs/07_blockage/`

| File | Description |
|------|-------------|
| `*_blockage.tif` | Pixel-level blockage map. Values: `1`=clear, `2`=partial, `3`=blocked |
| `*_canal_mask.tif` | Binary canal boundary. Values: `1`=canal, `0`=not canal |
| `canal_data.json` | Per-reach data: reach ID, status, structural_damage flag, GPS coords |

### Silt outputs — `outputs/08_silt/`

| File | Description |
|------|-------------|
| `*_silt.tif` | Classified silt depth. Values: `1`=low (<0.5m), `2`=medium (0.5–1m), `3`=high (>1m) |
| `*_chm_canal.tif` | Continuous silt depth in metres (DSM minus DTM) |

---

## Viewing Results in QGIS

1. **Load orthomosaic** as base layer: `Layer → Add Raster Layer`
2. **Load blockage raster:** `*_blockage.tif`
3. Open **Layer Properties → Symbology → Paletted/Unique Values**
4. Set colours:
   - `0` → Transparent
   - `1` → Green `#00AA00` (clear)
   - `2` → Orange `#FF8C00` (partial)
   - `3` → Red `#DC143C` (blocked)
5. Set **layer transparency to 60%** to see ortho underneath

---

## Project Report

The complete project report (methods, results, data description, action items) is available as:

```
PROJECT_REPORT.docx
```

To regenerate the report:

```bash
python build_report.py    # creates PROJECT_REPORT.docx
python make_pngs.py       # converts to PNG pages (requires MS Word on Windows)
```

---

## Author

**Prabhakar Yadav**  
prabhakaryadav0505@gmail.com  
Wells Lab — TLBC D95 Canal Survey, Raichur District, Karnataka  
May 2026
