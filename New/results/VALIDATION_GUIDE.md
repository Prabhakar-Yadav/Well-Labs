# Manual Accuracy Check in QGIS — Blockage Detection

**Goal:** walk along the canal in QGIS, compare the model's `status` (blocked/partial/clear)
against what you actually see in the imagery, and compute accuracy.

## Files
- Ortho: `New/NRBC_D10_Survey_1_ortho.tif`
- Detection: `New/results/nrbc_blockage_reaches.shp`  (EPSG:32643 — overlays the ortho exactly)

Each line feature = one ~50 m canal reach. Fields:
`canal`, `ctype` (main/distributary/lateral), `chainage` (m from junction),
`status` (model: blocked/partial/clear), `veg_frac`, **`actual`** (empty — you fill it),
**`notes`** (empty — optional).

## Step 1 — Load & overlay
1. Layer ▸ Add Raster Layer ▸ select the ortho.
2. Layer ▸ Add Vector Layer ▸ select `nrbc_blockage_reaches.shp`.
   (Same CRS, so they line up automatically.)

## Step 2 — Colour the reaches by status
Right-click the shapefile ▸ **Properties ▸ Symbology ▸ Categorized**
- Value = `status` ▸ **Classify**
- Set: blocked = red, partial = orange, clear = green. Line width ≈ 1.0–1.5.
- (Optional) **Labels** tab ▸ Single labels ▸ value = `chainage` to see distances.

## Step 3 — Validate, reach by reach
1. Click the **pencil (Toggle Editing)** on the shapefile.
2. Start at chainage 0 (the junction) and follow the canal downstream.
3. For each reach, look at the imagery under it and decide the TRUE state:
   - **clear** = open channel / water / bare bed visible
   - **partial** = some vegetation/silt, but channel still partly open
   - **blocked** = channel fully overgrown / choked
   *(Pick one rule and apply it consistently.)*
4. Use the **Identify** tool (ℹ️) to read the model's `status`, or open the Attribute Table.
5. Type your verdict into the **`actual`** field for that reach (clear/partial/blocked).
6. Save edits (Ctrl+S) every so often.

**Tip — don't do all 404.** Validate a representative **sample of ~60–80 reaches**
(some main canal, some distributary, a few laterals, mix of red/orange/green). That gives
a reliable accuracy estimate in ~1 hour instead of a full day.

## Step 4 — Compute accuracy
After filling `actual` for your sample:
- Attribute Table ▸ **Select by Expression**: `"actual" = "status" AND "actual" != ''`
  → number selected = correct.
- Accuracy = correct ÷ (total reaches you filled).
- For a confusion matrix, count each (status, actual) combination
  (e.g., select `"status"='blocked' AND "actual"='clear'` = false positives).

## Faster option
When you've filled the `actual` field and saved the shapefile, **send it to me** and I'll
compute overall accuracy, per-class recall/precision/F1, and the confusion matrix automatically.
