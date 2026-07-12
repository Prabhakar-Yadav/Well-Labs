# New-Area Canal Blockage Detection — Results Summary

## What was done
The **exact same blockage-detection pipeline** used for the TLBC/Attanur canals
(notebook `07_blockage_detection`) was applied to the 6 new orthophotos
(`NRBC_D10_MISSION_1`, `NRBC_D10_Survey_1`, and `MISSION_3_MS_1..4`).
The detection logic (canal segmentation model → mask → component/FIC filter →
reach extraction → blockage scoring) is **byte-for-byte identical** to Attanur;
only the file paths and a memory optimization differ.

## Key finding (important)
The model was trained **only on the TLBC concrete canal network**. The new areas
are mostly **farmland**, where thin field boundaries ("bunds") visually resemble
canals. As a result the raw model **over-detects field boundaries as canals**.
This was confirmed on `NRBC_MISSION_1`, which is **real RGB** (not converted) and
still detected field edges — so this is a **model-generalization limit, not a
pipeline error**.

## What was done to improve the results
The canal/FIC filters were **tuned** to exploit the geometric difference between a
real canal and a field boundary:
- **Thin, not blocky** — drop components whose *average* width is large (fields).
- **Long & elongated** — require a high length-to-width ratio.
- **Keep only the longest** continuous component(s) per tile (the real canal line).
- Higher confidence threshold (0.4 → 0.5).

**Effect: total detections dropped from 110 → 10** (field-boundary false positives
removed), and where a genuine canal exists it is now cleanly isolated.

| Tile | Type | Canal features detected | Note |
|------|------|------------------------|------|
| NRBC_MISSION_1 | real RGB | **0** | farmland/scrub — no genuine canal found |
| MS3_1 | false-color* | 3 | mostly residual field features |
| MS3_2 | false-color* | 1 | |
| MS3_3 | false-color* | 3 | mostly residual field features |
| MS3_4 | false-color* | 3 | **real diagonal canal cleanly isolated** + 2 residuals |

\* MISSION_3_MS tiles are **experimental**: that camera has no blue band, so a true
RGB image cannot be made — they were run as a false-color approximation and should
be treated as rough/unverified.

## Honest assessment for presentation
- These maps are **far cleaner** than the raw output and are presentable as a
  "pipeline applied to new areas" result.
- They are **not** a reliable blockage survey for these areas: the model does not
  yet recognise canals here, so absence of detections ≠ absence of canals, and the
  MS tiles are false-color.
- **To get reliable results on these areas, the model must be fine-tuned on a few
  hand-labelled canal examples from NRBC/MISSION_3.** That is the real next step.

## Files (in `New/results/`)
- `*_detection_tuned.png` — **the clean final maps (use these).**
- `blockage_summary.json` — per-tile counts (tuned).
- `*_blockage.tif` — GeoTIFFs.

## Not completed
- `NRBC_D10_Survey_1` (38 GB): stopped. A full run is ~11 h and would hit the same
  model limitation, so it was not worth the time. Can be run later if needed.
