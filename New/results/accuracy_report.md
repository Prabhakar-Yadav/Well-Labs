# NRBC D-10 Blockage — Accuracy Report
5-fold cross-validated on 405 field-verified reaches. Model: HistGradientBoosting on RGB channel-openness features.

## Headline accuracy
| Metric | Value |
|---|---|
| Weighted (partial slip=0.5, extreme=1.0) | **88.1%** |
| Within-one-class (only gross errors count) | 97.8% |
| Binary (blocked vs open) | 90.6% |
| Exact 3-class | 78.5% |
| Model fit to your labels | 100.0% |

## Confusion matrix (rows = model prediction, cols = your verified truth)
| pred \ truth | clear | partial | blocked |
|---|---|---|---|
| **clear**   | 22 | 11 | 2 |
| **partial** | 18 | 101 | 15 |
| **blocked** | 7 | 34 | 195 |

## Per-class (vs your verified truth)
| class | recall | precision | F1 |
|---|---|---|---|
| clear | 47% | 63% | 54% |
| partial | 69% | 75% | 72% |
| blocked | 92% | 83% | 87% |

## Notes
- Weighted accuracy = 1 − (total penalty / 405); partial-boundary slip penalised 0.5, gross clear↔blocked flip 1.0.
- Exact 3-class caps ~78% due to genuine partial↔blocked overlap in imagery (confirmed: richer features + 3 models all ~78%).
- Final NRBC map/shapefile use your VERIFIED labels directly; this report measures the trained model.
