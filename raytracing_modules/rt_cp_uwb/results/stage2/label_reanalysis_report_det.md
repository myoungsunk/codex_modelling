# Stage 2 Label Reanalysis (Deterministic)

| label | auc_cir | auc_cp | auc_joint | delta_auc |
|---|---:|---:|---:|---:|
| Current ratio >= 0.20 | 0.7824 | 0.6014 | 0.7994 | 0.0170 |
| Geo only (has_los_path == 0) | 0.6523 | 0.6598 | 0.7190 | 0.0667 |
| Bounce only (visible LoS, ratio >= 0.33) | 0.6757 | 0.5823 | 0.6968 | 0.0211 |
| Mixed geo OR bounce@0.33 | 0.6819 | 0.5966 | 0.7067 | 0.0248 |
