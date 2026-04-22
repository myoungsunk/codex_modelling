# Stage 2 Label Reanalysis

## Overall Comparison

| label | n | n_neg | n_pos | auc_cir | auc_cp | auc_joint | delta_auc | auc_rise_time_fp | auc_gamma_cp_3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current ratio >= 0.20 | 900 | 79 | 821 | 0.7896 | 0.6155 | 0.8093 | 0.0197 | 0.7165 | 0.4473 |
| Geo only (has_los_path == 0) | 900 | 833 | 67 | 0.6605 | 0.6491 | 0.7190 | 0.0584 | 0.5722 | 0.6216 |
| Bounce only (visible LoS, ratio >= 0.33) | 833 | 463 | 370 | 0.6791 | 0.5805 | 0.6989 | 0.0198 | 0.5876 | 0.5037 |
| Mixed geo OR bounce@0.33 | 900 | 463 | 437 | 0.6853 | 0.5900 | 0.7079 | 0.0225 | 0.5909 | 0.5221 |

## Room-wise Mixed@0.33

| room | n | n_neg | n_pos | auc_cir | auc_cp | auc_joint | delta_auc |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 300 | 165 | 135 | 0.6817 | 0.6070 | 0.7206 | 0.0389 |
| B | 300 | 162 | 138 | 0.7168 | 0.6001 | 0.7339 | 0.0171 |
| C | 300 | 136 | 164 | 0.7003 | 0.6135 | 0.7418 | 0.0415 |

## Disagreement Asymmetry

| label | cir_fail_cp_save | cp_fail_cir_save |
|---|---:|---:|
| current_0p20 | 0 | 2 |
| geo_only | 0 | 1 |
| bounce_0p33_visible | 137 | 180 |
| mixed_0p33 | 147 | 214 |

## Recommendation

- Preferred decomposition: `is_nlos_geo = ~has_los_path` and `is_nlos_bounce = has_los_path & (bounce_to_los_ratio_mid >= 0.33)`.
- Reporting aggregate label: `mixed_0p33 = is_nlos_geo OR is_nlos_bounce`.
- Avoid keeping `current_0p20` as the main Stage 2 label because `rise_time_fp` remains too close to the label-defining physics.
