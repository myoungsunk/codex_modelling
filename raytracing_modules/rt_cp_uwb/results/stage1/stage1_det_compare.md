# Stage 1 Deterministic Comparison

- delta_auc_orig: 0.040993
- delta_auc_det: 0.046258
- delta_auc_shift: 0.005265
- delta_stability_gate |shift| < 0.01: PASS
- auc_cir_shift: -0.005014
- auc_cp_shift: 0.003398
- auc_joint_shift: 0.000251
- top3_overlap: 1 / 3
- top10_overlap: 8 / 10

## Top 3 Original

| rank | x_var | x_label | y_var | y_label | n | delta_auc |
|---:|---|---|---|---|---:|---:|
| 1 | eps_r | [2.00, 3.60] | xpol_coupling_db | [20.01, 24.00] | 120 | 0.266582 |
| 2 | antenna_type | ideal | los_angle_from_anchor_bore_deg | [38.57, 45.77] | 294 | 0.200551 |
| 3 | antenna_type | ideal | slab_placement | wall_x | 371 | 0.192471 |

## Top 3 Deterministic

| rank | x_var | x_label | y_var | y_label | n | delta_auc |
|---:|---|---|---|---|---:|---:|
| 1 | los_angle_from_anchor_bore_deg | [53.49, 77.65] | xpol_coupling_db | [24.00, 28.00] | 122 | 0.214463 |
| 2 | los_angle_from_anchor_bore_deg | [38.57, 45.77] | eps_r | [5.20, 6.80] | 112 | 0.196581 |
| 3 | antenna_type | ideal | slab_placement | wall_x | 371 | 0.194003 |

## Top 10 Overlap IDs

- antenna_type |ideal |los_angle_from_anchor_bore_deg |[38.57, 45.77]
- antenna_type |ideal |slab_placement |wall_x
- los_angle_from_anchor_bore_deg |[53.49, 77.65] |xpol_coupling_db |[24.00, 28.00]
- antenna_type |ideal |slab_placement |ceiling
- antenna_type |ideal |los_angle_from_anchor_bore_deg |[53.49, 77.65]
- antenna_type |ideal |los_angle_from_anchor_bore_deg |[45.77, 53.49]
- los_angle_from_anchor_bore_deg |[38.57, 45.77] |slab_placement |wall_x
- antenna_type |ideal |slab_placement |wall_y
