# Stage 3 Theta0 CP16 Selection

This Stage 3 selection is regenerated from the theta0 peak-aligned CP16 Stage 2 full rerun. It does not overwrite the legacy `results/stage3` deterministic list.

- source_csv: `D:\codex\raytracing_modules\rt_cp_uwb\results\stage2_theta0_peak_aligned_cp16_phi_fastest\stage2_900_theta0_peak_aligned_cp16_phi_fastest_det.csv`
- valid_stage2_cases: 900
- selected_cases: 30
- primary_joint: `CIR12 + lambda_l_late_fraction + gamma_delay_linear + delta_f_l_minus_r_fp`
- groups: GEO and BOUNCE

## Selected Regimes

| group | regime | n | n_pos | n_neg | auc_cir | auc_joint | delta_auc | pr_auc_cir | pr_auc_joint |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GEO | dominant_wall_material_bin=concrete | 180 | 14 | 166 | 0.426850 | 0.993546 | 0.566695 | 0.067584 | 0.936080 |
| GEO | xpol_bin=(24.017, 27.995] | 180 | 13 | 167 | 0.432520 | 0.995854 | 0.563335 | 0.064218 | 0.939497 |
| GEO | los_angle_bin=(46.115, 52.704] | 180 | 14 | 166 | 0.547332 | 0.978916 | 0.431583 | 0.088744 | 0.887317 |
| BOUNCE | los_angle_bin=(40.556, 46.115] | 156 | 90 | 66 | 0.632828 | 0.836869 | 0.204040 | 0.733787 | 0.888266 |
| BOUNCE | xpol_bin=(31.988, 35.975] | 170 | 76 | 94 | 0.518337 | 0.654955 | 0.136618 | 0.461942 | 0.548833 |
| BOUNCE | snr_bin=(33.995, 39.984] | 169 | 75 | 94 | 0.603830 | 0.674894 | 0.071064 | 0.491349 | 0.575645 |

## Case Distribution

| group | room | n |
|---|---|---:|
| BOUNCE | A | 4 |
| BOUNCE | B | 7 |
| BOUNCE | C | 4 |
| GEO | A | 3 |
| GEO | B | 2 |
| GEO | C | 10 |
