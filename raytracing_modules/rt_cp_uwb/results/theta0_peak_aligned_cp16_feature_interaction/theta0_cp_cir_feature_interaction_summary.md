# Theta0 CP/CIR Feature Interaction Audit

Inputs are the full theta0 peak-aligned CP16 Stage 1 and Stage 2 rerun CSVs. Stage 1 uses stratified 5-fold CV; Stage 2 uses room-label stratified 5-fold CV as the primary room-aware estimate.

## Correlation Summary

| stage | mean_abs_cp_cir_spearman | median_abs_cp_cir_spearman | max_abs_cp_cir_spearman | mean_abs_cp_cp_spearman | mean_abs_cir_cir_spearman |
|---|---:|---:|---:|---:|---:|
| stage1 | 0.461014 | 0.470493 | 1.000000 | 0.623663 | 0.526079 |
| stage2 | 0.350772 | 0.393024 | 1.000000 | 0.492647 | 0.511164 |

## Main Feature-Set Metrics

| stage | cv_scheme | feature_set | n_features | AUC | PR_AUC |
|---|---|---|---:|---:|---:|
| stage1 | stratified5 | CIR_only_12 | 12 | 0.678617 | 0.826282 |
| stage1 | stratified5 | CP_only_16 | 16 | 0.700426 | 0.830601 |
| stage1 | stratified5 | Joint_CIR_plus_CP16 | 28 | 0.762476 | 0.872029 |
| stage1 | stratified5 | Joint_CIR_plus_greedy_CP | 18 | 0.756493 | 0.866537 |
| stage2 | room_label_stratified5 | CIR_only_12 | 12 | 0.639640 | 0.615663 |
| stage2 | room_label_stratified5 | CP_only_16 | 16 | 0.683202 | 0.713771 |
| stage2 | room_label_stratified5 | Joint_CIR_plus_CP16 | 28 | 0.714759 | 0.719408 |
| stage2 | room_label_stratified5 | Joint_CIR_plus_greedy_CP | 15 | 0.724511 | 0.711359 |

## Save / Harm Using Greedy Joint

| stage | cv_scheme | AUC_CIR | AUC_CP | AUC_Joint | Delta_AUC | Delta_AUC_CI_low | cp_save | cp_harm | net_recovery | net_recovery_CI_low | nlos_jaccard |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stage1 | stratified5 | 0.678617 | 0.700426 | 0.756493 | 0.077876 | 0.062851 | 0.097667 | 0.039333 | 0.058333 | 0.046325 | 0.868579 |
| stage2 | room_label_stratified5 | 0.639640 | 0.683202 | 0.724511 | 0.084871 | 0.054478 | 0.153333 | 0.105556 | 0.047778 | 0.012222 | 0.352294 |

## Greedy Joint Selection


### stage1

| step | added_feature | n_cp | AUC | delta_vs_cir |
|---:|---|---:|---:|---:|
| 0 |  | 0 | 0.678617 | 0.000000 |
| 1 | delta_tau_l_given_r_s | 1 | 0.694177 | 0.015560 |
| 2 | s3_fp | 2 | 0.716966 | 0.038349 |
| 3 | s3_all | 3 | 0.747589 | 0.068972 |
| 4 | delta_f_l_minus_r_fp | 4 | 0.753963 | 0.075346 |
| 5 | cp_phase_slope_delay_s | 5 | 0.755463 | 0.076846 |
| 6 | s3_late | 6 | 0.756493 | 0.077876 |

### stage2

| step | added_feature | n_cp | AUC | delta_vs_cir |
|---:|---|---:|---:|---:|
| 0 |  | 0 | 0.639640 | 0.000000 |
| 1 | lambda_l_late_fraction | 1 | 0.721066 | 0.081426 |
| 2 | gamma_delay_linear | 2 | 0.724145 | 0.084505 |
| 3 | delta_f_l_minus_r_fp | 3 | 0.724511 | 0.084871 |
| 4 | f_r_fp | 4 | 0.724392 | 0.084752 |
| 5 | f_l_fp | 5 | 0.724303 | 0.084663 |
| 6 | gamma_anchor_linear | 6 | 0.723053 | 0.083413 |
