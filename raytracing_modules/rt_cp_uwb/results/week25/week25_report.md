# Week 2.5 Report

Generated: 2026-04-21 17:51:26

Week 3 launch decision: **ADDITIONAL_CALIBRATION_NEEDED**

- rationale: patch gamma_cp AUC 0.965 is too high, so the scene is still too simple

## Checklist

- [x] Subset AUC diagnostic
- [x] Signal 1 decision recorded
- [x] Feature duplicate cleanup to canonical 18
- [x] NLoS kurtosis diversity diagnostic
- [x] 200-case smoke rerun with updated sweep design

## Key Numbers

- failed ratio: 0.0150
- patch gamma_cp_3 AUC: 0.9650
- patch mean gamma_cp_3 (LoS): 0.0273
- patch mean gamma_cp_3 (NLoS): 0.6060
- NLoS kurtosis std: 26.3993
- top 3 AUC: gamma_cp_3_fp_only, gamma_cp_2_freq_db, gamma_cp_1_freq_avg

## Failure Breakdown

- `no_paths`: 3

## Feature Cleanup

- canonical feature count: 18
- kept duplicate-side representatives: `fp_to_total_ratio`, `fp_kurtosis`, `rise_time_fp`, `a_fp_6_fp_to_2nd_peak`
- dropped aliases: `a_fp_1_norm_energy`, `a_fp_4_kurt_local`, `a_fp_5_rise_time`, `a_fp_3_peak_to_max`

## Artifacts

- `D:\codex\raytracing_modules\rt_cp_uwb\results\week25\smoke_sweep_200.csv`
- `D:\codex\raytracing_modules\rt_cp_uwb\results\week25\initial_auc.csv`
- `D:\codex\raytracing_modules\rt_cp_uwb\results\week25\subset_auc_report.md`
- `D:\codex\raytracing_modules\rt_cp_uwb\results\week25\nlos_kurtosis_diag.md`
