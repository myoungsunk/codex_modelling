# Week 2 Report

Generated: 2026-04-21 16:46:35

Week 3 launch status: **GO**

## Deliverables

- gamma convention fix and regression log: `D:\codex\raytracing_modules\rt_cp_uwb\results\week2_day1_convention_test.log`
- smoke sweep artifacts: `D:\codex\raytracing_modules\rt_cp_uwb\results\week2\smoke_sweep_200.csv`, `D:\codex\raytracing_modules\rt_cp_uwb\results\week2\smoke_sweep_200.mat`, `D:\codex\raytracing_modules\rt_cp_uwb\results\week2\smoke_diagnostics.png`
- initial per-feature AUC ranking: `D:\codex\raytracing_modules\rt_cp_uwb\results\week2\initial_auc.csv`
- analysis utilities: `+analysis/computeConditionalAuc.m`, `+analysis/computeDisagreementAnalysis.m`

## Launch Checklist

| Item | Status | Detail |
| --- | --- | --- |
| gamma_CP convention fixed and regression test passed | PASS | Week 2 Day 1 convention log exists and contains pass marker |
| materialsLibrary 8 materials work | PASS | All 8 predefined materials instantiate without error |
| Single-slab scene + runOneCase end-to-end | PASS | single_slab_paths=2, runOneCase_num_paths=2 |
| 200-case smoke sweep failed-case ratio < 5% | PASS | failed=0/200 (0.00%) |
| Per-feature AUC ranking available and top 3 identified | PASS | top3=gamma_cp_3_fp_only, kurtosis_total, peak_to_avg_ratio |
| Feature correlation checked and duplicate candidates identified | PASS | a_fp_1_norm_energy vs fp_to_total_ratio (\|r\|=1.0000) |
| Conditional AUC utility works on smoke set | PASS | finite bins=5/5, mean delta AUC=0.0763 |
| Disagreement analysis utility works | PASS | misc_rate=0.1150, auc_cir_cv=0.8945 |
| Failed-case analysis completed | PASS | 0 failed cases in the 200-case smoke sweep; no active geometry/tracer failure mode observed |

## Smoke Sweep Statistics

- Cases: 200
- Failed: 0 (0.00%)
- LoS: 100
- NLoS: 100
- Numeric non-finite count: 0

## Top AUC Features

| Rank | Feature | AUC | n_valid |
| --- | --- | ---: | ---: |
| 1 | gamma_cp_3_fp_only | 0.9273 | 200 |
| 2 | kurtosis_total | 0.8701 | 200 |
| 3 | peak_to_avg_ratio | 0.7974 | 200 |
| 4 | gamma_cp_4_total_energy | 0.6431 | 200 |
| 5 | gamma_cp_2_freq_db | 0.6424 | 200 |

## Correlation And Redundancy

| Feature A | Feature B | corr | exact duplicate |
| --- | --- | ---: | --- |
| a_fp_1_norm_energy | fp_to_total_ratio | 1.0000 | true |
| a_fp_3_peak_to_max | a_fp_6_fp_to_2nd_peak | 1.0000 | true |
| a_fp_4_kurt_local | fp_kurtosis | 1.0000 | true |
| a_fp_5_rise_time | rise_time_fp | 1.0000 | true |
| gamma_cp_4_total_energy | gamma_cp_5_post_fp | 1.0000 | false |
| a_fp_1_norm_energy | k_factor_estimate | 0.9992 | false |

## Conditional AUC Smoke Test

- Condition variable: `incidence_deg`
- Mean delta AUC (joint - CIR): 0.0763
- Finite bins: 5 / 5
| Bin | n | delta_auc |
| --- | ---: | ---: |
| 1 | 40 | 0.0800 |
| 2 | 40 | 0.0921 |
| 3 | 40 | 0.1250 |
| 4 | 40 | 0.0844 |
| 5 | 40 | 0.0000 |

## Disagreement Analysis Smoke Test

- CIR-only CV AUC: 0.8945
- Misclassification rate: 0.1150 (23 / 200)
- Strongest CP rescue signal: `gamma_cp_2_freq_db` with KS=0.4166, p=0.001079

## Failed Case Analysis

- 0 failed cases in the 200-case smoke sweep; no active geometry/tracer failure mode observed

## Known Issues And Limitations

- Smoke sweep is only 200 cases; Week 3 full run is still pending at n=3000.
- Stage 1 geometry is a single-slab scene with a synthetic blocker, so the current separation trends are not yet room-scale results.
- Patch antenna cases still use the synthetic patch surrogate rather than full angular pattern interpolation from measured/simulated FFD data.
- Several feature pairs are redundant enough to be pruning candidates before the 3000-case run.

