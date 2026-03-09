# Detailed Proposition-Experiment-Data-Plot Mapping (diag_protocol_repro_v1_20260309_r1)

## 0) 공통 플롯 규칙 요약
- 공통 지표: XPD_floor, XPD_target_ex, XPD_early_ex, XPD_late_ex, rho_early, L_pol, DS, EL_proxy
- window 규칙: W_floor(C0), W_target(A2/A3/A4), W_early/B, W_late
- 시나리오 역할: C0=floor, A2=odd, A3=even-mechanism, A4=material, A5=stress-response, B1/B2/B3=coverage-aware leverage map

## Summary
- Detailed plots ready: 45/46
- Proposition PASS: 43/46 (row-level reference)
- Proposition PARTIAL: 3/46
- Proposition FAIL: 0/46

## Mapping Table

| plot_id | 명제 | 시나리오(실험) | 필요한 데이터 | 플롯 | 데이터 CSV(x,y,data) | 통과 기준 | 명제 PASS/FAIL | 플롯 상태 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1-1 | M1 | C0 | co/cross PDP, W_floor | [M1-1__C0_raw_pdp_overlay_grid.png](figures/M1-1__C0_raw_pdp_overlay_grid.png) | [M1-1__data.csv](tables/plot_data/M1-1__data.csv) | LOS-only contamination low | PASS | READY |
| M1-2 | M1 | C0 | rays(C_floor) | [M1-2__C_floor_box_by_yaw.png](figures/M1-2__C_floor_box_by_yaw.png) | [M1-2__data.csv](tables/plot_data/M1-2__data.csv) | C_floor below -10/-15 dB | PASS | READY |
| M1-3 | M1 | C0 | XPD_floor,d | [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png) | [M1-3__data.csv](tables/plot_data/M1-3__data.csv) | weak distance dependence | PASS | READY |
| M1-4 | M1 | C0 | XPD_floor,yaw | [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png) | [M1-4__data.csv](tables/plot_data/M1-4__data.csv) | yaw-driven spread | PASS | READY |
| M1-5 | M1 | C0 | xpd_floor_curve,freq | [M1-5__xpd_floor_vs_frequency.png](figures/M1-5__xpd_floor_vs_frequency.png) | [M1-5__data.csv](tables/plot_data/M1-5__data.csv) | frequency baseline decision | PASS | READY |
| M1-6 | M1 | C0 | repeat_id | [M1-6__repeatability_strip_box.png](figures/M1-6__repeatability_strip_box.png) | [M1-6__data.csv](tables/plot_data/M1-6__data.csv) | Delta_repeat estimation | PASS | READY |
| M1-7 | M1 | C0 | Delta_floor,Delta_repeat | [M1-7__uncertainty_budget_summary.png](figures/M1-7__uncertainty_budget_summary.png) | [M1-7__data.csv](tables/plot_data/M1-7__data.csv) | Delta_ref=max() | PASS | READY |
| M2-1 | M2 | A2/A3/A4/A5/B | XPD_early_ex, XPD_late_ex | [M2-1__scenario_early_late_ex_box.png](figures/M2-1__scenario_early_late_ex_box.png) | [M2-1__data.csv](tables/plot_data/M2-1__data.csv) | beyond ±Delta_ref | PASS | READY |
| M2-2 | M2 | A2/A3/A4/A5/B | XPD_ex,Delta_ref | [M2-2__exceedance_rate_bar.png](figures/M2-2__exceedance_rate_bar.png) | [M2-2__data.csv](tables/plot_data/M2-2__data.csv) | channel-claim fraction | PASS | READY |
| M2-3 | M2 | C0 + indoor | floor vs indoor distributions | [M2-3__cdf_floor_vs_indoor_overlay.png](figures/M2-3__cdf_floor_vs_indoor_overlay.png) | [M2-3__data.csv](tables/plot_data/M2-3__data.csv) | separable distributions | PASS | READY |
| G1-1 | G1 | A2 | PDP, W_target/W_early | [G1-1__A2_pdp_overlay_target_early.png](figures/G1-1__A2_pdp_overlay_target_early.png) | [G1-1__data.csv](tables/plot_data/G1-1__data.csv) | odd cross-dominant trend | PASS | READY |
| G1-2 | G1 | A2 | XPD_early_ex | [G1-2__A2_xpd_early_ex_by_case.png](figures/G1-2__A2_xpd_early_ex_by_case.png) | [G1-2__data.csv](tables/plot_data/G1-2__data.csv) | sign stability | PASS | READY |
| G1-3 | G1 | A2 | rho_early(linear) + incidence | [G1-3__A2_rho_vs_incidence.png](figures/G1-3__A2_rho_vs_incidence.png) | [G1-3__data.csv](tables/plot_data/G1-3__data.csv) | odd supports rho increase | PASS | READY |
| G2-1 | G2 | A3 | PDP + target window | [G2-1__A3_pdp_overlay_target_early.png](figures/G2-1__A3_pdp_overlay_target_early.png) | [G2-1__data.csv](tables/plot_data/G2-1__data.csv) | even mechanism visibility | PASS | READY |
| G2-2 | G2 | A3 | target-window summary | [G2-2__A3_xpd_target_raw_summary.png](figures/G2-2__A3_xpd_target_raw_summary.png) | [G2-2__data.csv](tables/plot_data/G2-2__data.csv) | co-dominant tendency | PASS | READY |
| G2-3 | G2 | A2/A3 | target-window summary | [G2-3__A2_vs_A3_target_window_compare.png](figures/G2-3__A2_vs_A3_target_window_compare.png) | [G2-3__data.csv](tables/plot_data/G2-3__data.csv) | sign reversal | PASS | READY |
| G2-4 | G2 | A3 | target_in_early/first rates | [G2-4__A3_target_inearly_first_rate.png](figures/G2-4__A3_target_inearly_first_rate.png) | [G2-4__data.csv](tables/plot_data/G2-4__data.csv) | role split | PASS | READY |
| G3-1 | G3 | A2/A3 | XPD_ex | [G3-1__A2_A3_xpd_ex_violin.png](figures/G3-1__A2_A3_xpd_ex_violin.png) | [G3-1__data.csv](tables/plot_data/G3-1__data.csv) | not perfectly separable | PASS | READY |
| G3-2 | G3 | A2/A3/A4/A5 | |XPD_ex| + U | [G3-2__abs_xpd_ex_vs_el_scatter.png](figures/G3-2__abs_xpd_ex_vs_el_scatter.png) | [G3-2__data.csv](tables/plot_data/G3-2__data.csv) | conditional variation | PASS | READY |
| G3-3 | G3 | A2/A3/A4/A5 | variance | [G3-3__variance_xpd_ex_by_scenario.png](figures/G3-3__variance_xpd_ex_by_scenario.png) | [G3-3__data.csv](tables/plot_data/G3-3__data.csv) | leakage spread quantification | PASS | READY |
| L1-1 | L1 | A2-A5+B | early/late excess | [L1-1__early_late_paired_lines.png](figures/L1-1__early_late_paired_lines.png) | [L1-1__data.csv](tables/plot_data/L1-1__data.csv) | late dirtier trend | PASS | READY |
| L1-2 | L1 | A2-A5+B | L_pol | [L1-2__lpol_box_by_scenario.png](figures/L1-2__lpol_box_by_scenario.png) | [L1-2__data.csv](tables/plot_data/L1-2__data.csv) | L_pol > 0 tendency | PASS | READY |
| L2-M1 | L2 | A4 | material,angle,XPD_ex | [L2-M1__A4_material_angle_xpd_early_ex.png](figures/L2-M1__A4_material_angle_xpd_early_ex.png) | [L2-M1__data.csv](tables/plot_data/L2-M1__data.csv) | material dependence | PASS | READY |
| L2-M2 | L2 | A4 | |XPD_ex| | [L2-M2__A4_abs_xpd_ex_by_material.png](figures/L2-M2__A4_abs_xpd_ex_by_material.png) | [L2-M2__data.csv](tables/plot_data/L2-M2__data.csv) | toward 0dB/dispersion | PASS | READY |
| L2-M3 | L2 | A4 | variance | [L2-M3__A4_variance_by_material.png](figures/L2-M3__A4_variance_by_material.png) | [L2-M3__data.csv](tables/plot_data/L2-M3__data.csv) | dispersion growth | PASS | READY |
| L2-S1 | L2 | A5 | base/stress L_pol | [L2-S1__A5_base_stress_lpol_paired.png](figures/L2-S1__A5_base_stress_lpol_paired.png) | [L2-S1__data.csv](tables/plot_data/L2-S1__data.csv) | stress-response | PASS | MISSING |
| L2-S2 | L2 | A5 | base/stress late_ex | [L2-S2__A5_base_stress_xpd_late_ex.png](figures/L2-S2__A5_base_stress_xpd_late_ex.png) | [L2-S2__data.csv](tables/plot_data/L2-S2__data.csv) | late contamination | PASS | READY |
| L2-S3 | L2 | A5 | base/stress DS | [L2-S3__A5_base_stress_ds_tail.png](figures/L2-S3__A5_base_stress_ds_tail.png) | [L2-S3__data.csv](tables/plot_data/L2-S3__data.csv) | tail widening | PASS | READY |
| L3-1 | L3 | A3/A4/B | EL,XPD_ex | [L3-1__el_vs_xpd_ex_scatter_regression.png](figures/L3-1__el_vs_xpd_ex_scatter_regression.png) | [L3-1__data.csv](tables/plot_data/L3-1__data.csv) | a1<0 monotonic tendency | PASS | READY |
| L3-2 | L3 | A3/A4/B | EL bins | [L3-2__el_bin_conditional_box.png](figures/L3-2__el_bin_conditional_box.png) | [L3-2__data.csv](tables/plot_data/L3-2__data.csv) | nonparametric monotonicity | PASS | READY |
| L3-3 | L3 | stage1 fit | EL,fitted mean | [L3-3__residualized_effect_fit_ci.png](figures/L3-3__residualized_effect_fit_ci.png) | [L3-3__data.csv](tables/plot_data/L3-3__data.csv) | mu(U) visualization | PASS | READY |
| R1-1 | R1 | B1/B2/B3 | x,y,XPD_early_ex | [R1-1__B123_heatmap_xpd_early_ex.png](figures/R1-1__B123_heatmap_xpd_early_ex.png) | [R1-1__data.csv](tables/plot_data/R1-1__data.csv) | coverage-aware spatial pattern | PASS | READY |
| R1-2 | R1 | B1/B2/B3 | x,y,rho_early(linear) | [R1-2__B123_heatmap_rho_early.png](figures/R1-2__B123_heatmap_rho_early.png) | [R1-2__data.csv](tables/plot_data/R1-2__data.csv) | coverage-aware contamination map | PASS | READY |
| R1-3 | R1 | B1/B2/B3 | x,y,L_pol | [R1-3__B123_heatmap_lpol.png](figures/R1-3__B123_heatmap_lpol.png) | [R1-3__data.csv](tables/plot_data/R1-3__data.csv) | coverage-aware early-late structure map | PASS | READY |
| R1-4 | R1 | B pooled | LOS/NLOS grouped metrics | [R1-4__los_nlos_grouped_metrics.png](figures/R1-4__los_nlos_grouped_metrics.png) | [R1-4__data.csv](tables/plot_data/R1-4__data.csv) | group difference | PASS | READY |
| R2-1 | R2 | B pooled | XPD_early_ex,DS | [R2-1__xpd_early_ex_vs_ds_B123.png](figures/R2-1__xpd_early_ex_vs_ds_B123.png) | [R2-1__data.csv](tables/plot_data/R2-1__data.csv) | useful/risky relation | PASS | READY |
| R2-2 | R2 | B pooled | rho_early(linear),DS | [R2-2__rho_early_vs_ds_B123.png](figures/R2-2__rho_early_vs_ds_B123.png) | [R2-2__data.csv](tables/plot_data/R2-2__data.csv) | contamination relation | PASS | READY |
| R2-3 | R2 | B pooled | XPD,DS,L_pol | [R2-3__quadrant_useful_vs_risky.png](figures/R2-3__quadrant_useful_vs_risky.png) | [R2-3__data.csv](tables/plot_data/R2-3__data.csv) | coverage-aware useful vs risky regions | PASS | READY |
| R2-4 | R2 | B pooled | early_fraction,XPD | [R2-4__early_fraction_vs_xpd_early_ex.png](figures/R2-4__early_fraction_vs_xpd_early_ex.png) | [R2-4__data.csv](tables/plot_data/R2-4__data.csv) | early concentration link | PASS | READY |
| P1-1 | P1 | A3/A4/B | observed,predicted | [P1-1__observed_vs_predicted_cdf_overlay.png](figures/P1-1__observed_vs_predicted_cdf_overlay.png) | [P1-1__data.csv](tables/plot_data/P1-1__data.csv) | conditional>constant | PASS | READY |
| P1-2 | P1 | A3/A4/B | bin medians | [P1-2__predicted_vs_observed_bin_median.png](figures/P1-2__predicted_vs_observed_bin_median.png) | [P1-2__data.csv](tables/plot_data/P1-2__data.csv) | condition-wise fit | PASS | READY |
| P1-3 | P1 | A3/A4/B | ranks | [P1-3__rank_agreement_scatter.png](figures/P1-3__rank_agreement_scatter.png) | [P1-3__data.csv](tables/plot_data/P1-3__data.csv) | Spearman advantage | PASS | READY |
| P1-4 | P1 | A3/A4/B | residual | [P1-4__residual_vs_el.png](figures/P1-4__residual_vs_el.png) | [P1-4__data.csv](tables/plot_data/P1-4__data.csv) | reduced EL bias | PASS | READY |
| P2-1 | P2 | full vs minimal | effect-size set | [P2-1__full_vs_minimal_effect_size.png](figures/P2-1__full_vs_minimal_effect_size.png) | [P2-1__data.csv](tables/plot_data/P2-1__data.csv) | minimal reproducibility | PARTIAL | READY |
| P2-2 | P2 | subsampling | coefficients | [P2-2__coefficient_stability_subsampling.png](figures/P2-2__coefficient_stability_subsampling.png) | [P2-2__data.csv](tables/plot_data/P2-2__data.csv) | sign/scale stability | PARTIAL | READY |
| P2-3 | P2 | full vs minimal | CDF | [P2-3__full_vs_minimal_cdf_overlay.png](figures/P2-3__full_vs_minimal_cdf_overlay.png) | [P2-3__data.csv](tables/plot_data/P2-3__data.csv) | distribution reproducibility | PARTIAL | READY |

## Notes
- M1-1: C0 facet raw PDP
- M1-2: C_floor threshold lines
- M1-5: frequency-dependent floor
- M1-6: repeat strip+box
- M1-7: Delta_ref=max(Delta_floor,Delta_repeat)
- M2-1: with ±Delta_ref band + claim_caution markers
- M2-2: early/late exceedance
- M2-3: C0 floor vs indoor excess
- G1-1: W_target/W_early shown
- G1-2: A2 zoom + A3/A4 contrast
- G1-3: dominant-incidence proxy
- G2-1: A3 mechanism view
- G2-2: from A3_target_window_sign.csv (raw)
- G2-3: target-window median compare (raw)
- G2-4: diagnostic role split
- G3-1: A2/A3 variance view
- G3-2: conditioned by EL
- G3-3: variance summary
- L1-1: paired sampled cases
- L1-2: zero line shown
- L2-M1: material x angle-bin
- L2-M2: toward 0 dB check
- L2-M3: variance/IQR proxy
- L2-S2: late-ex branch
- L2-S3: stress tail widening proxy
- L3-1: subset=A3/A4/B
- L3-2: nonparametric monotonicity
- L3-3: stage1-style fit
- R1-1: facet B1/B2/B3 with support n
- R1-2: facet B1/B2/B3 with support n
- R1-3: facet B1/B2/B3 with support n
- R1-4: LOS/NLOS grouped CDFs with support n
- R2-1: B1/B2/B3 pooled
- R2-2: B1/B2/B3 pooled
- R2-3: median-threshold quadrants
- R2-4: system-surrogate link
- P1-1: simple linear proxy
- P1-2: EL tertiles
- P1-3: Spearman-oriented
- P1-4: EL bias check
- P2-1: G1/G2/L3/R1 effect compare
- P2-2: bootstrap slope distribution
- P2-3: distribution-level reproducibility