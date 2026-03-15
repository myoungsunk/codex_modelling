# Intermediate Report (diag_all_scenarios_plus_bridges_a6on_20260309)

## Final Scenario Structure (Agreed)

| unit | role | notes |
| --- | --- | --- |
| C0 | calibration only | floor_reference 강화 |
| A2_off | G1 primary evidence | odd isolation, keep fixed |
| A6 | G2 primary evidence | near-normal PEC, incidence <= 15 deg |
| A3_corner | supplementary mechanism | window mismatch documented; no sign-off |
| A4_iso | L2-M primary | late_panel=false, dispersion=off |
| A4_bridge | L2-M secondary | late_panel=true, dispersion=on |
| A5_pair | L2-S proxy stress-response | synthetic primary, geometric sensitivity |
| A2_on/A3_on/A4_on/A6_on | bridge observability set | LOS-on contrast bridge |
| B1/B2/B3 | R1/R2 coverage-aware leverage map | viable strata/support count required; no universal claim |

## Proposition Status

| proposition | status | data_status | definition |
| --- | --- | --- | --- |
| M1 | SUPPORTED | OK | C0 floor distance/yaw sensitivity |
| M2 | SUPPORTED | OK | Fraction inside/outside floor uncertainty band |
| G1 | SUPPORTED | OK | A2 odd-bounce increases cross dominance vs C0 |
| G2 | PARTIAL | OK | A3 even-bounce recovery vs A2 (supplementary only when A6 is absent) |
| L1 | SUPPORTED | OK | Early/Late leakage separation via XPD_early_ex/XPD_late_ex/L_pol |
| L2 | SUPPORTED | OK | A5 stress impact on center/tails (response-mode requires geometric path perturbation) |
| L3 | SUPPORTED | OK | Correlation between leakage excess and -EL_proxy |
| R1 | SUPPORTED | OK | Coverage-aware room-space leverage map (LOS/NLOS + heatmap; viable strata only, not universal) |
| R2 | SUPPORTED | OK | Coverage-aware DS/early relation leverage map (B1/B2/B3 viable strata subset) |
| P1 | SUPPORTED | OK | Two-stage model (EL first, then mechanism effects) vs constant baseline |
| P2 | SUPPORTED | OK | Stage-1 EL coefficient sign stability under subsampling |

## A5 Stress Semantics

| n_stress_rows | n_response | n_polarization_only | dominant_semantics | contamination_response_ready |
| --- | --- | --- | --- | --- |
| 30 | 30 | 0 | response | True |

- 해석 규칙: `synthetic`(polarization_only)는 편파축 stress만 의미하며 delay/path 구조 변화는 주장하지 않음.
- delay/path contamination-response 해석은 `geometry`/`hybrid`(`response`) 샘플에서만 사용.

## M1

- Definition: C0 floor distance/yaw sensitivity
- Status: **SUPPORTED**
- Details: `{"definition": "C0 floor distance/yaw sensitivity", "n": 9, "trend": {"slope": 0.1440218950604963, "intercept": 25.0608810368984, "r": 0.13530320107240518, "p": 0.7285330843668445, "n": 9}, "variance_decomp": {"n": 9, "distance_group_var": 0.018248195767375592, "yaw_group_var": 0.014373810388657949, "dominant": "distance"}, "status": "SUPPORTED"}`

## M2

- Definition: Fraction inside/outside floor uncertainty band
- Status: **SUPPORTED**
- Details: `{"definition": "Fraction inside/outside floor uncertainty band", "inside_ratio": 0.018867924528301886, "outside_ratio": 0.9811320754716981, "status": "SUPPORTED"}`

## G1

- Definition: A2 odd-bounce increases cross dominance vs C0
- Status: **SUPPORTED**
- Details: `{"definition": "A2 odd-bounce increases cross dominance vs C0", "delta_median_db": -33.06286457580818, "ks_wasserstein": {"ks_stat": 1.0, "ks_p": 0.0027972027972027976, "wasserstein": 33.11734175922813}, "status": "SUPPORTED"}`

## G2

- Definition: A3 even-bounce recovery vs A2 (supplementary only when A6 is absent)
- Status: **PARTIAL**
- Details: `{"definition": "A3 even-bounce recovery vs A2 (supplementary only when A6 is absent)", "primary_source": "A3_target_window_supplementary_only", "primary_status": "INCONCLUSIVE", "delta_median_db": 9.833969727424119e-08, "ks_wasserstein": {"ks_stat": 0.5, "ks_p": 0.4043956043956044, "wasserstein": 2.4537196456125496}, "status": "PARTIAL"}`

## L1

- Definition: Early/Late leakage separation via XPD_early_ex/XPD_late_ex/L_pol
- Status: **SUPPORTED**
- Details: `{"definition": "Early/Late leakage separation via XPD_early_ex/XPD_late_ex/L_pol", "median_early_ex_db": -27.84289456670919, "median_late_ex_db": -25.06286481744288, "median_lpol_db": -3.1920244866599794, "status": "SUPPORTED"}`

## L2

- Definition: A5 stress impact on center/tails (response-mode requires geometric path perturbation)
- Status: **SUPPORTED**
- Details: `{"definition": "A5 stress impact on center/tails (response-mode requires geometric path perturbation)", "base": {"n": 30, "mean": -25.619775992382117, "std": 0.2542914826869117, "p5": -25.934654844856272, "p10": -25.91430038269322, "p90": -25.381085294768408, "p95": -25.114477092709688}, "stress": {"n": 30, "mean": -25.07452246060837, "std": 0.9853981777285886, "p5": -26.69747757413488, "p10": -26.5091960248636, "p90": -24.159465947972087, "p95": -23.577047938719634}, "delta_mean_db": 0.5452535317737457, "var_ratio": 3.875073468118587, "stress_semantics": {"n_a5_rows": 60, "n_stress_rows": 30, "n_response": 30, "n_polarization_only": 0, "n_off": 0, "n_unknown": 0, "dominant_semantics": "response", "contamination_response_ready": true}, "status": "SUPPORTED"}`

## L3

- Definition: Correlation between leakage excess and -EL_proxy
- Status: **SUPPORTED**
- Details: `{"definition": "Correlation between leakage excess and -EL_proxy", "spearman": {"rho": 0.4836928047602144, "p": 1.3703697854955685e-11, "ci_lo": 0.33974879522249085, "ci_hi": 0.6057739678618167, "n": 174}, "status": "SUPPORTED"}`

## R1

- Definition: Coverage-aware room-space leverage map (LOS/NLOS + heatmap; viable strata only, not universal)
- Status: **SUPPORTED**
- Details: `{"definition": "Coverage-aware room-space leverage map (LOS/NLOS + heatmap; viable strata only, not universal)", "n_b_rows": 75, "n_los": 63, "n_nlos": 12, "ks_wasserstein": {"ks_stat": 0.46825396825396826, "ks_p": 0.015672436630230942, "wasserstein": 4.140077197704343}, "status": "SUPPORTED"}`

## R2

- Definition: Coverage-aware DS/early relation leverage map (B1/B2/B3 viable strata subset)
- Status: **SUPPORTED**
- Details: `{"definition": "Coverage-aware DS/early relation leverage map (B1/B2/B3 viable strata subset)", "spearman_ds_vs_xpd": {"rho": -0.19303783814901057, "p": 0.0423675379297573, "ci_lo": -0.40889803113350565, "ci_hi": 0.03153771080487853, "n": 111}, "status": "SUPPORTED"}`

## P1

- Definition: Two-stage model (EL first, then mechanism effects) vs constant baseline
- Status: **SUPPORTED**
- Details: `{"definition": "Two-stage model (EL first, then mechanism effects) vs constant baseline", "cv_two_stage": {"n_stage1": 111, "n_stage2": 174, "b1_el": 0.11326497005891632, "rmse_const": 3.578180932981289, "rmse_lin": 2.421287787345314, "nll_const": 2.693725390000246, "nll_lin": 2.291427753436268}, "cv_one_shot_reference": {"n": 174, "rmse_const": 3.4180909373632553, "rmse_lin": 2.2087737905035403, "nll_const": 2.6476571374811146, "nll_lin": 2.2013132304617984}, "status": "SUPPORTED"}`

## P2

- Definition: Stage-1 EL coefficient sign stability under subsampling
- Status: **SUPPORTED**
- Details: `{"definition": "Stage-1 EL coefficient sign stability under subsampling", "subset": ["A3", "A4", "B1", "B2", "B3"], "stability": {"n": 111, "base_sign": 1.0, "sign_keep_rate": 0.91}, "status": "SUPPORTED"}`

## Scenario Sections

### A2

- 의미: Odd parity isolation anchor (A2_off core evidence); A2_on is observability bridge set.

![A2 scene](figures/A2__0__scene.png)


### A2_on

- 의미: LOS-on bridge observability counterpart of A2_off; direct LOS + odd-path coexistence check.

![A2_on scene](figures/A2_on__0__scene.png)


### A3

- 의미: Corner 2-bounce supplementary mechanism scene (target-window evidence only, not system early baseline); A3_on is LOS-on observability bridge set.
- 보고 규칙: A3는 mechanism-only 시나리오로 `target-window` 지표를 1차로 사용하고 fixed system `W_early` 우세는 보조 진단으로만 사용.

![A3 scene](figures/A3__0__scene.png)


### A3_on

- 의미: LOS-on bridge observability counterpart of A3_corner; direct LOS + even-path coexistence check.

![A3_on scene](figures/A3_on__0__scene.png)


### A4

- 의미: Material branch: A4_iso primary (late_panel=off, dispersion=off), A4_bridge secondary (late_panel=on, dispersion=on); A4_on is LOS-on observability bridge set.

![A4 scene](figures/A4__0__scene.png)


### A4_on

- 의미: LOS-on bridge observability counterpart of A4 material branch; direct LOS under material contrast.

![A4_on scene](figures/A4_on__0__scene.png)


### A5

- 의미: Proxy stress response pair (A5_pair): synthetic is primary, geometric is sensitivity check.

![A5 scene](figures/A5__0__scene.png)


### A6

- 의미: Near-normal PEC parity benchmark; primary G2 sign evidence.
- 보고 규칙: A6는 near-normal parity benchmark로 G2의 primary sign evidence에 사용.

![A6 scene](figures/A6__0__scene.png)


### A6_on

- 의미: LOS-on bridge observability counterpart of A6 parity benchmark; direct LOS + odd/even coexistence check.

![A6_on scene](figures/A6_on__0__scene.png)


### B1

- 의미: Room grid LOS anchor for coverage-aware leverage mapping (viable strata only; not universal).

![B1 scene](figures/B1__GLOBAL__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3C0__ALL__xpd_early_ex_cdf.png](figures/A2A3C0__ALL__xpd_early_ex_cdf.png)
- [ALL__early_late_ex_box.png](figures/ALL__early_late_ex_box.png)
- [A5__ALL__base_vs_stress_xpd_early_ex_cdf.png](figures/A5__ALL__base_vs_stress_xpd_early_ex_cdf.png)
- [ALL__xpd_early_ex_vs_el_proxy.png](figures/ALL__xpd_early_ex_vs_el_proxy.png)
- [C0__ALL__xpd_floor_vs_logd.png](figures/C0__ALL__xpd_floor_vs_logd.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)
- [B__ALL__heatmap_lpol.png](figures/B__ALL__heatmap_lpol.png)
- [B__ALL__heatmap_xpd_early_ex.png](figures/B__ALL__heatmap_xpd_early_ex.png)
- [B__ALL__los_nlos_xpd_ex_cdf.png](figures/B__ALL__los_nlos_xpd_ex_cdf.png)
- [ALL__ds_vs_xpd_early_ex.png](figures/ALL__ds_vs_xpd_early_ex.png)
- [ALL__early_fraction_vs_rho.png](figures/ALL__early_fraction_vs_rho.png)

### B2

- 의미: Room grid with partition obstacle for coverage-aware NLOS leverage mapping (structural-hole aware).

![B2 scene](figures/B2__GLOBAL__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3C0__ALL__xpd_early_ex_cdf.png](figures/A2A3C0__ALL__xpd_early_ex_cdf.png)
- [ALL__early_late_ex_box.png](figures/ALL__early_late_ex_box.png)
- [A5__ALL__base_vs_stress_xpd_early_ex_cdf.png](figures/A5__ALL__base_vs_stress_xpd_early_ex_cdf.png)
- [ALL__xpd_early_ex_vs_el_proxy.png](figures/ALL__xpd_early_ex_vs_el_proxy.png)
- [C0__ALL__xpd_floor_vs_logd.png](figures/C0__ALL__xpd_floor_vs_logd.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)
- [B__ALL__heatmap_lpol.png](figures/B__ALL__heatmap_lpol.png)
- [B__ALL__heatmap_xpd_early_ex.png](figures/B__ALL__heatmap_xpd_early_ex.png)
- [B__ALL__los_nlos_xpd_ex_cdf.png](figures/B__ALL__los_nlos_xpd_ex_cdf.png)
- [ALL__ds_vs_xpd_early_ex.png](figures/ALL__ds_vs_xpd_early_ex.png)
- [ALL__early_fraction_vs_rho.png](figures/ALL__early_fraction_vs_rho.png)

### B3

- 의미: Room grid corner-obstacle stress region for coverage-aware leverage mapping (structural-hole aware).

![B3 scene](figures/B3__GLOBAL__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3C0__ALL__xpd_early_ex_cdf.png](figures/A2A3C0__ALL__xpd_early_ex_cdf.png)
- [ALL__early_late_ex_box.png](figures/ALL__early_late_ex_box.png)
- [A5__ALL__base_vs_stress_xpd_early_ex_cdf.png](figures/A5__ALL__base_vs_stress_xpd_early_ex_cdf.png)
- [ALL__xpd_early_ex_vs_el_proxy.png](figures/ALL__xpd_early_ex_vs_el_proxy.png)
- [C0__ALL__xpd_floor_vs_logd.png](figures/C0__ALL__xpd_floor_vs_logd.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)
- [B__ALL__heatmap_lpol.png](figures/B__ALL__heatmap_lpol.png)
- [B__ALL__heatmap_xpd_early_ex.png](figures/B__ALL__heatmap_xpd_early_ex.png)
- [B__ALL__los_nlos_xpd_ex_cdf.png](figures/B__ALL__los_nlos_xpd_ex_cdf.png)
- [ALL__ds_vs_xpd_early_ex.png](figures/ALL__ds_vs_xpd_early_ex.png)
- [ALL__early_fraction_vs_rho.png](figures/ALL__early_fraction_vs_rho.png)

### C0

- 의미: Calibration-only baseline: floor reference and alignment sensitivity (W_floor).

![C0 scene](figures/C0__0__scene.png)


