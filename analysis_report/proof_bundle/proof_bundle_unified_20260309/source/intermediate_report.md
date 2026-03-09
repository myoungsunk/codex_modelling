# Intermediate Report (diag_protocol_repro_v1_20260309_r1)

## Final Scenario Structure (Agreed)

| unit | role | notes |
| --- | --- | --- |
| C0 | calibration only | floor_reference 강화 |
| A2_off | G1 primary evidence | odd isolation, keep fixed |
| A6 | G2 primary evidence | near-normal PEC, incidence <= 15 deg |
| A3_supp | supplementary mechanism | mechanism-only scope; WARN is role lock, no sign-off |
| A4_iso | L2-M primary | late_panel=false, dispersion=off |
| A4_bridge | L2-M secondary support | bridge/support scope; WARN is role lock, not weakness |
| A5_pair | L2-S contamination-response | paired base/on contamination-response only |
| A2_on/A3_on/A4_on/A6_on | bridge observability set | LOS-on contrast bridge |
| B1/B2/B3 | R1/R2 coverage-aware leverage map | viable strata/support count required; no universal claim |

## Proposition Status

| proposition | status | data_status | definition |
| --- | --- | --- | --- |
| M1 | SUPPORTED | OK | C0 floor distance/yaw sensitivity |
| M2 | SUPPORTED | OK | Fraction inside/outside floor uncertainty band |
| G1 | SUPPORTED | OK | A2 odd-bounce increases cross dominance vs C0 |
| G2 | SUPPORTED | OK | A6 near-normal parity benchmark as primary G2 evidence; A3 kept as mechanism-only supplementary evidence. |
| L1 | SUPPORTED | OK | Early/Late leakage separation via XPD_early_ex/XPD_late_ex/L_pol |
| L2 | SUPPORTED | OK | A5 stress impact on center/tails (response-mode requires geometric path perturbation) |
| L3 | SUPPORTED | OK | Correlation between leakage excess and -EL_proxy |
| R1 | SUPPORTED | OK | Coverage-aware room-space leverage map (LOS/NLOS + heatmap; viable strata only, not universal) |
| R2 | SUPPORTED | OK | Coverage-aware DS/early relation leverage map (B1/B2/B3 viable strata subset) |
| P1 | SUPPORTED | OK | Two-stage model (EL first, then mechanism effects) vs constant baseline |
| P2 | PARTIAL | OK | Stage-1 EL coefficient sign stability under subsampling |

## A5 Stress Semantics

| n_stress_rows | n_response | n_polarization_only | dominant_semantics | contamination_response_ready |
| --- | --- | --- | --- | --- |
| 30 | 30 | 0 | response | True |

- 해석 규칙: `synthetic`(polarization_only)는 편파축 stress만 의미하며 delay/path 구조 변화는 주장하지 않음.
- delay/path contamination-response 해석은 `geometry`/`hybrid`(`response`) 샘플에서만 사용.

## M1

- Definition: C0 floor distance/yaw sensitivity
- Status: **SUPPORTED**
- Details: `{"definition": "C0 floor distance/yaw sensitivity", "n": 125, "trend": {"slope": 0.023238995337509685, "intercept": 25.029897963477662, "r": 0.012076647160113689, "p": 0.893663984013666, "n": 125}, "variance_decomp": {"n": 125, "distance_group_var": 0.006061532906614555, "yaw_group_var": 0.0163228124392014, "dominant": "yaw"}, "status": "SUPPORTED"}`

## M2

- Definition: Fraction inside/outside floor uncertainty band
- Status: **SUPPORTED**
- Details: `{"definition": "Fraction inside/outside floor uncertainty band", "inside_ratio": 0.24944320712694878, "outside_ratio": 0.7505567928730512, "status": "SUPPORTED"}`

## G1

- Definition: A2 odd-bounce increases cross dominance vs C0
- Status: **SUPPORTED**
- Details: `{"definition": "A2 odd-bounce increases cross dominance vs C0", "delta_median_db": -33.05244952454008, "ks_wasserstein": {"ks_stat": 1.0, "ks_p": 1.8166333859430377e-07, "wasserstein": 33.0395613384995}, "status": "SUPPORTED"}`

## G2

- Definition: A6 near-normal parity benchmark as primary G2 evidence; A3 kept as mechanism-only supplementary evidence.
- Status: **SUPPORTED**
- Details: `{"definition": "A6 near-normal parity benchmark as primary G2 evidence; A3 kept as mechanism-only supplementary evidence.", "primary_source": "A6_near_normal_benchmark", "primary_status": "PASS", "A6_hit_rate_odd": 1.0, "A6_hit_rate_even": 1.0, "A6_n_eval_total": 20, "A3_delta_median_db_supplementary": 9.833970437966855e-08, "A3_ks_wasserstein_supplementary": {"ks_stat": 0.5, "ks_p": 0.4043956043956044, "wasserstein": 2.453719645612548}, "status": "SUPPORTED"}`

## L1

- Definition: Early/Late leakage separation via XPD_early_ex/XPD_late_ex/L_pol
- Status: **SUPPORTED**
- Details: `{"definition": "Early/Late leakage separation via XPD_early_ex/XPD_late_ex/L_pol", "median_early_ex_db": -30.96598371159874, "median_late_ex_db": -22.57143715038745, "median_lpol_db": -8.2558502494477, "status": "SUPPORTED"}`

## L2

- Definition: A5 stress impact on center/tails (response-mode requires geometric path perturbation)
- Status: **SUPPORTED**
- Details: `{"definition": "A5 stress impact on center/tails (response-mode requires geometric path perturbation)", "base": {"n": 30, "mean": -25.609360941114023, "std": 0.2542914826869116, "p5": -25.92423979358817, "p10": -25.90388533142512, "p90": -25.370670243500307, "p95": -25.104062041441587}, "stress": {"n": 30, "mean": -25.076730637607827, "std": 1.0344917954738737, "p5": -27.106139150933817, "p10": -26.51349305015846, "p90": -23.899005997481847, "p95": -23.792885031375356}, "delta_mean_db": 0.5326303035061954, "var_ratio": 4.0681338774824765, "stress_semantics": {"n_a5_rows": 60, "n_stress_rows": 30, "n_response": 30, "n_polarization_only": 0, "n_off": 0, "n_unknown": 0, "dominant_semantics": "response", "contamination_response_ready": true}, "status": "SUPPORTED"}`

## L3

- Definition: Correlation between leakage excess and -EL_proxy
- Status: **SUPPORTED**
- Details: `{"definition": "Correlation between leakage excess and -EL_proxy", "spearman": {"rho": 0.37098108841563526, "p": 2.409000000861539e-09, "ci_lo": 0.2441908838557069, "ci_hi": 0.48628043095648066, "n": 243}, "status": "SUPPORTED"}`

## R1

- Definition: Coverage-aware room-space leverage map (LOS/NLOS + heatmap; viable strata only, not universal)
- Status: **SUPPORTED**
- Details: `{"definition": "Coverage-aware room-space leverage map (LOS/NLOS + heatmap; viable strata only, not universal)", "n_b_rows": 147, "n_los": 112, "n_nlos": 35, "ks_wasserstein": {"ks_stat": 0.5153342070773264, "ks_p": 5.448384662358456e-07, "wasserstein": 4.709880183650388}, "status": "SUPPORTED"}`

## R2

- Definition: Coverage-aware DS/early relation leverage map (B1/B2/B3 viable strata subset)
- Status: **SUPPORTED**
- Details: `{"definition": "Coverage-aware DS/early relation leverage map (B1/B2/B3 viable strata subset)", "spearman_ds_vs_xpd": {"rho": -0.29512453259661015, "p": 0.0005366878880914245, "ci_lo": -0.46089669800925603, "ci_hi": -0.11773345911478794, "n": 134}, "status": "SUPPORTED"}`

## P1

- Definition: Two-stage model (EL first, then mechanism effects) vs constant baseline
- Status: **SUPPORTED**
- Details: `{"definition": "Two-stage model (EL first, then mechanism effects) vs constant baseline", "cv_two_stage": {"n_stage1": 180, "n_stage2": 243, "b1_el": -0.05576966365672963, "rmse_const": 3.506766373908685, "rmse_lin": 2.473334619309229, "nll_const": 2.6732362992334755, "nll_lin": 2.3209956109621395}, "cv_one_shot_reference": {"n": 243, "rmse_const": 3.551525639134785, "rmse_lin": 2.432010631928581, "nll_const": 2.6858108902240643, "nll_lin": 2.30340510773064}, "status": "SUPPORTED"}`

## P2

- Definition: Stage-1 EL coefficient sign stability under subsampling
- Status: **PARTIAL**
- Details: `{"definition": "Stage-1 EL coefficient sign stability under subsampling", "subset": ["A3", "A4", "B1", "B2", "B3"], "stability": {"n": 180, "base_sign": -1.0, "sign_keep_rate": 0.735}, "status": "PARTIAL"}`

## Scenario Sections

### A2

- 의미: Odd parity isolation anchor (A2_off core evidence); A2_on is observability bridge set.

![A2 scene](figures/A2__0__scene.png)


### A2_on

- 의미: LOS-on bridge observability counterpart of A2_off; direct LOS + odd-path coexistence check.

![A2_on scene](figures/A2_on__0__scene.png)


### A3

- 의미: A3_supp supplementary mechanism scene (target-window evidence only, not system early baseline); A3_on is LOS-on observability bridge set.
- 보고 규칙: A3는 mechanism-only 시나리오로 `target-window` 지표를 1차로 사용하고 fixed system `W_early` 우세는 보조 진단으로만 사용.
- G2 본증거는 A6 near-normal parity benchmark를 사용하고, A3는 보조 메커니즘 증거로만 사용.

![A3 scene](figures/A3__0__scene.png)


### A3_on

- 의미: LOS-on bridge observability counterpart of A3_supp; direct LOS + even-path coexistence check.

![A3_on scene](figures/A3_on__0__scene.png)


### A4

- 의미: Material branch: A4_iso primary (late_panel=off, dispersion=off), A4_bridge secondary (late_panel=on, dispersion=on); A4_on is LOS-on observability bridge set.

![A4 scene](figures/A4__0__scene.png)


### A4_on

- 의미: LOS-on bridge observability counterpart of A4 material branch; direct LOS under material contrast.

![A4_on scene](figures/A4_on__0__scene.png)


### A5

- 의미: A5_pair contamination-response pair (base/on): use for paired contamination/stress-response only.

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


