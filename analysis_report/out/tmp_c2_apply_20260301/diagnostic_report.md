# Diagnostic Report (tmp_c2_apply_20260301)

## Dataset Summary

| scenario | n_links |
| --- | --- |
| A2 | 27 |
| A3 | 24 |
| A4 | 72 |
| A5 | 60 |
| B1 | 63 |
| B2 | 49 |
| B3 | 49 |
| C0 | 30 |

## Floor Reference

| xpd_floor_db | delta_floor_db | p5_db | p95_db | count | method |
| --- | --- | --- | --- | --- | --- |
| 24.93 | 0.6107 | 24.37 | 25.59 | 30 | p5_p95 |

## Diagnostics A-E

### A) Geometry / Path Validity

| scenario | los_rays | status | reason |
| --- | --- | --- | --- |
| A2 | 0 | PASS |  |
| A3 | 0 | PASS |  |
| A4 | 0 | PASS |  |
| A5 | 0 | PASS |  |

| scenario | target_n | hit | total | rate | status |
| --- | --- | --- | --- | --- | --- |
| A2 | 1 | 27 | 27 | 1 | PASS |
| A3 | 2 | 24 | 24 | 1 | PASS |

- A3 coordinate sanity: **WARN** (Coordinate penetration sanity needs scenario geometry file review; not inferable from standard outputs only.)

- A3 geometry manual review (ray-path visualization required before experiment)

| scenario_id | case_id | review_status | scene_debug_valid | los_rays | has_target_bounce_n2 | rays_topk_n | scene_debug_issues |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A3 | 0 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 1 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 10 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 11 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 12 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 13 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 14 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 15 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 16 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 17 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 18 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 19 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 2 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 20 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 21 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 22 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 23 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 3 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 4 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 5 | PASS | 1 | 0 | 1 | 2 |  |

#### A3 case 0

- review_status: **PASS**
![A3-case-0](figures/A3__0__scene.png)

#### A3 case 1

- review_status: **PASS**
![A3-case-1](figures/A3__1__scene.png)

#### A3 case 10

- review_status: **PASS**
![A3-case-10](figures/A3__10__scene.png)

#### A3 case 11

- review_status: **PASS**
![A3-case-11](figures/A3__11__scene.png)

#### A3 case 12

- review_status: **PASS**
![A3-case-12](figures/A3__12__scene.png)

#### A3 case 13

- review_status: **PASS**
![A3-case-13](figures/A3__13__scene.png)

#### A3 case 14

- review_status: **PASS**
![A3-case-14](figures/A3__14__scene.png)

#### A3 case 15

- review_status: **PASS**
![A3-case-15](figures/A3__15__scene.png)

#### A3 case 16

- review_status: **PASS**
![A3-case-16](figures/A3__16__scene.png)

#### A3 case 17

- review_status: **PASS**
![A3-case-17](figures/A3__17__scene.png)

#### A3 case 18

- review_status: **PASS**
![A3-case-18](figures/A3__18__scene.png)

#### A3 case 19

- review_status: **PASS**
![A3-case-19](figures/A3__19__scene.png)

#### A3 case 2

- review_status: **PASS**
![A3-case-2](figures/A3__2__scene.png)

#### A3 case 20

- review_status: **PASS**
![A3-case-20](figures/A3__20__scene.png)

#### A3 case 21

- review_status: **PASS**
![A3-case-21](figures/A3__21__scene.png)

#### A3 case 22

- review_status: **PASS**
![A3-case-22](figures/A3__22__scene.png)

#### A3 case 23

- review_status: **PASS**
![A3-case-23](figures/A3__23__scene.png)

#### A3 case 3

- review_status: **PASS**
![A3-case-3](figures/A3__3__scene.png)

#### A3 case 4

- review_status: **PASS**
![A3-case-4](figures/A3__4__scene.png)

#### A3 case 5

- review_status: **PASS**
![A3-case-5](figures/A3__5__scene.png)

### B) Time Resolution / Delay Separability

- `W_floor`(C0): `C_floor = Sum(P_nonLOS in W_floor) / P_LOS`
- `W_target`(A2-A5): `C_target = Sum(P_non-target in W_target) / P_target`
- `W_early`(B1-B3): `S(Te)=|mu_LOS-mu_NLOS|/sqrt(sig_LOS^2+sig_NLOS^2)`

| freq_source | dt_res_s | tau_max_s | Te_s | Tmax_s | W_floor_C_median_db | W_floor_status | A2_target_in_Wearly_rate | A3_target_in_Wearly_rate | A2_C_target_median_db | A3_C_target_median_db | A2_target_sign_hit_rate | A2_target_sign_status | A3_target_sign_hit_rate | A3_target_sign_status | A3_mechanism_status | A3_system_early_status | A5_target_mode | min_delay_gap_median_s | B2_status | B3_status | W3_best_te_ns | W3_best_S_xpd_early | W3_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| link_rows.xpd_floor_freq_hz[C0] | 2.5e-10 | 2.558e-07 | 3e-09 | 3e-08 | -237.2 | PASS | 1 | 0.1667 | -229.6 | -228.5 | 1 | PASS | 0 | FAIL | PASS | FAIL | contamination_response | 8.916e-10 | PASS | PASS | 3 | 0.8905 | WARN |

- W_floor(C0) contamination summary

| W_floor_s | C_floor_median_db | C_floor_p95_db | rate_below_m10_db | rate_below_m15_db | status | n_cases |
| --- | --- | --- | --- | --- | --- | --- |
| 1.5e-09 | -237.2 | -232.8 | 1 | 1 | PASS | 30 |

- W_target(controlled scenarios) summary

| scenario | target_n | target_exists_rate | target_is_first_rate | target_in_Wearly_rate | C_target_median_db | target_gap_median_s | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A2 | 1 | 1 | 1 | 1 | -229.6 | nan | PASS |
| A3 | 2 | 1 | 0 | 0.1667 | -228.5 | 4.67e-09 | WARN |
| A4 | 1 | 1 | 1 | 1 | -220.6 | 1.135e-08 | PASS |
| A5 | 2 | 1 | 0.2333 | 0.9667 | -0.1857 | 5.457e-10 | FAIL |

- W_early(room/grid) Te sweep separation

| Te_ns | S_xpd_early | S_rho_early_db | S_l_pol |
| --- | --- | --- | --- |
| 2 | 0.8572 | 0.8572 | 0.6925 |
| 3 | 0.8905 | 0.8905 | 0.8051 |
| 5 | 0.8168 | 0.8168 | 0.7716 |

- A2/A3 odd-even sign stability over Te sweep: **FAIL**

| scenario | expected_sign | min_hit_rate | median_hit_rate | status |
| --- | --- | --- | --- | --- |
| A2 | negative | 1 | 1 | PASS |
| A3 | positive | 0 | 0 | FAIL |

- Target-window sign metric (A2/A3)

| scenario | target_n | W_target_s | expected_sign | n_eval | expected_sign_hit_rate | median_xpd_target_ex_db | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A2 | 1 | 3e-09 | negative | 27 | 1 | -32.93 | PASS |
| A3 | 2 | 3e-09 | positive | 24 | 0 | -16.93 | FAIL |

### C) Effect Size vs Floor Uncertainty

- C2-M primary: `XPD_early_excess`; secondary: `XPD_late_excess`, `L_pol`
- C2-S primary: `L_pol`; secondary: `rho_early`, `DS`, `XPD_late_excess`; gate: `ΔP_target,total > -6 dB`

| floor_delta_db | repeat_delta_db | delta_ref_db | A3_minus_A2_delta_median_db | ratio_to_floor | C1_status | C2M_primary_span_db | C2M_primary_status | C2M_secondary_late_span_db | C2M_secondary_late_status | C2M_status | C2S_delta_lpol_db | C2S_primary_status | C2S_gate_delta_p_target_db | C2S_gate_status | C2S_status | C2_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.6107 | 0.2231 | 0.6107 | 3.49 | 5.714 | PASS | 8.189 | PASS | 8.145 | PASS | PASS | -5.853 | PASS | 0 | PASS | PASS | PASS |

### D) Identifiability

| status | EL_iqr_db | corr_d_vs_LOS | min_strata_n |
| --- | --- | --- | --- |
| PASS | 7.705 | -0.5151 | 7 |

- D1 split: EL-identifying coverage(global) + parity/stress isolation(local)

| component | status | role | EL_iqr_db | min_bin_n | n_rows | EL_std_db | delta_median_EL_stress_minus_base_db | n_base | n_stress | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1_global | PASS |  | 7.705 | 84 | 257 |  |  |  |  |  |
| A2_isolation | PASS |  |  |  | 27 | 6.828e-15 |  |  |  | small EL variation is desired for parity isolation |
| A5_isolation | PASS | stress_response |  |  |  |  | -2.489 | 30 | 30 | response mode: L_pol decrease is primary, EL shift can be non-zero |

- D2 split: material×angle coverage + stress×angle coverage + collinearity diagnostics

| status | material_x_angle_status | stress_x_angle_status | design_status | design_rank | design_cols | condition_number |
| --- | --- | --- | --- | --- | --- | --- |
| PASS | PASS | PASS | PASS | 8 | 8 | 86.29 |

| stage | status | n_rows | design_rank | design_cols | condition_number |
| --- | --- | --- | --- | --- | --- |
| stage1_EL_identifying | PASS | 257 | 8 | 8 | 86.29 |
| stage2_effects_after_EL | PASS | 344 | 5 | 5 | 8.015 |

| group | low | mid | high | NA |
| --- | --- | --- | --- | --- |
| glass | 2 | 10 | 12 | 0 |
| gypsum | 2 | 10 | 12 | 0 |
| wood | 2 | 10 | 12 | 0 |

| group | low | mid | high | NA |
| --- | --- | --- | --- | --- |
| base | 25 | 5 | 0 | 0 |
| stress | 25 | 5 | 0 | 0 |

- D3 split: LOS/NLOS×EL-bin strata coverage (viable-subset aware)

| status | n_rows | min_strata_all_n | min_strata_viable_n | qna_total | selected_rows_n |
| --- | --- | --- | --- | --- | --- |
| PASS | 161 | 0 | 7 | 0 | 0 |

| strata | n |
| --- | --- |
| LOS0_q1 | 28 |
| LOS0_q2 | 7 |
| LOS0_q3 | 0 |
| LOS0_qNA | 0 |
| LOS1_q1 | 26 |
| LOS1_q2 | 51 |
| LOS1_q3 | 49 |
| LOS1_qNA | 0 |

- D3 hole diagnosis (structural vs sampling)

| strata | pool_n | selected_n | hole_type | status |
| --- | --- | --- | --- | --- |
| LOS0_q1 | 28 |  | none | PASS |
| LOS0_q2 | 7 |  | none | PASS |
| LOS0_q3 | 0 |  | structural_hole | FAIL |
| LOS1_q1 | 26 |  | none | PASS |
| LOS1_q2 | 51 |  | none | PASS |
| LOS1_q3 | 49 |  | none | PASS |

- D3 per-scenario summary (B1/B2/B3)

| scenario_id | status | n_rows | q1_db | q2_db | min_strata_all_n | min_strata_viable_n | qna_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | FAIL | 63 | 3.589 | 5.5 | 0 | 1 | 0 |
| B2 | WARN | 49 | -0.2895 | 2.687 | 2 | 2 | 0 |
| B3 | PASS | 49 | 0.3647 | 2.72 | 0 | 4 | 0 |

- Legacy D strata view

| strata | n |
| --- | --- |
| LOS0_q1 | 28 |
| LOS0_q2 | 7 |
| LOS0_q3 | 0 |
| LOS0_qNA | 0 |
| LOS1_q1 | 26 |
| LOS1_q2 | 51 |
| LOS1_q3 | 49 |
| LOS1_qNA | 0 |

### E) Power-based Pipeline

- Status: **PASS**
- Note: No complex-phase fields are used by this report pipeline.
- Used metrics: XPD_early_db, XPD_late_db, rho_early_lin, L_pol_db, delay_spread_rms_s, early_energy_fraction, EL_proxy_db, LOSflag

## Scenario Sections

### A2

- 의미: LOS-blocked single-bounce(odd) control to probe early cross-leakage increase.

![A2 scene](figures/A2__0__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### A3

- 의미: LOS-blocked double-bounce(even) control to test co-dominant recovery trend.

![A3 scene](figures/A3__0__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### A4

- 의미: LOS-blocked material/angle sweep to isolate material-conditional leakage statistics.

![A4 scene](figures/A4__0__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### A5

- 의미: LOS-blocked depolarization-stress scenario (roughness/human/scatter) for tail-risk.

![A5 scene](figures/A5__0__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### B1

- 의미: Room grid baseline (mostly LOS) for spatial Z/U trend mapping.

![B1 scene](figures/B1__GLOBAL__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### B2

- 의미: Room grid with partition obstacle to induce partial NLOS/blocked regions.

![B2 scene](figures/B2__GLOBAL__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### B3

- 의미: Room grid with corner obstacles for stronger NLOS and multipath complexity.

![B3 scene](figures/B3__GLOBAL__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### C0

- 의미: Free-space LOS calibration baseline for floor/alignment uncertainty.

![C0 scene](figures/C0__0__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

## WARN

- B1/0: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/1: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/10: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/11: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/12: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/13: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/14: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/15: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/16: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/17: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/18: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/19: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/2: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/20: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/21: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/22: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/23: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/24: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/25: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/26: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/0: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/1: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/10: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/11: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/12: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/13: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/14: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/15: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/16: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/17: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/18: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/19: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/2: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/20: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/21: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/22: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/23: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/24: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/25: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/26: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/0: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/1: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/10: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/11: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/12: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/13: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/14: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/15: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/16: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/17: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/18: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/19: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/2: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/20: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/21: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/22: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/23: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/24: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/25: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/26: scene_debug missing; fallback layout used (ray polylines unavailable)

## Warning Case Drilldown

| scenario_id | case_id | case_label | warning | link_id | XPD_early_excess_db | XPD_late_excess_db | L_pol_db | EL_proxy_db | LOSflag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 0 | B1_0 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_0 | -31.74 | -23.15 | -8.594 | 4.828 | 1 |
| B1 | 1 | B1_1 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_1 | -31.83 | -21.47 | -10.36 | 5.533 | 1 |
| B1 | 10 | B1_10 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_10 | -31.28 | -22 | -9.283 | 6.342 | 1 |
| B1 | 11 | B1_11 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_11 | -31.43 | -22.31 | -9.119 | 8.855 | 1 |
| B1 | 12 | B1_12 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_12 | -31.62 | -21.84 | -9.783 | 12.51 | 1 |
| B1 | 13 | B1_13 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_13 | -31.27 | -23.2 | -8.062 | 15.39 | 1 |
| B1 | 14 | B1_14 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_14 | -32.34 | -22.35 | -9.989 | 12.51 | 1 |
| B1 | 15 | B1_15 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_15 | -31.95 | -22.47 | -9.485 | 8.855 | 1 |
| B1 | 16 | B1_16 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_16 | -32.56 | -22.17 | -10.39 | 6.342 | 1 |
| B1 | 17 | B1_17 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_17 | -32.45 | -22.08 | -10.37 | 4.651 | 1 |
| B1 | 18 | B1_18 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_18 | -31.21 | -22.33 | -8.883 | 4.19 | 1 |
| B1 | 19 | B1_19 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_19 | -31.21 | -22.07 | -9.141 | 5.483 | 1 |
| B1 | 2 | B1_2 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_2 | -32.99 | -23.52 | -9.472 | 9.71 | 1 |
| B1 | 20 | B1_20 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_20 | -30.71 | -22.13 | -8.58 | 7.124 | 1 |
| B1 | 21 | B1_21 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_21 | -31.96 | -21.94 | -10.02 | 8.854 | 1 |
| B1 | 22 | B1_22 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_22 | -32.4 | -22.54 | -9.863 | 9.71 | 1 |
| B1 | 23 | B1_23 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_23 | -31.98 | -22.12 | -9.863 | 8.854 | 1 |
| B1 | 24 | B1_24 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_24 | -32.22 | -21.85 | -10.37 | 7.124 | 1 |
| B1 | 25 | B1_25 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_25 | -32.63 | -22.07 | -10.56 | 5.483 | 1 |
| B1 | 26 | B1_26 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_26 | -32.06 | -22.3 | -9.768 | 4.19 | 1 |
| B2 | 0 | B2_0 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_0 | -31.79 | -20.46 | -11.33 | 0.3647 | 1 |
| B2 | 1 | B2_1 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_1 | -31.23 | -23.92 | -7.304 | 4.828 | 1 |
| B2 | 10 | B2_10 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_10 | -32.41 | -22.86 | -9.543 | 9.71 | 1 |
| B2 | 11 | B2_11 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_11 | -32.34 | -22.18 | -10.16 | 7.124 | 1 |
| B2 | 12 | B2_12 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_12 | -30.63 | -22.46 | -8.176 | 4.19 | 1 |
| B2 | 13 | B2_13 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_13 | -32.11 | -21.36 | -10.75 | 1.15 | 1 |
| B2 | 14 | B2_14 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_14 | -32.01 | -20.4 | -11.61 | 0.6058 | 1 |
| B2 | 15 | B2_15 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_15 | -31.65 | -21.08 | -10.57 | 1.452 | 1 |
| B2 | 16 | B2_16 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_16 | -30.59 | -21.14 | -9.455 | 2.687 | 1 |
| B2 | 17 | B2_17 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_17 | -32.83 | -21.13 | -11.7 | 3.348 | 1 |
| B2 | 18 | B2_18 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_18 | -32.03 | -20.77 | -11.26 | 2.687 | 1 |
| B2 | 19 | B2_19 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_19 | -31.22 | -21.25 | -9.974 | 1.452 | 1 |
| B2 | 2 | B2_2 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_2 | -31.52 | -23.77 | -7.752 | 9.71 | 1 |
| B2 | 20 | B2_20 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_20 | -32.52 | -20.33 | -12.2 | 0.6058 | 1 |
| B2 | 21 | B2_21 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_21 | -31.05 | -20.53 | -10.52 | 0.02432 | 1 |
| B2 | 22 | B2_22 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_22 | -32.29 | -21.63 | -10.67 | 2 | 1 |
| B2 | 23 | B2_23 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_23 | -30.72 | -21.57 | -9.143 | 2.499 | 1 |
| B2 | 24 | B2_24 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_24 | -31.4 | -22.38 | -9.018 | 2.72 | 1 |
| B2 | 25 | B2_25 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_25 | -32.02 | -21.56 | -10.46 | 2.499 | 1 |
| B2 | 26 | B2_26 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_26 | -30.6 | -21.43 | -9.165 | 2 | 1 |
| B3 | 0 | B3_0 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_0 | -31.79 | -18.84 | -12.95 | 0.3647 | 1 |
| B3 | 1 | B3_1 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_1 | -31.16 | -23.47 | -7.684 | 4.828 | 1 |
| B3 | 10 | B3_10 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_10 | -32.34 | -22.58 | -9.764 | 9.71 | 1 |
| B3 | 11 | B3_11 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_11 | -31.31 | -22.03 | -9.28 | 7.124 | 1 |
| B3 | 12 | B3_12 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_12 | -32.19 | -21.98 | -10.21 | 4.19 | 1 |
| B3 | 13 | B3_13 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_13 | -31.56 | -20.17 | -11.38 | 1.15 | 1 |
| B3 | 14 | B3_14 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_14 | -30.9 | -20.59 | -10.31 | 0.6058 | 1 |
| B3 | 15 | B3_15 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_15 | -30.32 | -21.64 | -8.673 | 2.993 | 1 |
| B3 | 16 | B3_16 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_16 | -32.95 | -21.69 | -11.26 | 4.186 | 1 |
| B3 | 17 | B3_17 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_17 | -32.59 | -22.33 | -10.25 | 4.828 | 1 |
| B3 | 18 | B3_18 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_18 | -31.86 | -22.01 | -9.858 | 4.186 | 1 |
| B3 | 19 | B3_19 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_19 | -31.49 | -21.64 | -9.849 | 1.452 | 1 |
| B3 | 2 | B3_2 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_2 | -32.38 | -22.98 | -9.395 | 9.71 | 1 |
| B3 | 20 | B3_20 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_20 | -31.71 | -19.63 | -12.08 | 0.6058 | 1 |
| B3 | 21 | B3_21 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_21 | -31.42 | -20.61 | -10.8 | 0.02432 | 1 |
| B3 | 22 | B3_22 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_22 | -31.43 | -21.98 | -9.45 | 2 | 1 |
| B3 | 23 | B3_23 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_23 | -31.51 | -21.37 | -10.14 | 2.499 | 1 |
| B3 | 24 | B3_24 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_24 | -30.55 | -19.9 | -10.65 | 0.7818 | 1 |
| B3 | 25 | B3_25 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_25 | -32.82 | -19.56 | -13.26 | -1.099 | 1 |
| B3 | 26 | B3_26 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_26 | -22.12 | -18.52 | -3.592 | -5.725 | 0 |

### WARN Case B1/0

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B1-0](figures/B1__0__scene.png)

### WARN Case B1/1

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B1-1](figures/B1__1__scene.png)

### WARN Case B1/10

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B1-10](figures/B1__10__scene.png)

### WARN Case B1/11

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B1-11](figures/B1__11__scene.png)

### WARN Case B1/12

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B1-12](figures/B1__12__scene.png)

### WARN Case B1/13

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B1-13](figures/B1__13__scene.png)

### WARN Case B1/14

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B1-14](figures/B1__14__scene.png)

### WARN Case B1/15

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B1-15](figures/B1__15__scene.png)

### WARN Case B1/16

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B1-16](figures/B1__16__scene.png)

### WARN Case B1/17

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B1-17](figures/B1__17__scene.png)

### WARN Case B1/18

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B1-18](figures/B1__18__scene.png)

### WARN Case B1/19

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B1-19](figures/B1__19__scene.png)

### WARN Case B1/2

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B1-2](figures/B1__2__scene.png)

### WARN Case B1/20

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B1-20](figures/B1__20__scene.png)

### WARN Case B1/21

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B1-21](figures/B1__21__scene.png)

### WARN Case B1/22

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B1-22](figures/B1__22__scene.png)

### WARN Case B1/23

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B1-23](figures/B1__23__scene.png)

### WARN Case B1/24

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B1-24](figures/B1__24__scene.png)

### WARN Case B1/25

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B1-25](figures/B1__25__scene.png)

### WARN Case B1/26

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B1-26](figures/B1__26__scene.png)

### WARN Case B2/0

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-0](figures/B2__0__scene.png)

### WARN Case B2/1

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-1](figures/B2__1__scene.png)

### WARN Case B2/10

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-10](figures/B2__10__scene.png)

### WARN Case B2/11

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-11](figures/B2__11__scene.png)

### WARN Case B2/12

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-12](figures/B2__12__scene.png)

### WARN Case B2/13

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-13](figures/B2__13__scene.png)

### WARN Case B2/14

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-14](figures/B2__14__scene.png)

### WARN Case B2/15

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-15](figures/B2__15__scene.png)

### WARN Case B2/16

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-16](figures/B2__16__scene.png)

### WARN Case B2/17

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-17](figures/B2__17__scene.png)

### WARN Case B2/18

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-18](figures/B2__18__scene.png)

### WARN Case B2/19

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-19](figures/B2__19__scene.png)

### WARN Case B2/2

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-2](figures/B2__2__scene.png)

### WARN Case B2/20

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-20](figures/B2__20__scene.png)

### WARN Case B2/21

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-21](figures/B2__21__scene.png)

### WARN Case B2/22

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-22](figures/B2__22__scene.png)

### WARN Case B2/23

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-23](figures/B2__23__scene.png)

### WARN Case B2/24

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-24](figures/B2__24__scene.png)

### WARN Case B2/25

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-25](figures/B2__25__scene.png)

### WARN Case B2/26

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-26](figures/B2__26__scene.png)

### WARN Case B3/0

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-0](figures/B3__0__scene.png)

### WARN Case B3/1

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-1](figures/B3__1__scene.png)

### WARN Case B3/10

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-10](figures/B3__10__scene.png)

### WARN Case B3/11

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-11](figures/B3__11__scene.png)

### WARN Case B3/12

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-12](figures/B3__12__scene.png)

### WARN Case B3/13

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-13](figures/B3__13__scene.png)

### WARN Case B3/14

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-14](figures/B3__14__scene.png)

### WARN Case B3/15

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-15](figures/B3__15__scene.png)

### WARN Case B3/16

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-16](figures/B3__16__scene.png)

### WARN Case B3/17

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-17](figures/B3__17__scene.png)

### WARN Case B3/18

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-18](figures/B3__18__scene.png)

### WARN Case B3/19

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-19](figures/B3__19__scene.png)

### WARN Case B3/2

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-2](figures/B3__2__scene.png)

### WARN Case B3/20

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-20](figures/B3__20__scene.png)

### WARN Case B3/21

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-21](figures/B3__21__scene.png)

### WARN Case B3/22

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-22](figures/B3__22__scene.png)

### WARN Case B3/23

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-23](figures/B3__23__scene.png)

### WARN Case B3/24

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-24](figures/B3__24__scene.png)

### WARN Case B3/25

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-25](figures/B3__25__scene.png)

### WARN Case B3/26

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-26](figures/B3__26__scene.png)

