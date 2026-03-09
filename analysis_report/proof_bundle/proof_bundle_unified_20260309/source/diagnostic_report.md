# Diagnostic Report (diag_protocol_repro_v1_20260309_archetype)

## Dataset Summary

| scenario | n_links |
| --- | --- |
| A2 | 4 |
| A2_on | 4 |
| A3 | 12 |
| A3_on | 12 |
| A4 | 24 |
| A4_on | 24 |
| A5 | 60 |
| A6 | 20 |
| A6_on | 20 |
| B1 | 30 |
| B2 | 120 |
| B3 | 25 |
| C0 | 125 |

- Figure metadata: [figure_metadata.csv](tables/figure_metadata.csv)

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

## Floor Reference

| xpd_floor_db | delta_floor_db | p5_db | p95_db | count | method |
| --- | --- | --- | --- | --- | --- |
| 25.05 | 0.7239 | 24.34 | 25.79 | 125 | p5_p95 |

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
| A2 | 1 | 4 | 4 | 1 | PASS |
| A3 | 2 | 12 | 12 | 1 | PASS |

- A3 coordinate sanity: **WARN** (Coordinate penetration sanity needs scenario geometry file review; not inferable from standard outputs only.)

- A5 stress semantics (path-structure vs polarization-only)

| status | dominant_semantics | n_stress_rows | n_response | n_polarization_only | contamination_response_ready | note |
| --- | --- | --- | --- | --- | --- | --- |
| PASS | response | 30 | 30 | 0 | True | Geometric path-structure stress is active; delay/path contamination-response interpretation is valid. |

- A3 geometry manual review (ray-path visualization required before experiment)

| scenario_id | case_id | review_status | scene_debug_valid | los_rays | has_target_bounce_n2 | rays_topk_n | scene_debug_issues |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A3 | 0 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 1 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 10 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 11 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 2 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 3 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 4 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 5 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 6 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 7 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 8 | PASS | 1 | 0 | 1 | 2 |  |
| A3 | 9 | PASS | 1 | 0 | 1 | 2 |  |

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

#### A3 case 2

- review_status: **PASS**
![A3-case-2](figures/A3__2__scene.png)

#### A3 case 3

- review_status: **PASS**
![A3-case-3](figures/A3__3__scene.png)

#### A3 case 4

- review_status: **PASS**
![A3-case-4](figures/A3__4__scene.png)

#### A3 case 5

- review_status: **PASS**
![A3-case-5](figures/A3__5__scene.png)

#### A3 case 6

- review_status: **PASS**
![A3-case-6](figures/A3__6__scene.png)

#### A3 case 7

- review_status: **PASS**
![A3-case-7](figures/A3__7__scene.png)

#### A3 case 8

- review_status: **PASS**
![A3-case-8](figures/A3__8__scene.png)

#### A3 case 9

- review_status: **PASS**
![A3-case-9](figures/A3__9__scene.png)

### B) Time Resolution / Delay Separability

- `W_floor`(C0): `C_floor = Sum(P_nonLOS in W_floor) / P_LOS`
- `W_target`(A2-A5): `C_target = Sum(P_non-target in W_target) / P_target`
- `W_early`(B1-B3): `S(Te)=|mu_LOS-mu_NLOS|/sqrt(sig_LOS^2+sig_NLOS^2)`

| freq_source | dt_res_s | tau_max_s | Te_s | Tmax_s | W_floor_C_median_db | W_floor_status | A2_target_in_Wearly_rate | A3_target_in_Wearly_rate | A2_C_target_median_db | A3_C_target_median_db | A2_target_sign_hit_rate | A2_target_sign_status | A3_target_sign_hit_rate | A3_target_sign_status | A3_target_sign_status_reporting | A2A3_sign_stability_raw | A2A3_sign_stability_reporting | A6_parity_status | A6_hit_rate_odd | A6_hit_rate_even | G2_primary_evidence_source | G2_primary_evidence_status | A3_mechanism_status | A3_system_early_status | A5_target_mode | A5_stress_semantics | A5_contamination_response_ready | min_delay_gap_median_s | B2_status | B3_status | W3_best_te_ns | W3_best_S_xpd_early | W3_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| link_rows.xpd_floor_freq_hz[C0] | 2.5e-10 | 3.175e-08 | 3e-09 | 3e-08 | -237.2 | PASS | 1 | 0.1667 | -230.4 | -227.8 | 1 | PASS | 0.6667 | WARN | WARN | FAIL | FAIL | PASS | 1 | 1 | A6_near_normal_benchmark | PASS | PASS | FAIL | isolation | response | True | 2.305e-10 | WARN | PASS | 5 | 0.6627 | WARN |

- A3 reporting rule: A3 is mechanism-only/supplementary: use target-window metrics for mechanism context; fixed system early-window dominance is not primary evidence. If A6 is present, use A6 as primary G2 sign evidence.

- G2 primary evidence: `A6_near_normal_benchmark` (status=PASS)

- A5 semantics note: Geometric path-structure stress is active; delay/path contamination-response interpretation is valid.

- W_floor(C0) contamination summary

| W_floor_s | C_floor_median_db | C_floor_p95_db | rate_below_m10_db | rate_below_m15_db | status | n_cases |
| --- | --- | --- | --- | --- | --- | --- |
| 1.5e-09 | -237.2 | -232.8 | 1 | 1 | PASS | 125 |

- W_target per-scenario configuration

| scenario | W_target_s | W_target_ns | mode |
| --- | --- | --- | --- |
| A2 | 3e-09 | 3 | default |
| A3 | 6e-09 | 6 | fixed |
| A4 | 3e-09 | 3 | default |
| A5 | 3e-09 | 3 | default |

- W_target(controlled scenarios) summary

| scenario | target_n | target_exists_rate | target_is_first_rate | target_in_Wearly_rate | C_target_median_db | target_gap_median_s | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A2 | 1 | 1 | 1 | 1 | -230.4 | nan | PASS |
| A3 | 2 | 1 | 0 | 0.1667 | -227.8 | 4.67e-09 | WARN |
| A4 | 1 | 1 | 1 | 1 | 0 | 0 | FAIL |
| A5 | 2 | 0.9661 | 0.07018 | 0.1228 | -228.8 | 3.141e-09 | WARN |

- W_early(room/grid) Te sweep separation

| Te_ns | S_xpd_early | S_rho_early_db | S_l_pol |
| --- | --- | --- | --- |
| 2 | 0.5321 | 0.5321 | 0.2883 |
| 3 | 0.6363 | 0.6363 | 0.2598 |
| 5 | 0.6627 | 0.6627 | 0.5117 |

- A2/A3 odd-even sign stability over Te sweep: **FAIL** (raw=FAIL)

| scenario | expected_sign | min_hit_rate | median_hit_rate | status |
| --- | --- | --- | --- | --- |
| A2 | negative | 1 | 1 | PASS |
| A3 | positive | 0 | 0 | FAIL |

- sign-stability reporting note: A2/A3 early-window sign stability interpreted directly from raw status.

- Target-window sign metric (A2/A3 and A6 parity benchmark when available)

| scenario | target_n | W_target_s | expected_sign | n_eval | sign_metric_for_status | expected_sign_hit_rate | expected_sign_hit_rate_raw | expected_sign_hit_rate_ex | median_xpd_target_raw_db | median_xpd_target_ex_db | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A2 | 1 | 3e-09 | negative | 4 | excess | 1 | 1 | 1 | -8 | -33.05 | PASS |
| A3 | 2 | 6e-09 | positive | 12 | raw | 0.6667 | 0.6667 | 0 | 8 | -17.05 | WARN |
| A6_odd | 1 | 3e-09 | negative | 10 | raw | 1 | 1 | 1 | -8 | -33.05 | PASS |
| A6_even | 2 | 3e-09 | positive | 10 | raw | 1 | 1 | 0 | 8 | -17.05 | PASS |

- A6 case-set comparison (full vs minimal, odd/even)

| case_set | mode | n_eval | median_xpd_target_raw_db | median_xpd_target_ex_db | expected_sign_hit_rate | status |
| --- | --- | --- | --- | --- | --- | --- |
| full | odd | 9 | -8 | -33.05 | 1 | PASS |
| full | even | 9 | 8 | -17.05 | 1 | PASS |
| minimal | odd | 1 | -8 | -33.05 | 1 | PASS |
| minimal | even | 1 | 8 | -17.05 | 1 | PASS |

- A6 full vs minimal delta summary (`minimal - full`)

| mode | delta_raw_minimal_minus_full_db | delta_ex_minimal_minus_full_db | delta_hit_rate_minimal_minus_full | n_eval_full | n_eval_minimal |
| --- | --- | --- | --- | --- | --- |
| odd | 0 | 0 | 0 | 9 | 1 |
| even | 0 | 0 | 0 | 9 | 1 |

### C) Effect Size vs Floor Uncertainty

- C2-M primary: `XPD_early_excess`; secondary: `XPD_late_excess`, `L_pol`
- C2-S primary: `L_pol`; secondary: `rho_early`, `DS`, `XPD_late_excess`; gate: `ΔP_target,total > -6 dB`

| floor_delta_db | repeat_delta_db | delta_ref_db | A3_minus_A2_delta_median_db | ratio_to_floor | C1_status | C2M_primary_span_db | C2M_primary_status | C2M_secondary_late_span_db | C2M_secondary_late_status | C2M_status | C2S_delta_lpol_db | C2S_primary_status | C2S_gate_delta_p_target_db | C2S_gate_status | C2S_semantics_gate_status | C2S_semantics | C2S_status | C2_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7239 | 0.4543 | 0.7239 | 9.834e-08 | 1.358e-07 | FAIL | 0.6738 | FAIL | 0 | FAIL | FAIL | 0.6817 | FAIL | 0 | PASS | PASS | response | FAIL | FAIL |

- C2-S semantics note: Geometric path-structure stress is active; delay/path contamination-response interpretation is valid.

### D) Identifiability

| status | EL_iqr_db | corr_d_vs_LOS | min_strata_n |
| --- | --- | --- | --- |
| FAIL | 5.535 | -0.35 | 10 |

- D1 split: EL-identifying coverage(global) + parity/stress isolation(local)

| component | status | role | EL_iqr_db | min_bin_n | n_rows | EL_std_db | delta_median_EL_stress_minus_base_db | n_base | n_stress | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1_global | PASS |  | 5.535 | 70 | 211 |  |  |  |  |  |
| A2_isolation | PASS |  |  |  | 4 | 7.105e-15 |  |  |  | small EL variation is desired for parity isolation |
| A5_isolation | FAIL | stress_response_proxy |  |  |  |  | 0 | 30 | 30 | response mode: L_pol decrease is primary, EL shift can be non-zero |

- D2 split: material×angle coverage + stress×angle coverage + collinearity diagnostics

| status | material_x_angle_status | stress_x_angle_status | design_status | design_rank | design_cols | condition_number |
| --- | --- | --- | --- | --- | --- | --- |
| FAIL | PASS | WARN | FAIL | 10 | 11 | 1.651e+16 |

| stage | status | n_rows | design_rank | design_cols | condition_number |
| --- | --- | --- | --- | --- | --- |
| stage1_EL_identifying | FAIL | 211 | 10 | 11 | 1.651e+16 |
| stage2_effects_after_EL | PASS | 275 | 5 | 5 | 12.4 |

| group | low | mid | high | NA |
| --- | --- | --- | --- | --- |
| glass | 0 | 4 | 4 | 0 |
| gypsum | 0 | 4 | 4 | 0 |
| wood | 0 | 4 | 4 | 0 |

| group | low | mid | high | NA |
| --- | --- | --- | --- | --- |
| base | 30 | 0 | 0 | 0 |
| stress | 27 | 2 | 0 | 1 |

- D3 split: LOS/NLOS×EL-bin strata coverage (viable-subset aware)

| status | status_physics_scene_objective | status_evidence_quality | n_rows | min_strata_all_n | min_strata_viable_n | qna_total | selected_rows_n | scene_debug_coverage | scene_debug_warn_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PASS | WARN | PASS | 175 | 10 | 10 | 0 | 0 | 1 | 0 |

| strata | n |
| --- | --- |
| LOS0_q1 | 48 |
| LOS0_q2 | 30 |
| LOS0_q3 | 47 |
| LOS0_qNA | 0 |
| LOS1_q1 | 10 |
| LOS1_q2 | 29 |
| LOS1_q3 | 11 |
| LOS1_qNA | 0 |

- D3 hole diagnosis (structural vs sampling)

| strata | pool_n | selected_n | hole_type | status |
| --- | --- | --- | --- | --- |
| LOS0_q1 | 48 |  | none | PASS |
| LOS0_q2 | 30 |  | none | PASS |
| LOS0_q3 | 47 |  | none | PASS |
| LOS1_q1 | 10 |  | none | PASS |
| LOS1_q2 | 29 |  | none | PASS |
| LOS1_q3 | 11 |  | none | PASS |

- D3 per-scenario summary (B1/B2/B3, scene-purpose one-line judgement)

- Scene-status counts: PASS=1, WARN=2, FAIL=0 (fail count is per-scenario objective, not per-metric sum)

| scenario_id | scene_axis | status | objective_hits | objective_total | support_rate | neg_tail_rate_xpd_early_ex_lt_0 | rho_early_q90_lin | early_fraction_median | early_fraction_q25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | good_condition_baseline | WARN | 1 | 4 | 1 | 1 | 5.768 | 0.7634 | 0.7581 |
| B2 | contamination_onset | PASS | 3 | 3 | 1 | 1 | 5.808 | 0.7781 | 0.6718 |
| B3 | high_risk_tail | WARN | 0 | 4 | 1 | 1 | 2.263 | 1 | 1 |

- D3 detailed support/tail summary (secondary evidence)

| scenario_id | support_core_n | support_core_rate | support_risk_n | support_risk_rate | xpd_early_ex_median_db | xpd_early_ex_q75_db | xpd_early_ex_q90_db | xpd_early_ex_neg_tail_frac_lt_m30 | rho_early_q75_lin | rho_early_q90_lin | ds_q75_ns | ds_q90_ns | l_pol_q75_db | l_pol_q90_db | early_fraction_median | early_fraction_q25 | status_strata_legacy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 30 | 1 | 30 | 1 | -31.75 | -31.36 | -31.1 | 1 | 5.227 | 5.768 | 4.44 | 4.742 | 0.5721 | 0.9194 | 0.7634 | 0.7581 | PASS |
| B2 | 120 | 1 | 120 | 1 | -31.63 | -30.5 | -27.28 | 0.8167 | 5.082 | 5.808 | 4.622 | 5.124 | -4.579 | -2.863 | 0.7781 | 0.6718 | PASS |
| B3 | 25 | 1 | 25 | 1 | -27.09 | -24.66 | -24.6 | 0.04 | 1.896 | 2.263 | 0 | 0 | 0.3887 | 0.4543 | 1 | 1 | PASS |

- Legacy D strata view

| strata | n |
| --- | --- |
| LOS0_q1 | 48 |
| LOS0_q2 | 30 |
| LOS0_q3 | 47 |
| LOS0_qNA | 0 |
| LOS1_q1 | 10 |
| LOS1_q2 | 29 |
| LOS1_q3 | 11 |
| LOS1_qNA | 0 |

### E) Power-based Pipeline

- Status: **FAIL**
- Note: No complex-phase fields are used by this report pipeline.
- basis values: `['circular']`
- convention values: `['IEEE-RHCP']`
- matrix_source values: `['A', 'J']`
- force_cp_swap_on_odd_reflection(any): `False`
- matrix_source rule: Use circular basis for CP interpretation and keep matrix_source fixed (A_f for antenna-embedded, J_f for de-embedded).
- main-result rule: force_cp_swap_on_odd_reflection must be false for main-result evidence.
- Used metrics: XPD_early_db, XPD_late_db, rho_early_lin, L_pol_db, delay_spread_rms_s, early_energy_fraction, EL_proxy_db, LOSflag
- C0 coupling-floor sanity: status=PASS, C0 floor=25.052 dB, nominal=35.000 dB, delta=-9.948 dB
- coupling-floor note: Heuristic sanity check only; measured C0 floor should be reviewed against configured antenna coupling/leakage model.
- A5 stress semantics: `response`
- A5 contamination-response ready: `True`
- A5 semantics note: Geometric path-structure stress is active; delay/path contamination-response interpretation is valid.
- Figure metadata table: [figure_metadata.csv](tables/figure_metadata.csv)

## Scenario Sections

### A2

- 의미: Odd parity isolation anchor (A2_off core evidence); A2_on is observability bridge set.

![A2 scene](figures/A2__0__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### A2_on

- 의미: LOS-on bridge observability counterpart of A2_off; direct LOS + odd-path coexistence check.

![A2_on scene](figures/A2_on__0__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### A3

- 의미: A3_supp supplementary mechanism scene (target-window evidence only, not system early baseline); A3_on is LOS-on observability bridge set.

![A3 scene](figures/A3__0__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### A3_on

- 의미: LOS-on bridge observability counterpart of A3_supp; direct LOS + even-path coexistence check.

![A3_on scene](figures/A3_on__0__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### A4

- 의미: Material branch: A4_iso primary (late_panel=off, dispersion=off), A4_bridge secondary (late_panel=on, dispersion=on); A4_on is LOS-on observability bridge set.

![A4 scene](figures/A4__0__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### A4_on

- 의미: LOS-on bridge observability counterpart of A4 material branch; direct LOS under material contrast.

![A4_on scene](figures/A4_on__0__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### A5

- 의미: A5_pair contamination-response pair (base/on): use for paired contamination/stress-response only.
- stress_semantics: `response` (response=30, polarization_only=0)
- contamination-response ready: `True`
- note: Geometric path-structure stress is active; delay/path contamination-response interpretation is valid.

![A5 scene](figures/A5__0__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### A6

- 의미: Near-normal PEC parity benchmark; primary G2 sign evidence.

![A6 scene](figures/A6__0__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### A6_on

- 의미: LOS-on bridge observability counterpart of A6 parity benchmark; direct LOS + odd/even coexistence check.

![A6_on scene](figures/A6_on__0__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### B1

- 의미: LOS-dominant practical archetype baseline in an open room (CP-valid anchor).

![B1 scene](figures/B1__GLOBAL__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### B2

- 의미: Partitioned transition archetype (soft-block mixed contamination with material-labelled partition).

![B2 scene](figures/B2__GLOBAL__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### B3

- 의미: Around-corner high-EL archetype (hard-NLOS risk; coverage-aware leverage map).

![B3 scene](figures/B3__GLOBAL__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### C0

- 의미: Calibration-only baseline: floor reference and alignment sensitivity (W_floor).

![C0 scene](figures/C0__0__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

