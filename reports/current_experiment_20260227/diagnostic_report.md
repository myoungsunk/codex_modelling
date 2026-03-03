# Diagnostic Report (report_20260227_current)

## Dataset Summary

| scenario | n_links |
| --- | --- |
| A2 | 27 |
| A3 | 24 |
| A4 | 72 |
| A5 | 60 |
| B1 | 49 |
| B2 | 121 |
| B3 | 121 |
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
| A3 | 2 | 12 | 14 | 0.8571 | PASS |

- A3 coordinate sanity: **WARN** (Coordinate penetration sanity needs scenario geometry file review; not inferable from standard outputs only.)

### B) Time Resolution / Delay Separability

| dt_res_s | tau_max_s | Te_s | Tmax_s | A2_target_in_early_rate | A3_target_in_early_rate | min_delay_gap_median_s | B2_status | B3_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-09 | 1e-06 | 1e-08 | 2e-07 | 0 | 0 | 0 | WARN | PASS |

### C) Effect Size vs Floor Uncertainty

| floor_delta_db | A3_minus_A2_delta_median_db | ratio_to_floor | C1_status | A4_material_shift_late_excess_db | A5_stress_delta_late_excess_db | A5_stress_var_ratio | C2_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.6107 | 6.335 | 10.37 | PASS | 0 | 0 | 5.891e+13 | PASS |

### D) Identifiability

| EL_iqr_db | corr_d_vs_LOS | min_strata_n | status |
| --- | --- | --- | --- |
| 2.243 | -0.5494 | 37 | WARN |

| strata | n |
| --- | --- |
| LOS0_q1 | 94 |
| LOS0_q2 | 106 |
| LOS0_q3 | 96 |
| LOS1_q1 | 59 |
| LOS1_q2 | 66 |
| LOS1_q3 | 37 |

### E) Power-based Pipeline

- Status: **PASS**
- Note: No complex-phase fields are used by this report pipeline.
- Used metrics: XPD_early_db, XPD_late_db, rho_early_lin, L_pol_db, delay_spread_rms_s, early_energy_fraction, EL_proxy_db, LOSflag

## Scenario Sections

### A2

![A2 scene](figures/A2__0__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### A3

![A3 scene](figures/A3__0__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### A4

![A4 scene](figures/A4__0__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### A5

![A5 scene](figures/A5__0__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### B1

![B1 scene](figures/B1__GLOBAL__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### B2

![B2 scene](figures/B2__GLOBAL__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### B3

![B3 scene](figures/B3__GLOBAL__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

### C0

![C0 scene](figures/C0__0__scene.png)

- [A2A3__ALL__lpol_box.png](figures/A2A3__ALL__lpol_box.png)
- [A2A3__ALL__xpd_early_ex_box.png](figures/A2A3__ALL__xpd_early_ex_box.png)
- [A2A3__ALL__xpd_early_ex_cdf.png](figures/A2A3__ALL__xpd_early_ex_cdf.png)
- [C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
- [C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
- [C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)

## WARN

- A2/0: scene_debug missing; fallback layout used (ray polylines unavailable)
- A2/1: scene_debug missing; fallback layout used (ray polylines unavailable)
- A2/10: scene_debug missing; fallback layout used (ray polylines unavailable)
- A2/11: scene_debug missing; fallback layout used (ray polylines unavailable)
- A2/12: scene_debug missing; fallback layout used (ray polylines unavailable)
- A2/13: scene_debug missing; fallback layout used (ray polylines unavailable)
- A2/14: scene_debug missing; fallback layout used (ray polylines unavailable)
- A2/15: scene_debug missing; fallback layout used (ray polylines unavailable)
- A3/0: scene_debug missing; fallback layout used (ray polylines unavailable)
- A3/1: scene_debug missing; fallback layout used (ray polylines unavailable)
- A3/10: scene_debug missing; fallback layout used (ray polylines unavailable)
- A3/11: scene_debug missing; fallback layout used (ray polylines unavailable)
- A3/12: scene_debug missing; fallback layout used (ray polylines unavailable)
- A3/13: scene_debug missing; fallback layout used (ray polylines unavailable)
- A3/14: scene_debug missing; fallback layout used (ray polylines unavailable)
- A3/15: scene_debug missing; fallback layout used (ray polylines unavailable)
- A4/0: scene_debug missing; fallback layout used (ray polylines unavailable)
- A4/1: scene_debug missing; fallback layout used (ray polylines unavailable)
- A4/10: scene_debug missing; fallback layout used (ray polylines unavailable)
- A4/11: scene_debug missing; fallback layout used (ray polylines unavailable)
- A4/12: scene_debug missing; fallback layout used (ray polylines unavailable)
- A4/13: scene_debug missing; fallback layout used (ray polylines unavailable)
- A4/14: scene_debug missing; fallback layout used (ray polylines unavailable)
- A4/15: scene_debug missing; fallback layout used (ray polylines unavailable)
- A5/0: scene_debug missing; fallback layout used (ray polylines unavailable)
- A5/1: scene_debug missing; fallback layout used (ray polylines unavailable)
- A5/10: scene_debug missing; fallback layout used (ray polylines unavailable)
- A5/11: scene_debug missing; fallback layout used (ray polylines unavailable)
- A5/12: scene_debug missing; fallback layout used (ray polylines unavailable)
- A5/13: scene_debug missing; fallback layout used (ray polylines unavailable)
- A5/14: scene_debug missing; fallback layout used (ray polylines unavailable)
- A5/15: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/0: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/1: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/10: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/11: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/12: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/13: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/14: scene_debug missing; fallback layout used (ray polylines unavailable)
- B1/15: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/0: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/1: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/10: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/100: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/101: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/102: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/103: scene_debug missing; fallback layout used (ray polylines unavailable)
- B2/104: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/0: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/1: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/10: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/100: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/101: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/102: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/103: scene_debug missing; fallback layout used (ray polylines unavailable)
- B3/104: scene_debug missing; fallback layout used (ray polylines unavailable)
- C0/0: scene_debug missing; fallback layout used (ray polylines unavailable)
- C0/1: scene_debug missing; fallback layout used (ray polylines unavailable)
- C0/10: scene_debug missing; fallback layout used (ray polylines unavailable)
- C0/11: scene_debug missing; fallback layout used (ray polylines unavailable)
- C0/12: scene_debug missing; fallback layout used (ray polylines unavailable)
- C0/13: scene_debug missing; fallback layout used (ray polylines unavailable)
- C0/14: scene_debug missing; fallback layout used (ray polylines unavailable)
- C0/15: scene_debug missing; fallback layout used (ray polylines unavailable)

## Warning Case Drilldown

| scenario_id | case_id | case_label | warning | link_id | XPD_early_excess_db | XPD_late_excess_db | L_pol_db | EL_proxy_db | LOSflag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A2 | 0 | A2_0 | scene_debug missing; fallback layout used (ray polylines unavailable) | A2_0 | -31.27 | -23.27 | -8 | 2.72 | 0 |
| A2 | 1 | A2_1 | scene_debug missing; fallback layout used (ray polylines unavailable) | A2_1 | -31.27 | -23.27 | -8 | 2.72 | 0 |
| A2 | 10 | A2_10 | scene_debug missing; fallback layout used (ray polylines unavailable) | A2_10 | -31.27 | -23.27 | -8 | 2.72 | 0 |
| A2 | 11 | A2_11 | scene_debug missing; fallback layout used (ray polylines unavailable) | A2_11 | -31.27 | -23.27 | -8 | 2.72 | 0 |
| A2 | 12 | A2_12 | scene_debug missing; fallback layout used (ray polylines unavailable) | A2_12 | -31.27 | -23.27 | -8 | 2.72 | 0 |
| A2 | 13 | A2_13 | scene_debug missing; fallback layout used (ray polylines unavailable) | A2_13 | -31.27 | -23.27 | -8 | 2.72 | 0 |
| A2 | 14 | A2_14 | scene_debug missing; fallback layout used (ray polylines unavailable) | A2_14 | -31.27 | -23.27 | -8 | 2.72 | 0 |
| A2 | 15 | A2_15 | scene_debug missing; fallback layout used (ray polylines unavailable) | A2_15 | -31.27 | -23.27 | -8 | 2.72 | 0 |
| A3 | 0 | A3_0 | scene_debug missing; fallback layout used (ray polylines unavailable) | A3_0 | -16.93 | -24.93 | 8 | 2.72 | 0 |
| A3 | 1 | A3_1 | scene_debug missing; fallback layout used (ray polylines unavailable) | A3_1 | -16.93 | -24.93 | 8 | 2.72 | 0 |
| A3 | 10 | A3_10 | scene_debug missing; fallback layout used (ray polylines unavailable) | A3_10 | -16.93 | -24.93 | 8 | 2.72 | 0 |
| A3 | 11 | A3_11 | scene_debug missing; fallback layout used (ray polylines unavailable) | A3_11 | -16.93 | -24.93 | 8 | 2.72 | 0 |
| A3 | 12 | A3_12 | scene_debug missing; fallback layout used (ray polylines unavailable) | A3_12 | -24.93 | -24.93 | 0 | nan | 0 |
| A3 | 13 | A3_13 | scene_debug missing; fallback layout used (ray polylines unavailable) | A3_13 | -24.93 | -24.93 | 0 | nan | 0 |
| A3 | 14 | A3_14 | scene_debug missing; fallback layout used (ray polylines unavailable) | A3_14 | -24.93 | -24.93 | 0 | nan | 0 |
| A3 | 15 | A3_15 | scene_debug missing; fallback layout used (ray polylines unavailable) | A3_15 | -24.93 | -24.93 | 0 | nan | 0 |
| A4 | 0 | A4_0 | scene_debug missing; fallback layout used (ray polylines unavailable) | A4_0 | -34.86 | -24.93 | -9.922 | 8.343 | 0 |
| A4 | 1 | A4_1 | scene_debug missing; fallback layout used (ray polylines unavailable) | A4_1 | -34.81 | -24.93 | -9.87 | 8.343 | 0 |
| A4 | 10 | A4_10 | scene_debug missing; fallback layout used (ray polylines unavailable) | A4_10 | -34.71 | -24.93 | -9.772 | 8.343 | 0 |
| A4 | 11 | A4_11 | scene_debug missing; fallback layout used (ray polylines unavailable) | A4_11 | -34.84 | -24.93 | -9.905 | 8.343 | 0 |
| A4 | 12 | A4_12 | scene_debug missing; fallback layout used (ray polylines unavailable) | A4_12 | -33.1 | -24.93 | -8.165 | 9.96 | 0 |
| A4 | 13 | A4_13 | scene_debug missing; fallback layout used (ray polylines unavailable) | A4_13 | -33.52 | -24.93 | -8.586 | 9.96 | 0 |
| A4 | 14 | A4_14 | scene_debug missing; fallback layout used (ray polylines unavailable) | A4_14 | -33.9 | -24.93 | -8.962 | 9.824 | 0 |
| A4 | 15 | A4_15 | scene_debug missing; fallback layout used (ray polylines unavailable) | A4_15 | -34 | -24.93 | -9.065 | 9.824 | 0 |
| A5 | 0 | A5_0 | scene_debug missing; fallback layout used (ray polylines unavailable) | A5_0 | -18.83 | -24.93 | 6.101 | 2.72 | 0 |
| A5 | 1 | A5_1 | scene_debug missing; fallback layout used (ray polylines unavailable) | A5_1 | -19.04 | -24.93 | 5.894 | 2.72 | 0 |
| A5 | 10 | A5_10 | scene_debug missing; fallback layout used (ray polylines unavailable) | A5_10 | -19.43 | -24.93 | 5.501 | 2.72 | 0 |
| A5 | 11 | A5_11 | scene_debug missing; fallback layout used (ray polylines unavailable) | A5_11 | -18.9 | -24.93 | 6.033 | 2.72 | 0 |
| A5 | 12 | A5_12 | scene_debug missing; fallback layout used (ray polylines unavailable) | A5_12 | -20.79 | -24.93 | 4.14 | 2.72 | 0 |
| A5 | 13 | A5_13 | scene_debug missing; fallback layout used (ray polylines unavailable) | A5_13 | -19.11 | -24.93 | 5.825 | 2.72 | 0 |
| A5 | 14 | A5_14 | scene_debug missing; fallback layout used (ray polylines unavailable) | A5_14 | -19.93 | -24.93 | 5.003 | 2.72 | 0 |
| A5 | 15 | A5_15 | scene_debug missing; fallback layout used (ray polylines unavailable) | A5_15 | -19.52 | -24.93 | 5.414 | 2.72 | 0 |
| B1 | 0 | B1_0 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_0 | -31.92 | -19.25 | -12.68 | 0.3647 | 1 |
| B1 | 1 | B1_1 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_1 | -31.7 | -23.58 | -8.128 | 4.828 | 1 |
| B1 | 10 | B1_10 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_10 | -32.06 | -22.66 | -9.401 | 9.71 | 1 |
| B1 | 11 | B1_11 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_11 | -30.66 | -22.05 | -8.613 | 7.124 | 1 |
| B1 | 12 | B1_12 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_12 | -32.51 | -22.25 | -10.27 | 4.19 | 1 |
| B1 | 13 | B1_13 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_13 | -30.53 | -20.84 | -9.693 | 1.15 | 1 |
| B1 | 14 | B1_14 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_14 | -31.97 | -20.54 | -11.43 | 0.6058 | 1 |
| B1 | 15 | B1_15 | scene_debug missing; fallback layout used (ray polylines unavailable) | B1_15 | -31.94 | -22.02 | -9.914 | 2.993 | 1 |
| B2 | 0 | B2_0 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_0 | -32.01 | -17.77 | -14.23 | 2.72 | 0 |
| B2 | 1 | B2_1 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_1 | -30.99 | -20.1 | -10.89 | 0.2158 | 1 |
| B2 | 10 | B2_10 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_10 | -29.48 | -17.91 | -11.57 | 2.72 | 0 |
| B2 | 100 | B2_100 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_100 | -24.93 | -24.93 | 0 | nan | 0 |
| B2 | 101 | B2_101 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_101 | -18.21 | -24.93 | 6.724 | 0.4832 | 0 |
| B2 | 102 | B2_102 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_102 | -19.1 | -24.93 | 5.833 | 0.7597 | 0 |
| B2 | 103 | B2_103 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_103 | -18.45 | -24.93 | 6.489 | 0.2801 | 0 |
| B2 | 104 | B2_104 | scene_debug missing; fallback layout used (ray polylines unavailable) | B2_104 | -17.27 | -24.93 | 7.662 | 0.007926 | 0 |
| B3 | 0 | B3_0 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_0 | -32.01 | -17.86 | -14.15 | 2.72 | 0 |
| B3 | 1 | B3_1 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_1 | -32.44 | -21.03 | -11.4 | 0.2158 | 1 |
| B3 | 10 | B3_10 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_10 | -33.53 | -17.27 | -16.26 | 2.72 | 0 |
| B3 | 100 | B3_100 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_100 | -20.04 | -24.93 | 4.893 | 2.72 | 0 |
| B3 | 101 | B3_101 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_101 | -21.77 | -24.93 | 3.162 | -3.347 | 0 |
| B3 | 102 | B3_102 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_102 | -21.65 | -24.93 | 3.289 | -3.117 | 0 |
| B3 | 103 | B3_103 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_103 | -22.23 | -24.93 | 2.708 | -3.381 | 0 |
| B3 | 104 | B3_104 | scene_debug missing; fallback layout used (ray polylines unavailable) | B3_104 | -17.07 | -24.93 | 7.867 | -3.329 | 0 |
| C0 | 0 | C0_0 | scene_debug missing; fallback layout used (ray polylines unavailable) | C0_0 | -0.07147 | -25.13 | 25.06 | 2.72 | 1 |
| C0 | 1 | C0_1 | scene_debug missing; fallback layout used (ray polylines unavailable) | C0_1 | -0.2004 | -25.13 | 24.93 | 2.72 | 1 |
| C0 | 10 | C0_10 | scene_debug missing; fallback layout used (ray polylines unavailable) | C0_10 | -0.1181 | -24.81 | 24.69 | 2.72 | 1 |
| C0 | 11 | C0_11 | scene_debug missing; fallback layout used (ray polylines unavailable) | C0_11 | 0.2142 | -24.81 | 25.02 | 2.72 | 1 |
| C0 | 12 | C0_12 | scene_debug missing; fallback layout used (ray polylines unavailable) | C0_12 | -1.297 | -25.13 | 23.84 | 2.72 | 1 |
| C0 | 13 | C0_13 | scene_debug missing; fallback layout used (ray polylines unavailable) | C0_13 | -0.2437 | -25.13 | 24.89 | 2.72 | 1 |
| C0 | 14 | C0_14 | scene_debug missing; fallback layout used (ray polylines unavailable) | C0_14 | -0.264 | -24.64 | 24.38 | 2.72 | 1 |
| C0 | 15 | C0_15 | scene_debug missing; fallback layout used (ray polylines unavailable) | C0_15 | -0.007134 | -24.64 | 24.63 | 2.72 | 1 |

### WARN Case A2/0

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A2-0](figures/A2__0__scene.png)

### WARN Case A2/1

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A2-1](figures/A2__1__scene.png)

### WARN Case A2/10

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A2-10](figures/A2__10__scene.png)

### WARN Case A2/11

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A2-11](figures/A2__11__scene.png)

### WARN Case A2/12

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A2-12](figures/A2__12__scene.png)

### WARN Case A2/13

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A2-13](figures/A2__13__scene.png)

### WARN Case A2/14

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A2-14](figures/A2__14__scene.png)

### WARN Case A2/15

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A2-15](figures/A2__15__scene.png)

### WARN Case A3/0

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A3-0](figures/A3__0__scene.png)

### WARN Case A3/1

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A3-1](figures/A3__1__scene.png)

### WARN Case A3/10

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A3-10](figures/A3__10__scene.png)

### WARN Case A3/11

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A3-11](figures/A3__11__scene.png)

### WARN Case A3/12

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A3-12](figures/A3__12__scene.png)

### WARN Case A3/13

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A3-13](figures/A3__13__scene.png)

### WARN Case A3/14

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A3-14](figures/A3__14__scene.png)

### WARN Case A3/15

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A3-15](figures/A3__15__scene.png)

### WARN Case A4/0

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A4-0](figures/A4__0__scene.png)

### WARN Case A4/1

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A4-1](figures/A4__1__scene.png)

### WARN Case A4/10

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A4-10](figures/A4__10__scene.png)

### WARN Case A4/11

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A4-11](figures/A4__11__scene.png)

### WARN Case A4/12

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A4-12](figures/A4__12__scene.png)

### WARN Case A4/13

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A4-13](figures/A4__13__scene.png)

### WARN Case A4/14

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A4-14](figures/A4__14__scene.png)

### WARN Case A4/15

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A4-15](figures/A4__15__scene.png)

### WARN Case A5/0

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A5-0](figures/A5__0__scene.png)

### WARN Case A5/1

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A5-1](figures/A5__1__scene.png)

### WARN Case A5/10

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A5-10](figures/A5__10__scene.png)

### WARN Case A5/11

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A5-11](figures/A5__11__scene.png)

### WARN Case A5/12

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A5-12](figures/A5__12__scene.png)

### WARN Case A5/13

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A5-13](figures/A5__13__scene.png)

### WARN Case A5/14

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A5-14](figures/A5__14__scene.png)

### WARN Case A5/15

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-A5-15](figures/A5__15__scene.png)

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

### WARN Case B2/0

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-0](figures/B2__0__scene.png)

### WARN Case B2/1

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-1](figures/B2__1__scene.png)

### WARN Case B2/10

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-10](figures/B2__10__scene.png)

### WARN Case B2/100

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-100](figures/B2__100__scene.png)

### WARN Case B2/101

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-101](figures/B2__101__scene.png)

### WARN Case B2/102

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-102](figures/B2__102__scene.png)

### WARN Case B2/103

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-103](figures/B2__103__scene.png)

### WARN Case B2/104

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B2-104](figures/B2__104__scene.png)

### WARN Case B3/0

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-0](figures/B3__0__scene.png)

### WARN Case B3/1

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-1](figures/B3__1__scene.png)

### WARN Case B3/10

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-10](figures/B3__10__scene.png)

### WARN Case B3/100

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-100](figures/B3__100__scene.png)

### WARN Case B3/101

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-101](figures/B3__101__scene.png)

### WARN Case B3/102

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-102](figures/B3__102__scene.png)

### WARN Case B3/103

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-103](figures/B3__103__scene.png)

### WARN Case B3/104

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-B3-104](figures/B3__104__scene.png)

### WARN Case C0/0

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-C0-0](figures/C0__0__scene.png)

### WARN Case C0/1

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-C0-1](figures/C0__1__scene.png)

### WARN Case C0/10

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-C0-10](figures/C0__10__scene.png)

### WARN Case C0/11

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-C0-11](figures/C0__11__scene.png)

### WARN Case C0/12

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-C0-12](figures/C0__12__scene.png)

### WARN Case C0/13

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-C0-13](figures/C0__13__scene.png)

### WARN Case C0/14

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-C0-14](figures/C0__14__scene.png)

### WARN Case C0/15

- reason: scene_debug missing; fallback layout used (ray polylines unavailable)
![warn-C0-15](figures/C0__15__scene.png)

