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

## WARN

- scene_debug missing: A2/0
- scene_debug missing: A2/1
- scene_debug missing: A2/10
- scene_debug missing: A2/11
- scene_debug missing: A2/12
- scene_debug missing: A2/13
- scene_debug missing: A2/14
- scene_debug missing: A2/15
- scene_debug missing: A3/0
- scene_debug missing: A3/1
- scene_debug missing: A3/10
- scene_debug missing: A3/11
- scene_debug missing: A3/12
- scene_debug missing: A3/13
- scene_debug missing: A3/14
- scene_debug missing: A3/15
- scene_debug missing: A4/0
- scene_debug missing: A4/1
- scene_debug missing: A4/10
- scene_debug missing: A4/11
- scene_debug missing: A4/12
- scene_debug missing: A4/13
- scene_debug missing: A4/14
- scene_debug missing: A4/15
- scene_debug missing: A5/0
- scene_debug missing: A5/1
- scene_debug missing: A5/10
- scene_debug missing: A5/11
- scene_debug missing: A5/12
- scene_debug missing: A5/13
- scene_debug missing: A5/14
- scene_debug missing: A5/15
- scene_debug missing: B1/0
- scene_debug missing: B1/1
- scene_debug missing: B1/10
- scene_debug missing: B1/11
- scene_debug missing: B1/12
- scene_debug missing: B1/13
- scene_debug missing: B1/14
- scene_debug missing: B1/15
- scene_debug missing: B2/0
- scene_debug missing: B2/1
- scene_debug missing: B2/10
- scene_debug missing: B2/100
- scene_debug missing: B2/101
- scene_debug missing: B2/102
- scene_debug missing: B2/103
- scene_debug missing: B2/104
- scene_debug missing: B3/0
- scene_debug missing: B3/1
- scene_debug missing: B3/10
- scene_debug missing: B3/100
- scene_debug missing: B3/101
- scene_debug missing: B3/102
- scene_debug missing: B3/103
- scene_debug missing: B3/104
- scene_debug missing: C0/0
- scene_debug missing: C0/1
- scene_debug missing: C0/10
- scene_debug missing: C0/11
- scene_debug missing: C0/12
- scene_debug missing: C0/13
- scene_debug missing: C0/14
- scene_debug missing: C0/15

