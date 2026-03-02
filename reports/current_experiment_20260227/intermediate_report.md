# Intermediate Report (report_20260227_current)

## Proposition Status

| proposition | status | data_status | definition |
| --- | --- | --- | --- |
| M1 | SUPPORTED | OK | C0 floor distance/yaw sensitivity |
| M2 | SUPPORTED | OK | Fraction inside/outside floor uncertainty band |
| G1 | SUPPORTED | OK | A2 odd-bounce increases cross dominance vs C0 |
| G2 | SUPPORTED | OK | A3 even-bounce recovery vs A2 |
| L1 | SUPPORTED | OK | Early/Late leakage separation via XPD_early_ex/XPD_late_ex/L_pol |
| L2 | SUPPORTED | OK | A5 stress impact on center/tails |
| L3 | PARTIAL | OK | Correlation between leakage excess and -EL_proxy |
| R1 | SUPPORTED | OK | Room-space LOS/NLOS and heatmap consistency |
| R2 | SUPPORTED | OK | DS/early-fraction relation with leakage |
| P1 | SUPPORTED | OK | Conditional model improvement vs constant baseline |
| P2 | SUPPORTED | OK | Sign stability of EL coefficient under subsampling |

## M1

- Definition: C0 floor distance/yaw sensitivity
- Status: **SUPPORTED**
- Details: `{"definition": "C0 floor distance/yaw sensitivity", "n": 30, "trend": {"slope": -0.09401463204579644, "intercept": 24.978348288120827, "r": -0.05736274400083174, "p": 0.7633487051129717, "n": 30}, "variance_decomp": {"n": 30, "distance_group_var": 0.05763991702654064, "yaw_group_var": 0.022595997370235613, "dominant": "distance"}, "status": "SUPPORTED"}`

## M2

- Definition: Fraction inside/outside floor uncertainty band
- Status: **SUPPORTED**
- Details: `{"definition": "Fraction inside/outside floor uncertainty band", "inside_ratio": 0.05510204081632653, "outside_ratio": 0.9448979591836735, "status": "SUPPORTED"}`

## G1

- Definition: A2 odd-bounce increases cross dominance vs C0
- Status: **SUPPORTED**
- Details: `{"definition": "A2 odd-bounce increases cross dominance vs C0", "delta_median_db": -31.27019954792553, "ks_wasserstein": {"ks_stat": 1.0, "ks_p": 1.4253754280521992e-16, "wasserstein": 31.34884510222977}, "status": "SUPPORTED"}`

## G2

- Definition: A3 even-bounce recovery vs A2
- Status: **SUPPORTED**
- Details: `{"definition": "A3 even-bounce recovery vs A2", "delta_median_db": 6.335361848931871, "ks_wasserstein": {"ks_stat": 0.9166666666666666, "ks_p": 1.1106662957837858e-11, "wasserstein": 8.107911002203206}, "status": "SUPPORTED"}`

## L1

- Definition: Early/Late leakage separation via XPD_early_ex/XPD_late_ex/L_pol
- Status: **SUPPORTED**
- Details: `{"definition": "Early/Late leakage separation via XPD_early_ex/XPD_late_ex/L_pol", "median_early_ex_db": -30.41909792597057, "median_late_ex_db": -24.934837698993658, "median_lpol_db": -5.8509891172287265, "status": "SUPPORTED"}`

## L2

- Definition: A5 stress impact on center/tails
- Status: **SUPPORTED**
- Details: `{"definition": "A5 stress impact on center/tails", "base": {"n": 30, "mean": -19.032027100100454, "std": 0.6584136378266066, "p5": -19.94015124287934, "p10": -19.6981530121707, "p90": -18.169538584463584, "p95": -17.98577312640226}, "stress": {"n": 30, "mean": -26.686283690330317, "std": 1.0050872712546133, "p5": -27.94015100518591, "p10": -27.561742999712436, "p90": -24.934837698993658, "p95": -24.934837698993658}, "delta_mean_db": -7.654256590229863, "var_ratio": 1.526528634146098, "status": "SUPPORTED"}`

## L3

- Definition: Correlation between leakage excess and -EL_proxy
- Status: **PARTIAL**
- Details: `{"definition": "Correlation between leakage excess and -EL_proxy", "spearman": {"rho": 0.21543610057151755, "p": 6.896601784337098e-06, "ci_lo": 0.1266967794361815, "ci_hi": 0.30204167970700874, "n": 428}, "status": "PARTIAL"}`

## R1

- Definition: Room-space LOS/NLOS and heatmap consistency
- Status: **SUPPORTED**
- Details: `{"definition": "Room-space LOS/NLOS and heatmap consistency", "n_b_rows": 291, "n_los": 146, "n_nlos": 145, "ks_wasserstein": {"ks_stat": 0.7310344827586207, "ks_p": 2.209648908220457e-36, "wasserstein": 7.5474307041073825}, "status": "SUPPORTED"}`

## R2

- Definition: DS/early-fraction relation with leakage
- Status: **SUPPORTED**
- Details: `{"definition": "DS/early-fraction relation with leakage", "spearman_ds_vs_xpd": {"rho": -0.3623856928208357, "p": 1.2916730986139675e-10, "ci_lo": -0.4528203752879176, "ci_hi": -0.26138334099518695, "n": 296}, "status": "SUPPORTED"}`

## P1

- Definition: Conditional model improvement vs constant baseline
- Status: **SUPPORTED**
- Details: `{"definition": "Conditional model improvement vs constant baseline", "cv": {"n": 428, "rmse_const": 5.508753947662446, "rmse_lin": 4.431223008607797, "nll_const": 3.124307824882634, "nll_lin": 2.9055556625650656}, "status": "SUPPORTED"}`

## P2

- Definition: Sign stability of EL coefficient under subsampling
- Status: **SUPPORTED**
- Details: `{"definition": "Sign stability of EL coefficient under subsampling", "stability": {"n": 428, "base_sign": -1.0, "sign_keep_rate": 1.0}, "status": "SUPPORTED"}`

## Scenario Sections

### A2

![A2 scene](figures/A2__0__scene.png)


### A3

![A3 scene](figures/A3__0__scene.png)


### A4

![A4 scene](figures/A4__0__scene.png)


### A5

![A5 scene](figures/A5__0__scene.png)


### B1

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

![C0 scene](figures/C0__0__scene.png)


## WARN

- scene_debug missing: A2/0 -> fallback layout used
- scene_debug missing: A2/1 -> fallback layout used
- scene_debug missing: A2/10 -> fallback layout used
- scene_debug missing: A2/11 -> fallback layout used
- scene_debug missing: A2/12 -> fallback layout used
- scene_debug missing: A2/13 -> fallback layout used
- scene_debug missing: A2/14 -> fallback layout used
- scene_debug missing: A2/15 -> fallback layout used
- scene_debug missing: A2/16 -> fallback layout used
- scene_debug missing: A2/17 -> fallback layout used
- scene_debug missing: A2/18 -> fallback layout used
- scene_debug missing: A2/19 -> fallback layout used
- scene_debug missing: A2/2 -> fallback layout used
- scene_debug missing: A2/20 -> fallback layout used
- scene_debug missing: A2/21 -> fallback layout used
- scene_debug missing: A2/22 -> fallback layout used
- scene_debug missing: A2/23 -> fallback layout used
- scene_debug missing: A2/24 -> fallback layout used
- scene_debug missing: A2/25 -> fallback layout used
- scene_debug missing: A2/26 -> fallback layout used
- scene_debug missing: A2/3 -> fallback layout used
- scene_debug missing: A2/4 -> fallback layout used
- scene_debug missing: A2/5 -> fallback layout used
- scene_debug missing: A2/6 -> fallback layout used
- scene_debug missing: A2/7 -> fallback layout used
- scene_debug missing: A2/8 -> fallback layout used
- scene_debug missing: A2/9 -> fallback layout used
- scene_debug missing: A3/0 -> fallback layout used
- scene_debug missing: A3/1 -> fallback layout used
- scene_debug missing: A3/10 -> fallback layout used
- scene_debug missing: A3/11 -> fallback layout used
- scene_debug missing: A3/12 -> fallback layout used
- scene_debug missing: A3/13 -> fallback layout used
- scene_debug missing: A3/14 -> fallback layout used
- scene_debug missing: A3/15 -> fallback layout used
- scene_debug missing: A3/16 -> fallback layout used
- scene_debug missing: A3/17 -> fallback layout used
- scene_debug missing: A3/18 -> fallback layout used
- scene_debug missing: A3/19 -> fallback layout used
- scene_debug missing: A3/2 -> fallback layout used
- scene_debug missing: A3/20 -> fallback layout used
- scene_debug missing: A3/21 -> fallback layout used
- scene_debug missing: A3/22 -> fallback layout used
- scene_debug missing: A3/23 -> fallback layout used
- scene_debug missing: A3/3 -> fallback layout used
- scene_debug missing: A3/4 -> fallback layout used
- scene_debug missing: A3/5 -> fallback layout used
- scene_debug missing: A3/6 -> fallback layout used
- scene_debug missing: A3/7 -> fallback layout used
- scene_debug missing: A3/8 -> fallback layout used
- scene_debug missing: A3/9 -> fallback layout used
- scene_debug missing: A4/0 -> fallback layout used
- scene_debug missing: A4/1 -> fallback layout used
- scene_debug missing: A4/10 -> fallback layout used
- scene_debug missing: A4/11 -> fallback layout used
- scene_debug missing: A4/12 -> fallback layout used
- scene_debug missing: A4/13 -> fallback layout used
- scene_debug missing: A4/14 -> fallback layout used
- scene_debug missing: A4/15 -> fallback layout used
- scene_debug missing: A4/16 -> fallback layout used
- scene_debug missing: A4/17 -> fallback layout used
- scene_debug missing: A4/18 -> fallback layout used
- scene_debug missing: A4/19 -> fallback layout used
- scene_debug missing: A4/2 -> fallback layout used
- scene_debug missing: A4/20 -> fallback layout used
- scene_debug missing: A4/21 -> fallback layout used
- scene_debug missing: A4/22 -> fallback layout used
- scene_debug missing: A4/23 -> fallback layout used
- scene_debug missing: A4/24 -> fallback layout used
- scene_debug missing: A4/25 -> fallback layout used
- scene_debug missing: A4/26 -> fallback layout used
- scene_debug missing: A4/27 -> fallback layout used
- scene_debug missing: A4/28 -> fallback layout used
- scene_debug missing: A4/29 -> fallback layout used
- scene_debug missing: A4/3 -> fallback layout used
- scene_debug missing: A4/30 -> fallback layout used
- scene_debug missing: A4/31 -> fallback layout used
- scene_debug missing: A4/32 -> fallback layout used
- scene_debug missing: A4/33 -> fallback layout used
- scene_debug missing: A4/34 -> fallback layout used
- scene_debug missing: A4/35 -> fallback layout used
- scene_debug missing: A4/36 -> fallback layout used
- scene_debug missing: A4/37 -> fallback layout used
- scene_debug missing: A4/38 -> fallback layout used
- scene_debug missing: A4/39 -> fallback layout used
- scene_debug missing: A4/4 -> fallback layout used
- scene_debug missing: A4/40 -> fallback layout used
- scene_debug missing: A4/41 -> fallback layout used
- scene_debug missing: A4/42 -> fallback layout used
- scene_debug missing: A4/43 -> fallback layout used
- scene_debug missing: A4/44 -> fallback layout used
- scene_debug missing: A4/45 -> fallback layout used
- scene_debug missing: A4/46 -> fallback layout used
- scene_debug missing: A4/47 -> fallback layout used
- scene_debug missing: A4/48 -> fallback layout used
- scene_debug missing: A4/49 -> fallback layout used
- scene_debug missing: A4/5 -> fallback layout used
- scene_debug missing: A4/50 -> fallback layout used
- scene_debug missing: A4/51 -> fallback layout used
- scene_debug missing: A4/52 -> fallback layout used
- scene_debug missing: A4/53 -> fallback layout used
- scene_debug missing: A4/54 -> fallback layout used
- scene_debug missing: A4/55 -> fallback layout used
- scene_debug missing: A4/56 -> fallback layout used
- scene_debug missing: A4/57 -> fallback layout used
- scene_debug missing: A4/58 -> fallback layout used
- scene_debug missing: A4/59 -> fallback layout used
- scene_debug missing: A4/6 -> fallback layout used
- scene_debug missing: A4/60 -> fallback layout used
- scene_debug missing: A4/61 -> fallback layout used
- scene_debug missing: A4/62 -> fallback layout used
- scene_debug missing: A4/63 -> fallback layout used
- scene_debug missing: A4/64 -> fallback layout used
- scene_debug missing: A4/65 -> fallback layout used
- scene_debug missing: A4/66 -> fallback layout used
- scene_debug missing: A4/67 -> fallback layout used
- scene_debug missing: A4/68 -> fallback layout used
- scene_debug missing: A4/69 -> fallback layout used
- scene_debug missing: A4/7 -> fallback layout used
- scene_debug missing: A4/70 -> fallback layout used
- scene_debug missing: A4/71 -> fallback layout used
- scene_debug missing: A4/8 -> fallback layout used
- scene_debug missing: A4/9 -> fallback layout used
- scene_debug missing: A5/0 -> fallback layout used
- scene_debug missing: A5/1 -> fallback layout used
- scene_debug missing: A5/10 -> fallback layout used
- scene_debug missing: A5/11 -> fallback layout used
- scene_debug missing: A5/12 -> fallback layout used
- scene_debug missing: A5/13 -> fallback layout used
- scene_debug missing: A5/14 -> fallback layout used
- scene_debug missing: A5/15 -> fallback layout used
- scene_debug missing: A5/16 -> fallback layout used
- scene_debug missing: A5/17 -> fallback layout used
- scene_debug missing: A5/18 -> fallback layout used
- scene_debug missing: A5/19 -> fallback layout used
- scene_debug missing: A5/2 -> fallback layout used
- scene_debug missing: A5/20 -> fallback layout used
- scene_debug missing: A5/21 -> fallback layout used
- scene_debug missing: A5/22 -> fallback layout used
- scene_debug missing: A5/23 -> fallback layout used
- scene_debug missing: A5/24 -> fallback layout used
- scene_debug missing: A5/25 -> fallback layout used
- scene_debug missing: A5/26 -> fallback layout used
- scene_debug missing: A5/27 -> fallback layout used
- scene_debug missing: A5/28 -> fallback layout used
- scene_debug missing: A5/29 -> fallback layout used
- scene_debug missing: A5/3 -> fallback layout used
- scene_debug missing: A5/4 -> fallback layout used
- scene_debug missing: A5/5 -> fallback layout used
- scene_debug missing: A5/6 -> fallback layout used
- scene_debug missing: A5/7 -> fallback layout used
- scene_debug missing: A5/8 -> fallback layout used
- scene_debug missing: A5/9 -> fallback layout used
- scene_debug missing: B1/0 -> fallback layout used
- scene_debug missing: B1/1 -> fallback layout used
- scene_debug missing: B1/10 -> fallback layout used
- scene_debug missing: B1/11 -> fallback layout used
- scene_debug missing: B1/12 -> fallback layout used
- scene_debug missing: B1/13 -> fallback layout used
- scene_debug missing: B1/14 -> fallback layout used
- scene_debug missing: B1/15 -> fallback layout used
- scene_debug missing: B1/16 -> fallback layout used
- scene_debug missing: B1/17 -> fallback layout used
- scene_debug missing: B1/18 -> fallback layout used
- scene_debug missing: B1/19 -> fallback layout used
- scene_debug missing: B1/2 -> fallback layout used
- scene_debug missing: B1/20 -> fallback layout used
- scene_debug missing: B1/21 -> fallback layout used
- scene_debug missing: B1/22 -> fallback layout used
- scene_debug missing: B1/23 -> fallback layout used
- scene_debug missing: B1/24 -> fallback layout used
- scene_debug missing: B1/25 -> fallback layout used
- scene_debug missing: B1/26 -> fallback layout used
- scene_debug missing: B1/27 -> fallback layout used
- scene_debug missing: B1/28 -> fallback layout used
- scene_debug missing: B1/29 -> fallback layout used
- scene_debug missing: B1/3 -> fallback layout used
- scene_debug missing: B1/30 -> fallback layout used
- scene_debug missing: B1/31 -> fallback layout used
- scene_debug missing: B1/32 -> fallback layout used
- scene_debug missing: B1/33 -> fallback layout used
- scene_debug missing: B1/34 -> fallback layout used
- scene_debug missing: B1/35 -> fallback layout used
- scene_debug missing: B1/36 -> fallback layout used
- scene_debug missing: B1/37 -> fallback layout used
- scene_debug missing: B1/38 -> fallback layout used
- scene_debug missing: B1/39 -> fallback layout used
- scene_debug missing: B1/4 -> fallback layout used
- scene_debug missing: B1/40 -> fallback layout used
- scene_debug missing: B1/41 -> fallback layout used
- scene_debug missing: B1/42 -> fallback layout used
- scene_debug missing: B1/43 -> fallback layout used
- scene_debug missing: B1/44 -> fallback layout used
- scene_debug missing: B1/45 -> fallback layout used
- scene_debug missing: B1/46 -> fallback layout used
- scene_debug missing: B1/47 -> fallback layout used
- scene_debug missing: B1/48 -> fallback layout used
- scene_debug missing: B1/5 -> fallback layout used
- scene_debug missing: B1/6 -> fallback layout used
- scene_debug missing: B1/7 -> fallback layout used
- scene_debug missing: B1/8 -> fallback layout used
- scene_debug missing: B1/9 -> fallback layout used
- scene_debug missing: B2/0 -> fallback layout used
- scene_debug missing: B2/1 -> fallback layout used
- scene_debug missing: B2/10 -> fallback layout used
- scene_debug missing: B2/100 -> fallback layout used
- scene_debug missing: B2/101 -> fallback layout used
- scene_debug missing: B2/102 -> fallback layout used
- scene_debug missing: B2/103 -> fallback layout used
- scene_debug missing: B2/104 -> fallback layout used
- scene_debug missing: B2/105 -> fallback layout used
- scene_debug missing: B2/106 -> fallback layout used
- scene_debug missing: B2/107 -> fallback layout used
- scene_debug missing: B2/108 -> fallback layout used
- scene_debug missing: B2/109 -> fallback layout used
- scene_debug missing: B2/11 -> fallback layout used
- scene_debug missing: B2/110 -> fallback layout used
- scene_debug missing: B2/111 -> fallback layout used
- scene_debug missing: B2/112 -> fallback layout used
- scene_debug missing: B2/113 -> fallback layout used
- scene_debug missing: B2/114 -> fallback layout used
- scene_debug missing: B2/115 -> fallback layout used
- scene_debug missing: B2/116 -> fallback layout used
- scene_debug missing: B2/117 -> fallback layout used
- scene_debug missing: B2/118 -> fallback layout used
- scene_debug missing: B2/119 -> fallback layout used
- scene_debug missing: B2/12 -> fallback layout used
- scene_debug missing: B2/120 -> fallback layout used
- scene_debug missing: B2/13 -> fallback layout used
- scene_debug missing: B2/14 -> fallback layout used
- scene_debug missing: B2/15 -> fallback layout used
- scene_debug missing: B2/16 -> fallback layout used
- scene_debug missing: B2/17 -> fallback layout used
- scene_debug missing: B2/18 -> fallback layout used
- scene_debug missing: B2/19 -> fallback layout used
- scene_debug missing: B2/2 -> fallback layout used
- scene_debug missing: B2/20 -> fallback layout used
- scene_debug missing: B2/21 -> fallback layout used
- scene_debug missing: B2/22 -> fallback layout used
- scene_debug missing: B2/23 -> fallback layout used
- scene_debug missing: B2/24 -> fallback layout used
- scene_debug missing: B2/25 -> fallback layout used
- scene_debug missing: B2/26 -> fallback layout used
- scene_debug missing: B2/27 -> fallback layout used
- scene_debug missing: B2/28 -> fallback layout used
- scene_debug missing: B2/29 -> fallback layout used
- scene_debug missing: B2/3 -> fallback layout used
- scene_debug missing: B2/30 -> fallback layout used
- scene_debug missing: B2/31 -> fallback layout used
- scene_debug missing: B2/32 -> fallback layout used
- scene_debug missing: B2/33 -> fallback layout used
- scene_debug missing: B2/34 -> fallback layout used
- scene_debug missing: B2/35 -> fallback layout used
- scene_debug missing: B2/36 -> fallback layout used
- scene_debug missing: B2/37 -> fallback layout used
- scene_debug missing: B2/38 -> fallback layout used
- scene_debug missing: B2/39 -> fallback layout used
- scene_debug missing: B2/4 -> fallback layout used
- scene_debug missing: B2/40 -> fallback layout used
- scene_debug missing: B2/41 -> fallback layout used
- scene_debug missing: B2/42 -> fallback layout used
- scene_debug missing: B2/43 -> fallback layout used
- scene_debug missing: B2/44 -> fallback layout used
- scene_debug missing: B2/45 -> fallback layout used
- scene_debug missing: B2/46 -> fallback layout used
- scene_debug missing: B2/47 -> fallback layout used
- scene_debug missing: B2/48 -> fallback layout used
- scene_debug missing: B2/49 -> fallback layout used
- scene_debug missing: B2/5 -> fallback layout used
- scene_debug missing: B2/50 -> fallback layout used
- scene_debug missing: B2/51 -> fallback layout used
- scene_debug missing: B2/52 -> fallback layout used
- scene_debug missing: B2/53 -> fallback layout used
- scene_debug missing: B2/54 -> fallback layout used
- scene_debug missing: B2/55 -> fallback layout used
- scene_debug missing: B2/56 -> fallback layout used
- scene_debug missing: B2/57 -> fallback layout used
- scene_debug missing: B2/58 -> fallback layout used
- scene_debug missing: B2/59 -> fallback layout used
- scene_debug missing: B2/6 -> fallback layout used
- scene_debug missing: B2/60 -> fallback layout used
- scene_debug missing: B2/61 -> fallback layout used
- scene_debug missing: B2/62 -> fallback layout used
- scene_debug missing: B2/63 -> fallback layout used
- scene_debug missing: B2/64 -> fallback layout used
- scene_debug missing: B2/65 -> fallback layout used
- scene_debug missing: B2/66 -> fallback layout used
- scene_debug missing: B2/67 -> fallback layout used
- scene_debug missing: B2/68 -> fallback layout used
- scene_debug missing: B2/69 -> fallback layout used
- scene_debug missing: B2/7 -> fallback layout used
- scene_debug missing: B2/70 -> fallback layout used
- scene_debug missing: B2/71 -> fallback layout used
- scene_debug missing: B2/72 -> fallback layout used
- scene_debug missing: B2/73 -> fallback layout used
- scene_debug missing: B2/74 -> fallback layout used
- scene_debug missing: B2/75 -> fallback layout used
- scene_debug missing: B2/76 -> fallback layout used
- scene_debug missing: B2/77 -> fallback layout used
- scene_debug missing: B2/78 -> fallback layout used
- scene_debug missing: B2/79 -> fallback layout used
- scene_debug missing: B2/8 -> fallback layout used
- scene_debug missing: B2/80 -> fallback layout used
- scene_debug missing: B2/81 -> fallback layout used
- scene_debug missing: B2/82 -> fallback layout used
- scene_debug missing: B2/83 -> fallback layout used
- scene_debug missing: B2/84 -> fallback layout used
- scene_debug missing: B2/85 -> fallback layout used
- scene_debug missing: B2/86 -> fallback layout used
- scene_debug missing: B2/87 -> fallback layout used
- scene_debug missing: B2/88 -> fallback layout used
- scene_debug missing: B2/89 -> fallback layout used
- scene_debug missing: B2/9 -> fallback layout used
- scene_debug missing: B2/90 -> fallback layout used
- scene_debug missing: B2/91 -> fallback layout used
- scene_debug missing: B2/92 -> fallback layout used
- scene_debug missing: B2/93 -> fallback layout used
- scene_debug missing: B2/94 -> fallback layout used
- scene_debug missing: B2/95 -> fallback layout used
- scene_debug missing: B2/96 -> fallback layout used
- scene_debug missing: B2/97 -> fallback layout used
- scene_debug missing: B2/98 -> fallback layout used
- scene_debug missing: B2/99 -> fallback layout used
- scene_debug missing: B3/0 -> fallback layout used
- scene_debug missing: B3/1 -> fallback layout used
- scene_debug missing: B3/10 -> fallback layout used
- scene_debug missing: B3/100 -> fallback layout used
- scene_debug missing: B3/101 -> fallback layout used
- scene_debug missing: B3/102 -> fallback layout used
- scene_debug missing: B3/103 -> fallback layout used
- scene_debug missing: B3/104 -> fallback layout used
- scene_debug missing: B3/105 -> fallback layout used
- scene_debug missing: B3/106 -> fallback layout used
- scene_debug missing: B3/107 -> fallback layout used
- scene_debug missing: B3/108 -> fallback layout used
- scene_debug missing: B3/109 -> fallback layout used
- scene_debug missing: B3/11 -> fallback layout used
- scene_debug missing: B3/110 -> fallback layout used
- scene_debug missing: B3/111 -> fallback layout used
- scene_debug missing: B3/112 -> fallback layout used
- scene_debug missing: B3/113 -> fallback layout used
- scene_debug missing: B3/114 -> fallback layout used
- scene_debug missing: B3/115 -> fallback layout used
- scene_debug missing: B3/116 -> fallback layout used
- scene_debug missing: B3/117 -> fallback layout used
- scene_debug missing: B3/118 -> fallback layout used
- scene_debug missing: B3/119 -> fallback layout used
- scene_debug missing: B3/12 -> fallback layout used
- scene_debug missing: B3/120 -> fallback layout used
- scene_debug missing: B3/13 -> fallback layout used
- scene_debug missing: B3/14 -> fallback layout used
- scene_debug missing: B3/15 -> fallback layout used
- scene_debug missing: B3/16 -> fallback layout used
- scene_debug missing: B3/17 -> fallback layout used
- scene_debug missing: B3/18 -> fallback layout used
- scene_debug missing: B3/19 -> fallback layout used
- scene_debug missing: B3/2 -> fallback layout used
- scene_debug missing: B3/20 -> fallback layout used
- scene_debug missing: B3/21 -> fallback layout used
- scene_debug missing: B3/22 -> fallback layout used
- scene_debug missing: B3/23 -> fallback layout used
- scene_debug missing: B3/24 -> fallback layout used
- scene_debug missing: B3/25 -> fallback layout used
- scene_debug missing: B3/26 -> fallback layout used
- scene_debug missing: B3/27 -> fallback layout used
- scene_debug missing: B3/28 -> fallback layout used
- scene_debug missing: B3/29 -> fallback layout used
- scene_debug missing: B3/3 -> fallback layout used
- scene_debug missing: B3/30 -> fallback layout used
- scene_debug missing: B3/31 -> fallback layout used
- scene_debug missing: B3/32 -> fallback layout used
- scene_debug missing: B3/33 -> fallback layout used
- scene_debug missing: B3/34 -> fallback layout used
- scene_debug missing: B3/35 -> fallback layout used
- scene_debug missing: B3/36 -> fallback layout used
- scene_debug missing: B3/37 -> fallback layout used
- scene_debug missing: B3/38 -> fallback layout used
- scene_debug missing: B3/39 -> fallback layout used
- scene_debug missing: B3/4 -> fallback layout used
- scene_debug missing: B3/40 -> fallback layout used
- scene_debug missing: B3/41 -> fallback layout used
- scene_debug missing: B3/42 -> fallback layout used
- scene_debug missing: B3/43 -> fallback layout used
- scene_debug missing: B3/44 -> fallback layout used
- scene_debug missing: B3/45 -> fallback layout used
- scene_debug missing: B3/46 -> fallback layout used
- scene_debug missing: B3/47 -> fallback layout used
- scene_debug missing: B3/48 -> fallback layout used
- scene_debug missing: B3/49 -> fallback layout used
- scene_debug missing: B3/5 -> fallback layout used
- scene_debug missing: B3/50 -> fallback layout used
- scene_debug missing: B3/51 -> fallback layout used
- scene_debug missing: B3/52 -> fallback layout used
- scene_debug missing: B3/53 -> fallback layout used
- scene_debug missing: B3/54 -> fallback layout used
- scene_debug missing: B3/55 -> fallback layout used
- scene_debug missing: B3/56 -> fallback layout used
- scene_debug missing: B3/57 -> fallback layout used
- scene_debug missing: B3/58 -> fallback layout used
- scene_debug missing: B3/59 -> fallback layout used
- scene_debug missing: B3/6 -> fallback layout used
- scene_debug missing: B3/60 -> fallback layout used
- scene_debug missing: B3/61 -> fallback layout used
- scene_debug missing: B3/62 -> fallback layout used
- scene_debug missing: B3/63 -> fallback layout used
- scene_debug missing: B3/64 -> fallback layout used
- scene_debug missing: B3/65 -> fallback layout used
- scene_debug missing: B3/66 -> fallback layout used
- scene_debug missing: B3/67 -> fallback layout used
- scene_debug missing: B3/68 -> fallback layout used
- scene_debug missing: B3/69 -> fallback layout used
- scene_debug missing: B3/7 -> fallback layout used
- scene_debug missing: B3/70 -> fallback layout used
- scene_debug missing: B3/71 -> fallback layout used
- scene_debug missing: B3/72 -> fallback layout used
- scene_debug missing: B3/73 -> fallback layout used
- scene_debug missing: B3/74 -> fallback layout used
- scene_debug missing: B3/75 -> fallback layout used
- scene_debug missing: B3/76 -> fallback layout used
- scene_debug missing: B3/77 -> fallback layout used
- scene_debug missing: B3/78 -> fallback layout used
- scene_debug missing: B3/79 -> fallback layout used
- scene_debug missing: B3/8 -> fallback layout used
- scene_debug missing: B3/80 -> fallback layout used
- scene_debug missing: B3/81 -> fallback layout used
- scene_debug missing: B3/82 -> fallback layout used
- scene_debug missing: B3/83 -> fallback layout used
- scene_debug missing: B3/84 -> fallback layout used
- scene_debug missing: B3/85 -> fallback layout used
- scene_debug missing: B3/86 -> fallback layout used
- scene_debug missing: B3/87 -> fallback layout used
- scene_debug missing: B3/88 -> fallback layout used
- scene_debug missing: B3/89 -> fallback layout used
- scene_debug missing: B3/9 -> fallback layout used
- scene_debug missing: B3/90 -> fallback layout used
- scene_debug missing: B3/91 -> fallback layout used
- scene_debug missing: B3/92 -> fallback layout used
- scene_debug missing: B3/93 -> fallback layout used
- scene_debug missing: B3/94 -> fallback layout used
- scene_debug missing: B3/95 -> fallback layout used
- scene_debug missing: B3/96 -> fallback layout used
- scene_debug missing: B3/97 -> fallback layout used
- scene_debug missing: B3/98 -> fallback layout used
- scene_debug missing: B3/99 -> fallback layout used
- scene_debug missing: C0/0 -> fallback layout used
- scene_debug missing: C0/1 -> fallback layout used
- scene_debug missing: C0/10 -> fallback layout used
- scene_debug missing: C0/11 -> fallback layout used
- scene_debug missing: C0/12 -> fallback layout used
- scene_debug missing: C0/13 -> fallback layout used
- scene_debug missing: C0/14 -> fallback layout used
- scene_debug missing: C0/15 -> fallback layout used
- scene_debug missing: C0/16 -> fallback layout used
- scene_debug missing: C0/17 -> fallback layout used
- scene_debug missing: C0/18 -> fallback layout used
- scene_debug missing: C0/19 -> fallback layout used
- scene_debug missing: C0/2 -> fallback layout used
- scene_debug missing: C0/20 -> fallback layout used
- scene_debug missing: C0/21 -> fallback layout used
- scene_debug missing: C0/22 -> fallback layout used
- scene_debug missing: C0/23 -> fallback layout used
- scene_debug missing: C0/24 -> fallback layout used
- scene_debug missing: C0/25 -> fallback layout used
- scene_debug missing: C0/26 -> fallback layout used
- scene_debug missing: C0/27 -> fallback layout used
- scene_debug missing: C0/28 -> fallback layout used
- scene_debug missing: C0/29 -> fallback layout used
- scene_debug missing: C0/3 -> fallback layout used
- scene_debug missing: C0/4 -> fallback layout used
- scene_debug missing: C0/5 -> fallback layout used
- scene_debug missing: C0/6 -> fallback layout used
- scene_debug missing: C0/7 -> fallback layout used
- scene_debug missing: C0/8 -> fallback layout used
- scene_debug missing: C0/9 -> fallback layout used

