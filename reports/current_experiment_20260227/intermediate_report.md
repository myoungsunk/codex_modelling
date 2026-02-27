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

- WARN: scene plot missing


### A3

- WARN: scene plot missing


### A4

- WARN: scene plot missing


### A5

- WARN: scene plot missing


### B1

- WARN: scene plot missing

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

- WARN: scene plot missing

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

- WARN: scene plot missing

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

- WARN: scene plot missing


