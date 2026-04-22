# Stage 3 HFSS 30-Case Reselection

## Principle

- Total cases: 30
- GEO group: 15 cases (3 regimes x 5 cases)
- BOUNCE group: 15 cases (3 regimes x 5 cases)
- Selection basis: Stage 2 relabeled dataset with dual-label decomposition.
- Reporting label remains `mixed@0.33`, but Stage 3 selection is split into physical sub-regimes.

## Selected Regimes

| group | regime | n | n_neg | n_pos | auc_cir | auc_joint | delta_auc |
|---|---|---:|---:|---:|---:|---:|---:|
| GEO | room type=C, los angle from anchor bore=[40.56, 46.12] | 63 | 44 | 15 | 0.8153 | 1.0000 | 0.1847 |
| GEO | room type=C, los angle from anchor bore=[33.29, 40.56] | 58 | 43 | 15 | 0.8321 | 1.0000 | 0.1679 |
| GEO | room type=C, xpol coupling db=[20.04, 24.00] | 60 | 49 | 10 | 0.8660 | 1.0000 | 0.1340 |
| BOUNCE | room type=C, los angle from anchor bore=[11.63, 33.29] | 49 | 21 | 26 | 0.7250 | 0.9183 | 0.1933 |
| BOUNCE | room type=A, xpol coupling db=[32.05, 35.96] | 58 | 30 | 28 | 0.7702 | 0.9143 | 0.1440 |
| BOUNCE | room type=A, dominant wall material=wood | 60 | 43 | 17 | 0.7839 | 0.9261 | 0.1423 |

## Outputs

- Cell summary: `D:\codex\raytracing_modules\rt_cp_uwb\results\stage3\stage3_reselection_cells.csv`
- HFSS case list: `D:\codex\raytracing_modules\rt_cp_uwb\results\stage3\hfss_case_list.csv`
