# Stage 2 Relabel Rerun

- source dataset: `D:\codex\raytracing_modules\rt_cp_uwb\results\stage2\stage2_900_ffd.mat`
- relabeled dataset: `D:\codex\raytracing_modules\rt_cp_uwb\results\stage2\stage2_900_ffd_relabel.mat`
- primary label in relabeled export: `mixed_0p33`
- auxiliary labels retained: `current_0p20`, `geo_only`, `bounce_0p33`

## Label Balance (valid cases only)

| label | n_neg | n_pos |
|---|---:|---:|
| current_0p20 | 79 | 821 |
| geo_only | 833 | 67 |
| bounce_0p33 | 530 | 370 |
| mixed_0p33 | 463 | 437 |

## Global Metrics

| set | n | n_los | n_nlos | auc_cir | auc_cp | auc_joint | delta_auc |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL_mixed_0p33 | 900 | 463 | 437 | 0.6853 | 0.5900 | 0.7079 | 0.0225 |
| ALL_geo_only | 900 | 833 | 67 | 0.6605 | 0.6491 | 0.7190 | 0.0584 |

## Room Metrics

| set | n | n_los | n_nlos | auc_cir | auc_cp | auc_joint | delta_auc |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_mixed_0p33 | 300 | 165 | 135 | 0.6817 | 0.6070 | 0.7206 | 0.0389 |
| A_geo_only | 300 | 300 | 0 | NaN | NaN | NaN | NaN |
| B_mixed_0p33 | 300 | 162 | 138 | 0.7168 | 0.6001 | 0.7339 | 0.0171 |
| B_geo_only | 300 | 293 | 7 | 0.8181 | 0.6558 | 0.8333 | 0.0151 |
| C_mixed_0p33 | 300 | 136 | 164 | 0.7003 | 0.6135 | 0.7418 | 0.0415 |
| C_geo_only | 300 | 240 | 60 | 0.6901 | 0.6535 | 0.7394 | 0.0494 |
