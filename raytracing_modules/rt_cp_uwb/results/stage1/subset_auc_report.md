# Stage 1 Subset AUC

Primary feature: `gamma_cp_3_fp_only`

| subset | n | n_los | n_nlos | mean_gamma_los | mean_gamma_nlos | auc_gamma_cp_3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ideal_all | 100 | 33 | 67 | 0.0115 | 0.0145 | 0.5712 |
| patch_all | 100 | 31 | 69 | 0.0133 | 0.0118 | 0.5175 |
| ideal_ceiling | 27 | 4 | 23 | 0.0049 | 0.0158 | 0.7283 |
| ideal_floor | 24 | 10 | 14 | 0.0134 | 0.0116 | 0.4357 |
| ideal_wall_x | 27 | 10 | 17 | 0.0151 | 0.0167 | 0.5647 |
| ideal_wall_y | 22 | 9 | 13 | 0.0082 | 0.0122 | 0.6154 |
| patch_ceiling | 23 | 4 | 19 | 0.0256 | 0.0075 | 0.3158 |
| patch_floor | 26 | 8 | 18 | 0.0138 | 0.0100 | 0.4861 |
| patch_wall_x | 23 | 8 | 15 | 0.0066 | 0.0179 | 0.7417 |
| patch_wall_y | 28 | 11 | 17 | 0.0132 | 0.0132 | 0.5455 |

- patch overall AUC verdict: **too_weak**
