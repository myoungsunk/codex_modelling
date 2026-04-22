# Week 3 Day 4 Subset AUC (FFD)

- primary feature: `gamma_cp_3_fp_only`
- patch_ffd_all verdict: **target_range**

| subset | n | n_los | n_nlos | mean_gamma_los | mean_gamma_nlos | auc_gamma_cp_3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ideal_all | 100 | 33 | 67 | 0.0115 | 0.0145 | 0.5712 |
| patch_ffd_all | 100 | 31 | 69 | 0.7084 | 1.0427 | 0.6204 |
| patch_ffd_ceiling | 23 | 4 | 19 | 0.7991 | 0.7862 | 0.4605 |
| patch_ffd_floor | 26 | 8 | 18 | 0.5516 | 1.1081 | 0.6667 |
| patch_ffd_wall_x | 23 | 8 | 15 | 0.7527 | 1.2021 | 0.6833 |
| patch_ffd_wall_y | 28 | 11 | 17 | 0.7573 | 1.1194 | 0.6150 |
| patch_ffd_angle_bin1_[0.4,37.1] | 25 | 11 | 14 | 0.8754 | 1.1354 | 0.4870 |
| patch_ffd_angle_bin2_[37.1,43.1] | 25 | 7 | 18 | 0.7152 | 0.8325 | 0.5000 |
| patch_ffd_angle_bin3_[43.1,52.0] | 25 | 5 | 20 | 0.9399 | 1.3903 | 0.6300 |
| patch_ffd_angle_bin4_[52.0,73.5] | 25 | 8 | 17 | 0.3283 | 0.7799 | 0.8971 |
