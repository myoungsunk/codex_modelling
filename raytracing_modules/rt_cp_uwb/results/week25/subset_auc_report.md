# Subset AUC Diagnostic

Primary feature: `gamma_cp_3_fp_only`

Subset mode: `slab_placement (slab_placement_cases_tbl)`

| subset | antenna_type | los_blockage | n | n_los | n_nlos | mean_gamma_los | mean_gamma_nlos | auc_gamma_cp_3 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ideal_ceiling | ideal | all | 27 | 4 | 23 | 0.0049 | 0.0158 | 0.7283 |
| ideal_floor | ideal | all | 24 | 10 | 14 | 0.0134 | 0.0116 | 0.4357 |
| ideal_wall_x | ideal | all | 27 | 10 | 17 | 0.0151 | 0.0167 | 0.5647 |
| ideal_wall_y | ideal | all | 22 | 9 | 13 | 0.0082 | 0.0122 | 0.6154 |
| patch_ceiling | patch | all | 23 | 4 | 19 | 1.0206 | 0.6147 | 0.3684 |
| patch_floor | patch | all | 26 | 8 | 18 | 0.5981 | 1.0257 | 0.6597 |
| patch_wall_x | patch | all | 23 | 8 | 15 | 0.7053 | 1.0735 | 0.6833 |
| patch_wall_y | patch | all | 28 | 11 | 17 | 0.8732 | 1.0094 | 0.4011 |
| ideal_all | ideal | all | 100 | 33 | 67 | 0.0115 | 0.0145 | 0.5712 |
| patch_all | patch | all | 100 | 31 | 69 | 0.7779 | 0.9189 | 0.5264 |

## Signal 1 Decision

- ideal gamma_cp_3 AUC: 0.5712
- patch gamma_cp_3 AUC: 0.5264
- verdict: **partial**
- rationale: patch subset AUC is weak, suggesting limited CP usefulness in this setup
- patch mean gamma_cp_3 (LoS): 0.7779
- patch mean gamma_cp_3 (NLoS): 0.9189
