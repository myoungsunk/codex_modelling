# Theta0 Peak-Aligned CP/CIR/Joint Comparison

Source: `theta0_cp16_primary_metrics.csv`; variant: `new_theta0_posz_cp16`; scope: `ALL`.

## AUC / PR-AUC

| stage | cv_scheme | model | feature_set | n | n_pos | AUC | PR_AUC |
|---|---|---|---|---:|---:|---:|---:|
| stage1 | stratified5 | cir_only | CIR_only_12 | 3000.000000 | 2120.000000 | 0.678617 | 0.826282 |
| stage1 | stratified5 | cp_only | CP_all_RH_LH_16 | 3000.000000 | 2120.000000 | 0.700426 | 0.830601 |
| stage1 | stratified5 | joint | Joint_CIR_plus_CP_all | 3000.000000 | 2120.000000 | 0.762476 | 0.872029 |
| stage2 | fit_all | cir_only | CIR_only_12 | 900.000000 | 437.000000 | 0.662657 | 0.641354 |
| stage2 | fit_all | cp_only | CP_all_RH_LH_16 | 900.000000 | 437.000000 | 0.696502 | 0.724628 |
| stage2 | fit_all | joint | Joint_CIR_plus_CP_all | 900.000000 | 437.000000 | 0.744582 | 0.746724 |
| stage2 | stratified5 | cir_only | CIR_only_12 | 900.000000 | 437.000000 | 0.637885 | 0.610470 |
| stage2 | stratified5 | cp_only | CP_all_RH_LH_16 | 900.000000 | 437.000000 | 0.683123 | 0.714215 |
| stage2 | stratified5 | joint | Joint_CIR_plus_CP_all | 900.000000 | 437.000000 | 0.721056 | 0.723878 |
| stage2 | room_label_stratified5 | cir_only | CIR_only_12 | 900.000000 | 437.000000 | 0.639640 | 0.615663 |
| stage2 | room_label_stratified5 | cp_only | CP_all_RH_LH_16 | 900.000000 | 437.000000 | 0.683202 | 0.713771 |
| stage2 | room_label_stratified5 | joint | Joint_CIR_plus_CP_all | 900.000000 | 437.000000 | 0.714759 | 0.719408 |
| stage2 | leave_one_room_out | cir_only | CIR_only_12 | 900.000000 | 437.000000 | 0.617943 | 0.596697 |
| stage2 | leave_one_room_out | cp_only | CP_all_RH_LH_16 | 900.000000 | 437.000000 | 0.674173 | 0.682005 |
| stage2 | leave_one_room_out | joint | Joint_CIR_plus_CP_all | 900.000000 | 437.000000 | 0.709234 | 0.695084 |

## Paired Delta / Recovery

| stage | cv_scheme | AUC_CIR | AUC_CP | AUC_Joint | Delta_AUC | Delta_AUC_CI_low | Net_recovery | Net_recovery_CI_low |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stage1 | stratified5 | 0.678617 | 0.700426 | 0.762476 | 0.083859 | 0.068564 | 0.055333 | 0.043333 |
| stage2 | stratified5 | 0.637885 | 0.683123 | 0.721056 | 0.083171 | 0.055498 | 0.045556 | 0.011083 |
| stage2 | room_label_stratified5 | 0.639640 | 0.683202 | 0.714759 | 0.075119 | 0.047336 | 0.038889 | 0.005556 |
| stage2 | leave_one_room_out | 0.617943 | 0.674173 | 0.709234 | 0.091291 | 0.063014 | 0.057778 | 0.023333 |
