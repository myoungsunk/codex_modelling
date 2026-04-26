# Theta0 Peak-Aligned CP16 Stage 1/2 Comparison

The new rerun uses local `theta=0, phi=90`, i.e. local `+z`, as the nominal peak and maps it to Tx `-z` and Rx `+z`.

## Sources

- stage1 / old_local_negx: loaded / `D:\codex\raytracing_modules\rt_cp_uwb\results\stage1_peak_aligned_ffd_phi_fastest\stage1_3000_ffd_peak_aligned_phi_fastest_det.csv`
- stage1 / new_theta0_posz_cp16: loaded / `D:\codex\raytracing_modules\rt_cp_uwb\results\stage1_theta0_peak_aligned_cp16_phi_fastest\stage1_3000_theta0_peak_aligned_cp16_phi_fastest_det.csv`
- stage2 / old_local_negx: loaded / `D:\codex\raytracing_modules\rt_cp_uwb\results\stage2_peak_aligned_phi_fastest\stage2_900_ffd_peak_aligned_phi_fastest_det.csv`
- stage2 / new_theta0_posz_cp16: loaded / `D:\codex\raytracing_modules\rt_cp_uwb\results\stage2_theta0_peak_aligned_cp16_phi_fastest\stage2_900_theta0_peak_aligned_cp16_phi_fastest_det.csv`

## Primary Metrics

| stage | variant | scope | cv_scheme | feature_set | n | n_pos | AUC | PR_AUC | AUC_CIR | AUC_CP | AUC_Joint | Delta_AUC | Delta_AUC_boot_CI_low | Delta_AUC_boot_CI_high | Net_recovery | Net_recovery_CI_low | Net_recovery_CI_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage1 | old_local_negx | ALL | stratified5 | CIR_only_12 | 3000.000000 | 2120.000000 | 0.703599 | 0.842414 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage1 | old_local_negx | ALL | stratified5 | CP6_canonical | 3000.000000 | 2120.000000 | 0.658951 | 0.793526 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage1 | old_local_negx | ALL | stratified5 | Joint_CIR_plus_CP6 | 3000.000000 | 2120.000000 | 0.747627 | 0.862671 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage1 | old_local_negx | ALL | stratified5 | CP_all_RH_LH_16 | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage1 | old_local_negx | ALL | stratified5 | CP_pruned_6 | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage1 | old_local_negx | ALL | stratified5 | CP_low_overlap_4 | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage1 | old_local_negx | ALL | stratified5 | Joint_CIR_plus_CP_all | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage1 | new_theta0_posz_cp16 | ALL | stratified5 | CIR_only_12 | 3000.000000 | 2120.000000 | 0.678617 | 0.826282 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage1 | new_theta0_posz_cp16 | ALL | stratified5 | CP6_canonical | 3000.000000 | 2120.000000 | 0.635481 | 0.778409 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage1 | new_theta0_posz_cp16 | ALL | stratified5 | Joint_CIR_plus_CP6 | 3000.000000 | 2120.000000 | 0.722951 | 0.853210 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage1 | new_theta0_posz_cp16 | ALL | stratified5 | CP_all_RH_LH_16 | 3000.000000 | 2120.000000 | 0.700426 | 0.830601 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage1 | new_theta0_posz_cp16 | ALL | stratified5 | CP_pruned_6 | 3000.000000 | 2120.000000 | 0.646143 | 0.778820 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage1 | new_theta0_posz_cp16 | ALL | stratified5 | CP_low_overlap_4 | 3000.000000 | 2120.000000 | 0.651398 | 0.782398 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage1 | new_theta0_posz_cp16 | ALL | stratified5 | Joint_CIR_plus_CP_all | 3000.000000 | 2120.000000 | 0.762476 | 0.872029 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage1 | new_theta0_posz_cp16 | ALL | stratified5 | Joint_CIR_plus_CP_all | nan | nan | nan | nan | 0.678617 | 0.700426 | 0.762476 | 0.083859 | 0.068564 | 0.099969 | 0.055333 | 0.043333 | 0.067675 |
| stage2 | old_local_negx | ALL | fit_all | CIR_only_12 | 900.000000 | 437.000000 | 0.645739 | 0.607621 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | fit_all | CP6_canonical | 900.000000 | 437.000000 | 0.712654 | 0.709579 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | fit_all | Joint_CIR_plus_CP6 | 900.000000 | 437.000000 | 0.738740 | 0.724267 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | fit_all | CP_all_RH_LH_16 | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | fit_all | CP_pruned_6 | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | fit_all | CP_low_overlap_4 | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | fit_all | Joint_CIR_plus_CP_all | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | stratified5 | CIR_only_12 | 900.000000 | 437.000000 | 0.625890 | 0.582489 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | stratified5 | CP6_canonical | 900.000000 | 437.000000 | 0.703204 | 0.699293 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | stratified5 | Joint_CIR_plus_CP6 | 900.000000 | 437.000000 | 0.718110 | 0.702897 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | stratified5 | CP_all_RH_LH_16 | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | stratified5 | CP_pruned_6 | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | stratified5 | CP_low_overlap_4 | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | stratified5 | Joint_CIR_plus_CP_all | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | room_label_stratified5 | CIR_only_12 | 900.000000 | 437.000000 | 0.618778 | 0.572676 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | room_label_stratified5 | CP6_canonical | 900.000000 | 437.000000 | 0.705878 | 0.702920 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | room_label_stratified5 | Joint_CIR_plus_CP6 | 900.000000 | 437.000000 | 0.718481 | 0.701587 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | room_label_stratified5 | CP_all_RH_LH_16 | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | room_label_stratified5 | CP_pruned_6 | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | room_label_stratified5 | CP_low_overlap_4 | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | room_label_stratified5 | Joint_CIR_plus_CP_all | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | leave_one_room_out | CIR_only_12 | 900.000000 | 437.000000 | 0.590216 | 0.555304 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | leave_one_room_out | CP6_canonical | 900.000000 | 437.000000 | 0.703960 | 0.695225 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | leave_one_room_out | Joint_CIR_plus_CP6 | 900.000000 | 437.000000 | 0.697738 | 0.676580 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | leave_one_room_out | CP_all_RH_LH_16 | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | leave_one_room_out | CP_pruned_6 | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | leave_one_room_out | CP_low_overlap_4 | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | old_local_negx | ALL | leave_one_room_out | Joint_CIR_plus_CP_all | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | fit_all | CIR_only_12 | 900.000000 | 437.000000 | 0.662657 | 0.641354 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | fit_all | CP6_canonical | 900.000000 | 437.000000 | 0.668044 | 0.700484 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | fit_all | Joint_CIR_plus_CP6 | 900.000000 | 437.000000 | 0.726957 | 0.731560 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | fit_all | CP_all_RH_LH_16 | 900.000000 | 437.000000 | 0.696502 | 0.724628 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | fit_all | CP_pruned_6 | 900.000000 | 437.000000 | 0.679016 | 0.710710 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | fit_all | CP_low_overlap_4 | 900.000000 | 437.000000 | 0.672181 | 0.707646 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | fit_all | Joint_CIR_plus_CP_all | 900.000000 | 437.000000 | 0.744582 | 0.746724 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | stratified5 | CIR_only_12 | 900.000000 | 437.000000 | 0.637885 | 0.610470 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | stratified5 | CP6_canonical | 900.000000 | 437.000000 | 0.659652 | 0.689342 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | stratified5 | Joint_CIR_plus_CP6 | 900.000000 | 437.000000 | 0.699127 | 0.702742 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | stratified5 | CP_all_RH_LH_16 | 900.000000 | 437.000000 | 0.683123 | 0.714215 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | stratified5 | CP_pruned_6 | 900.000000 | 437.000000 | 0.671380 | 0.704993 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | stratified5 | CP_low_overlap_4 | 900.000000 | 437.000000 | 0.667159 | 0.704630 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | stratified5 | Joint_CIR_plus_CP_all | 900.000000 | 437.000000 | 0.721056 | 0.723878 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | room_label_stratified5 | CIR_only_12 | 900.000000 | 437.000000 | 0.639640 | 0.615663 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | room_label_stratified5 | CP6_canonical | 900.000000 | 437.000000 | 0.659889 | 0.682366 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | room_label_stratified5 | Joint_CIR_plus_CP6 | 900.000000 | 437.000000 | 0.701202 | 0.706039 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | room_label_stratified5 | CP_all_RH_LH_16 | 900.000000 | 437.000000 | 0.683202 | 0.713771 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | room_label_stratified5 | CP_pruned_6 | 900.000000 | 437.000000 | 0.672191 | 0.706744 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | room_label_stratified5 | CP_low_overlap_4 | 900.000000 | 437.000000 | 0.668296 | 0.705732 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | room_label_stratified5 | Joint_CIR_plus_CP_all | 900.000000 | 437.000000 | 0.714759 | 0.719408 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | leave_one_room_out | CIR_only_12 | 900.000000 | 437.000000 | 0.617943 | 0.596697 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | leave_one_room_out | CP6_canonical | 900.000000 | 437.000000 | 0.650405 | 0.667085 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | leave_one_room_out | Joint_CIR_plus_CP6 | 900.000000 | 437.000000 | 0.696369 | 0.689952 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | leave_one_room_out | CP_all_RH_LH_16 | 900.000000 | 437.000000 | 0.674173 | 0.682005 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | leave_one_room_out | CP_pruned_6 | 900.000000 | 437.000000 | 0.643782 | 0.636062 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | leave_one_room_out | CP_low_overlap_4 | 900.000000 | 437.000000 | 0.626641 | 0.625517 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | leave_one_room_out | Joint_CIR_plus_CP_all | 900.000000 | 437.000000 | 0.709234 | 0.695084 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stage2 | new_theta0_posz_cp16 | ALL | stratified5 | paired_primary_joint_cp16 | nan | nan | nan | nan | 0.637885 | 0.683123 | 0.721056 | 0.083171 | 0.055498 | 0.110753 | 0.045556 | 0.011083 | 0.077778 |
| stage2 | new_theta0_posz_cp16 | ALL | room_label_stratified5 | paired_primary_joint_cp16 | nan | nan | nan | nan | 0.639640 | 0.683202 | 0.714759 | 0.075119 | 0.047336 | 0.103916 | 0.038889 | 0.005556 | 0.071111 |
| stage2 | new_theta0_posz_cp16 | ALL | leave_one_room_out | paired_primary_joint_cp16 | nan | nan | nan | nan | 0.617943 | 0.674173 | 0.709234 | 0.091291 | 0.063014 | 0.120936 | 0.057778 | 0.023333 | 0.087833 |

## CP16 Single Features

| stage | cv_scheme | feature_set | n | n_pos | AUC | PR_AUC |
| --- | --- | --- | --- | --- | --- | --- |
| stage1 | stratified5 | CP16_single:delta_tau_l_given_r_s | 3000 | 2120 | 0.593176 | 0.745544 |
| stage1 | stratified5 | CP16_single:cp_phase_slope_delay_s | 3000 | 2120 | 0.590544 | 0.745232 |
| stage1 | stratified5 | CP16_single:f_r_fp | 3000 | 2120 | 0.560333 | 0.732747 |
| stage1 | stratified5 | CP16_single:delta_f_l_minus_r_fp | 3000 | 2120 | 0.540000 | 0.724622 |
| stage1 | stratified5 | CP16_single:xpr_fp_db | 3000 | 2120 | 0.526256 | 0.708457 |
| stage1 | stratified5 | CP16_single:s3_all | 3000 | 2120 | 0.519486 | 0.704354 |
| stage1 | stratified5 | CP16_single:s3_late | 3000 | 2120 | 0.518448 | 0.702313 |
| stage1 | stratified5 | CP16_single:xpr_all_db | 3000 | 2120 | 0.515033 | 0.699875 |
| stage1 | stratified5 | CP16_single:xpr_late_db | 3000 | 2120 | 0.513106 | 0.697701 |
| stage1 | stratified5 | CP16_single:delta_p_l_given_r_db | 3000 | 2120 | 0.512328 | 0.699813 |
| stage1 | stratified5 | CP16_single:s3_fp | 3000 | 2120 | 0.510482 | 0.704978 |
| stage1 | stratified5 | CP16_single:lambda_l_late_fraction | 3000 | 2120 | 0.510090 | 0.695881 |
| stage1 | stratified5 | CP16_single:f_l_fp | 3000 | 2120 | 0.509898 | 0.710350 |
| stage1 | stratified5 | CP16_single:gamma_delay_linear | 3000 | 2120 | 0.502627 | 0.700109 |
| stage1 | stratified5 | CP16_single:gamma_anchor_linear | 3000 | 2120 | 0.493528 | 0.694188 |
| stage1 | stratified5 | CP16_single:cp_phase_residual_circvar | 3000 | 2120 | 0.492771 | 0.699198 |
| stage2 | fit_all | CP16_single:lambda_l_late_fraction | 900 | 437 | 0.660779 | 0.692240 |
| stage2 | stratified5 | CP16_single:lambda_l_late_fraction | 900 | 437 | 0.660433 | 0.692171 |
| stage2 | room_label_stratified5 | CP16_single:lambda_l_late_fraction | 900 | 437 | 0.659326 | 0.690712 |
| stage2 | leave_one_room_out | CP16_single:lambda_l_late_fraction | 900 | 437 | 0.658149 | 0.687738 |
| stage2 | fit_all | CP16_single:xpr_late_db | 900 | 437 | 0.630694 | 0.672336 |
| stage2 | fit_all | CP16_single:s3_late | 900 | 437 | 0.630694 | 0.672336 |
| stage2 | room_label_stratified5 | CP16_single:xpr_late_db | 900 | 437 | 0.630274 | 0.671860 |
| stage2 | room_label_stratified5 | CP16_single:s3_late | 900 | 437 | 0.629854 | 0.671638 |
| stage2 | stratified5 | CP16_single:xpr_late_db | 900 | 437 | 0.629810 | 0.671814 |
| stage2 | stratified5 | CP16_single:s3_late | 900 | 437 | 0.628979 | 0.671541 |
| stage2 | leave_one_room_out | CP16_single:xpr_late_db | 900 | 437 | 0.625757 | 0.657885 |
| stage2 | leave_one_room_out | CP16_single:s3_late | 900 | 437 | 0.625263 | 0.666069 |
| stage2 | fit_all | CP16_single:gamma_delay_linear | 900 | 437 | 0.609857 | 0.655401 |
| stage2 | stratified5 | CP16_single:gamma_delay_linear | 900 | 437 | 0.604554 | 0.625707 |
| stage2 | fit_all | CP16_single:f_l_fp | 900 | 437 | 0.591318 | 0.606601 |
| stage2 | room_label_stratified5 | CP16_single:f_l_fp | 900 | 437 | 0.590335 | 0.605744 |
| stage2 | stratified5 | CP16_single:f_l_fp | 900 | 437 | 0.589727 | 0.606039 |
| stage2 | room_label_stratified5 | CP16_single:gamma_delay_linear | 900 | 437 | 0.588323 | 0.603686 |
| stage2 | fit_all | CP16_single:xpr_all_db | 900 | 437 | 0.586405 | 0.637393 |
| stage2 | fit_all | CP16_single:s3_all | 900 | 437 | 0.586405 | 0.637393 |
| stage2 | room_label_stratified5 | CP16_single:xpr_all_db | 900 | 437 | 0.585471 | 0.637050 |
| stage2 | stratified5 | CP16_single:xpr_all_db | 900 | 437 | 0.585313 | 0.636063 |
| stage2 | room_label_stratified5 | CP16_single:s3_all | 900 | 437 | 0.585115 | 0.636729 |
| stage2 | stratified5 | CP16_single:s3_all | 900 | 437 | 0.583944 | 0.636094 |
| stage2 | fit_all | CP16_single:delta_p_l_given_r_db | 900 | 437 | 0.583776 | 0.638273 |
| stage2 | stratified5 | CP16_single:delta_p_l_given_r_db | 900 | 437 | 0.582926 | 0.636904 |
| stage2 | room_label_stratified5 | CP16_single:delta_p_l_given_r_db | 900 | 437 | 0.582620 | 0.636282 |
| stage2 | fit_all | CP16_single:delta_tau_l_given_r_s | 900 | 437 | 0.577867 | 0.570494 |
| stage2 | leave_one_room_out | CP16_single:s3_all | 900 | 437 | 0.575142 | 0.609978 |
| stage2 | stratified5 | CP16_single:delta_tau_l_given_r_s | 900 | 437 | 0.574324 | 0.587899 |
| stage2 | room_label_stratified5 | CP16_single:delta_tau_l_given_r_s | 900 | 437 | 0.570753 | 0.587257 |
| stage2 | fit_all | CP16_single:cp_phase_residual_circvar | 900 | 437 | 0.564995 | 0.542567 |
| stage2 | leave_one_room_out | CP16_single:xpr_all_db | 900 | 437 | 0.559163 | 0.545028 |
| stage2 | leave_one_room_out | CP16_single:f_l_fp | 900 | 437 | 0.558140 | 0.504040 |
| stage2 | fit_all | CP16_single:delta_f_l_minus_r_fp | 900 | 437 | 0.552723 | 0.568946 |
| stage2 | room_label_stratified5 | CP16_single:delta_f_l_minus_r_fp | 900 | 437 | 0.550034 | 0.559752 |
| stage2 | stratified5 | CP16_single:delta_f_l_minus_r_fp | 900 | 437 | 0.547741 | 0.554237 |
| stage2 | fit_all | CP16_single:xpr_fp_db | 900 | 437 | 0.547168 | 0.607714 |
| stage2 | fit_all | CP16_single:s3_fp | 900 | 437 | 0.547168 | 0.607714 |
| stage2 | fit_all | CP16_single:gamma_anchor_linear | 900 | 437 | 0.547168 | 0.607714 |
| stage2 | stratified5 | CP16_single:xpr_fp_db | 900 | 437 | 0.546585 | 0.606994 |
| stage2 | room_label_stratified5 | CP16_single:cp_phase_residual_circvar | 900 | 437 | 0.546046 | 0.516377 |
| stage2 | stratified5 | CP16_single:s3_fp | 900 | 437 | 0.545171 | 0.606528 |
| stage2 | room_label_stratified5 | CP16_single:s3_fp | 900 | 437 | 0.544795 | 0.605958 |
| stage2 | room_label_stratified5 | CP16_single:xpr_fp_db | 900 | 437 | 0.544578 | 0.604253 |
| stage2 | leave_one_room_out | CP16_single:delta_p_l_given_r_db | 900 | 437 | 0.542225 | 0.529293 |
| stage2 | stratified5 | CP16_single:cp_phase_residual_circvar | 900 | 437 | 0.541178 | 0.522214 |
| stage2 | leave_one_room_out | CP16_single:gamma_delay_linear | 900 | 437 | 0.527813 | 0.504167 |
| stage2 | leave_one_room_out | CP16_single:cp_phase_residual_circvar | 900 | 437 | 0.526256 | 0.500618 |
| stage2 | fit_all | CP16_single:f_r_fp | 900 | 437 | 0.520686 | 0.489987 |
| stage2 | stratified5 | CP16_single:gamma_anchor_linear | 900 | 437 | 0.517627 | 0.514907 |
| stage2 | room_label_stratified5 | CP16_single:f_r_fp | 900 | 437 | 0.516762 | 0.490210 |
| stage2 | stratified5 | CP16_single:f_r_fp | 900 | 437 | 0.508281 | 0.485468 |
| stage2 | leave_one_room_out | CP16_single:delta_tau_l_given_r_s | 900 | 437 | 0.507258 | 0.523871 |
| stage2 | room_label_stratified5 | CP16_single:gamma_anchor_linear | 900 | 437 | 0.487187 | 0.506963 |
| stage2 | fit_all | CP16_single:cp_phase_slope_delay_s | 900 | 437 | 0.481938 | 0.539139 |
| stage2 | room_label_stratified5 | CP16_single:cp_phase_slope_delay_s | 900 | 437 | 0.479936 | 0.531248 |
| stage2 | leave_one_room_out | CP16_single:f_r_fp | 900 | 437 | 0.473941 | 0.477142 |
| stage2 | stratified5 | CP16_single:cp_phase_slope_delay_s | 900 | 437 | 0.472869 | 0.523451 |
| stage2 | leave_one_room_out | CP16_single:s3_fp | 900 | 437 | 0.462109 | 0.464798 |
| stage2 | leave_one_room_out | CP16_single:delta_f_l_minus_r_fp | 900 | 437 | 0.444178 | 0.450966 |
| stage2 | leave_one_room_out | CP16_single:gamma_anchor_linear | 900 | 437 | 0.421048 | 0.451463 |
| stage2 | leave_one_room_out | CP16_single:cp_phase_slope_delay_s | 900 | 437 | 0.415087 | 0.440205 |
| stage2 | leave_one_room_out | CP16_single:xpr_fp_db | 900 | 437 | 0.408721 | 0.446433 |
