# Theta0 CP16 Model Zoo Evaluation

Feature families are evaluated as CIR-only, CP-only variants, and joint variants. Stage 1 uses stratified 5-fold CV; Stage 2 uses room-label stratified 5-fold CV plus leave-one-room-out.

## stage1 / stratified5

| rank | family | feature_set | model | n_features | AUC | PR_AUC | bal_acc@0.5 | F1@0.5 |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | joint | Joint_CIR12_CP16_all | hist_gradient_boosting | 28 | 0.899893 | 0.953939 | 0.770937 | 0.886013 |
| 2 | joint | Joint_CIR12_CP_greedy | hist_gradient_boosting | 18 | 0.899124 | 0.953979 | 0.766209 | 0.882981 |
| 3 | joint | Joint_CIR12_CP_greedy | extra_trees | 18 | 0.898655 | 0.954039 | 0.799700 | 0.873321 |
| 4 | joint | Joint_CIR12_CP16_all | random_forest | 28 | 0.898242 | 0.955132 | 0.771998 | 0.873371 |
| 5 | joint | Joint_CIR12_CP_pruned6 | random_forest | 18 | 0.897840 | 0.954885 | 0.783308 | 0.878675 |
| 6 | joint | Joint_CIR12_CP_greedy | random_forest | 18 | 0.897804 | 0.952491 | 0.782697 | 0.877570 |
| 7 | joint | Joint_CIR12_CP_pruned6 | hist_gradient_boosting | 18 | 0.896782 | 0.953923 | 0.762232 | 0.881579 |
| 8 | joint | Joint_CIR12_CP_low_overlap4 | hist_gradient_boosting | 16 | 0.895176 | 0.952853 | 0.760967 | 0.878560 |
| 9 | joint | Joint_CIR12_CP_late_energy5 | extra_trees | 17 | 0.894894 | 0.953511 | 0.789365 | 0.862261 |
| 10 | joint | Joint_CIR12_CP_low_overlap4 | random_forest | 16 | 0.893933 | 0.952064 | 0.775740 | 0.874534 |
| 11 | joint | Joint_CIR12_CP_late_energy5 | random_forest | 17 | 0.892602 | 0.952532 | 0.777659 | 0.870200 |
| 12 | joint | Joint_CIR12_CP16_all | extra_trees | 28 | 0.891619 | 0.951783 | 0.784348 | 0.869896 |
| 13 | joint | Joint_CIR12_CP_pruned6 | extra_trees | 18 | 0.891413 | 0.950180 | 0.787489 | 0.867735 |
| 14 | joint | Joint_CIR12_CP_low_overlap4 | extra_trees | 16 | 0.890102 | 0.949528 | 0.788057 | 0.867943 |
| 15 | joint | Joint_CIR12_CP_late_energy5 | hist_gradient_boosting | 17 | 0.889661 | 0.948988 | 0.761632 | 0.878449 |

Best by family:

| family | feature_set | model | AUC | PR_AUC |
|---|---|---|---:|---:|
| joint | Joint_CIR12_CP16_all | hist_gradient_boosting | 0.899893 | 0.953939 |
| cp_only | CP16_all_only | random_forest | 0.812481 | 0.907036 |
| cir_only | CIR12_only | gradient_boosting | 0.770315 | 0.889268 |

## stage2 / leave_one_room_out

| rank | family | feature_set | model | n_features | AUC | PR_AUC | bal_acc@0.5 | F1@0.5 |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | joint | Joint_CIR12_CP16_all | extra_trees | 28 | 0.773544 | 0.765628 | 0.687344 | 0.664269 |
| 2 | joint | Joint_CIR12_CP16_all | random_forest | 28 | 0.773273 | 0.742893 | 0.700624 | 0.677885 |
| 3 | joint | Joint_CIR12_CP_pruned6 | hist_gradient_boosting | 18 | 0.763511 | 0.730069 | 0.703298 | 0.685579 |
| 4 | joint | Joint_CIR12_CP16_all | hist_gradient_boosting | 28 | 0.763180 | 0.726703 | 0.696690 | 0.679245 |
| 5 | joint | Joint_CIR12_CP_pruned6 | random_forest | 18 | 0.762350 | 0.710431 | 0.700945 | 0.681710 |
| 6 | joint | Joint_CIR12_CP_low_overlap4 | gradient_boosting | 16 | 0.761925 | 0.741829 | 0.692551 | 0.665857 |
| 7 | joint | Joint_CIR12_CP_low_overlap4 | hist_gradient_boosting | 16 | 0.761159 | 0.730035 | 0.692049 | 0.672209 |
| 8 | joint | Joint_CIR12_CP_late_energy5 | random_forest | 17 | 0.760748 | 0.744217 | 0.686393 | 0.665077 |
| 9 | joint | Joint_CIR12_CP_pruned6 | gradient_boosting | 18 | 0.759891 | 0.743808 | 0.691150 | 0.660934 |
| 10 | joint | Joint_CIR12_CP_low_overlap4 | random_forest | 16 | 0.759572 | 0.704379 | 0.704506 | 0.687868 |
| 11 | joint | Joint_CIR12_CP_late_energy5 | gradient_boosting | 17 | 0.756182 | 0.749200 | 0.681045 | 0.648582 |
| 12 | joint | Joint_CIR12_CP_late_energy5 | hist_gradient_boosting | 17 | 0.756024 | 0.735671 | 0.688308 | 0.675926 |
| 13 | joint | Joint_CIR12_CP_greedy | gradient_boosting | 15 | 0.755233 | 0.740852 | 0.691407 | 0.664234 |
| 14 | joint | Joint_CIR12_CP16_all | gradient_boosting | 28 | 0.754852 | 0.731978 | 0.688861 | 0.657635 |
| 15 | joint | Joint_CIR12_CP_greedy | hist_gradient_boosting | 15 | 0.750557 | 0.722777 | 0.674321 | 0.654028 |

Best by family:

| family | feature_set | model | AUC | PR_AUC |
|---|---|---|---:|---:|
| joint | Joint_CIR12_CP16_all | extra_trees | 0.773544 | 0.765628 |
| cp_only | CP16_all_only | extra_trees | 0.743752 | 0.745767 |
| cir_only | CIR12_only | gradient_boosting | 0.664199 | 0.621198 |

## stage2 / room_label_stratified5

| rank | family | feature_set | model | n_features | AUC | PR_AUC | bal_acc@0.5 | F1@0.5 |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | joint | Joint_CIR12_CP16_all | extra_trees | 28 | 0.767851 | 0.761870 | 0.675465 | 0.655621 |
| 2 | joint | Joint_CIR12_CP16_all | gradient_boosting | 28 | 0.759953 | 0.752856 | 0.684092 | 0.648379 |
| 3 | joint | Joint_CIR12_CP_pruned6 | gradient_boosting | 18 | 0.758935 | 0.753022 | 0.678821 | 0.646116 |
| 4 | joint | Joint_CIR12_CP_pruned6 | random_forest | 18 | 0.757630 | 0.754927 | 0.682394 | 0.665885 |
| 5 | joint | Joint_CIR12_CP16_all | random_forest | 28 | 0.757536 | 0.754943 | 0.686521 | 0.666667 |
| 6 | joint | Joint_CIR12_CP_low_overlap4 | gradient_boosting | 16 | 0.757348 | 0.756394 | 0.678757 | 0.645241 |
| 7 | joint | Joint_CIR12_CP_pruned6 | extra_trees | 18 | 0.757111 | 0.750109 | 0.677123 | 0.663573 |
| 8 | joint | Joint_CIR12_CP_late_energy5 | gradient_boosting | 17 | 0.756691 | 0.752218 | 0.680916 | 0.646840 |
| 9 | joint | Joint_CIR12_CP16_all | hist_gradient_boosting | 28 | 0.754205 | 0.755854 | 0.679347 | 0.665893 |
| 10 | joint | Joint_CIR12_CP_greedy | gradient_boosting | 15 | 0.750466 | 0.747284 | 0.674373 | 0.641184 |
| 11 | joint | Joint_CIR12_CP_low_overlap4 | random_forest | 16 | 0.749168 | 0.748145 | 0.671017 | 0.650888 |
| 12 | joint | Joint_CIR12_CP_pruned6 | hist_gradient_boosting | 18 | 0.747701 | 0.750965 | 0.666569 | 0.646154 |
| 13 | joint | Joint_CIR12_CP_phase_delay4 | gradient_boosting | 16 | 0.747515 | 0.748782 | 0.659254 | 0.630303 |
| 14 | joint | Joint_CIR12_CP_phase_delay4 | svm_rbf | 16 | 0.747327 | 0.717208 | 0.677701 | 0.670455 |
| 15 | joint | Joint_CIR12_CP_low_overlap4 | extra_trees | 16 | 0.746075 | 0.739994 | 0.655898 | 0.640279 |

Best by family:

| family | feature_set | model | AUC | PR_AUC |
|---|---|---|---:|---:|
| joint | Joint_CIR12_CP16_all | extra_trees | 0.767851 | 0.761870 |
| cp_only | CP16_all_only | extra_trees | 0.743094 | 0.744376 |
| cir_only | CIR12_only | gradient_boosting | 0.681893 | 0.636935 |

