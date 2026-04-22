# Week 4 Day 2 Room-wise AUC

- verdict: **A < B < C confirmed for CP-only AUC**

| room | n | n_los | n_nlos | auc_cir | auc_cp | auc_joint | delta_auc |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 75 | 5 | 70 | 1.0000 | 0.7200 | 1.0000 | 0.0000 |
| B | 75 | 6 | 69 | 1.0000 | 0.7319 | 1.0000 | 0.0000 |
| C | 75 | 5 | 70 | 1.0000 | 0.9600 | 1.0000 | 0.0000 |

## Top Features Per Room

### Room A

| feature | auc |
|---|---:|
| energy_concentration_50ns | 0.7257 |
| gamma_cp_6_phase_circvar | 0.6714 |
| rise_time_fp | 0.5986 |
| gamma_cp_1_freq_avg | 0.5800 |
| max_excess_delay | 0.5714 |

### Room B

| feature | auc |
|---|---:|
| rise_time_fp | 0.7222 |
| gamma_cp_6_phase_circvar | 0.6836 |
| rms_delay_spread | 0.6498 |
| gamma_cp_1_freq_avg | 0.6473 |
| mean_excess_delay | 0.6449 |

### Room C

| feature | auc |
|---|---:|
| rise_time_fp | 0.7614 |
| max_excess_delay | 0.7500 |
| gamma_cp_6_phase_circvar | 0.7171 |
| gamma_cp_1_freq_avg | 0.6486 |
| a_fp_6_fp_to_2nd_peak | 0.6029 |

