# Simulation Success Criteria

This document defines pre-measurement simulation success checks using standard outputs v1.

## Core Rule

- RT is a latent-variable generator (`A` ray table).  
- Validation is done at `Z/U` level only (distribution/correlation over link summaries), not waveform matching.

## Required Checks

1. C0 floor behavior:
   - estimate `XPD_floor` mean/spread (`std` or `5-95%` range)
   - report `delta_floor`

2. Parity-sign trend:
   - `A2`: median(`XPD_early_db`) < 0
   - `A3`: median(`XPD_early_db`) > 0

3. Breaking/depolarization trend:
   - under stress (`roughness_flag`/`human_flag`), `|XPD_early|` decreases
   - and/or `XPD_late` increases (late cross increase)

4. Spatial consistency in B scenarios:
   - Spearman correlation between `XPD_early` and `-EL_proxy`
   - LOS vs NLOS `XPD_early` distribution summary (CDF/boxplot and optional KS)

## Report Outputs

`scripts/make_success_report.py` must emit:

- `success_report.md`
- `success_report.json`
- plot references under `plots/`
