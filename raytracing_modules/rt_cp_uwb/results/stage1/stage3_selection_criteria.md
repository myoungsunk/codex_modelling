# Stage 3 HFSS SBR+ Selection Criteria

## Principle

- Total Stage 3 target: 45 cases
- Group 1: 15 cases from Stage 1 top patch-only Delta AUC regimes
- Group 2: 15 cases from Stage 1 patch-weak regimes
- Group 3: 15 cases reserved for Stage 2 top room-regime Delta AUC cells

## Group 1: Stage 1 Top Delta AUC Regimes

| Regime | n | AUC CIR | AUC Joint | Delta AUC |
| --- | ---: | ---: | ---: | ---: |
| los angle from anchor bore=[38.69, 45.86], slab placement=floor | 67 | 0.7596 | 0.9596 | 0.2000 |
| los angle from anchor bore=[38.69, 45.86], xpol coupling db=[31.80, 35.90] | 73 | 0.8240 | 1.0000 | 0.1760 |
| los angle from anchor bore=[1.27, 29.65], xpol coupling db=[23.97, 27.71] | 67 | 0.8178 | 0.9516 | 0.1337 |

Selection rule: choose 5 representative samples per regime by nearest-to-medoid distance within the cell.

## Group 2: Patch-Weak Regimes

| Regime | n | AUC CIR | AUC Joint | Delta AUC |
| --- | ---: | ---: | ---: | ---: |
| slab placement=ceiling, snr db=[34.34, 39.99] | 72 | 0.9336 | 0.8047 | -0.1289 |
| los angle from anchor bore=[38.69, 45.86], slab placement=ceiling | 82 | 1.0000 | 1.0000 | 0.0000 |
| slab placement=wall_y, snr db=[28.54, 34.34] | 79 | 0.8727 | 0.8870 | 0.0143 |

Selection rule: choose 5 representative samples per weak regime to test whether weakness is scene-driven or antenna-driven.

## Group 3: Stage 2 Deferred Group

- Reserve 15 Stage 3 slots for top Stage 2 room-regime Delta AUC cells once Week 4 results exist
- Prioritize room/position regimes where patch_ffd Joint - CIR lift is strongest under richer multipath
- Use 5 representative cases per selected Stage 2 regime (3 regimes x 5 cases)

## Outputs

- Cell summary CSV: `D:\codex\raytracing_modules\rt_cp_uwb\results\stage1\stage3_selection_cells.csv`
- Candidate case CSV: `D:\codex\raytracing_modules\rt_cp_uwb\results\stage1\stage3_selection_candidates.csv`
