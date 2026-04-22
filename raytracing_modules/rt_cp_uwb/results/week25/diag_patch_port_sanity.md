# Patch Port Sanity

| link | |H(1,1)| mid | |H(2,1)| mid | gamma_mid | gamma_fp | tx coupling ratio | rx coupling ratio | H11 dominant |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ideal_to_ideal | 0.00183513 | 0 | 0 | 0 | 0 | 0 | yes |
| ideal_to_patch | 0.00396532 | 0.00107459 | 0.270997 | 0.270997 | 0 | 0.270997 | yes |
| patch_to_ideal | 0.00396532 | 0.00107459 | 0.270997 | 0.270997 | 0.270997 | 0 | yes |
| patch_to_patch | 0.00919745 | 0 | 0 | 6.11579e-18 | 0.270997 | 0.270997 | yes |

- Interpretation: LoS should remain same-hand dominant, so `|H(1,1)| > |H(2,1)|` for every link.
- The synthetic patch baseline is set by the coupling matrix, so `gamma_mid` should stay below `1` and be on the same order as the coupling ratios.
