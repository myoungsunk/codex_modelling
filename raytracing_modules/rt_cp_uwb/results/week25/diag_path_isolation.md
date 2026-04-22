# Path Isolation Diagnostic

| case | bounce_count | |H(1,1)| mid | |H(2,1)| mid | gamma_mid | gamma_fp | path_length_m | delay_ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ideal_los_only | 0 | 0.00183513 | 0 | 0 | 0 | 2 | 6.67128 |
| patch_los_only | 0 | 0.00919745 | 0 | 0 | 6.11579e-18 | 2 | 6.67128 |
| ideal_1bounce_only | 1 | 7.29714e-06 | 0.00129763 | 177.828 | 177.828 | 2.82843 | 9.43462 |
| patch_1bounce_only | 1 | 3.65723e-05 | 0.00650358 | 177.828 | 177.828 | 2.82843 | 9.43462 |

## Blocked Scene Check

- blocked scene bounce counts: `1`
- direct path present: **no**
