# Asymmetric Boresight Diagnostic

- TX anchor boresight: `[1; 0; 0]`
- RX tag boresight: `[0; 0; 1]`
- Geometry note: RX is placed near the floor (`z = 0.2 m`) to avoid the exact `90 deg` side-null of the synthetic `cos^n` patch pattern.
- Expectation: `patch_los` should show non-zero handedness leakage, while `ideal_los` remains near zero.

| case | path | |H(1,1)| mid | |H(2,1)| mid | gamma mid | gamma FP | TX AR/XPD (dB) | RX AR/XPD (dB) |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| ideal_los | LoS | 2.1684e-19 | 0.00170388 | 7.85775e+15 | 1.3368e+16 | 1.38 / 52.83 | 8.62 / 15.17 |
| patch_los | LoS | 0.000361463 | 0.00292243 | 8.085 | 8.085 | 3.97 / 18.34 | 9.03 / 9.66 |
| ideal_pec_1bounce | PEC 1-bounce | 0.00157361 | 8.84908e-06 | 0.00562341 | 0.00562341 | 2.65 / 46.24 | 10.00 / 8.00 |
| patch_pec_1bounce | PEC 1-bounce | 0 | 0 | NaN | NaN | 4.85 / 16.82 | 10.00 / 8.00 |

## Day 1 Checkpoint

- patch-patch gamma_LoS (FP) = 8.085
- verdict: **GO**
