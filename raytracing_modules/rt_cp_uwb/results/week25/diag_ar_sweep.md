# AR Sweep Diagnostic

- XPD fixed at `35 dB`.
- Geometry: pure LoS, boresight aligned.
- If matched `patch -> patch` stays near zero while mixed links increase with AR, coupling is active but cancels in the matched link.

| AR (dB) | gamma mid patch-patch | gamma mid ideal-patch | gamma mid patch-ideal | gamma FP patch-patch | gamma FP ideal-patch | coupling ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0.0177828 | 0.0177828 | 1.13099e-17 | 0.0177828 | 0.0177828 |
| 1 | 0 | 0.0752839 | 0.0752839 | 2.623e-18 | 0.0752839 | 0.0752839 |
| 3 | 9.43046e-17 | 0.18878 | 0.18878 | 6.31179e-18 | 0.18878 | 0.18878 |
| 6 | 4.71523e-17 | 0.350062 | 0.350062 | 3.41639e-18 | 0.350062 | 0.350062 |
| 10 | 0 | 0.49 | 0.49 | 1.28879e-18 | 0.49 | 0.49 |
