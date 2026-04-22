# XPD Sweep Diagnostic

- AR fixed at `0 dB` to isolate XPD.
- Cases per XPD: `50` (`25` LoS + `25` isolated 1-bounce NLoS)
- Runtime: `2.946 s`
- LoS samples: empty scene over incidence/height sweep.
- NLoS samples: isolated single PEC bounce over the same geometry sweep.

| XPD (dB) | mean gamma LoS | mean gamma NLoS | median gamma LoS | median gamma NLoS | AUC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 60 | 8.36753e-18 | 167.547 | 8.19115e-18 | 167.547 | 1 |
| 40 | 9.9133e-18 | 48.1291 | 8.00041e-18 | 48.1291 | 1 |
| 30 | 7.68169e-18 | 15.7336 | 6.91384e-18 | 15.7336 | 1 |
| 25 | 8.41971e-18 | 8.85229 | 8.11435e-18 | 8.85229 | 1 |
| 20 | 5.96477e-18 | 4.94809 | 5.08713e-18 | 4.94809 | 1 |
| 15 | 8.09921e-18 | 2.72248 | 6.65439e-18 | 2.72248 | 1 |
| 10 | 6.0131e-18 | 1.42299 | 5.9227e-18 | 1.42299 | 1 |

- Interpretation: monotonic degradation from high XPD to low XPD supports a real antenna-impairment mechanism.
