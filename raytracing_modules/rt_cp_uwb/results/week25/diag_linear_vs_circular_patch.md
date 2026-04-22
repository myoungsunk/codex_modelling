# Linear vs Circular Patch Diagnostic

- Geometry: pure LoS, boresight aligned.
- Parameters: `AR=3 dB`, `XPD=20 dB`.
- Mixed links reveal whether coupling is active; matched links reveal whether it cancels.

| link | basis | |H(1,1)| mid | |H(2,1)| mid | gamma mid |
| --- | --- | ---: | ---: | ---: |
| linear_ideal_to_linear_patch | linear | 0.00177125 | 0.000480003 | 0.270997 |
| linear_patch_to_linear_ideal | linear | 0.00177125 | 0.000480003 | 0.270997 |
| linear_patch_to_linear_patch | linear | 0.00183513 | 5.42101e-20 | 2.95402e-17 |
| circular_ideal_to_circular_patch | circular | 0.00177125 | 0.000480003 | 0.270997 |
| circular_patch_to_circular_ideal | circular | 0.00177125 | 0.000480003 | 0.270997 |
| circular_patch_to_circular_patch | circular | 0.00183513 | 0 | 0 |
