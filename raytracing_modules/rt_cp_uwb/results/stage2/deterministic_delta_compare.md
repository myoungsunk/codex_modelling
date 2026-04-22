# Deterministic Rerun vs Original Comparison

- mixed@0.33 global delta shift: 0.002254
- mixed@0.33 stability gate (|shift| < 0.01): PASS
- regime overlap count: 5 / 6
- HFSS candidate overlap count: 25 / 30

## Delta AUC Comparison

| label | room | delta orig | delta det | shift | cir orig | cir det | joint orig | joint det |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| current_0p20 | ALL | 0.019704 | 0.017037 | -0.002667 | 0.789621 | 0.782389 | 0.809325 | 0.799426 |
| geo_only | ALL | 0.058447 | 0.066707 | 0.008260 | 0.660533 | 0.652327 | 0.718980 | 0.719034 |
| bounce_0p33_visible | ALL | 0.019789 | 0.021149 | 0.001360 | 0.679096 | 0.675658 | 0.698885 | 0.696807 |
| mixed_0p33 | ALL | 0.022532 | 0.024786 | 0.002254 | 0.685337 | 0.681893 | 0.707870 | 0.706679 |
| mixed_0p33 | A | 0.038923 | 0.036588 | -0.002334 | 0.681706 | 0.679506 | 0.720629 | 0.716094 |
| mixed_0p33 | B | 0.017087 | 0.017400 | 0.000313 | 0.716765 | 0.698247 | 0.733852 | 0.715647 |
| mixed_0p33 | C | 0.041472 | 0.046539 | 0.005066 | 0.700278 | 0.688173 | 0.741750 | 0.734711 |

## Regime Overlap

- GEO | room type=C, los angle from anchor bore=[40.56, 46.12]
- GEO | room type=C, los angle from anchor bore=[33.29, 40.56]
- GEO | room type=C, xpol coupling db=[20.04, 24.00]
- BOUNCE | room type=C, los angle from anchor bore=[11.63, 33.29]
- BOUNCE | room type=A, dominant wall material=wood

## Candidate Overlap

- case_ids: [879 226 774 61 303 854 630 701 444 799 852 199 52 505 899 315 565 324 249 633 545 135 605 190 480]
