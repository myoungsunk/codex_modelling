# Stage 1 Conditional AUC Summary (Deterministic)

## Global AUC

| Model | AUC |
| --- | ---: |
| CIR-only | 0.6745 |
| CP-only | 0.6519 |
| Joint | 0.7208 |

## Top Conditional Cells

| Pair | X | Y | n | Delta AUC |
| --- | --- | --- | ---: | ---: |
| los angle from anchor bore x xpol coupling db | [53.49, 77.65] | [24.00, 28.00] | 122 | 0.2145 |
| los angle from anchor bore x eps r | [38.57, 45.77] | [5.20, 6.80] | 112 | 0.1966 |
| antenna type x slab placement | ideal | wall_x | 371 | 0.1940 |
| antenna type x los angle from anchor bore | ideal | [45.77, 53.49] | 302 | 0.1811 |
| antenna type x los angle from anchor bore | ideal | [38.57, 45.77] | 294 | 0.1801 |
| antenna type x slab placement | ideal | wall_y | 367 | 0.1695 |
| los angle from anchor bore x slab placement | [38.57, 45.77] | wall_x | 152 | 0.1655 |
| antenna type x slab placement | ideal | ceiling | 383 | 0.1640 |
| antenna type x los angle from anchor bore | ideal | [53.49, 77.65] | 294 | 0.1635 |
| eps r x snr db | [2.00, 3.60] | [10.00, 16.00] | 116 | 0.1626 |

## Disagreement

- CIR-only CV AUC: 0.6774
- Misclassification rate: 0.3000
