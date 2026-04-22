# Stage 1 Conditional AUC Summary

## Global AUC

| Model | AUC |
| --- | ---: |
| CIR-only | 0.6795 |
| CP-only | 0.6485 |
| Joint | 0.7205 |

## Top Conditional Cells

| Pair | X | Y | n | Delta AUC |
| --- | --- | --- | ---: | ---: |
| eps r x xpol coupling db | [2.00, 3.60] | [20.01, 24.00] | 120 | 0.2666 |
| antenna type x los angle from anchor bore | ideal | [38.57, 45.77] | 294 | 0.2006 |
| antenna type x slab placement | ideal | wall_x | 371 | 0.1925 |
| los angle from anchor bore x xpol coupling db | [53.49, 77.65] | [24.00, 28.00] | 122 | 0.1865 |
| antenna type x slab placement | ideal | ceiling | 383 | 0.1858 |
| antenna type x los angle from anchor bore | ideal | [53.49, 77.65] | 294 | 0.1641 |
| antenna type x los angle from anchor bore | ideal | [45.77, 53.49] | 302 | 0.1618 |
| los angle from anchor bore x xpol coupling db | [38.57, 45.77] | [28.00, 32.00] | 126 | 0.1582 |
| los angle from anchor bore x slab placement | [38.57, 45.77] | wall_x | 152 | 0.1549 |
| antenna type x slab placement | ideal | wall_y | 367 | 0.1521 |

## Disagreement

- CIR-only CV AUC: 0.6803
- Misclassification rate: 0.3000
