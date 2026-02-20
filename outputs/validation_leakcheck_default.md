# RT Validation Report

- basis: linear
- xpd_matrix_source: A
- antenna_config: {'convention': 'IEEE-RHCP', 'tx_cross_pol_leakage_db': 35.0, 'rx_cross_pol_leakage_db': 35.0, 'tx_axial_ratio_db': 0.0, 'rx_axial_ratio_db': 0.0, 'enable_coupling': True}
- predicted_leakage_floor_db: 28.979 (eps_tx=0.01778, eps_rx=0.01778)

## C0

- case 0: paths=1, bounce_dist={0: 1}
- strongest path: tau=1.001e-08s, power=5.288e-07
- LOS exists: True
- case 1: paths=1, bounce_dist={0: 1}
- strongest path: tau=2.001e-08s, power=1.322e-07
- LOS exists: True
- case 2: paths=1, bounce_dist={0: 1}
- strongest path: tau=3.002e-08s, power=5.875e-08
- LOS exists: True
- parity XPD stats (exact_bounce=None): {'even': {'mu': 28.982138791832057, 'sigma': 6.5646083438579166e-06, 'n': 3}}
- leakage-limited check: median_xpd_db=28.982, sigma_db=0.000, delta_to_floor_db=0.003, floor_db=28.979
- WARNING: XPD appears leakage-limited; use --physics-validation-mode and/or --xpd-matrix-source J for propagation-only analysis.

## A1

- case 0: paths=1, bounce_dist={0: 1}
- strongest path: tau=1.336e-08s, power=2.967e-07
- LOS exists: True
- case 1: paths=1, bounce_dist={0: 1}
- strongest path: tau=2.002e-08s, power=1.320e-07
- LOS exists: True
- case 2: paths=1, bounce_dist={0: 1}
- strongest path: tau=2.669e-08s, power=7.431e-08
- LOS exists: True
- parity XPD stats (exact_bounce=None): {'even': {'mu': 28.982139386210893, 'sigma': 4.351526318131701e-06, 'n': 3}}
- leakage-limited check: median_xpd_db=28.982, sigma_db=0.000, delta_to_floor_db=0.003, floor_db=28.979
- WARNING: XPD appears leakage-limited; use --physics-validation-mode and/or --xpd-matrix-source J for propagation-only analysis.

## A2

- case 0: paths=1, bounce_dist={1: 1}
- strongest path: tau=1.668e-08s, power=1.904e-07
- LOS exists: False
- case 1: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.238e-08s, power=1.058e-07
- LOS exists: False
- case 2: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.850e-08s, power=6.519e-08
- LOS exists: False
- case 3: paths=1, bounce_dist={1: 1}
- strongest path: tau=1.887e-08s, power=1.487e-07
- LOS exists: False
- case 4: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.405e-08s, power=9.152e-08
- LOS exists: False
- case 5: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.983e-08s, power=5.949e-08
- LOS exists: False
- case 6: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.136e-08s, power=1.161e-07
- LOS exists: False
- case 7: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.605e-08s, power=7.802e-08
- LOS exists: False
- case 8: paths=1, bounce_dist={1: 1}
- strongest path: tau=3.147e-08s, power=5.347e-08
- LOS exists: False
- parity XPD stats (exact_bounce=1): {'odd': {'mu': 28.98213638543161, 'sigma': 3.971581855488219e-06, 'n': 9}}
- leakage-limited check: median_xpd_db=28.982, sigma_db=0.000, delta_to_floor_db=0.003, floor_db=28.979
- WARNING: XPD appears leakage-limited; use --physics-validation-mode and/or --xpd-matrix-source J for propagation-only analysis.

## A2R

- case 0: paths=1, bounce_dist={1: 1}
- strongest path: tau=1.296e-07s, power=3.147e-09
- LOS exists: False
- case 1: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.625e-08s, power=7.668e-08
- LOS exists: False
- case 2: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.542e-08s, power=8.196e-08
- LOS exists: False
- case 3: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.790e-08s, power=6.801e-08
- LOS exists: False
- case 4: paths=0, bounce_dist={}
- WARNING: no paths matched exact_bounce=1 for stats
- case 5: paths=1, bounce_dist={1: 1}
- strongest path: tau=3.477e-08s, power=4.374e-08
- LOS exists: False
- case 6: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.406e-08s, power=9.149e-08
- LOS exists: False
- case 7: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.620e-08s, power=7.710e-08
- LOS exists: False
- case 8: paths=0, bounce_dist={}
- WARNING: no paths matched exact_bounce=1 for stats
- case 9: paths=1, bounce_dist={1: 1}
- strongest path: tau=5.753e-08s, power=1.599e-08
- LOS exists: False
- case 10: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.277e-08s, power=1.021e-07
- LOS exists: False
- case 11: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.445e-08s, power=8.857e-08
- LOS exists: False
- case 12: paths=0, bounce_dist={}
- WARNING: no paths matched exact_bounce=1 for stats
- case 13: paths=0, bounce_dist={}
- WARNING: no paths matched exact_bounce=1 for stats
- case 14: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.173e-08s, power=1.122e-07
- LOS exists: False
- case 15: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.286e-08s, power=1.014e-07
- LOS exists: False
- parity XPD stats (exact_bounce=1): {'odd': {'mu': 11.708909606460624, 'sigma': 9.04971168375865, 'n': 12}}
- leakage-limited check: median_xpd_db=11.517, sigma_db=9.050, delta_to_floor_db=17.462, floor_db=28.979

## A3

- case 0: paths=1, bounce_dist={2: 1}
- strongest path: tau=2.405e-08s, power=9.152e-08
- LOS exists: False
- case 1: paths=1, bounce_dist={2: 1}
- strongest path: tau=2.405e-08s, power=9.152e-08
- LOS exists: False
- case 2: paths=1, bounce_dist={2: 1}
- strongest path: tau=2.582e-08s, power=7.939e-08
- LOS exists: False
- case 3: paths=1, bounce_dist={2: 1}
- strongest path: tau=2.780e-08s, power=6.853e-08
- LOS exists: False
- parity XPD stats (exact_bounce=2): {'even': {'mu': 28.982135841684514, 'sigma': 1.4964949185936964e-06, 'n': 4}}
- leakage-limited check: median_xpd_db=28.982, sigma_db=0.000, delta_to_floor_db=0.003, floor_db=28.979
- WARNING: XPD appears leakage-limited; use --physics-validation-mode and/or --xpd-matrix-source J for propagation-only analysis.

## A3R

- case 0: paths=1, bounce_dist={2: 1}
- strongest path: tau=3.493e-08s, power=4.335e-08
- LOS exists: False
- case 1: paths=1, bounce_dist={2: 1}
- strongest path: tau=4.691e-08s, power=2.400e-08
- LOS exists: False
- case 2: paths=1, bounce_dist={2: 1}
- strongest path: tau=6.773e-08s, power=1.153e-08
- LOS exists: False
- case 3: paths=1, bounce_dist={2: 1}
- strongest path: tau=8.357e-08s, power=7.570e-09
- LOS exists: False
- parity XPD stats (exact_bounce=2): {'even': {'mu': -2.7129067422803774, 'sigma': 4.9425716969222835, 'n': 4}}
- leakage-limited check: median_xpd_db=-0.971, sigma_db=4.943, delta_to_floor_db=29.950, floor_db=28.979

## A4

- case 0: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.238e-08s, power=2.524e-08
- LOS exists: False
- case 1: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.405e-08s, power=1.969e-08
- LOS exists: False
- case 2: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.605e-08s, power=1.592e-08
- LOS exists: False
- case 3: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.238e-08s, power=1.123e-08
- LOS exists: False
- case 4: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.405e-08s, power=6.531e-09
- LOS exists: False
- case 5: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.605e-08s, power=4.338e-09
- LOS exists: False
- case 6: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.238e-08s, power=1.446e-08
- LOS exists: False
- case 7: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.405e-08s, power=9.191e-09
- LOS exists: False
- case 8: paths=1, bounce_dist={1: 1}
- strongest path: tau=2.605e-08s, power=6.510e-09
- LOS exists: False
- parity XPD stats (exact_bounce=1): {'odd': {'mu': 33.50343102561886, 'sigma': 2.338648570633866, 'n': 9}}
- leakage-limited check: median_xpd_db=33.421, sigma_db=2.339, delta_to_floor_db=4.442, floor_db=28.979
- material sub-summary: {'glass': {'mu': 35.70955251445628, 'sigma': 2.2809869812149737, 'n': 3}, 'wood': {'mu': 32.024986457454766, 'sigma': 1.521591488654668, 'n': 3}, 'gypsum': {'mu': 32.77575410494553, 'sigma': 1.7278715758428427, 'n': 3}}

## A5

- case 0: paths=1, bounce_dist={2: 1}
- strongest path: tau=2.582e-08s, power=7.938e-08
- LOS exists: False
- case 1: paths=1, bounce_dist={2: 1}
- strongest path: tau=2.582e-08s, power=7.936e-08
- LOS exists: False
- case 2: paths=1, bounce_dist={2: 1}
- strongest path: tau=2.582e-08s, power=7.934e-08
- LOS exists: False
- case 3: paths=1, bounce_dist={2: 1}
- strongest path: tau=2.582e-08s, power=7.932e-08
- LOS exists: False
- case 4: paths=1, bounce_dist={2: 1}
- strongest path: tau=2.582e-08s, power=7.931e-08
- LOS exists: False
- parity XPD stats (exact_bounce=None): {'even': {'mu': 8.208189020679189, 'sigma': 1.7867074489710089, 'n': 5}}
- leakage-limited check: median_xpd_db=7.928, sigma_db=1.787, delta_to_floor_db=21.052, floor_db=28.979
