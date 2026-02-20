# Measurement Bridge (G2)

This guide explains what RT matrix to compare against measurements.

## Which Matrix To Compare

- If your measurement is antenna-included transfer data (for example VNA `S21` with real Tx/Rx antennas in place), compare against `A_f` (embedded channel).
- If your measurement is antenna de-embedded (or you want propagation-only physics validation), compare against `J_f` (propagation-only matrix).

## Basis / Convention Labeling Rule

Always report these together and keep them identical between RT and measurement post-processing:

- `basis`: `linear` or `circular`
- `convention`: for CP, use explicit label such as `IEEE-RHCP`

Do not interpret CP same-hand/opposite-hand metrics from linear-basis data unless you explicitly convert basis first and label that conversion.

## Mismatch Diagnosis Order

When RT and measurement disagree, diagnose in this order:

1. Geometry and delay: path existence, LOS blocking, `tau = length / c0`, occlusion.
2. Scalar path loss: FSPL / distance scaling (`scalar_gain_f`).
3. Material reflection: Fresnel `Gamma_s`, `Gamma_p` (magnitude/phase).
4. Polarization transform: local basis handling (`s/p`, linear/circular convention, parity interpretation).
5. Antenna embedding: coupling, leakage floor, axial-ratio/cross-pol effects (`G_tx_f`, `G_rx_f`).

This ordering avoids conflating geometry or loss errors with polarization-model errors.
