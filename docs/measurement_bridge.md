# Measurement Bridge (G2)

This guide explains what RT matrix to compare against measurements.

## Which Matrix To Compare

- If your measurement is antenna-included transfer data (for example VNA `S21` with real Tx/Rx antennas in place), compare against `A_f` (embedded channel).
- If your measurement is antenna de-embedded (or you want propagation-only physics validation), compare against `J_f` (propagation-only matrix).

Channel-definition mapping in CLI:

- `--measurement-channel-definition embedded` -> compare against `A_f`
- `--measurement-channel-definition propagation` -> compare against `J_f`

Recommended metrics:

- `PDP/CIR` overlays and `|H_ij(f)|` overlays can be compared in either definition, but the definition must match the measurement calibration state.
- `XPD/XPR` should use the same `basis` and `convention`; CP same/opp interpretation is only valid in circular basis.

## Basis / Convention Labeling Rule

Always report these together and keep them identical between RT and measurement post-processing:

- `basis`: `linear` or `circular`
- `convention`: for CP, use explicit label such as `IEEE-RHCP`

Do not interpret CP same-hand/opposite-hand metrics from linear-basis data unless you explicitly convert basis first and label that conversion.

## Measurement Import Formats

`scenarios.runner` provides optional measurement compare mode (off by default):

- Single matrix CSV:
  `--measurement-format matrix_csv --measurement-matrix-csv <file>`
- Four trace CSVs:
  `--measurement-format four_csv --measurement-hh-csv <file> --measurement-hv-csv <file> --measurement-vh-csv <file> --measurement-vv-csv <file>`

Supported complex column styles:

- `*_re`, `*_im` (or `*_real`, `*_imag`)
- `*_mag_db`, `*_phase_deg`

Frequency column aliases:

- `f_hz`, `freq_hz`, `frequency_hz`, `f`

## Mismatch Diagnosis Order

When RT and measurement disagree, diagnose in this order:

1. Geometry and delay: path existence, LOS blocking, `tau = length / c0`, occlusion.
2. Scalar path loss: FSPL / distance scaling (`scalar_gain_f`).
3. Material reflection: Fresnel `Gamma_s`, `Gamma_p` (magnitude/phase).
4. Polarization transform: local basis handling (`s/p`, linear/circular convention, parity interpretation).
5. Antenna embedding: coupling, leakage floor, axial-ratio/cross-pol effects (`G_tx_f`, `G_rx_f`).

This ordering avoids conflating geometry or loss errors with polarization-model errors.
