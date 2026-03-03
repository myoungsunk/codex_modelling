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
- Dual-CP sequential CSVs:
  `--measurement-format dualcp_two_csv --measurement-co-csv <file> --measurement-cross-csv <file>`
  - `co_csv`: RHCP->RHCP
  - `cross_csv`: RHCP->LHCP
  - imported as `H_f[:,0,0]=H_co`, `H_f[:,1,0]=H_cross`, other entries zero-filled.

Supported complex column styles:

- `*_re`, `*_im` (or `*_real`, `*_imag`)
- `*_mag_db`, `*_phase_deg`

Frequency column aliases:

- `f_hz`, `freq_hz`, `frequency_hz`, `f`

## Dual-CP Sequential Measurement Scope

- This dual-CP bridge is intentionally power-metric-focused.
- Do not over-claim per-ray complex 2x2 matrix recovery from sequential co/cross captures.
- Unless explicit phase-lock evidence exists, treat co/cross inter-trace phase as inconsistent; use power-domain XPD/XPR style metrics.

## Mismatch Diagnosis Order

When RT and measurement disagree, diagnose in this order:

1. Geometry and delay: path existence, LOS blocking, `tau = length / c0`, occlusion.
2. Scalar path loss: FSPL / distance scaling (`scalar_gain_f`).
3. Material reflection: Fresnel `Gamma_s`, `Gamma_p` (magnitude/phase).
4. Polarization transform: local basis handling (`s/p`, linear/circular convention, parity interpretation).
5. Antenna embedding: coupling, leakage floor, axial-ratio/cross-pol effects (`G_tx_f`, `G_rx_f`).

This ordering avoids conflating geometry or loss errors with polarization-model errors.

## Band-Limited CIR Reminder

- `H(f) -> h(t)` in this repo is computed on the available sampled RF band.
- For UWB sweeps such as 6-10 GHz:
  - practical delay-bin spacing is `dt = 1/(nfft*df)`,
  - multipath separability is fundamentally limited by `~1/BW` (`BW=f_max-f_min`).
- This is a known measurement/synthesis limitation (not a bug). Keep it explicit in reports.

## Diffuse Scattering Scope (Important)

- Current `diffuse` branch in `rt_core.tracer` is an empirical proxy:
  - depolarization mixer + controlled power scaling + delay jitter around parent specular path.
  - it does **not** synthesize new BRDF-consistent outgoing directions / clusters.
- Therefore:
  - suitable for sensitivity and stress-response studies,
  - not suitable as a physically faithful angle-resolved diffuse scattering model.
- In `--release-mode`, diffuse is blocked by default unless explicitly acknowledged:
  - add `--allow-proxy-diffuse-release` to proceed intentionally.

## Reflection Jones Scope (Diagonal vs Effective Cross Coupling)

- Baseline reflection Jones matrix is diagonal in local `(s,p)`:
  - `R = diag(Gamma_s, Gamma_p)`
  - This corresponds to smooth, isotropic, homogeneous interface assumption.
- For expanded studies, an empirical off-diagonal hook is available at material level:
  - `xpol_coupling_db`
  - `xpol_coupling_phase_deg`
- This adds effective cross-polar reflection terms in `jones_reflection`, but is not a full
  rough-surface/multilayer anisotropic solver (e.g., BRDF/Kirchhoff/cluster scattering).
- PEC `Gamma_p` sign is convention-sensitive in local `(s,p)` basis.
  - tracer default keeps legacy continuity: `Gamma_s=-1`, `Gamma_p=-1`
  - optional override exists via material field `pec_tm_sign=+1.0`

## Antenna Coupling Frequency Dependence

- Antenna coupling is complex-valued and can be frequency-dependent:
  - leakage level: `cross_pol_leakage_db + slope*(f-f_ref)`
  - axial-ratio term: `axial_ratio_db + slope*(f-f_ref)`
  - cross-coupling phase: `cross_coupling_phase_deg + slope*(f-f_ref)`
- Supported runner CLI keys (Tx/Rx each):
  - `--*-coupling-ref-freq-hz`
  - `--*-cross-pol-leakage-db-slope-per-ghz`
  - `--*-axial-ratio-db-slope-per-ghz`
  - `--*-cross-coupling-phase-deg`
  - `--*-cross-coupling-phase-slope-deg-per-ghz`
- Defaults keep previous behavior (frequency-flat, real-symmetric coupling).

## FSPL and Antenna Pattern Assumptions

- `scalar_gain_f` in tracer uses amplitude-domain Friis scaling: `lambda/(4*pi*R)`.
- Path power therefore scales with `|lambda/(4*pi*R)|^2`, consistent with FSPL.
- By default, directional pattern gain is isotropic:
  - `tx_peak_gain_dbi=0`, `rx_peak_gain_dbi=0`
  - `tx_pattern_cos_exp=0`, `rx_pattern_cos_exp=0`
- Optional directional gain can be enabled with a simple boresight model:
  - `G(psi) = G_peak * max(cos(psi), 0)^n` (power gain)
  - CLI (`scenarios.runner`): `--tx-peak-gain-dbi`, `--rx-peak-gain-dbi`,
    `--tx-pattern-cos-exp`, `--rx-pattern-cos-exp`
- If you compare against measured datasets with strong off-boresight attenuation,
  keep isotropic defaults only for sanity checks; use directional settings for
  physically tighter absolute power matching.

## Wave-Basis Gauge Choice

- Tracer supports two wave-basis update modes:
  - `transport` (default): segment-to-segment basis is smoothly transported.
  - `global_up`: recompute each segment basis from fixed `global_up`.
- Both are gauge choices; observable power metrics should remain close, but
  `transport` reduces basis-jump sensitivity in multi-bounce geometries.
- CLI (`scenarios.runner`): `--wave-basis-mode transport|global_up`
