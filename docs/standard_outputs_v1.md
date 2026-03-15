# Standard Outputs v1

Schema version: `standard_outputs_v1`

This schema is the common output contract for all simulation families:

- calibration (`C0`)
- controlled geometry (`A2`-`A5`)
- spatial grid (`B1`-`B3`)

RT is used as a latent-variable generator (`tau_m`, `n_m`, `EL_m`, ...).  
Validation is performed at link-summary metric level `Z` (distribution/correlation),
not waveform-level CIR matching.

## Common IDs and Provenance

Mandatory identifiers:

- `run_id`
- `scenario_id`
- `case_id`
- `link_id` (anchor-tag pair or unique link key)

Run metadata (`/runs/{run_id}/meta`) must include:

- `schema_version`
- `git_commit`
- `git_branch`
- `cmdline`
- `seed`
- `timestamp_utc`
- scenario/run options used for generation

## Layer A: Ray Table (Latent Variables)

Per-link ray table (`Nrays` rows).

Mandatory columns:

- `tau_s` [s]
- `L_m` [m]
- `n_bounce` [int]
- `P_lin` [linear power proxy]
- `EL_db` [dB], where  
  `EL_m = 10log10(P_fs(L_m) / P_m)`

Optional columns:

- `material_class`
- `incidence_deg`
- `surface_seq`
- `parity` (`even`/`odd`)
- `los_flag_ray`

## Layer B: Measurement-Compatible Dual-CP PDP/CIR

Per-link delay-domain vectors (`Ntau` bins):

- `delay_tau_s`
- `P_co`
- `P_cross`

Optional:

- `XPD_tau_db = 10log10(P_co / P_cross)`

## Layer C: Link Summary Metrics `Z`

Mandatory metrics:

- `XPD_early_db = 10log10(sum_early(P_co)/sum_early(P_cross))`
- `XPD_late_db = 10log10(sum_late(P_co)/sum_late(P_cross))`
- `rho_early_lin = sum_early(P_cross)/sum_early(P_co)`
- `rho_early_db = 10log10(rho_early_lin)`
- `L_pol_db = XPD_early_db - XPD_late_db`
- `delay_spread_rms_s`
- `early_energy_fraction = sum_early(P_total)/sum_total(P_total)` (recommended)

Window parameters must be saved:

- `tau0_s`
- `Te_s`
- `Tmax_s`
- `tau0_method`
- `noise_floor_def`

Delay spread default reference (fixed for v1): `total` (`P_total = P_co + P_cross`).

## Link Conditions `U`

Mandatory:

- `d_m`
- `LOSflag`
- `EL_proxy_db`

Optional:

- `material_class`
- `roughness_flag`
- `human_flag`
- `obstacle_flag`
- `dominant_parity_early`

## HDF5 Layout (Recommended)

`/runs/{run_id}/meta/...`  
`/runs/{run_id}/links/{link_id}/rays/...`  
`/runs/{run_id}/links/{link_id}/pdp/delay_tau_s`  
`/runs/{run_id}/links/{link_id}/pdp/P_co`  
`/runs/{run_id}/links/{link_id}/pdp/P_cross`  
`/runs/{run_id}/links/{link_id}/metrics/...`  
`/runs/{run_id}/links/{link_id}/U/...`

CSV exports:

- `link_metrics.csv` (1 row/link)
- `rays.csv` (1 row/ray + link columns)
