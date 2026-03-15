# Dual-CP Proxy Model

## Scope

This flow targets dual-CP sequential UWB measurements where co/cross are captured in separate sweeps.

- Use power-domain metrics.
- Do not claim per-ray complex 2x2 recovery from sequential traces.
- Keep basis/convention explicit: `basis=circular`, `convention=IEEE-RHCP`.

## Pipeline

1. Floor calibration (C0 LOS)
2. Metric extraction (`H(f) -> CIR -> early/late metrics`)
3. Conditional proxy fit (`Z | U`)
4. RT->Proxy bridge evaluation (distribution-level, not waveform-level)

## Step 1: Floor Calibration

Estimate `XPD_floor(f)` from LOS/free-space cases:

```bash
python3 scripts/dualcp_calibrate_floor.py \
  --measurement-h5 outputs/measurement_dualcp.h5 \
  --scenario-id C0 \
  --out-json outputs/calibration_floor.json \
  --plots-dir outputs/plots_floor
```

Output includes:

- `frequency_hz`
- `xpd_floor_db`
- `xpd_floor_uncert_db` (percentile-width based)

## Step 2: Dual-CP Metrics

Run RT and emit case-level dual-CP proxy metrics:

```bash
python3 -m scenarios.runner \
  --basis circular \
  --dualcp-metrics on \
  --calibration-floor-json outputs/calibration_floor.json \
  --early-window-ns 3.0 \
  --tmax-ns 30.0 \
  --noise-tail-ns 8.0 \
  --threshold-db 6.0 \
  --dualcp-metrics-csv outputs/dualcp_metrics.csv \
  --dualcp-metrics-json outputs/dualcp_metrics.json
```

Core outputs:

- `xpd_early_db`, `xpd_late_db`
- `l_pol_db`
- `rho_early_linear`, `rho_early_db`
- `tau_rms_ns`, `early_energy_concentration`
- `xpd_early_excess_db`, `xpd_late_excess_db` (if floor JSON is provided)
- `claim_caution_early/late` (true when excess is within floor uncertainty)

CTF/CIR interpretation limits (important):

- `ctf_to_cir` uses the sampled RF band grid directly (e.g. 6-10 GHz), so CIR is band-limited.
- Delay-bin spacing: `dt = 1/(nfft*df)`.
- Fundamental path separability is bandwidth-limited (`~1/BW`, `BW=f_max-f_min`), not improved by zero padding alone.
- For strict reporting, include `analysis.ctf_cir.cir_bandlimit_info(...)` in metadata.

## Step 3: Conditional Proxy Fit

Fit `Z|U` using binned statistics + numeric regression fallback:

```bash
python3 scripts/fit_dualcp_proxy.py \
  --metrics-csv outputs/dualcp_metrics.csv \
  --calibration-json outputs/calibration_floor.json \
  --out-model-json outputs/proxy_model.json \
  --out-report outputs/proxy_report.md
```

Default target variables `Z`:

- `xpd_early_excess_db`
- `xpd_late_excess_db`
- `l_pol_db`
- `rho_early_db`

Default condition variables `U`:

- `los_blocked`, `material`, `scatter_stress`
- `distance_d_m`, `pathloss_proxy_db`
- `delay_bin`
- `parity`, `incidence_angle_bin`, `excess_loss_proxy_db`, `bounce_count`

GOF summary includes KS, QQ-correlation, and Wasserstein.

## Step 4: RT->Proxy Bridge

Bridge target is distribution agreement of `Z`, not CIR waveform identity.

- Compute `U` from RT/metadata.
- Predict `Z` distribution with proxy model.
- Compare measured-vs-predicted distributions via Wasserstein + rank correlation.

## Sensitivity Guidance

Run at least 2-3 parameter sets:

- `T_e` (early window): e.g. `2.0, 3.0, 4.0 ns`
- `noise_tail_ns`: e.g. `6, 8, 10 ns`
- `threshold_db`: e.g. `3, 6, 9 dB`

Check stability of:

- `xpd_early_excess_db` sign
- `l_pol_db` ordering across conditions
- bridge metrics (Wasserstein, rank correlation)
