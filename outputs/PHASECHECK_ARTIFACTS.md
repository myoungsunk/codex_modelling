# Latest Output Set (phasecheck)

This folder keeps only the most recent validation run artifacts.

## Core artifacts
- `rt_dataset_phasecheck.h5`
  - Main RT dataset (per-case/per-path matrices, delays, metadata).
- `rt_dataset_phasecheck_model_params.json`
  - Fitted stochastic model parameters from RT statistics (for RT→Synth generation).
- `rt_dataset_phasecheck_synthetic_compare.json`
  - Quantitative RT vs synthetic comparison metrics (F3/F5/F6 and related diagnostics).
- `report_phasecheck.md`
  - Full validation report with PASS/FAIL gates, scenario summaries, diagnostics.
- `report_phasecheck_tap_path_outliers.csv`
  - Outlier cases for tap-wise vs path-wise consistency checks.
- `report_phasecheck_tap_path_repro_bundle.json`
  - Minimal reproducibility bundle for tap/path inconsistency debugging.

## Plot folder
- `plots_phasecheck/`
  - All validation figures for this run (PNG/PDF pairs).

## Plot meaning guide
### Channel geometry / path sanity
- `P0_geometry_overlay`: Scene and traced rays; checks reflection geometry and blocking.
- `P1_tau_power`: Path delay vs power scatter.
- `P14_tau_error_hist`: Error of `tau` vs geometric path length/c0.
- `P15_incidence_distribution`: Incidence-angle distribution.
- `P19_reciprocity_sanity`: Forward/reverse reciprocity mismatch summary.

### Frequency/time channel behavior
- `P2_Hij_magnitude`: Magnitude of 2x2 channel elements over frequency.
- `P3_PDP`: PDP/CIR power profile.
- `P4_main_taps`: Zoom on dominant taps.
- `P21_tap_vs_path_consistency`: Tap-wise vs path-wise XPD consistency.

### Polarization/XPD statistics
- `P5_cp_same_vs_opp`: CP same-hand vs opposite-hand response.
- `P6_parity_xpd`: XPD by odd/even parity.
- `P7_xpd_vs_bounce`: XPD vs bounce count.
- `P8_xpd_vs_f_material`: XPD(f) by material.
- `P9_subband_mu_sigma`: Subband-wise XPD mean/sigma.
- `P10_parity_collapse`: Parity separation collapse view.
- `P11_xpd_var_vs_rho`: XPD variance vs depolarization strength.
- `P12_delay_conditioned`: Delay-conditioned XPD stats.
- `P16_fresnel_curves`: Fresnel (|Gamma_s|/|Gamma_p| and phase) behavior.
- `P17_A6_bounce_compare`: Controlled CP benchmark (1-bounce vs 2-bounce).
- `P17_cp_same_opp_vs_bounce`: Same/opp-hand power trend vs bounce.
- `P18_singular_values_delay`: Singular values/conditioning vs delay.
- `P20_xpd_fit_gof`: GOF of XPD distribution fit.

### Scenario richness / model stress
- `P13_k_factor`: Scenario-wise K-factor trend.
- `P22_material_dispersion_impact`: Dispersion OFF/ON impact on response.
- `P23_path_count_vs_bounce`: Path-count distribution by bounce.
- `P24_rms_delay_spread_diffuse_compare`: Delay-spread shift by diffuse mode.
- `P25_diffuse_power_accounting`: Specular/diffuse power accounting sanity.

### RT vs Synth model comparison
- `F2_rt_vs_synth_pdp_overlay`: RT vs synthetic PDP overlay.
- `F3_rt_vs_synth_xpd_cdf`: RT vs synthetic XPD CDF.
- `F4_rt_vs_synth_parity_box`: Parity-conditioned RT vs synthetic XPD boxes.
- `F5_rt_vs_synth_subband_xpd`: RT vs synthetic subband XPD mean/sigma overlay.
- `F5_rt_vs_synth_subband_sigma`: RT vs synthetic subband sigma comparison.
- `F6_offdiag_phase_uniformity`: Off-diagonal phase histogram and Kuiper uniformity diagnostics.
