# Polarimetric UWB Ray Tracing

## File Tree & Responsibilities

- `rt_core/geometry.py`: plane/material primitives, normal handling, intersections, reflection and path length.
- `rt_core/rays.py`: ray and path metadata accumulator structures.
- `rt_core/polarization.py`: transverse basis, local s/p basis, Fresnel/Jones reflection, depolarization layer.
- `rt_core/antenna.py`: idealized H/V or R/L port bases and projection matrices.
- `rt_core/tracer.py`: deterministic path enumeration (`max_bounce=0,1,2`), delay/`A_f`/meta computation.
- `rt_io/hdf5_io.py`: HDF5 schema save/load + roundtrip reproducibility self-test.
- `analysis/ctf_cir.py`: `H(f)` synthesis, basis conversion, IFFT CIR and PDP.
- `analysis/xpd_stats.py`: tap/path XPD, parity labels, conditional normal fitting and JSON export.
- `scenarios/*.py`: C0/A1/A2/A3/A4/A5 scenario generators (`build_scene`, `build_sweep_params`, `run_case`).
- `scenarios/runner.py`: end-to-end sweep execution, HDF5 save, P0~P13 plot generation, validation report.
- `plots/p0_p13.py`: matplotlib-only automated plotting pipeline for P0~P13.
- `tests/test_tracer.py`: unit tests for LOS, 1-bounce, and 2-bounce cases.

## Quick Start

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m scenarios.runner --output outputs/rt_dataset.h5 --plots-dir outputs/plots --report outputs/validation_report.md
make canonical_release
make rich_release
make parity_map
```

## MATLAB Port (Phase 1)

MATLAB core rewrite is available under `matlab/`:

```matlab
addpath('/Users/kimmyoungsun/Documents/codex/matlab');
ok = run_all_tests();

dataset = run_scenario_sweep('scenario', 'C0', 'basis', 'linear', ...
  'out_mat', '/Users/kimmyoungsun/Documents/codex/outputs/matlab_c0.mat', ...
  'out_json_summary', '/Users/kimmyoungsun/Documents/codex/outputs/matlab_c0_summary.json');

report = run_xpd_analysis('dataset', dataset, ...
  'num_subbands', 4, ...
  'out_json', '/Users/kimmyoungsun/Documents/codex/outputs/matlab_c0_xpd.json');

summary = validate_matlab2025('output_dir', ...
  '/Users/kimmyoungsun/Documents/codex/outputs/matlab_validation_2025');
```

Current MATLAB coverage includes `rt_core`, scenario generation/execution (`A1/C0/A2/A2R/A3/A3R/A4/A5/A6/B0`), and core `analysis` functions (`ctf_cir` + key `xpd_stats`).

Optional dispersive material mode (defaults keep legacy behavior):
```bash
python -m scenarios.runner \
  --materials-db materials/materials_db.json \
  --material-dispersion on \
  --output outputs/rt_dataset_dispersion.h5 \
  --plots-dir outputs/plots_dispersion \
  --report outputs/report_dispersion.md
```

## Measurement Bridge

- See `docs/measurement_bridge.md` for `A_f` vs `J_f` comparison rules and measurement mismatch diagnosis order.
- Optional measurement compare mode (off by default):
```bash
python -m scenarios.runner \
  --measurement-compare \
  --measurement-format matrix_csv \
  --measurement-matrix-csv /path/to/measured_2x2.csv
```

Dual-CP sequential measurement compare:
```bash
python3 -m scenarios.runner \
  --measurement-compare \
  --measurement-format dualcp_two_csv \
  --measurement-co-csv /path/to/co.csv \
  --measurement-cross-csv /path/to/cross.csv \
  --measurement-basis circular \
  --measurement-convention IEEE-RHCP
```

## Dual-CP Proxy Workflow

1) Ingest dual-CP measurement CSVs into measurement HDF5:
```bash
python3 scripts/ingest_measurement_dualcp.py \
  --scenario-id C0 \
  --case-id 0 \
  --co-csv /path/to/co.csv \
  --cross-csv /path/to/cross.csv \
  --out-h5 outputs/measurement_dualcp.h5 \
  --meta-json '{"basis":"circular","convention":"IEEE-RHCP","label":"C0 LOS"}'
```

2) Calibrate LOS floor:
```bash
python3 scripts/dualcp_calibrate_floor.py \
  --measurement-h5 outputs/measurement_dualcp.h5 \
  --scenario-id C0 \
  --out-json outputs/calibration_floor.json \
  --plots-dir outputs/plots_floor
```

3) Run RT with dual-CP power metrics:
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

4) Fit conditional proxy model:
```bash
python3 scripts/fit_dualcp_proxy.py \
  --metrics-csv outputs/dualcp_metrics.csv \
  --calibration-json outputs/calibration_floor.json \
  --out-model-json outputs/proxy_model.json \
  --out-report outputs/proxy_report.md
```

See `docs/dualcp_proxy_model.md` for floor->metrics->conditional-fit->bridge details and sensitivity guidance.

## Standard Outputs Runner (v1)

Run standardized A/B/C/U outputs (HDF5 + CSV) for each scenario.
`run_standard_sim.py` now accepts protocol aliases directly:

- `_off`: `A2_off`, `A6_off` (LOS blocked/off profile)
- `_on`: `A2_on`, `A3_on`, `A4_on`, `A6_on` (LOS bridge profile)
- split labels: `A3_supp`, `A4_iso`, `A4_bridge`
- paired label: `A5_pair` (base+stress in one run)

Examples:

```bash
python3 scripts/run_standard_sim.py --scenario C0 --out-h5 outputs/std_c0.h5 --out-dir outputs/std_c0 --basis circular --convention IEEE-RHCP --matrix-source A --force-cp-swap-on-odd-reflection false
python3 scripts/run_standard_sim.py --scenario A2_off --out-h5 outputs/std_a2_off.h5 --out-dir outputs/std_a2_off --strict-los-blocked --basis circular --convention IEEE-RHCP --matrix-source A --force-cp-swap-on-odd-reflection false
python3 scripts/run_standard_sim.py --scenario A6_off --out-h5 outputs/std_a6_off.h5 --out-dir outputs/std_a6_off --strict-los-blocked --a6-modes odd,even --a6-case-set both --a6-even-layout localized --a6-even-path-policy canonical --a6-incidence-max-deg 15 --basis circular --convention IEEE-RHCP --matrix-source J --force-cp-swap-on-odd-reflection false
python3 scripts/run_standard_sim.py --scenario A3_supp --out-h5 outputs/std_a3_supp.h5 --out-dir outputs/std_a3_supp --strict-los-blocked --basis circular --convention IEEE-RHCP --matrix-source A --force-cp-swap-on-odd-reflection false
python3 scripts/run_standard_sim.py --scenario A4_iso --out-h5 outputs/std_a4_iso.h5 --out-dir outputs/std_a4_iso --material-list glass,wood --a4-dispersion-modes off,on --basis circular --convention IEEE-RHCP --matrix-source A --force-cp-swap-on-odd-reflection false
python3 scripts/run_standard_sim.py --scenario A4_bridge --out-h5 outputs/std_a4_bridge.h5 --out-dir outputs/std_a4_bridge --material-list glass,wood --a4-dispersion-modes off,on --basis circular --convention IEEE-RHCP --matrix-source A --force-cp-swap-on-odd-reflection false
python3 scripts/run_standard_sim.py --scenario A5_pair --out-h5 outputs/std_a5_pair.h5 --out-dir outputs/std_a5_pair --strict-los-blocked --a5-stress-mode geometry --a5-scatterer-count 3 --a5-diffuse-enabled false --basis circular --convention IEEE-RHCP --matrix-source A --force-cp-swap-on-odd-reflection false
python3 scripts/run_standard_sim.py --scenario A2_on --out-h5 outputs/std_a2_on.h5 --out-dir outputs/std_a2_on --basis circular --convention IEEE-RHCP --matrix-source A --force-cp-swap-on-odd-reflection false
python3 scripts/run_standard_sim.py --scenario A3_on --out-h5 outputs/std_a3_on.h5 --out-dir outputs/std_a3_on --basis circular --convention IEEE-RHCP --matrix-source A --force-cp-swap-on-odd-reflection false
python3 scripts/run_standard_sim.py --scenario A4_on --out-h5 outputs/std_a4_on.h5 --out-dir outputs/std_a4_on --material-list glass,wood --a4-layout-modes iso,bridge --a4-dispersion-modes off,on --basis circular --convention IEEE-RHCP --matrix-source A --force-cp-swap-on-odd-reflection false
python3 scripts/run_standard_sim.py --scenario A6_on --out-h5 outputs/std_a6_on.h5 --out-dir outputs/std_a6_on --a6-modes odd,even --a6-case-set both --a6-even-layout localized --a6-even-path-policy canonical --a6-incidence-max-deg 15 --basis circular --convention IEEE-RHCP --matrix-source J --force-cp-swap-on-odd-reflection false
python3 scripts/run_standard_sim.py --scenario B1 --out-h5 outputs/std_b1.h5 --out-dir outputs/std_b1
python3 scripts/run_standard_sim.py --scenario B2 --out-h5 outputs/std_b2.h5 --out-dir outputs/std_b2
python3 scripts/run_standard_sim.py --scenario B3 --out-h5 outputs/std_b3.h5 --out-dir outputs/std_b3
```

Protocol reproducibility (manifest + wrapper):
```bash
python3 scripts/run_protocol_manifest.py \
  --manifest analysis_report/manifests/protocol_repro_manifest.v1.json \
  --output-root outputs/protocol_repro_v1 \
  --analysis-config-out analysis_report/config.protocol_repro_v1.json
```
Use `--only A6_off,A6_on` for partial reruns and `--tag <name>` when you need parallel output folders.

Reporting rule: use `A6` as the primary G2 (odd/even sign) evidence in standard reports; keep `A3` as supplementary mechanism evidence.
For A4 reporting, split `A4_iso` (primary material reflection, `include_late_panel=false`) and `A4_bridge` (secondary delayed effect, `include_late_panel=true`); make dispersion claims only when bridge runs include `material_dispersion=on|debye`.
For B1/B2/B3 reporting, interpret R1/R2 as a `coverage-aware leverage map` over viable strata; do not claim an absolute universal map while structural holes remain.

Generate success checks + plots:

```bash
python3 scripts/make_success_report.py \
  --link-metrics-csv outputs/std_b1/link_metrics.csv \
  --out-report outputs/success_report.md \
  --out-json outputs/success_report.json \
  --plots-dir outputs/plots_standard
```

Select candidate measurement points (stratified by EL/LOS):

```bash
python3 scripts/select_measurement_points.py \
  --link-metrics-csv outputs/std_b1/link_metrics.csv \
  --out-csv outputs/selected_points.csv \
  --bins EL_proxy_db:4,LOSflag:2 \
  --per-bin 5
```
