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
