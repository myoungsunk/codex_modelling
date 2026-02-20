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
```

## Measurement Bridge

- See `docs/measurement_bridge.md` for `A_f` vs `J_f` comparison rules and measurement mismatch diagnosis order.
