# Ray-Tracing Modules

This folder is a consolidated copy of the current ray-tracing code.
Original files in `D:\codex` were not modified.

## Layout

- `rt_core_bundle`
  - Low-level deterministic ray tracer
  - Includes `rt_core`, `scenarios`, `requirements.txt`, and `tests/test_tracer.py`

- `dualpol_rt_bundle`
  - Higher-level realistic dual-polarized ray-tracing package
  - Includes `dualpol_rt`, `examples`, `tests`, and `pyproject.toml`

## Notes

- `rt_core_bundle\scenarios\runner.py` was copied as part of the module set, but it references extra top-level analysis/report code that is not included here.
- Core solver files and the main `dualpol_rt` package files are included intact.
- MATLAB validation scripts are included under:
  - `dualpol_rt_bundle\dualpol_rt\validation`

## Quick Start

- `rt_core_bundle`
  - Run from: `D:\codex\raytracing_modules\rt_core_bundle`
  - Example test: `python -m pytest tests\\test_tracer.py -q`

- `dualpol_rt_bundle`
  - Run from: `D:\codex\raytracing_modules\dualpol_rt_bundle`
  - Example test: `python -m pytest tests\\test_geometry_paths.py -q`
