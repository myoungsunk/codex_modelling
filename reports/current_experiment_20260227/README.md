# Current Experiment Report Bundle

This folder contains the uploaded report artifacts for the current experiment review.

## Files
- `diagnostic_report.md`
- `intermediate_report.md`
- `warning_report.md` (diagnostic WARN/FAIL case-by-case)
- `index.csv`, `index.md`
- `tables/` key numeric summaries
- `figures/` figures referenced by the reports
- `config_used.json` (reproducibility)

## Source generation commands
- `python analysis_report/generate_diagnostic_report.py --config /tmp/analysis_report_current.json`
- `python analysis_report/generate_intermediate_report.py --config /tmp/analysis_report_current.json`
