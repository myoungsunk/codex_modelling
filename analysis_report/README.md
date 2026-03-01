# analysis_report

`standard outputs`(run_summary/link_metrics/rays/scene_debug)를 읽어 진단 보고서와 중간 보고서를 생성합니다.

## Inputs
- 각 run 폴더의 `run_summary.json` (필수)
- `link_metrics.csv` (필수)
- `rays.csv` (권장)
- `scene_debug/*.json` (scene plot 생성용, 권장)

## Config
- 예시: `/Users/kimmyoungsun/Documents/codex/analysis_report/config.example.yaml`
- 실행 시 `--config`로 전달

## Commands
- 진단 보고서:
```bash
python analysis_report/generate_diagnostic_report.py --config analysis_report/config.yaml
```
- 중간 보고서:
```bash
python analysis_report/generate_intermediate_report.py --config analysis_report/config.yaml
```
- 경고 보고서(진단 WARN/FAIL 케이스별 리뷰):
```bash
python analysis_report/generate_warning_report.py --config analysis_report/config.yaml
```

## Outputs
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/diagnostic_report.md`
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/intermediate_report.md`
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/warning_report.md`
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/figures/*.png`
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/tables/*.csv`
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/index.csv`

## Notes
- 기본 구현은 **power 기반 지표(Z/U)**만 사용합니다.
- `scene_debug.json`이 없으면 해당 케이스 scene plot은 WARN으로 보고서에 기록됩니다.
- 진단 B는 목적별 3-window를 분리해 계산합니다.
  - `W_floor` (C0): LOS peak 주변 contamination `C_floor`
  - `W_target` (A2-A5): target path 주변 contamination `C_target`
  - `W_early` (B1-B3): `Te` sweep(기본 `2/3/5 ns`) 분리력 `S(Te)`
