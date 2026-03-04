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

# 4) 최종 판정 보고서(템플릿 기반, 문서용)
python analysis_report/generate_final_decision_report.py --config analysis_report/config.yaml

# 5) 명제-실험-데이터-플롯 매칭표 + PASS/FAIL
python analysis_report/generate_proposition_matrix_report.py --run-group <run_group>

# 6) 상세 명제(M/G/L/R/P) 플롯 매칭표(세부 Plot ID 기준)
python analysis_report/generate_proposition_plot_mapping_detailed.py --run-group <run_group>
```

## Outputs
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/diagnostic_report.md`
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/intermediate_report.md`
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/warning_report.md`
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/figures/*.png`
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/tables/*.csv`
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/index.csv`
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/proposition_plot_mapping_report.md`
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/tables/proposition_plot_mapping.csv`
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/proposition_plot_mapping_detailed_report.md`
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/tables/proposition_plot_mapping_detailed.csv`
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/tables/plot_data/<plot_id>__data.csv` (`x,y,data` 포함)

## Notes
- 기본 구현은 **power 기반 지표(Z/U)**만 사용합니다.
- `scene_debug.json`이 없으면 해당 케이스 scene plot은 WARN으로 보고서에 기록됩니다.
- 진단 B는 목적별 3-window를 분리해 계산합니다.
  - `W_floor` (C0): LOS peak 주변 contamination `C_floor`
  - `W_target` (A2-A5): target path 주변 contamination `C_target`
  - `W_early` (B1-B3): `Te` sweep(기본 `2/3/5 ns`) 분리력 `S(Te)`
- 진단 C는 endpoint를 분리합니다.
  - `C2-M`(material): primary=`XPD_early_excess`, secondary=`XPD_late_excess`/`L_pol`
  - `C2-S`(stress): primary=`L_pol`, secondary=`rho_early`/`DS`/`XPD_late_excess`, gate=`ΔP_target,total`
- A5 stress 해석 규칙:
  - `stress_semantics=response`(geometry/hybrid): delay/path contamination-response 해석 가능
  - `stress_semantics=polarization_only`(synthetic): 편파축 stress만 의미하며 delay/path 구조 변화 주장 금지
- 최종 보고서 산출물:
  - `analysis_report/out/<run_group>/final_diagnostic_decision.md`
  - `analysis_report/out/<run_group>/scenario_space_plots.md`
  - `analysis_report/out/<run_group>/figures/<scenario>__ALL__scene_montage.png`
