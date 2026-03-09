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
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/tables/target_level.csv`
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/tables/case_level.csv`
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/tables/sensitivity_level.csv`
- `/Users/kimmyoungsun/Documents/codex/analysis_report/out/<run_group>/tables/A6_case_set_sign_compare.csv` (A6 full/minimal odd-even 비교)

## Notes
- 기본 구현은 **power 기반 지표(Z/U)**만 사용합니다.
- 최종 시나리오 구조(합의):
  - `C0`: calibration only
  - `A2_off`: G1 primary evidence
  - `A6`: G2 primary evidence (near-normal PEC, incidence <= 15 deg)
    - A6 실행 시 `--a6-case-set both`를 쓰면 full/minimal을 한 번에 돌리고 `A6_case_set_sign_compare.csv`에서 odd/even 비교 가능
    - `--a6-even-path-policy canonical`(기본)으로 even 대칭 다중경로를 단일 canonical path로 고정 가능
  - `A3_corner`: supplementary mechanism only
  - `A4_iso`: L2-M primary (`late_panel=false`, `dispersion=off`)
  - `A4_bridge`: L2-M secondary (`late_panel=true`, `dispersion=on`)
  - `A5_pair`: L2-S proxy stress response (synthetic primary, geometric sensitivity)
  - `A2_on/A3_on/A4_on/A6_on`: LOS-on observability bridge set
  - `B1/B2/B3`: R1/R2 coverage-aware leverage map (viable strata + support count mandatory, no absolute universal-map claim)
- 표준 리포팅 규칙:
  - G2 sign-off는 `A6` target-window raw sign(odd/even) 기준을 1차로 사용
  - `A3`는 메커니즘 보조 증거로만 보고
  - A4는 반드시 `A4_iso`(primary: `include_late_panel=false`)와 `A4_bridge`(secondary: `include_late_panel=true`)로 분리 표기
  - dispersion claim은 `A4_bridge`에서 `material_dispersion=on|debye` 데이터가 있을 때만 허용
  - B1/B2/B3 기반 R1/R2는 structural-hole을 반영한 `coverage-aware leverage map`으로만 해석하고, absolute universal map 주장은 금지
- `scene_debug.json`이 없으면 해당 케이스 scene plot은 WARN으로 보고서에 기록됩니다.
- 진단 B는 목적별 3-window를 분리해 계산합니다.
  - `W_floor` (C0): LOS peak 주변 contamination `C_floor`
  - `W_target` (A2-A5): target path 주변 contamination `C_target`
  - `W_early` (B1-B3): `Te` sweep(기본 `2/3/5 ns`) 분리력 `S(Te)`
  - A3는 `target_window_ns_by_scenario.A3`(또는 `target_window_mode_by_scenario.A3=adaptive`)로 별도 `W_target` 설정 가능
  - A3 sign-off는 `windows.target_sign_metric_by_scenario.A3=raw` 권장 (`A3_target_window_sign.csv`에 raw/ex 동시 출력)
  - 보고서에서 A3는 mechanism-only로 분리 표기하며, fixed system `W_early` 우세 여부는 보조 진단으로만 사용
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

## Reproducibility Manifest
- 고정 프로토콜 manifest: `analysis_report/manifests/protocol_repro_manifest.v1.json`
- 실행 wrapper:
```bash
python3 scripts/run_protocol_manifest.py \
  --manifest analysis_report/manifests/protocol_repro_manifest.v1.json \
  --output-root outputs/protocol_repro_v1 \
  --analysis-config-out analysis_report/config.protocol_repro_v1.json
```
