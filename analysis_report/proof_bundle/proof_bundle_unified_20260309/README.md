# Proof Bundle Unified (2026-03-09)

이 폴더는 `diag_protocol_repro_v1_20260309_r1` 진단 결과를 재현/검토 가능한 형태로 단일화한 증명 번들입니다.

## 1) Repro Provenance

- Manifest: `source/protocol_repro_manifest.v1.json`
- Analysis config: `source/config.protocol_repro_v1_20260309_r1.json`
- Source run-group: `analysis_report/out/diag_protocol_repro_v1_20260309_r1`
- Final decision: `source/final_diagnostic_decision.md`
- Diagnostic core checks: `tables/diagnostic_checks.json`

Executed command:
```bash
python3 scripts/run_protocol_manifest.py \
  --manifest analysis_report/manifests/protocol_repro_manifest.v1.json \
  --output-root outputs/protocol_repro_v1 \
  --tag 20260309_r1 \
  --run-group diag_protocol_repro_v1_20260309_r1 \
  --analysis-config-out analysis_report/config.protocol_repro_v1_20260309_r1.json \
  --skip-existing
```

## 2) Scenario Validity (Requested 13 units)

- PASS: 6
- WARN: 6 (scope-control=2)
- FAIL: 1

| Scenario | Status | n_links | Why | Evidence |
| --- | --- | ---: | --- | --- |
| C0 | PASS | 125 | Calibration floor/uncertainty 추정이 안정적으로 수집됨. | `tables/floor_reference_used.json; source/diagnostic_report.md#Floor-Reference` |
| A2_off | PASS | 4 | odd parity target-window sign(raw/excess) 검증이 충족됨. | `tables/target_level.csv (scenario=A2); source/final_diagnostic_decision.md#L124` |
| A6_off | PASS | 20 | near-normal odd/even parity raw sign hit-rate(odd/even=1.0)가 모두 PASS. | `tables/diagnostic_checks.json:B_time_resolution.A6_parity_benchmark; source/final_diagnostic_decision.md#L129` |
| A3_supp | WARN(scope-control) | 12 | mechanism-only 보조 시나리오로 고정(시스템 early baseline/G2 sign-off 용도 제외). | `tables/target_level.csv (scenario=A3); source/final_diagnostic_decision.md#L128` |
| A4_iso | PASS | 12 | material primary 분기로 채택(iso: include_late_panel=False)되어 재현 가능. | `tables/case_level.csv (scenario=A4, iso subset via diagnostic_link_rows); source/final_diagnostic_decision.md#L130` |
| A4_bridge | WARN(scope-control) | 12 | bridge/support 증거로만 사용하도록 역할 고정(primary material claim 제외). | `tables/diagnostic_checks.json:C_effect_vs_floor (A4_bridge_*); source/final_diagnostic_decision.md#L131` |
| A5_pair | PASS | 60 | paired base/on contamination-response 해석이 유효(stress semantics=response). | `tables/diagnostic_checks.json:A5_stress_semantics; source/final_diagnostic_decision.md#L133` |
| A2_on | WARN | 4 | LOS-under bridge 관측 확인용이며 G1 sign-off 용도로는 제외. | `tables/case_level.csv (A2/A2_on); source/final_diagnostic_decision.md#L125` |
| Even-LOS-on bridge (A6_on + A3_on) | WARN | 32 | LOS-under even bridge 관측 확인용이며 G2 sign-off는 A6_off core evidence를 사용. | `tables/case_level.csv (A6/A6_on/A3_on); source/final_diagnostic_decision.md#L126-L127` |
| A4_on | WARN | 24 | LOS-under material bridge 관측 확인용(최종 material sign-off 용도 아님). | `tables/case_level.csv (A4/A4_on); source/final_diagnostic_decision.md#L132` |
| B1 | FAIL | 49 | per-scenario strata coverage hole(LOS0_q2=0, LOS0_q3=0)로 absolute map 해석 불가. | `tables/B_per_scenario_summary.csv (B1 row); source/final_diagnostic_decision.md#L134` |
| B2 | WARN | 49 | coverage는 있으나 strata support가 최소 수준(min_strata=2)이라 leverage 해석으로 제한. | `tables/B_per_scenario_summary.csv (B2 row); source/final_diagnostic_decision.md#L135` |
| B3 | PASS | 49 | viable strata support(min_strata_viable_n=4) 확보로 coverage-aware high-risk map 역할 수행 가능. | `tables/B_per_scenario_summary.csv (B3 row); source/final_diagnostic_decision.md#L136` |

원본 상세 표: `tables/scenario_validity_summary.csv` / `tables/scenario_validity_summary.json`

## 3) PASS/WARN/FAIL Causes (Concise)

- PASS 핵심: C0/A2_off/A6_off/A4_iso/A5_pair/B3는 고정변수 조건과 스윕 커버리지가 충족되고, 핵심 지표가 목적에 맞게 계산됨.
- WARN 핵심(bridge): A2_on/A3_on+A6_on/A4_on은 LOS-under 관측가능성 확인용으로는 유효하지만 core sign-off 용도로는 제외됨.
- WARN 핵심(scope-control): A3_supp/A4_bridge는 의도적으로 보조 역할로 고정한 상태이며, 약점이 아니라 범위 통제 규칙임.
- FAIL 핵심: B1은 per-scenario strata hole(`LOS0_q2=0`, `LOS0_q3=0`)로 universal/absolute map 주장을 할 수 없음.

## 4) Which FAIL/WARN are solvable by more simulation?

| Item | Type | Additional simulation helps? | Why |
| --- | --- | --- | --- |
| B1 FAIL | coverage_hole (data + geometry design) | yes | B1 strata LOS0_q2/q3 empty. Adding B1 grid points/orientation near LOS0 high-EL regime can fill holes. |
| B-all FAIL (D3) | reporting-input gap + support policy | partial | selected_rows_n=0 (selected_points_csv 미제공)와 일부 낮은 strata support가 결합. 추가 시뮬레이션 + selected subset 제공 필요. |
| A3_supp WARN(scope-control) | intentional role lock | no (not target) | A6를 G2 core로 쓰는 정책상 A3는 mechanism-only로 고정. |
| A4_bridge WARN(scope-control) | intentional role lock | no (not target) | A4_bridge는 bridge/support로만 사용하도록 합의. |

참조: `tables/fail_warn_fixability.csv`

## 5) Included Artifacts

- Reports: `source/diagnostic_report.md`, `source/intermediate_report.md`, `source/proposition_plot_mapping_report.md`, `source/proposition_plot_mapping_detailed_report.md`, `source/final_diagnostic_decision.md`
- Tables: `tables/diagnostic_checks.json`, `tables/case_level.csv`, `tables/target_level.csv`, `tables/B_per_scenario_summary.csv`, `tables/D3_hole_analysis.csv`, `tables/floor_reference_used.json`, `tables/intermediate_proposition_status.csv`, `tables/proposition_plot_mapping.csv`, `tables/proposition_plot_mapping_detailed.csv`

## 6) Notes for Review

- `A6_on`은 이번 번들에서 실제로 LOS-on(`LOSflag=1`)으로 생성되었음.
- `A4_iso`/`A4_bridge`는 `include_late_panel=0/1`로 분리되어 저장됨.
- B-series 결과는 `coverage-aware leverage map`으로만 해석해야 하며 absolute universal map claim은 금지.