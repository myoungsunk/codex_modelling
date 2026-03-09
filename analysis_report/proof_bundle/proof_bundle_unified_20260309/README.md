# Proof Bundle Unified (2026-03-09)

이 번들은 공개 검토용 고정 산출물입니다.
기준 run-group은 `diag_protocol_repro_v1_20260309_archetype`이며, B 진단 계약은 2026-03-09 업데이트(상대 anchor + support-aware risk) 기준입니다.

## 1) Repro Provenance

- Manifest: `source/protocol_repro_manifest.v1.json`
- Analysis config: `source/config.protocol_repro_v1_20260309_archetype.json`
- Source run-group (local): `analysis_report/out/diag_protocol_repro_v1_20260309_archetype`
- Final decision: `source/final_diagnostic_decision.md`
- Diagnostic checks: `tables/diagnostic_checks.json`

## 2) Public Verification Pointers

아래 파일에서 3/9 업데이트 토큰을 직접 확인할 수 있습니다.

- `source/diagnostic_report.md`
  - `good_condition_baseline`
  - `contamination_onset`
  - `high_risk_tail`
  - `support_rate`
  - `neg_tail_rate_xpd_early_ex_lt_0`
- `tables/diagnostic_checks.json`
  - `status_evidence_quality`
  - `tail_threshold_med_anchor_db_from_B1_minus_delta_ref`
  - `rho_q90_B2/B3`
  - `early_fraction_q25_B2/B3`

## 3) Scenario Validity (13 units)

최신 요약표:
- `tables/scenario_validity_summary.csv`
- `tables/scenario_validity_summary.json`

현재 상태 요약:
- PASS: `C0, A2_off, A6_off, A4_iso, A5_pair, B2`
- WARN(scope-control): `A3_supp, A4_bridge`
- WARN(bridge): `A2_on, A3_on, A6_on, A4_on`
- WARN(B practical): `B1, B3`

## 4) PASS/WARN/FAIL Cause Tables

- `tables/fail_warn_fixability.csv`
  - 추가 시뮬레이션으로 개선 가능한 WARN과 역할고정 WARN를 분리

## 5) Included Artifacts

- Reports:
  - `source/diagnostic_report.md`
  - `source/intermediate_report.md`
  - `source/final_diagnostic_decision.md`
  - `source/proposition_plot_mapping_report.md`
  - `source/proposition_plot_mapping_detailed_report.md`
- Tables:
  - `tables/diagnostic_checks.json`
  - `tables/B_per_scenario_summary.csv`
  - `tables/D3_hole_analysis.csv`
  - `tables/case_level.csv`
  - `tables/target_level.csv`
  - `tables/floor_reference_used.json`
  - `tables/intermediate_proposition_status.csv`
  - `tables/intermediate_proposition_details.json`
  - `tables/proposition_plot_mapping.csv`
  - `tables/proposition_plot_mapping_detailed.csv`

## 6) Interpretation Rules (Current)

- B1: absolute positive gate 제거, 상대 good-condition anchor로 판정.
- B2: transition-zone distribution shift 중심으로 판정.
- B3: `B1 anchored relative tail + rho upper-tail + early_fraction lower-tail + support drop` 중심 판정.
- L_pol/DS: 보조지표(secondary evidence)로만 사용.
- physics status와 evidence-quality status 분리 보고.
