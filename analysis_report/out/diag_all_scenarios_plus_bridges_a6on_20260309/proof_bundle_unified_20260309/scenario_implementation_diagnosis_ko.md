# 시나리오 구현 총괄 진단 (최신 통합 데이터)

- 대상 run_group: `diag_all_scenarios_plus_bridges_a6on_20260309`
- 목적: 요청하신 시나리오 정의(고정변수/스윕변수/핵심 지표/실험 목적)가 현재 데이터에서 실제로 구현되었는지 총체적으로 점검
- 증명 자료 폴더: `proof_bundle_unified_20260309`

## 총괄 결과

- PASS: 5, WARN: 8, FAIL: 0 (총 13)
- `A6_on` 포함 Even-LOS-on bridge는 이제 데이터로 직접 검증 가능
- `B1/B2/B3`는 여전히 structural-hole 제약으로 coverage-aware leverage map 해석만 허용

## 요청 표 기반 구현 진단

| 시나리오 | 고정변수 | 스윕변수 | 핵심 지표 | 실험 목적 | 근거 | 고정 진단 | 스윕 진단 | 지표 진단 | 목적 진단 | 최종 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | free-space LOS, W_floor, 정적 반복 측정 | 거리, yaw, 반복수 | XPD_floor, Delta_floor, Delta_repeat | calibration floor와 불확실도 추정 | [GitHub][1] | PASS (LOSflag=1 9/9) | WARN (d_m=3, yaw=3, rep=1) | PASS (XPD_floor=9/9, Delta_floor=9/9) | PASS (calibration floor 추정은 성립, 반복 불확실도(Delta_repeat)는 rep 다양성 부족으로 제한) | **WARN** |
| A2_off | single PEC plane, exact 1-bounce, LOS blocked/off | reflector 위치, 거리 | XPD_target_raw, XPD_early_ex, rho_early | odd parity cause proof | [GitHub][2] | PASS (LOS=0 4/4, bounce=1 odd 확인) | PASS (d_m=2, cases=4) | PASS (XPD_target_raw=4/4, XPD_early_ex/rho_early complete) | PASS (odd parity 원인 고립 구현) | **PASS** |
| A6_off | near-normal PEC benchmark, odd/even exact target bounce, LOS off, incidence 제한 | mode(odd/even), small Rx perturbation | XPD_target_raw | clean even/odd parity benchmark, 특히 G2-core | [GitHub][3] | PASS (LOS=0 100/100, mode=2) | PASS (case_set=2, d_m=6) | PASS (XPD_target_raw sign hit odd=1.000, even=1.000) | PASS (G2-core near-normal parity benchmark 구현) | **PASS** |
| A3_supp | 2-plane corner, exact 2-bounce, Rx가 두 plane의 same side, LOS blocked/off | corner offset, Rx 위치 | XPD_target_raw | even-path existence/supporting mechanism scene, not core theorem | [GitHub][4] | PASS (LOS=0 12/12, bounce=2 even 확인) | PASS (case_n=12, d_m=2) | PASS (XPD_target_raw=12/12, sign_hit_raw=0.667) | WARN (supporting mechanism 용도로 구현(코어 정리에는 보조)) | **WARN** |
| A4_iso | 1-bounce material plane, LOS blocked/off, include_late_panel=False 권장 | material, incidence/plane position, dispersion on/off | XPD_target_raw, XPD_target_ex, XPD_early_ex | target reflection 자체의 material dependence | [GitHub][5] | PASS (LOS=0 12/12, include_late_panel=0) | PASS (material=3, dispersion=2) | PASS (XPD_target_raw/ex=12/12, XPD_early_ex=12/12) | WARN (target reflection material dependence는 구현되었으나 effect-size는 경계권) | **WARN** |
| A4_bridge | same as A4 but include_late_panel=True | material, late-panel offset | XPD_late_ex, L_pol, XPD_early_ex | material effect가 late/leakage 구조까지 바뀌는지 확인 | [GitHub][5] | PASS (LOS=0 12/12, include_late_panel=1) | PASS (material=3, dispersion=2) | PASS (XPD_late_ex=12/12, L_pol=12/12) | WARN (late/leakage 구조 관측은 구현; 1차(material primary) 주장에는 보조로만 사용) | **WARN** |
| A5_pair | corner stress scene, paired base/on, LOS blocked/off | rho, rep, stress_mode, scatterer_count | Delta L_pol, Delta rho_early, Delta XPD_late_ex, Delta DS | stress-isolation이 아니라 contamination/stress response 검증 | [GitHub][6] | PASS (LOS=0 60/60, base=30, stress=30) | PASS (stress_mode=2, scatterer_count=2) | WARN (ΔL_pol/Δrho/ΔXPD_late: full, ΔDS finite base/stress=30/30 & 29/30) | PASS (paired contamination/stress-response 검증 구성 완료) | **WARN** |
| A2_on | A2와 동일 geometry, LOS present | reflector 위치, 거리 | matched baseline 대비 ΔXPD_early_ex, Δrho_early, early energy fraction | odd mechanism의 LOS-under observability | [GitHub][2] | PASS (LOS=1 4/4) | PASS (d_m=2, cases=4) | PASS (Δmetric source rows=4 (XPD_early_ex/rho/early_fraction complete)) | PASS (odd mechanism LOS-under observability 구현) | **PASS** |
| Even-LOS-on bridge | A6_on 권장, A3_on exploratory | mode/geometry | ΔXPD_early_ex, L_pol, co/cross contrast | even mechanism이 LOS 아래에서 어떻게 묻히거나 나타나는지 확인 | [GitHub][3] | PASS (A6_on=100 (LOS+target 동시 링크 100/100), A3_on=12) | PASS (a6_mode=2, d_m=6) | PASS (ΔXPD_early_ex/L_pol finite=112/112) | PASS (A6_on primary bridge + A3_on exploratory bridge 동시 확보) | **PASS** |
| A4_on | material plane + LOS present | material, incidence, dispersion on/off | XPD_early_ex, L_pol, 필요시 XPD_target_raw | material effect의 LOS-under system relevance | [GitHub][5] | PASS (LOS=1 24/24) | PASS (material=3, dispersion=2) | PASS (XPD_early_ex/L_pol finite=24/24) | PASS (material effect LOS-under system relevance 구현) | **PASS** |
| B1 | practical LOS-dominant environment | measurement/grid positions | XPD_early_ex, rho_early, L_pol, DS, early energy fraction | good region / practical LOS baseline | [GitHub][1] | PASS (n=25, LOS1=24, LOS0=1) | PASS (grid rx_x=5, rx_y=5) | WARN (XPD/rho/L_pol full, DS=1/25, early_frac=1/25) | WARN (coverage-aware leverage/risk map 용도로 구현(absolute universal map 아님)) | **WARN** |
| B2 | partition/wall-blocked practical NLOS | measurement/grid positions | same as B1 | contamination region; B1 대비 성능 열화 방향 확인 | [GitHub][1] | PASS (n=25, LOS1=19, LOS0=6) | PASS (grid rx_x=5, rx_y=5) | WARN (XPD/rho/L_pol full, DS=6/25, early_frac=6/25) | PASS (coverage-aware leverage/risk map 용도로 구현(absolute universal map 아님)) | **WARN** |
| B3 | corner/high-EL practical NLOS | measurement/grid positions | same as B1 | high-risk region; leverage/risk map 공급 | [GitHub][1] | PASS (n=25, LOS1=20, LOS0=5) | PASS (grid rx_x=5, rx_y=5) | WARN (XPD/rho/L_pol full, DS=5/25, early_frac=5/25) | WARN (coverage-aware leverage/risk map 용도로 구현(absolute universal map 아님)) | **WARN** |

## PASS/WARN/FAIL 사유 요약

- `C0`: WARN / fixed=PASS, sweep=WARN, metric=PASS, objective=PASS
- `A2_off`: PASS / fixed=PASS, sweep=PASS, metric=PASS, objective=PASS
- `A6_off`: PASS / fixed=PASS, sweep=PASS, metric=PASS, objective=PASS
- `A3_supp`: WARN / fixed=PASS, sweep=PASS, metric=PASS, objective=WARN
- `A4_iso`: WARN / fixed=PASS, sweep=PASS, metric=PASS, objective=WARN
- `A4_bridge`: WARN / fixed=PASS, sweep=PASS, metric=PASS, objective=WARN
- `A5_pair`: WARN / fixed=PASS, sweep=PASS, metric=WARN, objective=PASS
- `A2_on`: PASS / fixed=PASS, sweep=PASS, metric=PASS, objective=PASS
- `Even-LOS-on bridge`: PASS / fixed=PASS, sweep=PASS, metric=PASS, objective=PASS
- `A4_on`: PASS / fixed=PASS, sweep=PASS, metric=PASS, objective=PASS
- `B1`: WARN / fixed=PASS, sweep=PASS, metric=WARN, objective=WARN
- `B2`: WARN / fixed=PASS, sweep=PASS, metric=WARN, objective=PASS
- `B3`: WARN / fixed=PASS, sweep=PASS, metric=WARN, objective=WARN

## 증명 데이터(한 폴더) 구성

### 보고서 원문
- `evidence/reports/diagnostic_report.md`
- `evidence/reports/intermediate_report.md`
- `evidence/reports/proposition_plot_mapping_report.md`
- `evidence/reports/proposition_plot_mapping_detailed_report.md`
- `evidence/reports/final_diagnostic_decision.md`

### 테이블/JSON 원본
- `evidence/tables/diagnostic_checks.json`
- `evidence/tables/case_level.csv`
- `evidence/tables/target_level.csv`
- `evidence/tables/sensitivity_level.csv`
- `evidence/tables/intermediate_link_rows.csv`
- `evidence/tables/intermediate_ray_rows.csv`
- `evidence/tables/A3_target_window_sign.csv`
- `evidence/tables/A6_case_set_sign_compare.csv`
- `evidence/tables/B_per_scenario_summary.csv`
- `evidence/tables/D3_hole_analysis.csv`
- `evidence/tables/config.all_scenarios_plus_bridges_a6on_20260309.json`

### 진단 산출물(본 문서 기준)
- `scenario_diagnosis_summary.csv` (정량 요약)
- `scenario_implementation_diagnosis_ko.md` (현재 문서)

## 해석 원칙

1. A6는 G2 core sign-off의 primary evidence, A3는 supplementary evidence
2. A4는 A4_iso(primary)와 A4_bridge(secondary)를 분리 보고
3. B1/B2/B3는 absolute universal map이 아니라 coverage-aware leverage map으로 해석

[1]: https://github.com/myoungsunk/codex_modelling/blob/feature/dualcp-proxy-bridge/analysis_report/out/tmp_c2_apply_20260301/diagnostic_report.md
[2]: https://github.com/myoungsunk/codex_modelling/blob/feature/dualcp-proxy-bridge/scenarios/A2_pec_plane.py
[3]: https://raw.githubusercontent.com/myoungsunk/codex_modelling/feature/dualcp-proxy-bridge/scenarios/A6_cp_parity_benchmark.py
[4]: https://github.com/myoungsunk/codex_modelling/blob/feature/dualcp-proxy-bridge/scenarios/A3_corner_2bounce.py
[5]: https://github.com/myoungsunk/codex_modelling/blob/feature/dualcp-proxy-bridge/scenarios/A4_dielectric_plane.py
[6]: https://github.com/myoungsunk/codex_modelling/blob/feature/dualcp-proxy-bridge/scenarios/A5_depol_stress.py
