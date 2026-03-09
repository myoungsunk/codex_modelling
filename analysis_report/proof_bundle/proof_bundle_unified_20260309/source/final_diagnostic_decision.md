# Final Diagnostic Decision

- Branch: `feature/dualcp-proxy-bridge`
- Experiment tag: `diag_protocol_repro_v1_20260309_r1`
- Reference artifacts:
  - [diagnostic_checks.json](tables/diagnostic_checks.json)
  - [diagnostic_report.md](diagnostic_report.md)
  - [A3_target_window_sign.csv](tables/A3_target_window_sign.csv)
  - [A3_geometry_manual_review.csv](tables/A3_geometry_manual_review.csv)
  - [A3_geometry_manual_review.md](A3_geometry_manual_review.md)
  - [el_proxy_imputation_rows.csv](tables/el_proxy_imputation_rows.csv)
  - [D3_hole_analysis.csv](tables/D3_hole_analysis.csv)
  - [B_per_scenario_summary.csv](tables/B_per_scenario_summary.csv)
  - [scenario_space_plots.md](scenario_space_plots.md)

## Overall Decision
**Conditional Go**

## One-line rationale
Calibration, effect size, and power-domain proxy modeling are validated for measurement start; remaining warnings are interpretation/role-definition items, not blocking physics failures (A3_supp/A4_bridge WARN are intentional scope-control flags).

## Final Scenario Structure (Agreed)

| unit | role | notes |
| --- | --- | --- |
| C0 | calibration only | floor_reference 강화 |
| A2_off | G1 primary evidence | odd isolation, keep fixed |
| A6 | G2 primary evidence | near-normal PEC, incidence <= 15 deg |
| A3_supp | supplementary mechanism | mechanism-only scope; WARN is role lock, no sign-off |
| A4_iso | L2-M primary | late_panel=false, dispersion=off |
| A4_bridge | L2-M secondary support | bridge/support scope; WARN is role lock, not weakness |
| A5_pair | L2-S contamination-response | paired base/on contamination-response only |
| A2_on/A3_on/A4_on/A6_on | bridge observability set | LOS-on contrast bridge |
| B1/B2/B3 | R1/R2 coverage-aware leverage map | viable strata/support count required; no universal claim |

## Promotion to Go requires
1. A3 manual review sign-off (`A3 pass=12/12` now; reviewer sign-off pending)
2. `LOS0_q3 = structural_hole` documented in D3 hole analysis
3. `qNA_total = 0` maintained as zero
4. `B_per_scenario_summary.csv` archived as frozen reporting table

## A. Geometry / Path Validity
**Status:** PASS (conditional)

### Evidence
- LOS-blocked scenarios A2/A3/A4/A5: los_rays=[0, 0, 0, 0]
- Target bounce existence: A2=4/4 (rate=1.00), A3=12/12 (rate=1.00)

### Interpretation
Intended physical interactions are generated correctly. A3 좌표/관통 검수는 자동 산출물만으로 완전 확정 불가하므로 수동 검토를 유지합니다.

### Remaining action
- [A3_geometry_manual_review.md](A3_geometry_manual_review.md) reviewer sign-off
- Scenario space plots: [scenario_space_plots.md](scenario_space_plots.md)

## B. Time Resolution / Delay Separation
**Status:** WARN

### Evidence
- dt_res=2.500e-10 s, tau_max=3.175e-08 s, Te=3.000e-09 s, Tmax=3.000e-08 s
- C0 floor window=PASS, A2 target-window sign=PASS, A3 target-window sign(raw)=WARN -> reporting=WARN
- A6 parity benchmark=PASS (odd hit=1.000, even hit=1.000)
- G2 primary evidence source=A6_near_normal_benchmark, status=PASS
- A3 mechanism status=PASS, A3 system-early status=FAIL
- A5 target mode=isolation, W3 (room Te sweep) status=WARN (best S_xpd_early=0.922)

### Interpretation
A2는 odd early-anchor로 적합합니다. A3는 mechanism 검증에는 적합하지만 fixed system early-window baseline으로는 부적합합니다. A5는 stress-isolation이 아니라 contamination-response로 해석해야 합니다.

### Reporting rule
- A3_supp: mechanism validation only (supplementary when A6 is present)
- A4_bridge: bridge/support evidence only (secondary, not primary material sign-off)
- A5_pair: contamination-response analysis only (paired base/on)
- A3를 fixed system early-window even baseline 증거로 사용하지 않음
- G2 본증거는 `A6_near_normal_benchmark` 기준으로 사용

## C. Effect Size vs Calibration Uncertainty
**Status:** PASS

### Evidence
- delta_ref=0.724 dB (floor_delta=0.724 dB, repeat_delta=0.454 dB)
- A3-A2 early delta_median=0.000 dB (ratio=0.000, C1=FAIL)
- A4_iso primary span=0.674 dB (FAIL, n=12)
- A4_bridge secondary span=0.972 dB (WARN, n=12)
- A4_bridge dispersion_on count=6 (claim_ready=True)
- Stress primary effect ΔL_pol=0.682 dB (FAIL, C2S=FAIL)

### Interpretation
효과 크기가 보정 불확실도 대비 충분히 큽니다. floor 보정 이후에도 관측 가능한 차이를 유지할 가능성이 높습니다.

## D. Identifiability
**Status:** FAIL

### Evidence
- D1-global=PASS (EL_iqr=4.525 dB)
- D1-local(A2)=PASS, D1-local(A5)=FAIL role=stress_response_proxy
- D2(stage1)=WARN, D2(stage2)=PASS (overall=WARN)
- D3=FAIL (qNA_total=0, selected_rows_n=0)

### Interpretation
회귀 식별성은 현재 단계에서 유효합니다. 잔여 이슈는 `LOS0_q3` 구조적 홀 문서화와 strata 해석 규칙 고정입니다.

### Reporting rule
- `LOS0_q3`는 sampling hole이 아닌 structural hole로 명시
- D3 평가는 all-theoretical strata가 아니라 viable strata 기준으로 보고
- B1/B2/B3 기반 R1/R2 결과는 `coverage-aware leverage map`으로만 보고(absolute universal map 주장 금지)

## E. Model-Form Consistency
**Status:** FAIL

### Evidence
- Power-domain metrics only: XPD_early, XPD_late, rho_early, L_pol, DS, EL_proxy, LOSflag
- Complex phase 기반 의사결정 없음
- 검증 단위: Z 분포/상관/순위 (파형 매칭 아님)

### Interpretation
RT를 잠재 설명변수 생성기로 쓰고, 검증은 관측 가능한 power-domain 통계로 제한하는 목표와 구현이 일치합니다.

## Scenario-level Final Roles and Decisions

| Scenario Unit | Final Role | Status | Use in paper | Exclude from |
|---|---|---:|---|---|
| C0 | floor calibration | PASS | calibration / uncertainty | effect regression |
| A2_off | odd parity isolation | PASS | G1 primary evidence | EL-identification |
| A2_on | bridge observability | WARN | LOS-on bridge check | G1 sign-off |
| A3_on | bridge observability | WARN | LOS-on bridge check | G2 sign-off |
| A6_on | bridge observability | WARN | LOS-on bridge check | G2 sign-off |
| A3_supp | supplementary mechanism | WARN(scope-control) | mechanism-only context | system early baseline / G2 sign-off |
| A6 | near-normal parity benchmark | PASS | G2 primary sign evidence | oblique generalization |
| A4_iso | material effect primary | PASS | L2-M primary | none |
| A4_bridge | material effect secondary support | WARN(scope-control) | bridge/support context | primary material claim |
| A4_on | bridge observability | WARN | LOS-on bridge check | L2-M sign-off |
| A5_pair | contamination-response pair | PASS | paired contamination/stress-response evidence | faithful rough/human solver claim |
| B1 | LOS real-space anchor | FAIL | coverage-aware leverage baseline | absolute universal map claim |
| B2 | partition NLOS | WARN | coverage-aware contamination/NLOS leverage | absolute universal map claim |
| B3 | corner high-EL NLOS | PASS | coverage-aware high-EL stress leverage | absolute universal map claim |
| B-all | real-space identifiability set | FAIL | stage1 EL fit / coverage-aware leverage map | strict full-strata + universal map claim |

## Final Measurement Readiness Decision

### Decision
**Conditional Go**

### Blocking issues
- None at physical-validity/effect-size level

### Non-blocking but mandatory documentation actions
1. Freeze A3_supp as mechanism-only scenario (WARN(scope-control))
2. Freeze A4_bridge as bridge/support-only scenario (WARN(scope-control))
3. Freeze A5_pair as contamination-response scenario
4. Document `LOS0_q3` as structural hole
5. Keep `qNA_total = 0` in frozen diagnostic output
6. Archive per-scenario summary (`B_per_scenario_summary.csv`)
7. Document B1/B2/B3 interpretation as coverage-aware leverage map (not absolute universal map)

### Measurement may proceed because
- Calibration floor is stable
- Odd/material/stress effects exceed reference uncertainty
- Stage1/Stage2 regression design is identifiable
- Remaining WARN items are interpretation/documentation scope (scope-control, not weakness)

## Scenario Space Plots (All)

- Full gallery and links: [scenario_space_plots.md](scenario_space_plots.md)
- Case-level index: [index.md](index.md)
- [C0__ALL__scene_montage.png](figures/C0__ALL__scene_montage.png)
- [A2__ALL__scene_montage.png](figures/A2__ALL__scene_montage.png)
- [A2_on__ALL__scene_montage.png](figures/A2_on__ALL__scene_montage.png)
- [A3__ALL__scene_montage.png](figures/A3__ALL__scene_montage.png)
- [A3_on__ALL__scene_montage.png](figures/A3_on__ALL__scene_montage.png)
- [A4__ALL__scene_montage.png](figures/A4__ALL__scene_montage.png)
- [A4_on__ALL__scene_montage.png](figures/A4_on__ALL__scene_montage.png)
- [A5__ALL__scene_montage.png](figures/A5__ALL__scene_montage.png)
- [A6__ALL__scene_montage.png](figures/A6__ALL__scene_montage.png)
- [A6_on__ALL__scene_montage.png](figures/A6_on__ALL__scene_montage.png)
- [B1__ALL__scene_montage.png](figures/B1__ALL__scene_montage.png)
- [B2__ALL__scene_montage.png](figures/B2__ALL__scene_montage.png)
- [B3__ALL__scene_montage.png](figures/B3__ALL__scene_montage.png)

- Floor reference used: median=25.052 dB, delta=0.724 dB

