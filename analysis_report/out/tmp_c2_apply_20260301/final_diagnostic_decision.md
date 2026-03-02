# Final Diagnostic Decision

- Branch: `feature/dualcp-proxy-bridge`
- Experiment tag: `tmp_c2_apply_20260301`
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
Calibration, effect size, and power-domain proxy modeling are validated for measurement start; remaining warnings are interpretation/role-definition items, not blocking physics failures.

## Promotion to Go requires
1. A3 manual review sign-off (`A3 pass=20/20` now; reviewer sign-off pending)
2. `LOS0_q3 = structural_hole` documented in D3 hole analysis
3. `qNA_total = 0` maintained as zero
4. `B_per_scenario_summary.csv` archived as frozen reporting table

## A. Geometry / Path Validity
**Status:** PASS (conditional)

### Evidence
- LOS-blocked scenarios A2/A3/A4/A5: los_rays=[0, 0, 0, 0]
- Target bounce existence: A2=27/27 (rate=1.00), A3=24/24 (rate=1.00)

### Interpretation
Intended physical interactions are generated correctly. A3 좌표/관통 검수는 자동 산출물만으로 완전 확정 불가하므로 수동 검토를 유지합니다.

### Remaining action
- [A3_geometry_manual_review.md](A3_geometry_manual_review.md) reviewer sign-off
- Scenario space plots: [scenario_space_plots.md](scenario_space_plots.md)

## B. Time Resolution / Delay Separation
**Status:** WARN

### Evidence
- dt_res=2.500e-10 s, tau_max=2.558e-07 s, Te=3.000e-09 s, Tmax=3.000e-08 s
- C0 floor window=PASS, A2 target-window sign=PASS, A3 target-window sign=FAIL
- A3 mechanism status=PASS, A3 system-early status=FAIL
- A5 target mode=contamination_response, W3 (room Te sweep) status=WARN (best S_xpd_early=0.891)

### Interpretation
A2는 odd early-anchor로 적합합니다. A3는 mechanism 검증에는 적합하지만 fixed system early-window baseline으로는 부적합합니다. A5는 stress-isolation이 아니라 contamination-response로 해석해야 합니다.

### Reporting rule
- A3: mechanism validation only
- A5: stress-response / contamination analysis
- A3를 fixed system early-window even baseline 증거로 사용하지 않음

## C. Effect Size vs Calibration Uncertainty
**Status:** PASS

### Evidence
- delta_ref=0.611 dB (floor_delta=0.611 dB, repeat_delta=0.223 dB)
- A3-A2 early delta_median=3.490 dB (ratio=5.714, C1=PASS)
- Material primary span=8.189 dB (PASS)
- Stress primary effect ΔL_pol=-5.853 dB (PASS, C2S=PASS)

### Interpretation
효과 크기가 보정 불확실도 대비 충분히 큽니다. floor 보정 이후에도 관측 가능한 차이를 유지할 가능성이 높습니다.

## D. Identifiability
**Status:** PASS

### Evidence
- D1-global=PASS (EL_iqr=7.705 dB)
- D1-local(A2)=PASS, D1-local(A5)=PASS role=stress_response
- D2(stage1)=PASS, D2(stage2)=PASS (overall=PASS)
- D3=PASS (qNA_total=0, selected_rows_n=0)
  - LOS0_q3: structural_hole (pool_n=0, selected_n=)

### Interpretation
회귀 식별성은 현재 단계에서 유효합니다. 잔여 이슈는 `LOS0_q3` 구조적 홀 문서화와 strata 해석 규칙 고정입니다.

### Reporting rule
- `LOS0_q3`는 sampling hole이 아닌 structural hole로 명시
- D3 평가는 all-theoretical strata가 아니라 viable strata 기준으로 보고

## E. Model-Form Consistency
**Status:** PASS

### Evidence
- Power-domain metrics only: XPD_early, XPD_late, rho_early, L_pol, DS, EL_proxy, LOSflag
- Complex phase 기반 의사결정 없음
- 검증 단위: Z 분포/상관/순위 (파형 매칭 아님)

### Interpretation
RT를 잠재 설명변수 생성기로 쓰고, 검증은 관측 가능한 power-domain 통계로 제한하는 목표와 구현이 일치합니다.

## Scenario-level Final Roles and Decisions

| Scenario | Final Role | Status | Use in paper | Exclude from |
|---|---|---:|---|---|
| C0 | floor calibration | PASS | calibration / uncertainty | effect regression |
| A2 | odd parity isolation | PASS | parity anchor | EL-identification |
| A3 | even mechanism validation | WARN | mechanism-only | system early baseline |
| A4 | material effect | PASS | material coefficient / target-window stats | none |
| A5 | stress response | PASS | stress-response / contamination analysis | stress-isolation claim |
| B1 | LOS real-space baseline | FAIL | leverage baseline | strict full-strata claim |
| B2 | partition NLOS | WARN | contamination / NLOS comparison | none |
| B3 | corner high-EL NLOS | PASS | high-EL / structural stress region | none |
| B-all | real-space identifiability set | PASS | stage1 EL fit / leverage map | strict full-strata claim |

## Final Measurement Readiness Decision

### Decision
**Conditional Go**

### Blocking issues
- None at physical-validity/effect-size level

### Non-blocking but mandatory documentation actions
1. Freeze A3 as mechanism-only scenario
2. Freeze A5 as contamination-response scenario
3. Document `LOS0_q3` as structural hole
4. Keep `qNA_total = 0` in frozen diagnostic output
5. Archive per-scenario summary (`B_per_scenario_summary.csv`)

### Measurement may proceed because
- Calibration floor is stable
- Odd/material/stress effects exceed reference uncertainty
- Stage1/Stage2 regression design is identifiable
- Remaining WARN items are interpretation/documentation scope

## Scenario Space Plots (All)

- Full gallery and links: [scenario_space_plots.md](scenario_space_plots.md)
- Case-level index: [index.md](index.md)
- [C0__ALL__scene_montage.png](figures/C0__ALL__scene_montage.png)
- [A2__ALL__scene_montage.png](figures/A2__ALL__scene_montage.png)
- [A3__ALL__scene_montage.png](figures/A3__ALL__scene_montage.png)
- [A4__ALL__scene_montage.png](figures/A4__ALL__scene_montage.png)
- [A5__ALL__scene_montage.png](figures/A5__ALL__scene_montage.png)
- [B1__ALL__scene_montage.png](figures/B1__ALL__scene_montage.png)
- [B2__ALL__scene_montage.png](figures/B2__ALL__scene_montage.png)
- [B3__ALL__scene_montage.png](figures/B3__ALL__scene_montage.png)

- Floor reference used: median=24.935 dB, delta=0.611 dB

