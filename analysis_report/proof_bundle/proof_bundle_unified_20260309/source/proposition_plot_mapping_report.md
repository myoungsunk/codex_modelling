# Proposition Mapping Report (diag_protocol_repro_v1_20260309_r1)

명제-실험-데이터-플롯 매칭과 PASS/FAIL 결과를 정리한 표입니다.

## Summary
- PASS: 11/12
- PARTIAL: 1/12
- FAIL: 0/12
- Plot missing propositions: 0/12

## Proposition Table

| 명제 | 실험 | 필요한 데이터 | 핵심 플롯/검정 | 통과 기준 | PASS/FAIL | 플롯 준비 |
| --- | --- | --- | --- | --- | --- | --- |
| M1 | C0 거리x각도 sweep | co/cross PDP, XPD(floor) | XPD_floor vs 거리/각도 + box/CDF | 거리 의존 약함 + 각도 민감도 | PASS | READY |
| M2 | C0 + 전체 시나리오 | XPD_excess, Delta_floor | inside/outside floor band ratio | 채널 주장 구간/불확실 구간 분리 | PASS | READY |
| G1 | A2 odd target-window sign | target-window raw sign, co/cross PDP | A2 target-window raw sign + PDP/tap sanity | raw target-window sign<0 (A2) | PASS | READY |
| G2 | A6 core theorem (odd/even) raw sign | target-window raw sign (A6 target_n=1,2) | A6 odd/even target-window raw sign | A6 odd<0 & even>0 with raw hit-rate PASS | PASS | READY |
| G3 | A2/A3 + A4/A5 | XPD_ex 분산/꼬리 | 조건별 CDF + 분산 비교 | 완전분리 아님 + 조건별 변동 | PASS | READY |
| L1 | A2-A5 + room | XPD_early_ex, XPD_late_ex, L_pol | L_pol box + early-late scatter | 기본 L_pol>0, stress 예외 | PASS | READY |
| L2 | A4_iso(primary) + A4_bridge(secondary) / A5 stress | A4 branch label(include_late_panel, dispersion) + stress label + XPD_ex | A4_iso/A4_bridge material CDF + A5 stress CDF | primary는 A4_iso로 판정, A4_bridge는 보조(지연/dispersion) 증거 | PASS | READY |
| L3 | 통제+room + EL proxy | EL, XPD_ex | scatter + Spearman | EL 증가에 따라 XPD/XPR 감소 단조 | PASS | READY |
| R1 | room grid LOS/NLOS (coverage-aware leverage) | Z 맵(XPD_ex, rho, L_pol, DS) + viable strata support | heatmap + LOS/NLOS CDF | 공간 분포 + LOS/NLOS 차이(viable strata 기준, universal claim 금지) | PASS | READY |
| R2 | room grid + 지표 연결 (coverage-aware leverage) | Z와 DS/early집중도 + viable strata support | DS vs XPD_ex, DS vs rho | 유효조건 영역 분리(coverage-aware map) | PASS | READY |
| P1 | 전체 데이터(통제+room) | Z, U(EL/material/late) | 조건부모델 vs 상수모델 | 조건부 모델 예측 우수 | PASS | READY |
| P2 | 최소세트 subsampling | 부분 샘플 반복 | 계수 부호 안정성 | 표본 축소에도 결론 유지 | PARTIAL | READY |

## Plot Files

### M1
- Found:
  - [figures/C0__ALL__xpd_floor_vs_distance.png](figures/C0__ALL__xpd_floor_vs_distance.png)
  - [figures/C0__ALL__xpd_floor_vs_yaw.png](figures/C0__ALL__xpd_floor_vs_yaw.png)
  - [figures/C0__ALL__xpd_floor_cdf.png](figures/C0__ALL__xpd_floor_cdf.png)
  - [figures/M1__C0_xpd_floor_box_by_yaw.png](figures/M1__C0_xpd_floor_box_by_yaw.png)

### M2
- Found:
  - [figures/M2__inside_outside_ratio.png](figures/M2__inside_outside_ratio.png)

### G1
- Found:
  - [figures/G1__A2_target_window_raw_sign_summary.png](figures/G1__A2_target_window_raw_sign_summary.png)
  - [figures/G1__A2_case0__pdp_overlay.png](figures/G1__A2_case0__pdp_overlay.png)
  - [figures/G1__A2_case0__tap_xpd_tau.png](figures/G1__A2_case0__tap_xpd_tau.png)
- Notes: target(raw) median=-8.000 dB, hit_rate_raw=1.000, PASS>=0.8/WARN>=0.6

### G2
- Found:
  - [figures/G2__A6_target_window_raw_sign_summary.png](figures/G2__A6_target_window_raw_sign_summary.png)
- Notes: A6 odd median=-8.000 dB (hit=1.000), even median=8.000 dB (hit=1.000), min_hit_raw=1.000, PASS>=0.8/WARN>=0.6

### G3
- Found:
  - [figures/G3__xpd_early_ex__scenario_cdf.png](figures/G3__xpd_early_ex__scenario_cdf.png)
  - [figures/G3__xpd_early_ex__std_bar.png](figures/G3__xpd_early_ex__std_bar.png)
- Notes: median_span=7.570 dB, delta_floor=0.724 dB, iqr_overlap=True

### L1
- Found:
  - [figures/ALL__early_late_ex_box.png](figures/ALL__early_late_ex_box.png)
  - [figures/L1__early_vs_late_ex_scatter.png](figures/L1__early_vs_late_ex_scatter.png)

### L2
- Found:
  - [figures/L2__A4_iso_material_xpd_early_ex_cdf.png](figures/L2__A4_iso_material_xpd_early_ex_cdf.png)
  - [figures/L2__A4_bridge_material_xpd_early_ex_cdf.png](figures/L2__A4_bridge_material_xpd_early_ex_cdf.png)
  - [figures/L2__A5_base_vs_stress_lpol_cdf.png](figures/L2__A5_base_vs_stress_lpol_cdf.png)
- Notes: A4_iso_n=12, A4_bridge_n=12, A4_bridge_dispersion_on_n=6

### L3
- Found:
  - [figures/ALL__xpd_early_ex_vs_el_proxy.png](figures/ALL__xpd_early_ex_vs_el_proxy.png)

### R1
- Found:
  - [figures/B__ALL__heatmap_xpd_early_ex.png](figures/B__ALL__heatmap_xpd_early_ex.png)
  - [figures/B__ALL__heatmap_lpol.png](figures/B__ALL__heatmap_lpol.png)
  - [figures/B__ALL__los_nlos_xpd_ex_cdf.png](figures/B__ALL__los_nlos_xpd_ex_cdf.png)

### R2
- Found:
  - [figures/ALL__ds_vs_xpd_early_ex.png](figures/ALL__ds_vs_xpd_early_ex.png)
  - [figures/R2__ds_vs_rho_early_db.png](figures/R2__ds_vs_rho_early_db.png)

### P1
- Found:
  - [figures/P1__model_vs_constant_metrics.png](figures/P1__model_vs_constant_metrics.png)

### P2
- Found:
  - [figures/P2__subsampling_sign_keep_rate.png](figures/P2__subsampling_sign_keep_rate.png)
