# 수정 후 검토 보고서 (v3 — 코드 병합 후 재검증)

> 날짜: 2026-02-26
> 대상: `feature/dualcp-proxy-bridge` 브랜치 병합 후, 13개 수정 항목 재검증
> 범위: 병합된 코드 전체 정독 기반

---

## 요약: 수정 상태 총괄

| # | 항목 | 판정 | 비고 |
|---|------|------|------|
| 1 | C0 거리/각도/반복 | **해결** | 5거리×3yaw×1pitch×5rep=75케이스, 안테나 회전 구현 |
| 2 | A3 표본 확대 | **해결** | 4→12 기하 조합 (3 offset × 4 rx) |
| 3 | LOS 차단 물리성 | **해결** | `--los-block-mode occluder`, `make_los_blocker_plane()` |
| 4 | A5 stress_mode 확장 | **해결** | none/synthetic/geometry/hybrid, scatterer 추가 |
| 5 | B LOSflag 재계산 | **해결** | ray table 기반 `los_flag_ray` 집계 |
| 6 | Student-t/Laplace 분포 | **해결** | `--dist-family` CLI, GOF 분포별 적응 |
| 7 | Huber IRLS 강건 회귀 | **해결** | `--robust-regression` 기본 True, MAD 스케일링 |
| 8 | bucket 최소 표본 | **해결** | `--min-bucket-n 3`, 미달 시 regression fallback |
| 9 | 주파수/서브밴드 floor | **해결** | `_build_floor_reference_with_curve()`, subband별 excess/caution |
| 10 | claim_caution 모드 | **해결** | `--claim-caution-mode {scaled,half_width,off}` |
| 11 | drift check (3-CSV) | **해결** | `load_measurement_dualcp_three_csv()`, `--max-drift-db` |
| 12 | LP baseline | **해결** | `run_cp_lp_baseline.py`, `lp_baseline_compare.py` |
| 13 | ANOVA/FDR/혼동진단 | **해결** | `_anova_oneway()`, `_fdr_bh()`, `distance_vs_losflag_rank_corr` |

**13건 전부 해결됨.** 다만 아래의 잔존 문제 및 설계 우려 사항이 있음.

---

## 1. 항목별 상세 검증

### 1-1. C0 보정: 거리 1-5m, yaw/pitch sweep, --n-rep

**코드 확인**:

- `C0_free_space.py:17-18` — `build_sweep_params()` → `[1.0, 2.0, 3.0, 4.0, 5.0]` ✅
- `C0_free_space.py:31-48` — `run_case()`가 `yaw_deg`, `pitch_deg`를 params에서 읽어 Rx 안테나 boresight/h_axis/v_axis에 회전 적용 (`_rot_z`, `_rot_y`) ✅
- `run_standard_sim.py:689` — `--n-rep` 기본값 5 ✅
- `run_standard_sim.py:462-463` — C0 기본: `dlist=[1.0,2.0,3.0,4.0,5.0]`, `yaw_list=[-10.0,0.0,10.0]` ✅
- `run_standard_sim.py:466-488` — 4중 루프: distance × yaw × pitch × rep ✅
- `run_standard_sim.py:476-481` — yaw_deg, pitch_deg가 params와 meta 양쪽에 기록 ✅

**기본 케이스 수**: 5d × 3yaw × 1pitch × 5rep = **75 케이스** — 계획서 요구 75+에 부합

**잔존 사항 (Minor)**: `--pitch-list` 기본값이 `"0"` (단일 값). 계획서의 pitch 민감도 분석을 위해서는 CLI에서 별도 지정 필요 (예: `--pitch-list "-5,0,5"`). `AngleSensitiveFloorXPD`의 `pitch_slope_db_per_deg` 보정 데이터가 기본 설정으로는 생성되지 않음.

---

### 1-2. A3 표본 확대

**코드 확인**:

- `A3_corner_2bounce.py:39-44` — `build_sweep_params()`:
  ```python
  for off in [3.0, 3.5, 4.0]:
      for rx_x, rx_y in [(3.0, 4.0), (3.5, 4.5), (4.0, 5.0), (4.5, 4.0)]:
  ```
  3 offset × 4 rx = **12 기하 조합** ✅ (기존 4개에서 3배 확장)
- `run_standard_sim.py:525-557` — n_rep 루프 적용 ✅

**기본 케이스 수**: 12 × 5rep = **60 데이터 포인트** — KS 검정에 적절한 표본 크기

---

### 1-3. --los-block-mode 물리적 차단 (occluder)

**코드 확인**:

- `scenarios/common.py:69-102` — `make_los_blocker_plane()`:
  - Tx-Rx 중간점에 유한 평면(흡수체 프록시) 배치 ✅
  - Normal = Tx→Rx 방향, u/v 축 자동 계산 ✅
  - `Material.dielectric(eps_r=1.15, tan_delta=1.0, name="absorber_proxy")` ✅
- `A2_pec_plane.py:55-56,62-63` — `los_blocker` 파라미터 수용, `make_los_blocker_plane()` 호출 ✅
- `A3_corner_2bounce.py:55-56,62-63` — 동일 패턴 ✅
- `A4_dielectric_plane.py:70-71,81-82` — 동일 패턴 ✅
- `A5_depol_stress.py:96-97,105-106` — 동일 패턴 ✅
- `run_standard_sim.py:692` — `--los-block-mode {synthetic,occluder}` 기본 "occluder" ✅

---

### 1-4. A5 stress_mode 확장

**코드 확인**:

- `A5_depol_stress.py:86-136` — `run_case()` 확장:
  - `stress_mode` (none/synthetic/geometry/hybrid) ✅
  - `scatterer_count` ✅
  - `los_blocker` ✅
  - mode="geometry"/"hybrid" → `_append_stress_scatterers()` 호출 (lines 103-104) ✅
  - mode="synthetic"/"hybrid" → `DepolConfig(rho_func=rho_hook)` 생성 (lines 108-120) ✅
- `A5_depol_stress.py:44-66` — `_append_stress_scatterers()`: 랜덤 소형 PEC 평면 추가 ✅
- `run_standard_sim.py:693-695` — CLI: `--a5-stress-mode hybrid`, `--a5-scatterer-count 3`, `--a5-max-cases 0` (전체) ✅
- `run_standard_sim.py:626-629` — meta에 `roughness_flag`, `human_flag`, `stress_mode` 기록 ✅

**설계 우려 (아래 잔존 문제 2-A 참조)**: `--stress-flag`가 실행 단위 전역. base vs stress 비교는 별도 실행 필요.

---

### 1-5. B LOSflag 재계산

**코드 확인**:

- `run_standard_sim.py:735` — `los_link = int(any(int(r.get("los_flag_ray", 0)) == 1 for r in ray_rows))` ✅
- `run_standard_sim.py:803` — `link_meta["LOSflag"] = int(los_link)` ✅
- `link_conditions.py:42-44` — meta에서 LOSflag 읽되, 없으면 ray 기반 fallback ✅
- `success_checks.py:263` — confounding 진단: `distance_vs_losflag_rank_corr` ✅

---

### 1-6. Student-t / Laplace 분포

**코드 확인**:

- `conditional_proxy.py:158` — `fit_proxy_model()` 파라미터: `dist_family: str = "normal"` ✅
- `conditional_proxy.py:228-247` — 분포별 분기:
  - `"student_t"`: `stats.t.fit()`, 분포별 KS 검정, `{"family":"student_t","df","loc","scale"}` 저장 ✅
  - `"laplace"`: `stats.laplace.fit()`, 분포별 KS 검정, `{"family":"laplace","loc","scale"}` 저장 ✅
  - default (normal): `stats.kstest(resid_std, "norm")` ✅
- `conditional_proxy.py:256` — `model["residual_dist"] = resid_dist` ✅
- `fit_dualcp_proxy.py:207` — `--dist-family` CLI, default `"student_t"` ✅
- `fit_dualcp_proxy.py:85-97` — `_sample_resid()` 분포별 샘플링 ✅

---

### 1-7. Huber IRLS 강건 회귀

**코드 확인**:

- `conditional_proxy.py:40-45` — `_weighted_lstsq()` 가중 최소제곱 헬퍼 ✅
- `conditional_proxy.py:48-108` — `_fit_regression()`:
  - `robust=False, huber_delta=1.5, max_iter=30` 파라미터 ✅
  - 초기 OLS → robust=True 시 반복:
    - 잔차 계산 → MAD 스케일링 → Huber 가중치 (`u <= 1.0 → w=1.0, else w=1/u`) ✅
    - 수렴 검사: `np.linalg.norm(beta_new - beta) <= 1e-8 * (1.0 + np.linalg.norm(beta))` ✅
- `conditional_proxy.py:196` — `fit_proxy_model()` → `_fit_regression(..., robust=bool(robust_regression))` ✅
- `fit_dualcp_proxy.py:208-211` — `--robust-regression` / `--no-robust-regression`, default True ✅

---

### 1-8. bucket 최소 표본

**코드 확인**:

- `conditional_proxy.py:160` — `fit_proxy_model(..., min_bucket_n: int = 3)` ✅
- `conditional_proxy.py:209` — model에 `"bucket_min_n": int(max(1, min_bucket_n))` 저장 ✅
- `conditional_proxy.py:117` — `predict_distribution()`에서 `min_bucket_n = int(model.get("bucket_min_n", 1))` 읽음 ✅
- `conditional_proxy.py:123` — `n_b >= min_bucket_n` 체크 → 미달 시 regression fallback ✅
- `fit_dualcp_proxy.py:210` — `--min-bucket-n 3` ✅

---

### 1-9. 주파수/서브밴드 floor 보존

**코드 확인**:

- `run_standard_sim.py:223-249` — `_build_floor_subbands()`: subband별 floor/uncert 계산 ✅
- `run_standard_sim.py:252-303` — `_build_floor_reference_with_curve()`:
  - `frequency_hz`, `xpd_floor_db` (주파수별 벡터) ✅
  - `xpd_floor_uncert_db`, `xpd_floor_p_lo_db`, `xpd_floor_p_hi_db` ✅
  - `subbands` 리스트 (index, f_lo_hz, f_hi_hz, xpd_floor_db, uncert) ✅
- `run_standard_sim.py:871-907` — 번들별 extras에 저장:
  - Per-freq: `xpd_floor_curve_db`, `XPD_early_excess_curve_db`, `claim_caution_early_curve` ✅
  - Per-subband: `xpd_floor_subband_db`, `XPD_early_excess_subband_db`, `claim_caution_early_subband` ✅

**참고**: `dualcp_metrics.py:_resolve_floor_band()`(측정 경로)은 여전히 스칼라 압축하지만, 시뮬레이션 경로에서 주파수별/서브밴드별 상세 데이터를 extras에 보존하므로, 서브밴드 분석이 가능함.

---

### 1-10. claim_caution 모드

**코드 확인**:

- `run_standard_sim.py:358-367` — `_claim_caution_flag()`:
  - `mode="off"` → 항상 False ✅
  - `mode="scaled"` → threshold = |uncert| × scale ✅
  - `mode="half_width"` → threshold = |uncert| (scale 무시) ✅
  - 판정: `|excess| ≤ threshold` ✅
- `run_standard_sim.py:684-685` — `--claim-caution-mode {scaled,half_width,off}`, `--claim-caution-scale 1.0` ✅
- 스칼라/per-freq/per-subband 수준에서 모두 적용 (lines 860-907) ✅

---

### 1-11. Drift check (3-CSV)

**코드 확인**:

- `measurement_compare.py:209-255` — `load_measurement_dualcp_three_csv()`:
  - co_pre/cross/co_post 3개 CSV 로드 ✅
  - drift 계산: `20*log10(|co_post|/|co_pre|)` ✅
  - meta에 `drift_co_db` (median), `drift_co_p95_db` (95th pctl) 저장 ✅
- `dualcp_calibrate_floor.py:44-66` — `_load_cases_from_csv_pairs()`:
  - `co_post_csv` 파라미터 수용 ✅
  - 존재 시 `load_measurement_dualcp_three_csv()` 호출 ✅
- `dualcp_calibrate_floor.py:147` — `--max-drift-db` CLI ✅
- `dualcp_calibrate_floor.py:163-183` — drift 필터링 + summary 기록 ✅

---

### 1-12. LP baseline 스크립트

**코드 확인**:

- `scripts/run_cp_lp_baseline.py` (79 lines):
  - `--basis circular` + `--basis linear` 두 번 `run_standard_sim.py` 호출 ✅
  - `compare_cp_lp_metrics()` 호출하여 비교 리포트 생성 ✅
- `analysis/lp_baseline_compare.py` (146 lines):
  - CP/LP 쌍을 (scenario_id, case_id)로 매칭 ✅
  - 8개 주요 지표에 대해 delta(CP-LP) 계산 ✅
  - 페어 CSV + 마크다운 리포트 출력 ✅

---

### 1-13. ANOVA / FDR / 혼동 진단

**코드 확인**:

- `success_checks.py:82-108` — `_anova_oneway()`:
  - `scipy.stats.f_oneway()` + eta² 효과 크기 ✅
  - 그룹별 SS_between, SS_total 계산 ✅
- `success_checks.py:111-129` — `_fdr_bh()`:
  - Benjamini-Hochberg FDR 보정 ✅
  - NaN-safe, 순위 기반 조정 ✅
- `success_checks.py:143-144` — C0 floor 검증에 ANOVA 적용:
  - `anova_distance_p`, `anova_distance_eta2` ✅
  - `anova_yaw_p`, `anova_yaw_eta2` ✅
- `success_checks.py:263` — B 공간 일관성에 confounding 진단:
  - `distance_vs_losflag_rank_corr` ✅
- `success_checks.py:271-295` — `evaluate_success_criteria()`:
  - 4개 p-value 수집 → FDR-BH 일괄 보정 ✅
  - `multiple_testing` 섹션 출력 ✅
- `fit_dualcp_proxy.py:100-119,342-349` — proxy 보고서에도 FDR-BH 적용 ✅

---

## 2. 잔존 문제 및 설계 우려

### 2-A. A5 stress_flag는 실행 단위 전역 (설계 우려, Major)

`run_standard_sim.py:600` — `stress_on = bool(args.stress_flag)` → 한 실행 내 모든 A5 케이스가 동일한 stress on/off.

**결과**:
- base vs stress 비교를 위해 **2번 실행** 필요 (한 번은 `--stress-flag` 없이, 한 번은 있이)
- `success_checks.py`의 `check_A4_A5_breaking()`이 정상 동작하려면, 두 실행의 CSV를 **수동 병합** 해야 함
- 자동화 파이프라인에서는 이 2-pass 구조를 명시적으로 처리해야 함

**권고**: 파이프라인 문서에 A5 2-pass 실행 절차를 명시하거나, A5 러너에서 `--stress-flag`가 없을 때 자동으로 base+stress 양쪽을 생성하는 옵션 추가 검토.

---

### 2-B. pitch 기본값이 단일 (Minor)

`--pitch-list` 기본값 = `"0"`. 계획서의 "pitch 민감도" 분석을 위해서는 CLI에서 별도 지정 필요.
`AngleSensitiveFloorXPD.pitch_slope_db_per_deg` 보정 데이터가 기본 설정으로 생성되지 않음.

**권고**: 계획서에서 pitch 분석을 명시적으로 포함한다면, 기본값을 `"-5,0,5"`로 변경하거나, 실행 지침서에 `--pitch-list` 인자 사용을 명기.

---

### 2-C. B0_room_box.py와 run_standard_sim.py의 부분 분리 (Minor)

`run_standard_sim.py`의 `_run_room_case()` (lines 407-451)가 `B0_room_box.run_case()`를 **사용하지 않고** 직접 장면을 구성:
- B2/B3 장애물은 러너에 하드코딩
- `B0_room_box.py`의 `run_case()`는 `los_blocker` 파라미터를 받지 않음

이로 인해 `B0_room_box.py` 수정이 B1/B2/B3 시뮬레이션에 반영되지 않는 유지보수 위험.

---

### 2-D. A5 scatterer 좌표축 직교성 (Minor)

`A5_depol_stress.py:61-62`:
```python
u_axis=np.array([0.0, 0.0, 1.0], dtype=float),
v_axis=np.array([-n[1], n[0], 0.0], dtype=float),
```
scatterer의 normal `n`이 z 성분을 포함(`0.2*(rng.random()-0.5)`)하므로, `u_axis=[0,0,1]`이 normal에 직교하지 않을 수 있음. ray tracer가 u/v 축의 직교성을 전제한다면 미세한 기하 오차 발생 가능.

**영향**: 실용적으로 z 성분이 매우 작아(±0.1) 큰 문제는 아니나, 엄밀히는 Gram-Schmidt 정규화 필요.

---

### 2-E. dualcp_metrics.py 측정 경로의 스칼라 floor 압축 (Minor, 기존 이슈 잔존)

`dualcp_metrics.py:131-159`의 `_resolve_floor_band()`가 주파수별 floor를 median 스칼라로 압축. 시뮬레이션 경로에서는 extras에 주파수별 상세 데이터가 보존되므로 **분석에는 지장 없으나**, 측정 데이터를 직접 처리할 때는 서브밴드 정보가 손실됨.

**권고**: 측정 경로 분석 시 `dualcp_calibrate_floor.py`의 서브밴드별 floor을 명시적으로 사용하도록 문서화.

---

### 2-F. B grid에서 거리-LOSflag 상관 가능성 (기존 이슈 부분 잔존)

B2/B3 시나리오에서 장애물이 특정 x 좌표에 배치되므로, Tx에서 먼 Rx일수록 NLOS가 될 확률이 높음 → d_m과 LOSflag의 상관.

`success_checks.py:263`의 `distance_vs_losflag_rank_corr`가 이를 **진단**하므로, 상관이 높으면 회귀 결과 해석 시 주의 필요. 진단 도구는 구현되었으나, 상관 완화 설계(distance-matched LOS/NLOS 쌍)는 미구현.

**권고**: 보고서에서 confounding rank correlation 값을 명시적으로 보고하고, |r|>0.5이면 회귀 계수 해석에 caveat 추가.

---

## 3. 검증 질문(Q1-Q3) 준비 상태 업데이트

| 질문 | 데이터 충분성 | 코드 지원 | 주요 Gap |
|------|-------------|-----------|---------|
| Q1-1 (Floor 기준선) | **충분** — 75+ 케이스 | `dualcp_calibration.py` + floor reference | — |
| Q1-2 (정렬 민감도) | **충분** (yaw) / **부족** (pitch) | `AngleSensitiveFloorXPD` + ANOVA | pitch 기본값 단일 |
| Q2-1 (Odd leakage) | **충분** — A2 9×5rep | 전체 파이프라인 | — |
| Q2-2 (Even 회복) | **충분** — A3 12×5rep=60 | KS 검정 + 분포 비교 | — |
| Q2-3 (재질 효과) | **충분** — A4 재질별 sweep | Cliff's delta 가능 | — |
| Q2-4 (Depol stress) | **가능** — 2-pass 실행 필요 | Student-t/Laplace + 검증 로직 | A5 2-pass 수동 병합 |
| Q3-1 (Early/Late 구분) | **충분** | L_pol 계산 | — |
| Q3-2 (Z↔DS 연결) | **충분** | Spearman + 조건별 분석 | — |
| Q3-3 (Early energy) | **충분** | early_energy_fraction | — |

---

## 4. 최종 우선순위 권고

### 실행 전 필수 (파이프라인 무결성)

1. **A5 2-pass 실행 절차 문서화** (2-A): `--stress-flag` 없이 + 있이 2회 실행 → CSV 병합 → `check_A4_A5_breaking()` 정상 동작

### 권장 (정확도 향상)

2. **pitch 기본값 확장** (2-B): Q1-2 pitch 분석이 필요하면 `--pitch-list "-5,0,5"` 추가
3. **B grid confounding 보고** (2-F): rank correlation이 높으면 caveat 명시

### 중기 (유지보수)

4. **B0 모듈과 러너 통합** (2-C): `_run_room_case()`를 `B0_room_box.run_case()` 기반으로 리팩터링
5. **A5 scatterer 좌표축 정규화** (2-D): Gram-Schmidt 적용

---

## 5. 결론

1차 검토에서 지적한 **5개 치명적 문제, 7개 주요 문제, 4개 경미한 문제** 중:

- **치명적 5건**: 전부 해결 (C0 확장, A5 stress_mode, 분포 확장, Room NLOS, floor 서브밴드)
- **주요 7건**: 전부 해결 (A3 확장, Huber, min_bucket, LP baseline, A5 플래그, LOS 차단, claim_caution)
- **경미 4건**: 전부 해결 (ANOVA, FDR, drift check, sensitivity)

잔존 사항은 모두 **설계 선택** 또는 **Minor** 수준으로, 파이프라인의 핵심 기능에 영향을 주지 않음. A5 2-pass 실행 절차만 문서화하면 전체 Q1-Q3 검증이 실행 가능한 상태.

---

## 6. 핵심 파일 참조 (병합 후)

| 파일 | 역할 | 상태 |
|------|------|------|
| `scenarios/C0_free_space.py` | 보정 시나리오 | ✅ 5거리, yaw/pitch 회전 |
| `scenarios/A3_corner_2bounce.py` | Even-bounce | ✅ 12 기하 조합 |
| `scenarios/A5_depol_stress.py` | Depol stress | ✅ 4 stress_mode, scatterer |
| `scenarios/common.py` | 공유 헬퍼 | ✅ `make_los_blocker_plane()` |
| `scenarios/A2_pec_plane.py` | Odd-bounce | ✅ los_blocker 지원 |
| `scenarios/A4_dielectric_plane.py` | 재질 sweep | ✅ los_blocker 지원 |
| `scenarios/B0_room_box.py` | Room grid | ⚠️ 러너와 부분 분리 |
| `scripts/run_standard_sim.py` | 시뮬레이션 러너 | ✅ 전체 CLI 확장 완료 |
| `scripts/fit_dualcp_proxy.py` | 프록시 모델 피팅 | ✅ dist_family, robust, FDR |
| `scripts/run_cp_lp_baseline.py` | LP baseline | ✅ 신규 생성 |
| `scripts/dualcp_calibrate_floor.py` | Floor 보정 | ✅ 3-CSV, drift 필터 |
| `analysis/conditional_proxy.py` | 조건부 모델 | ✅ Student-t, Huber, min_n |
| `analysis/success_checks.py` | 검증 로직 | ✅ ANOVA, FDR, confounding |
| `analysis/measurement_compare.py` | 측정 비교 | ✅ 3-CSV drift 지원 |
| `analysis/lp_baseline_compare.py` | CP/LP 비교 | ✅ 신규 생성 |
| `analysis/dualcp_metrics.py` | 지표 계산 | ⚠️ 측정 경로 scalar 압축 유지 |
| `analysis/link_conditions.py` | 조건 변수 | ✅ A5 플래그 연동 |
| `calibration/floor_model.py` | Floor 모델 클래스 | ✅ 변경 불요 |
