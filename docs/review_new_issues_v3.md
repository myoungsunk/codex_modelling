# 신규 발견 문제 보고서 (v3 — 병합 후 코드에서 새로 발견된 버그/설계 결함)

> 날짜: 2026-02-26
> 범위: `feature/dualcp-proxy-bridge` 병합 후 코드에서 기존 13개 수정 항목과 무관하게 **새로 발견된** 문제
> 방법: 코드 정독 + 필드명 추적 + 통계 방법론 검증

---

## 요약

| # | 심각도 | 제목 | 영향 범위 |
|---|--------|------|-----------|
| N-1 | **치명적** | `fit_dualcp_proxy.py` z_keys 대소문자 불일치 → 3/4 변수 실패 | 프록시 모델 전체 |
| N-2 | **치명적** | `fit_dualcp_proxy.py` u_keys 이름 불일치 → 9/10 조건변수 실패 | 조건부 모델 전체 |
| N-3 | **주요** | 스칼라/곡선 floor uncertainty 2배 불일치 (claim_caution) | 보정 판정 체계 |
| N-4 | **주요** | 광대역 스칼라 XPD에서 주파수별 floor 곡선 빼기 → 차원 불일치 | excess 곡선 계산 |
| N-5 | **주요** | `_dualcp_tap_powers` 비대칭 power floor → late-window XPD 편향 | dualcp_metrics 지표 |
| N-6 | **주요** | `measurement_compare.py`의 XPD 공식이 다른 모듈과 불일치 | 측정-RT 비교 |
| N-7 | **경미** | `_fdr_bh`가 총 검정 수 대신 유효 p-value 수 사용 → FDR 과소보정 | 다중검정 보정 |
| N-8 | **경미** | `--nf` 기본값 128 vs 나머지 코드의 256 불일치 | 주파수 해상도 |
| N-9 | **경미** | `rng` 변수가 lambda 정의 후에 생성 → 순서 의존성 | conditional_proxy |
| N-10 | **경미** | `_claim_caution_flag`의 `half_width` 모드가 implicit fall-through | 유지보수 |

---

## 치명적 문제 (Critical)

### N-1. `fit_dualcp_proxy.py` z_keys 대소문자 불일치

**위치**: `scripts/fit_dualcp_proxy.py:215`

**문제**: 기본 z_keys가 소문자(`xpd_early_excess_db`)이지만, CSV 컬럼명은 대문자(`XPD_early_excess_db`). Python `csv.DictReader`는 **대소문자 구분**하므로, `r.get("xpd_early_excess_db")`는 `None`/`NaN` 반환.

| 기본 z_key | CSV 실제 컬럼명 | 일치? |
|---|---|---|
| `xpd_early_excess_db` | `XPD_early_excess_db` | **불일치** |
| `xpd_late_excess_db` | `XPD_late_excess_db` | **불일치** |
| `l_pol_db` | `L_pol_db` | **불일치** |
| `rho_early_db` | `rho_early_db` | 일치 |

**결과**: `fit_proxy_model()`에서 3개 z-변수의 유한 표본이 0개 → `ValueError("No finite samples for z_key=...")` 예외, 또는 조건부 모델 3/4 미생성.

**근거**:
- `rt_types/standard_outputs.py:72-82` — `LinkMetricsZ.to_dict()`: `"XPD_early_db"`, `"L_pol_db"` (대문자)
- `run_standard_sim.py:864-865` — extras: `"XPD_early_excess_db"`, `"XPD_late_excess_db"` (대문자)
- `rt_io/standard_outputs_hdf5.py:168-175` — `export_csv()`가 `to_dict()` 결과를 그대로 CSV 기록

**필요 조치**: z_keys 기본값을 CSV 컬럼명과 일치시킴:
```
XPD_early_excess_db,XPD_late_excess_db,L_pol_db,rho_early_db
```

---

### N-2. `fit_dualcp_proxy.py` u_keys 이름 완전 불일치

**위치**: `scripts/fit_dualcp_proxy.py:220-224`

**문제**: 기본 u_keys가 `run_standard_sim.py`의 CSV 출력 컬럼명과 **거의 전부 불일치**.

| 기본 u_key | CSV 실제 컬럼명 | 일치? | 비고 |
|---|---|---|---|
| `los_blocked` | `LOSflag` | **불일치** | 이름과 의미 반전 (LOSflag=1은 LOS 존재) |
| `material` | `material_class` | **불일치** | 접미사 누락 |
| `scatter_stress` | *(없음)* | **부재** | CSV에 없는 필드 |
| `distance_d_m` | `d_m` | **불일치** | 접두사 초과 |
| `pathloss_proxy_db` | *(없음)* | **부재** | `EL_proxy_db`와 유사하나 다른 이름 |
| `delay_bin` | *(런타임 주입)* | OK | fit 스크립트에서 직접 삽입 |
| `parity` | `dominant_parity_early` | **불일치** | 축약 |
| `incidence_angle_bin` | *(없음)* | **부재** | link 수준에 없는 필드 |
| `excess_loss_proxy_db` | `EL_proxy_db` | **불일치** | |
| `bounce_count` | *(없음)* | **부재** | ray 수준 필드 (link 수준 없음) |

**결과**: 10개 u_key 중 9개가 NaN → 버킷 키가 `"NA|NA|NA|..."` 단일 버킷으로 붕괴 → 회귀에 numeric feature 0개 → **조건부 모델이 global mean/sigma만 되돌림** → 조건부 프록시의 핵심 목적(조건별 Z|U 예측) 완전 상실.

**추가 위험 — LOSflag vs los_blocked 의미 반전**:
- `LOSflag = 1` → LOS **존재**
- `los_blocked` → 이름 의미: LOS **차단됨**
- 수동으로 매핑해도 부호가 반전되어 회귀 계수 해석이 역전됨

**필요 조치**: u_keys 기본값을 CSV 실제 컬럼명으로 교체:
```
LOSflag,material_class,roughness_flag,d_m,EL_proxy_db,delay_bin,dominant_parity_early,human_flag
```

---

## 주요 문제 (Major)

### N-3. 스칼라/곡선 floor uncertainty 2배 불일치

**위치**:
- `run_standard_sim.py:151` — 스칼라: `delta_floor_db = p95 - p5` (**전체 폭**)
- `run_standard_sim.py:290` — 곡선: `uncert = 0.5 * (p_hi - p_lo)` (**반폭**)

**문제**: 두 값 모두 `_claim_caution_flag()`의 `uncert_db` 인자로 전달됨:
- 스칼라 경로 (line 860): `_claim_caution_flag(ex_e, float(delta), ...)` — delta = p95-p5
- 곡선 경로 (line 875): `_claim_caution_flag(float(xx), float(uu), ...)` — uu = 0.5*(p95-p5)

즉, **동일한 `_claim_caution_flag` 함수에 전폭(scalar)과 반폭(curve)이 혼용 전달**되어:
- 스칼라 caution 임계값 = |p95 - p5| (넓음, 보수적)
- 곡선 caution 임계값 = |0.5*(p95 - p5)| (절반, 공격적)

같은 데이터에서 스칼라 caution = True이고 곡선 caution = False인 경우가 발생 → 보고서 내 모순.

**필요 조치**: 둘 중 하나를 통일. 반폭(`0.5*Δ`)을 표준으로 삼는다면 `_build_floor_reference()` line 151도 `0.5*(p95-p5)`로 변경, 또는 `_lookup_floor()`에서 반으로 나누기.

---

### N-4. 광대역 스칼라 XPD에서 주파수별 floor 곡선 빼기 → 차원 불일치

**위치**: `run_standard_sim.py:872-873`

```python
ex_e_curve = np.asarray(b.metrics.XPD_early_db - f_curve_db, dtype=float)
```

- `b.metrics.XPD_early_db` = **스칼라** (시간 도메인 early window에서 합산된 광대역 XPD)
- `f_curve_db` = **주파수별 벡터** [Nf] (주파수 도메인 floor 곡선)

NumPy 브로드캐스팅으로 `[스칼라 - floor_f1, 스칼라 - floor_f2, ...]` 생성. 이것은 "주파수별 excess XPD"가 아닌 "광대역 XPD에서 주파수별 floor만 뺀 값"으로, **물리적으로 의미가 다름**. 진정한 주파수별 excess를 계산하려면 주파수별 XPD를 먼저 계산해야 함.

Line 891-892의 서브밴드도 동일 문제:
```python
ex_e_sb = np.asarray(b.metrics.XPD_early_db - sb_floor, dtype=float)
```

**결과**: `XPD_early_excess_curve_db` 라벨이 실제 내용과 불일치 → 서브밴드별 excess 분석 시 오해 소지.

**필요 조치**: (a) 변수명을 `XPD_early_minus_floor_curve_db`로 명확화, 또는 (b) PDP에서 서브밴드별 XPD를 별도 계산하여 진정한 주파수별 excess 산출.

---

### N-5. `_dualcp_tap_powers` 비대칭 power floor

**위치**: `analysis/dualcp_metrics.py:56-57`

```python
p_co = p[:, 0, 0].astype(float)                                    # floor 없음
p_cross = np.maximum(p[:, 1, 0].astype(float), float(power_floor)) # floor 적용
```

- `p_co`에는 power floor 미적용 → 0에 도달 가능
- `p_cross`에만 `power_floor` (기본 1e-18) 적용

**결과**: 순수 노이즈 tap에서 `p_co ≈ 0`, `p_cross = 1e-18` → `XPD = 10*log10(0/1e-18) → -∞ dB`. late window에 이런 tap이 포함되면 `xpd_late_db`가 체계적으로 하방 편향.

**필요 조치**: 양쪽 모두 floor 적용, 또는 양쪽 모두 미적용.

---

### N-6. `measurement_compare.py`의 XPD 공식 불일치

**위치**: `analysis/measurement_compare.py:301-304`

```python
co = np.abs(H_f[:, 0, 0]) ** 2 + np.abs(H_f[:, 1, 1]) ** 2   # 대각 합
cr = np.abs(H_f[:, 0, 1]) ** 2 + np.abs(H_f[:, 1, 0]) ** 2   # 비대각 합
```

다른 모든 모듈은 단일 컬럼 정의:
- `dualcp_metrics.py:56-57` — `p_co = |H[0,0]|²`, `p_cross = |H[1,0]|²`
- `dualcp_calibration.py:24` — 동일 단일 컬럼

**문제**: RT 데이터의 2×2 채널 행렬에서 `H[1,1]`과 `H[0,1]`이 비영이면, `measurement_compare`의 XPD와 다른 모듈의 XPD가 다른 값을 반환. `compare_measured_to_dataset()`의 KS 검정이 일치 기준(apples)과 다른 기준(oranges)을 비교하게 됨.

dual-CP 2-CSV 측정(`H[:,0,1]=0`, `H[:,1,1]=0`)에서는 우연히 일치하지만, 전체 2×2 행렬(RT 또는 full 측정)에서는 불일치.

**필요 조치**: `_xpd_over_frequency()`를 다른 모듈과 동일한 단일 컬럼 정의로 통일, 또는 dual-CP 전용임을 명시.

---

## 경미한 문제 (Minor)

### N-7. `_fdr_bh` — 총 검정 수 대신 유효 p-value 수 사용

**위치**: `success_checks.py:123`, `fit_dualcp_proxy.py:113`

```python
q_ord = p_ord * float(len(p_ord)) / np.arange(1, len(p_ord) + 1, dtype=float)
```

`len(p_ord)` = NaN 제외 후의 유효 p-value 수. BH 절차의 `m`은 **전체 검정 수**여야 함. NaN 검정이 있으면 보정이 과소적용(anti-conservative).

현재 코드에서 `evaluate_success_criteria()`가 전달하는 4개 p-value 중 NaN이 드물어 실용적 영향은 작지만, 원칙적으로 부정확.

---

### N-8. `--nf` 기본값 불일치 (128 vs 256)

**위치**:
- `run_standard_sim.py:675` — `--nf` default=128
- `scenarios/common.py:45` — `uwb_frequency(nf=256)`
- `scenarios/runner.py:1489` — `--nf` default=256

새 러너가 기존 코드의 절반 해상도를 기본으로 사용. `dualcp_calibrate_floor.py`가 256점 floor 곡선을 생성하고 `run_standard_sim.py`가 128점에서 사용하면, interpolation이 필요하나 일부 경로에서 길이 불일치 오류 발생 가능.

---

### N-9. `rng` late binding 의존

**위치**: `analysis/conditional_proxy.py:234,238,243,247` (lambda 정의) vs line 254 (`rng` 생성)

Lambda가 `rng`를 참조하지만 `rng`는 lambda 정의 시점에 존재하지 않음. Python late-binding으로 호출 시점(line 255)에서 정상 동작하나, 코드 리팩토링 시 `UnboundLocalError` 위험.

---

### N-10. `_claim_caution_flag` half_width 모드 implicit fall-through

**위치**: `run_standard_sim.py:358-367`

`mode == "half_width"`일 때 `"off"`와 `"scaled"` 어디에도 매칭되지 않아 기본 동작(thr = |uncert|)으로 빠짐. 의도대로 동작하지만 명시적 분기가 없어 향후 모드 추가 시 오류 가능.

---

## 우선순위 권고

### 즉시 수정 (파이프라인 완전 불능)

1. **N-1, N-2**: `fit_dualcp_proxy.py`의 `--z-keys`, `--u-keys` 기본값을 CSV 실제 컬럼명으로 교체. 이것 없이는 프록시 모델 피팅이 **완전히 무의미한 결과**를 생성.

### 출판 전 수정 (결과 편향/모순)

2. **N-3**: floor uncertainty 전폭/반폭 통일
3. **N-4**: excess curve 라벨링 또는 계산 방식 명확화
4. **N-5**: power floor 대칭 적용
5. **N-6**: XPD 공식 통일

### 편의 수정 (정확성 미세 향상)

6. **N-7 ~ N-10**: FDR 분모, nf 기본값, rng 순서, half_width 분기
