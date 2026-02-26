# 수정 후 2차 검토 보고서

> 날짜: 2026-02-26
> 대상: 1차 검토에서 지적된 치명적/잠재적 문제에 대한 수정사항 검증
> 범위: 코드 전체 grep + 핵심 파일 정독 기반

---

## 요약: 수정 상태 총괄

| # | 항목 | 주장 상태 | 실제 코드 상태 | 판정 |
|---|------|----------|---------------|------|
| 0 | C0 거리/각도/반복 | 부분 해결 | CLI args 존재, **각도가 트레이서에 미반영** | **미해결** |
| 1 | A3 표본 확대 | 해결 | `A3_corner_2bounce.py` **여전히 4개 케이스** | **미해결** |
| 2 | LOS 차단 물리성 | 해결 | `--strict-los-blocked` = 사후 검증만, **occluder 미구현** | **부분** |
| 3 | A5 합성↔물리 괴리 | 부분 해결 | `--stress-flag` 일괄 적용, **stress_mode 미구현** | **미해결** |
| 4 | B LOSflag 고정 | 해결 | ray table 기반 재계산 **구현됨** | **해결** |
| 5 | Normal-only 분포 | 해결 | `conditional_proxy.py` **변경 없음**, Normal만 지원 | **미해결** |
| 6 | OLS 강건성(Huber) | 해결 | `_fit_regression()` **여전히 lstsq**, Huber 없음 | **미해결** |
| 7 | bucket 최소 표본 | 해결(3→) | `conditional_proxy.py`에 **min_bucket_n 파라미터 없음** | **미해결** |
| 8 | 주파수/서브밴드 floor | 부분 해결 | `_resolve_floor_band()`가 **여전히 median 스칼라 압축** | **미해결** |
| 9 | claim_caution 모드 | 해결 | `--claim-caution-mode` 파라미터 **없음**, 하드코딩 유지 | **미해결** |
| 10 | drift check | 해결 | 3-CSV ingest **없음**, `--max-drift-db` **없음** | **미해결** |
| 11 | LP baseline | 해결 | `run_cp_lp_baseline.py` **없음** | **미해결** |
| 12 | ANOVA/FDR/혼동진단 | 부분 해결 | ANOVA, FDR-BH, confounding 진단 **전부 없음** | **미해결** |

**실제 해결: 1건 (B LOSflag), 부분 해결: 1건 (LOS 차단 검증), 미해결: 10건**

---

## 1. 상세 검증

### 1-0. C0 보정: 거리/각도/반복

**주장**: C0 거리 1-5m 반영, `--n-rep` 추가, yaw/pitch sweep 옵션 추가

**실제 코드**:

`run_standard_sim.py:420-422`에 CLI args가 존재:
```python
parser.add_argument("--dist-list", type=str, default="")
parser.add_argument("--yaw-list", type=str, default="0")
parser.add_argument("--pitch-list", type=str, default="0")
```

**문제 1 - 각도가 RT에 반영되지 않음** (`run_standard_sim.py:274`):
```python
paths = C0_free_space.run_case({"distance_m": d}, f_hz, basis="circular")
```
`yaw`와 `pitch`가 `params`에 전달되지 않습니다. `C0_free_space.run_case()`는 Rx를 `[d, 0, 1.5]`에 고정 배치하며 안테나 방향 변경 로직이 없습니다. yaw/pitch는 **metadata에만 기록**되고 실제 시뮬레이션에는 영향을 주지 않습니다.

**문제 2 - 기본 거리가 여전히 미변경**:
- `--dist-list` 기본값: `""` → `_parse_float_list("", [3.0, 6.0, 9.0])` = **[3, 6, 9]m** 그대로
- 실험 계획서의 1-5m 범위와 여전히 불일치

**문제 3 - `--n-rep` 없음**:
전체 codebase에 `n-rep` 또는 `n_rep` CLI 파라미터 검색 결과 **없음**. A5에만 `N_REPS_PER_RHO = 6`이 하드코딩되어 있음.

**잔존 위험**: C0 Floor의 각도 민감도 분석(Q1-2)이 불가능. metadata에 yaw=15를 기록하더라도 실제 시뮬레이션은 yaw=0과 동일한 결과를 생성하므로, `check_C0_floor()`의 `yaw_rank_corr`이 무의미한 값(상수 XPD에 대한 상관)을 반환합니다.

---

### 1-1. A3 표본 확대

**주장**: A3 sweep case 확대 (다중 조합)

**실제 코드**:

`A3_corner_2bounce.py:39-45` — **변경 없음**, 여전히 4개 고정 케이스:
```python
def build_sweep_params() -> list[dict[str, Any]]:
    return [
        {"offset": 3.0, "rx_x": 3.0, "rx_y": 4.0},
        {"offset": 3.0, "rx_x": 4.0, "rx_y": 3.0},
        {"offset": 3.5, "rx_x": 3.5, "rx_y": 4.5},
        {"offset": 4.0, "rx_x": 4.0, "rx_y": 5.0},
    ]
```

`run_standard_sim.py:312-329` — A3 러너가 `build_sweep_params()`를 그대로 사용:
```python
base = A3_corner_2bounce.build_sweep_params()
for p in base:  # 4개만 순회
```

**잔존 위험**: Q2-2 (even-bounce 회복)의 분포 비교에 4×N_rep 점만 사용 → KS 검정 검정력 매우 낮음. A2(9 케이스) 대비 A3의 표본이 절반 이하.

---

### 1-2. LOS 차단 물리성

**주장**: `--los-block-mode {synthetic,occluder}`, occluder 경로 구현

**실제 코드**:

- `--los-block-mode` 파라미터: **없음**
- `occluder` 구현: **없음**
- 존재하는 것: `--strict-los-blocked` (boolean flag) — `run_standard_sim.py:461-463`에서 **사후 검증**만 수행:
```python
if scenario_id in {"A2", "A3", "A4", "A5"} and bool(args.strict_los_blocked):
    if los_link == 1:
        raise SystemExit(...)
```

B2/B3에서는 `_run_room_case()` (lines 216-260)에 PEC 장벽(Plane)을 추가하여 물리적 차단을 시뮬레이션하지만, 이것은 A2-A5와 무관한 별도 구현.

**잔존 위험**: A2-A5의 LOS 차단은 `los_enabled=False`로 ray tracer에서 LOS ray를 제거하는 방식. 이는 실제 흡수체와 물리적으로 다릅니다(흡수체는 diffraction/scattering을 일으킬 수 있음). 현재 수준에서는 "synthetic LOS removal"로 명확히 한정 필요.

---

### 1-3. A5 stress_mode

**주장**: stress_mode를 none/synthetic/geometry/hybrid로 확장, geometry scatterer 추가

**실제 코드**:

- `stress_mode` 파라미터: **없음**
- `geometry scatterer`: **없음**
- 존재하는 것: `--stress-flag` (boolean) — `run_standard_sim.py:424`:
```python
parser.add_argument("--stress-flag", action="store_true")
```

이 플래그가 켜지면 **모든** A5 케이스에 `roughness_flag=1, human_flag=1`을 일괄 설정 (line 367-368). 합성 depol rho와 물리적 scatter 구분이 불가능.

`A5_depol_stress.py` — **변경 없음**: 여전히 `DepolConfig(rho_func=rho_hook)`로 합성 depolarization만 사용.

**잔존 위험**:
- `--stress-flag`가 꺼져 있으면 `roughness_flag=0, human_flag=0` → `check_A4_A5_breaking()`의 stress 그룹이 0건 → 검증 공허
- `--stress-flag`가 켜져 있으면 모든 A5 케이스가 stress → base 그룹이 0건 → 역시 비교 불가
- 즉, **base vs stress 비교가 구조적으로 불가능**합니다

---

### 1-4. B LOSflag 고정 → 해결됨

**주장**: ray table 기반 LOSflag 재계산

**실제 코드** (`run_standard_sim.py:460, 527`):
```python
los_link = int(any(int(r.get("los_flag_ray", 0)) == 1 for r in ray_rows))
...
link_meta["LOSflag"] = int(los_link)
```

B1(장애물 없음)은 LOS ray 존재 → LOSflag=1, B2/B3(장벽 추가)는 일부 Rx에서 LOS ray 부재 → LOSflag=0으로 자연스럽게 분리됩니다.

`check_B_space_consistency()` (success_checks.py:183-210)도 LOSflag 기반 LOS/NLOS 분리를 수행하여 정상 동작.

**판정: 해결됨**. 다만 B1 grid에서 모든 점이 LOS인 경우 NLOS 변동이 0건이 될 수 있으므로, B2/B3를 반드시 함께 실행해야 의미 있는 비교가 가능합니다.

---

### 1-5. 분포 모델: Normal → Student-t/Laplace

**주장**: normal/student_t/laplace 지원, 기본을 student_t로 설정

**실제 코드**:

`conditional_proxy.py` — **완전히 변경 없음** (1차 검토 시점과 동일):
- `fit_proxy_model()`: Normal assumption만 사용
- `predict_distribution()`: docstring이 `"Predict Normal(mu,sigma) parameters"` 그대로
- GOF: `stats.kstest(resid_std, "norm")` — Normal에 대해서만 검정
- Student-t, Laplace 관련 코드: **전체 codebase에서 0건**

**참고**: `analysis/xpd_stats.py`에 `gof_model_selection_db()`가 GMM/truncnorm 등 다양한 분포를 지원하지만, 이는 **path-level** 분석용이며 `conditional_proxy.py`의 link-level proxy와 연결되지 않음.

**잔존 위험**: Q2-4 (depol stress의 꼬리 분석)에서 Normal 가정이 체계적으로 heavy tail을 과소추정. 계획서의 "하위 5-10% quantile" 비교는 모델 예측이 아닌 경험적 CDF로만 유효.

---

### 1-6. OLS → Huber IRLS

**주장**: Huber IRLS 기반 robust regression 옵션 추가(기본 on)

**실제 코드**:

`conditional_proxy.py:65` — **변경 없음**:
```python
beta, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
```
전체 codebase에서 "huber", "irls", "robust_reg", "iteratively_reweighted" 검색: **0건**

**잔존 위험**: A5 극단 케이스(고-rho)의 outlier가 회귀 계수를 왜곡할 수 있음.

---

### 1-7. bucket 최소 표본

**주장**: min_bucket_n 파라미터화, 기본값 5→3

**실제 코드**:

`conditional_proxy.py`의 bucket 통계 계산 (lines 140-147):
```python
bucket_stats[k] = {
    "u": bucket_u[k],
    "n": int(len(v)),
    "mu": float(np.mean(v)),
    "sigma": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
}
```
**최소 N 제한 없음**. N=1인 버킷도 mu를 계산하고 `predict_distribution()`에서 우선 사용됨.

`min_bucket_n` 파라미터: `conditional_proxy.py`에 **없음**.

**참고**: `analysis/xpd_stats.py`에는 `min_n=20`이 있으나 이는 GOF 테스트용이며, proxy model의 bucket 통계와 무관.

**잔존 위험**: 소수 표본 버킷의 sigma=0.0이 prediction interval을 점 추정으로 퇴화시킴.

---

### 1-8. 주파수/서브밴드 floor 정보

**주장**: floor reference에서 curve/subband를 metrics extras에 저장

**실제 코드**:

`dualcp_calibrate_floor.py`는 주파수별 벡터 `xpd_floor_db[Nf]`를 JSON으로 출력 → **이 부분은 정상**

하지만 `run_standard_sim.py`의 `_build_floor_reference()` (lines 127-178)는:
```python
x = np.asarray([r["xpd_early_db"] for r in rows], dtype=float)
...
"xpd_floor_db": float(np.nanmedian(x)),
```
`xpd_early_db`는 이미 시간 영역에서 합산된 **단일 스칼라**이므로, 주파수별 floor 정보가 이 시점에서 이미 소실됨.

`_lookup_floor()` (lines 181-213)도 스칼라 `xpd_floor_db`만 반환.

`dualcp_metrics.py:_resolve_floor_band()` (lines 131-159)도 주파수별 floor를 `median` 스칼라로 압축:
```python
floor_band = float(np.median(floor_i))
```

**잔존 위험**: UWB 대역(3-10+ GHz)에서 안테나 XPD floor가 주파수 의존적일 경우, 단일 스칼라 floor로는 특정 서브밴드에서 체계적 편향 발생.

---

### 1-9. claim_caution 모드

**주장**: `--claim-caution-mode {scaled,half_width,off}`, `--claim-caution-scale` 추가

**실제 코드**:

전체 codebase에서 `claim_caution_mode`, `caution_mode`, `caution_scale` 검색: **0건**

기존 하드코딩 유지 (`dualcp_metrics.py:294`):
```python
out["claim_caution_early"] = bool(abs(xpd_early_excess) <= u)
```

`run_standard_sim.py:576-577`도 동일 로직:
```python
caution_e = bool(np.isfinite(delta) and abs(ex_e) <= abs(delta))
```

---

### 1-10. 순차 측정 drift check

**주장**: 3-CSV ingest 추가, drift metric 저장, `--max-drift-db` 필터

**실제 코드**:

- 측정 포맷 (`scenarios/runner.py:1530`): `choices=["matrix_csv", "four_csv", "dualcp_two_csv"]` — 3-CSV 없음
- `co_pre`, `co_post`, `max_drift` 검색: **0건**
- drift 관련 코드: `success_checks.py:103`에 `"distance_or_drift"` 라벨이 있으나, 이는 C0 floor 변동의 원인 분류 라벨일 뿐 실제 drift 검출이 아님

---

### 1-11. LP baseline

**주장**: CP/LP 동시 실행 및 비교 리포트 스크립트 추가

**실제 코드**:

`run_cp_lp_baseline.py` 검색: **없음**
`lp_baseline`, `cp_lp`, `linear_baseline` 검색: **0건**

---

### 1-12. ANOVA/FDR/혼동 진단

**주장**: C0 distance/yaw ANOVA + eta2, FDR-BH, distance-vs-LOS confounding

**실제 코드**:

`anova`, `eta_sq`, `eta2`, `fdr`, `benjamini`, `confound` 검색: **전부 0건**

---

## 2. 새로 발견된 문제

### 2-A. A5 stress_flag의 base/stress 비교 불가 구조 (신규, Critical)

`run_standard_sim.py:367-368`에서 `--stress-flag`가 **전역** 적용:
```python
"roughness_flag": int(bool(args.stress_flag)),
"human_flag": int(bool(args.stress_flag)),
```

결과:
- `--stress-flag` OFF: 모든 A5 → roughness=0, human=0 → stress 그룹 0건
- `--stress-flag` ON: 모든 A5 → roughness=1, human=1 → base 그룹 0건

`check_A4_A5_breaking()`의 `base` vs `stress` 비교가 **구조적으로 항상 한쪽이 빈 집합**입니다.

**해결 방향**: rho 값 기반으로 stress 레벨을 분류하거나, stress_flag를 케이스별로 설정해야 합니다.

### 2-B. C0 yaw/pitch가 "phantom 변수"로 작동 (신규, Critical)

`run_standard_sim.py:274`에서 yaw/pitch가 RT 시뮬레이션에 전달되지 않으므로, `--yaw-list 0,15,30`으로 실행해도 모든 케이스가 동일한 XPD를 생성합니다. 그런데 metadata에는 서로 다른 yaw 값이 기록되므로:
- `check_C0_floor()`의 `yaw_rank_corr`이 **0에 가까운 무의미한 상관**을 반환
- 이를 "각도 민감도가 없다"는 결론으로 오해할 위험

### 2-C. A5가 처음 8개 케이스만 사용 (신규, Minor)

`run_standard_sim.py:353`:
```python
params = A5_depol_stress.build_sweep_params()[:8]
```
A5는 5 rho × 6 rep = 30 케이스를 생성하지만, 러너가 **처음 8개만** 사용. 이는 rho=0.05 (6개) + rho=0.15 (2개)만 포함하여, 고-rho 범위의 depolarization stress를 전혀 테스트하지 못합니다.

### 2-D. B2/B3의 NLOS 분류가 기하 의존적 (신규, Minor)

`_run_room_case()`에서 B2는 x=5.0에 벽 하나, B3는 x=5.5에 벽 두 개를 배치합니다. Tx가 x=2.0에 있으므로, Rx의 x 좌표가 5.0 미만이면 대부분 LOS가 유지됩니다. 즉, NLOS 포인트는 grid의 한쪽 끝에만 집중되어, **거리와 LOSflag가 강하게 상관**됩니다. 이는 회귀 모델에서 d_m과 LOSflag의 계수 분리를 어렵게 합니다.

---

## 3. 우선순위 권고

### 즉시 수정 필요 (파이프라인 무결성)

1. **A5 stress_flag → 케이스별 분류** (2-A): rho threshold 기반으로 base(rho<0.1)/stress(rho>=0.1) 분리
2. **C0 yaw/pitch RT 반영** (1-0): `run_case()`에 안테나 방향 전달, 또는 phantom 변수 제거
3. **A5 케이스 수 확대** (2-C): `[:8]` 제한 제거 또는 rho 범위 균등 샘플링
4. **A3 케이스 확대** (1-1): `build_sweep_params()`에 offset × rx_position 조합 추가

### 통계 모델 보강 (논문 출판 전)

5. **Student-t 분포** (1-5): `conditional_proxy.py`에 분포 선택 옵션
6. **Huber regression** (1-6): `_fit_regression()`에 robust 옵션
7. **bucket min_n** (1-7): N < threshold일 때 regression fallback 강제

### 중기 과제 (논문 범위 조정으로 우회 가능)

8. LP baseline (1-11): 논문 scope를 "CP validity conditions"로 한정
9. Drift check (1-10): 측정 프로토콜 문서로 대체 가능
10. ANOVA/FDR (1-12): 탐색적 분석으로 명시하면 수용 가능

---

## 4. 코드 참조 요약

| 파일 | 라인 | 현재 상태 | 필요 조치 |
|------|------|----------|----------|
| `scenarios/C0_free_space.py` | 15-16, 19-40 | 거리 3점, 각도 없음 | yaw/pitch 파라미터 추가 |
| `scenarios/A3_corner_2bounce.py` | 39-45 | 4개 케이스 고정 | sweep 확장 |
| `scenarios/A5_depol_stress.py` | 44-58 | 합성 rho only | stress 레벨 분류 메타 추가 |
| `scripts/run_standard_sim.py` | 274 | yaw/pitch 미전달 | params에 각도 포함 |
| `scripts/run_standard_sim.py` | 353 | `[:8]` 하드 제한 | 제거 또는 파라미터화 |
| `scripts/run_standard_sim.py` | 367-368 | stress_flag 전역 | 케이스별 분류 |
| `analysis/conditional_proxy.py` | 40-76, 118-196 | Normal-only, OLS, min_n 없음 | 분포/회귀/버킷 전부 보강 |
| `analysis/dualcp_metrics.py` | 131-159, 291-298 | floor 스칼라 압축, caution 하드코딩 | 서브밴드 보존, 모드 파라미터화 |
