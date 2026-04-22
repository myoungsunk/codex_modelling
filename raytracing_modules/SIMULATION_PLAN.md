시뮬레이션 계획서: CP-CIR Conditional Complementarity Analysis
Part 1. 목표와 산출물
1.1 분석 목표
주장 (Claim): CP feature와 CIR feature는 글로벌하게는 redundant하지만, 다음 5개 conditioning axis 위에서 정의되는 특정 regime 안에서는 conditionally complementary하다.
1.2 최종 산출물 (논문 figure 단위)
Figure내용데이터 출처F1Conceptual: CP/CIR feature 정의 및 직교성 도식도식F2Stage 1 setup: single-slab geometry + sweep gridRTF3Global ROC: CIR-only vs CIR+CP fusion (baseline)RT allF4Conditional ΔAUC heatmap (axis i × axis j) ★메인RTF5Disagreement analysis: CIR-only가 틀린 sample의 CP feature 분포RTF6Mechanism plot: 1-bounce ratio vs γ_CPRTF7Stage 2: room A/B/C에서 regime 재현성RTF8Stage 3: HFSS SBR+ 검증 (선별 case)SBRF9LP measurement vs RT CIR statistics (validation)LP실측 + RT
★ F4가 논문의 코어. F8이 시뮬-only 비판의 방어선.
1.3 시간표
Month 1 : RT pipeline 정비 + UWB 합성 + feature extractor 작성
Month 2 : Stage 1 (single-slab wide sweep)
Month 3 : Stage 1 분석, regime 가설 도출
Month 4 : Stage 2 (room A/B/C) + Stage 3 시작 (HFSS)
Month 5 : Stage 3 완료 + LP 실측 fidelity validation
Month 6 : 논문화 + revision 1차

Part 2. RT 코드 수정 포인트 (우선순위 순)
P0 (반드시): Feature extractor 모듈 신규 작성
05_dualpol_rt_channel_builder.py가 H(f) 까지만 생성. 그 다음 단계가 없음. 다음 모듈을 신규 작성해야 함:
feature_extractor.py
  ├── ifft_to_cir(H_f, freqs, window='hann') → h(t)
  ├── extract_first_path(h_t, threshold='leading_edge') → (t_FP, idx_FP)
  ├── compute_gamma_cp_variants(H_f_RHCP, H_f_LHCP, ...) → dict
  ├── compute_a_fp_variants(h_t, idx_FP, ...) → dict
  ├── compute_cir_baseline_features(h_t, ...) → dict
  └── extract_all_features(channel_dict) → flat dict
이게 (a) 설계의 핵심이고 stage 일관성의 근간.
P1 (높음): Cross-pol coupling을 sweep variable로 승격
현재 material.xpol_coupling_db가 "있어도 되고 없어도 되는" 옵션. 이걸 명시적 sweep 축으로 olds승격하고, 모든 material에 default 값 (예: 35 dB) 강제.
python# materials_db.py (신규)
MATERIALS = {
    "wood":    Material(eps_r=2.0, tan_delta=0.01, xpol_coupling_db=30, ...),
    "glass":   Material(eps_r=6.0, tan_delta=0.001, xpol_coupling_db=40, ...),
    "concrete":Material(eps_r=5.5, tan_delta=0.05, xpol_coupling_db=25, ...),
    "metal":   Material(kind="PEC", pec_tm_sign=-1.0, xpol_coupling_db=45, ...),
}
Sweep 시 xpol_coupling_db를 [20, 25, 30, 35, 40] 5단계로.
P2 (높음): PEC sign convention assertion
python# config.py 또는 sanity check 모듈
def validate_material(mat: Material):
    if mat.kind.upper() == "PEC":
        assert mat.pec_tm_sign == -1.0, "PEC TM sign must be -1.0 (IEEE convention)"
P3 (중간): Grazing band 명시적 처리
Sweep grid에서 입사각 [0°, 75°]만 사용. 75-90° 구간은 별도 "grazing regime" sweep으로 분리.
P4 (중간): Ideal CP antenna 옵션 추가 — 너의 질문 7번
Antenna dataclass에 두 가지 mode를 명시적으로 분리:
pythondef make_ideal_cp_antenna(handedness="right", pos=..., boresight=...) -> Antenna:
    """이상적 CP 안테나: AR=0, cross-pol leakage=∞ (실제로 60dB 이상)"""
    return Antenna(
        position=pos, boresight=boresight,
        h_axis=..., v_axis=...,
        basis="circular",
        convention="IEEE-RHCP",
        cross_pol_leakage_db=60.0,    # 사실상 0
        axial_ratio_db=0.0,           # 완벽 AR
        enable_coupling=False,        # cross-coupling 비활성
        tx_pattern_cos_exp=0.0,       # 등방성
        rx_pattern_cos_exp=0.0,
    )

def make_realistic_patch_cp_antenna(pattern_data, ...) -> Antenna:
    """HFSS 시뮬한 패치 패턴 기반"""
    # Antenna 객체 + pattern 별도 객체로 wrap
    # axial_ratio_db, cross_pol_leakage_db를 패턴에서 추출
    return Antenna(
        ...
        axial_ratio_db=measured_AR,           # 패턴에서
        cross_pol_leakage_db=measured_XPD,    # 패턴에서
        tx_pattern_cos_exp=fitted_exp,        # 패턴 fit
        ...
    )
→ Sweep에서 두 안테나 모두 돌려서 비교. *"이상적 CP에서는 보이는 효과가 현실 패치에서 얼마나 살아남는가"*가 한 conditioning axis가 됨. 이게 너의 질문 7번에 대한 답: 별도의 antenna factory function 두 개, 같은 RT pipeline에 plug-in.
P5 (낮음): Depolarization 비활성화 명시
DepolConfig(enabled=False)를 default로 강제. Limitation으로 논문에 명시.

Part 3. RT Sanity Check Checklist
Stage 1 시작 전에 반드시 통과해야 하는 12개 check. 각 check는 자동화된 unit test 또는 plot으로 검증.
Group A: Geometric correctness (4개)
A1. LoS 단독 시나리오의 free-space path loss

Setup: 빈 공간, TX-RX 1~10 m, single LoS path만
Expected: |H(f)|² ∝ 1/d² (Friis)
Plot: log-log d vs |H| 직선, slope -1
Tolerance: ±0.5 dB

A2. Single bounce path 길이의 image method 정확도

Setup: 단일 슬랩, TX-image-RX 기하
Expected: path_length_m == 2 × specular geometry 계산값
Tolerance: 1e-6 m

A3. Snell's law 검증

Setup: 다양한 입사각 single bounce
Expected: 입사각 == 반사각 (PathRecord.incidence_angles_rad)
Tolerance: 1e-9 rad

A4. Path enumeration completeness

Setup: 4면 박스, max_reflections=2
Expected: 1-bounce 4개, 2-bounce 12개 (4×3, no immediate repeat)
코드 검증: _enumerate_sequences 결과 카운트

Group B: Polarization correctness (4개)
B1. Single bounce CP handedness reversal

Setup: TX RHCP, single PEC plate, RX 위치에 RHCP/LHCP 두 안테나
Expected: |H_LHCP| >> |H_RHCP| (handedness reversal)
Specifically: cross-pol ratio > 30 dB (ideal CP ant 사용 시)
Plot: 입사각 0~70° 변화 시 cross-pol ratio

B2. Even bounce (2-bounce) CP handedness preservation

Setup: 두 PEC 평면 90° 각도, 2-bounce path
Expected: |H_RHCP| >> |H_LHCP| (handedness preserved)
Plot: bounce count vs handedness ratio

B3. Fresnel coefficient at Brewster angle (dielectric)

Setup: ε_r=4 슬랩, TM 편파
Expected: Brewster angle = arctan(√4) = 63.43°에서 |Γ_p| ≈ 0
Plot: 입사각 vs |Γ_s|, |Γ_p|

B4. Normal incidence reflection coefficient

Setup: ε_r=4, theta_i=0
Expected: |Γ| = |(n-1)/(n+1)| = 1/3 (n=2)
Tolerance: ±1%

Group C: Frequency / UWB correctness (2개)
C1. Free space delay → CIR peak

Setup: TX-RX 3 m LoS
Process: H(f) → IFFT → h(t)
Expected: peak at t = 3/c = 10 ns
Tolerance: ±1 sample

C2. UWB pulse shape after IFFT

Setup: Flat H(f) over [6.25, 6.75] GHz
Expected: sinc-like pulse, main lobe width ~ 1/BW = 2 ns
Plot: h(t) magnitude

Group D: Antenna correctness (2개)
D1. Ideal CP vs realistic patch comparison

Setup: 동일 LoS geometry, 두 안테나로 동일 측정
Expected: ideal CP에서 cross-pol > 50 dB, realistic patch는 패턴이 정한 XPD
Plot: 두 안테나의 H(f) magnitude bar chart

D2. Coupling matrix unitarity check

Setup: _coupling_matrix(f_hz) 출력 M
Expected: M @ M.conj().T ≈ I (energy preserving)
Tolerance: 1e-6


Sanity check 산출 plot (논문에는 안 들어감, 내부 검증용)
체크리스트 검증을 위해 다음 5개 plot을 자동 생성하는 sanity check script 작성:

plot_los_pathloss.png: A1 검증, log-log free-space loss
plot_handedness_reversal.png: B1+B2, 입사각 vs cross-pol ratio (single/even bounce)
plot_fresnel_brewster.png: B3, Brewster angle 부근 |Γ_s|, |Γ_p|
plot_cir_los.png: C1+C2, LoS CIR peak 위치 + pulse shape
plot_antenna_compare.png: D1, ideal vs patch H(f)

이 5개 plot이 다 합리적이면 Stage 1 진행 가능.

Part 4. Conditioning Variables & Sweep Grid (RT/SBR 공통)
4.1 5개 conditioning axes 최종 spec
Axis변수범위샘플비고A1. GeometryTX-Slab 거리0.5~5 m8 (log)RX-Slab 거리0.5~5 m8 (log)입사각0~75°8 (10° step)grazing 분리A2. Materialε_r2, 4, 6, 104tan δ0.001, 0.01, 0.053슬랩 두께inf, 20mm, 5mm3finite slab은 phase 2A3. Surfacexpol_coupling_db20, 25, 30, 35, 405★새 sweepA4. Antennatypeideal CP, patch2★새 sweepA5. SNRSNR_total10, 20, 30, 40 dB4post-RT noise injection
Total full grid: 8×8×8×4×3×3×5×2×4 ≈ 737k cases — 너무 많음.
4.2 Latin Hypercube Sampling (LHS) 적용
LHS로 ~3000 case로 축소. Continuous variables (거리, 입사각, ε_r, xpol_db, SNR)에 LHS, categorical (slab 두께, antenna type)은 stratify.
python# sweep_design.py
def design_sweep_lhs(n_samples=3000, seed=42) -> pd.DataFrame:
    ...
4.3 Bandwidth는 fixed

Center: 6.5 GHz
BW: 500 MHz
N_freq: 257 points (FFT 친화)
합성: Frequency sweep + IFFT (너의 결정 4)


Part 5. CP/CIR Feature 정의 (질문 5에 대한 답: 다중 variant)
너의 결정 5번 — "동일 지표를 다양한 해석에 따라 gamma_1/gamma_2 등 여럿 포함" — 에 따라, 각 feature를 multiple variant로 정의해 모두 추출. 분석 단계에서 어떤 variant가 가장 informative한지 선별.
5.1 γ_CP (handedness ratio) variants
pythondef compute_gamma_cp_variants(H_RHCP, H_LHCP, h_RHCP, h_LHCP, idx_FP):
    return {
        "gamma_cp_1_freq_avg":    np.mean(np.abs(H_LHCP) / np.abs(H_RHCP)),
        "gamma_cp_2_freq_db":     20*np.log10(np.mean(np.abs(H_LHCP)) / np.mean(np.abs(H_RHCP))),
        "gamma_cp_3_fp_only":     np.abs(h_LHCP[idx_FP]) / np.abs(h_RHCP[idx_FP]),
        "gamma_cp_4_total_energy":np.sum(np.abs(h_LHCP)**2) / np.sum(np.abs(h_RHCP)**2),
        "gamma_cp_5_post_fp":     np.sum(np.abs(h_LHCP[idx_FP:])**2) / np.sum(np.abs(h_RHCP[idx_FP:])**2),
        "gamma_cp_6_phase_consistency": circular_var(np.angle(h_LHCP[FP_window]) - np.angle(h_RHCP[FP_window])),
    }
각 variant의 의미:

1, 2: 평균적 handedness 분포
3: First path만 (LoS 의존성 강함)
4: 전체 에너지 (multipath structure 통합)
5: First path 이후 (NLoS 정보 분리)
6: 위상 consistency (coherent vs scattered 구분)

5.2 a_FP (first path prominence) variants
pythondef compute_a_fp_variants(h_t, idx_FP, fs):
    fp_window = slice(idx_FP-2, idx_FP+3)
    return {
        "a_fp_1_norm_energy":    np.sum(|h[fp_window]|^2) / np.sum(|h|^2),
        "a_fp_2_peak_to_total":  |h[idx_FP]|^2 / np.sum(|h|^2),
        "a_fp_3_peak_to_max":    |h[idx_FP]| / np.max(|h|),
        "a_fp_4_kurt_local":     kurtosis(|h[fp_window]|),
        "a_fp_5_rise_time":      time from 10% to 90% of peak,
        "a_fp_6_fp_to_2nd_peak": |h[idx_FP]| / |h[idx_2nd_peak]|,
    }
각각 LoS/NLoS sensitivity가 다름. variant 6은 multipath dominance에 민감, variant 5는 dispersion에 민감.
5.3 Stage 일관성 보장
모든 variant는 H(f) 기반 (stage 1/2/3 공통). Path-level info는 추가 메커니즘 분석용으로만 사용 (PathRecord.bounce_count, incidence_angles_rad 등).

Part 6. CIR Baseline Feature Set (질문 6: hand-crafted 모음)
pythondef compute_cir_baseline_features(h_t, fs):
    return {
        # 시간 분포
        "rms_delay_spread": ...,
        "mean_excess_delay": ...,
        "max_excess_delay": ...,
        # First path 형상
        "fp_to_total_ratio": ...,    # τ
        "rise_time_fp": ...,
        "fp_kurtosis": ...,
        # 전체 형상
        "kurtosis_total": ...,        # NLoS 시 감소
        "skewness_total": ...,
        "energy_concentration_50ns": ...,
        # 다중경로 indicators
        "num_significant_peaks": ...,  # threshold 기반
        "peak_to_avg_ratio": ...,
        # K-factor estimate
        "k_factor_estimate": ...,
    }
총 ~12 features. CP variant는 6+6=12 features. Joint set은 24 features.
비교 대상:

CIR-only (12)
CP-only (12)
CIR+CP joint (24)
각각에 대해 logistic regression + random forest 두 분류기


Part 7. Complementarity Metric (앞서 합의한 4단계)
Level 1: Global ΔAUC
ΔAUC_global = AUC(CIR+CP) - AUC(CIR)
보고용. 작아도 무방.
Level 2: Conditional ΔAUC heatmap (★메인)
ΔAUC(c) for c in (axis_i, axis_j) bins
Heatmap으로 시각화. F4 figure.
Level 3: Conditional Mutual Information
I(Y; X_CP | X_CIR) per bin
Binning approximation 사용.
Level 4: Disagreement Analysis (★ 너의 "맹점 찾기" 동기와 직결)
1. CIR-only classifier 학습
2. Misclassified samples 추출
3. 이 sample들의 CP feature 분포 vs correctly-classified의 CP feature 분포
4. KS test 또는 Wasserstein distance
이게 *"CIR이 못 보는 곳을 CP가 보는가"*에 대한 직접 답변.

Part 8. Stage별 실행 계획
Stage 1: Single-slab wide sweep (Month 2-3)

Tool: MATLAB RT (또는 너의 Python RT, 코드 검토 결과로 보아 Python이 더 통제 가능)
Cases: LHS 3000개 from 5-axis grid
Per case: H(f) (RHCP, LHCP) → CIR → 24 features → label (LoS/NLoS)
Output: features.parquet, ~3000 rows × 25 cols
Analysis: ΔAUC heatmap (5 axes × 5 axes = 10 pairs, 상삼각)

Stage 2: Room A/B/C (Month 4)

Tool: 동일 Python RT
Cases: 룸당 50 TX-RX positions × 3 rooms × 2 antennas = 300 cases
Goal: Stage 1에서 식별한 critical regime이 룸 환경에서 재현되는가
Output: regime_persistence.csv

Stage 3: HFSS SBR+ 정밀 검증 (Month 4-5)

Tool: HFSS 23R2 SBR+
Cases: Stage 1/2에서 ΔAUC 가장 큰 30 케이스 선별
Goal: RT 결과의 정성적 결론이 SBR에서 유지되는가
Output: sbr_validation.csv, RT vs SBR scatter plot

LP Measurement Validation (Month 5)

Source: 너의 기존 LP UWB 실측 데이터
Goal: RT-simulated CIR의 통계적 분포가 LP 실측과 매칭됨을 확인
Method: K-S test on (RMS delay spread, fp_to_total_ratio) distributions
이게 reviewer 방어선: "시뮬레이션이 LP 실측과 통계적으로 매칭됨을 보였으므로, CP 부분의 시뮬레이션 외삽도 신뢰 가능"


Part 9. 즉시 실행할 첫 작업 (Week 1)

Sanity check 5개 plot 생성하는 script 작성 — sanity_check.py
Feature extractor 모듈 prototype — feature_extractor.py
Ideal CP antenna factory 함수 추가 — 02_rt_core_antenna.py에 make_ideal_cp_antenna() patch
PEC sign convention assertion 추가
Sanity check 결과 plot 5개 보면서 결과 검증

상기 내용 넣어 simulation plan 문서 저장
