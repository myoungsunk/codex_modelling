# Stage 1 Results Summary

## Setup

- 3000 LHS cases
- Realistic patch FFD antennas from HFSS
- Symmetric-boresight Stage 1 geometry with slab placement sweep
- Canonical feature set: 18 features (12 CIR + 6 CP)
- Runtime: 16.81 min total, 336.30 ms/case

## Global Results

- CIR-only AUC: 0.6795
- CP-only AUC: 0.6485
- Joint AUC: 0.7205
- Delta AUC (Joint - CIR): +0.0410

## Conditional Regimes Identified

1. `eps r x xpol coupling db` @ `[2.00, 3.60]` / `[20.01, 24.00]`: Delta AUC = +0.2666 (n=120)
2. `antenna type x los angle from anchor bore` @ `ideal` / `[38.57, 45.77]`: Delta AUC = +0.2006 (n=294)
3. `antenna type x slab placement` @ `ideal` / `wall_x`: Delta AUC = +0.1925 (n=371)

## Mechanism

- LoS AR vs gamma correlation: 0.0575
- NLoS AR vs gamma correlation: -0.0476
- LoS XPD vs gamma correlation: -0.0680
- NLoS XPD vs gamma correlation: -0.1368
- Mechanism plot: `D:\codex\raytracing_modules\rt_cp_uwb\results\stage1\stage1_mechanism_scatter.png`
- Interpretation: FFD patch quality induces a non-zero LoS gamma floor, while NLoS improvement remains regime-dependent rather than uniformly dominant.
- This is consistent with measurement-side observations that CP multipath discrimination can be marginal once realistic patch pattern quality is imposed.

## Disagreement Analysis

- CIR-only CV AUC: 0.6803
- Misclassification rate: 0.3000 (900 / 3000)
- Full heatmap figure: `D:\codex\raytracing_modules\rt_cp_uwb\results\stage1\conditional_auc_heatmaps.png`

## Outlook To Stage 2

- Stage 1 remains a limiting single-slab case and already shows that realistic FFD patches reduce CP-only separability.
- Stage 2 room scenes should test whether richer multipath strengthens or further weakens CP utility under realistic patch patterns.
- Stage 2 should use `patch_ffd` as the main antenna condition; ideal CP can be reduced to a small sanity subset rather than a full parallel sweep.
- Week 4 prep document: `D:\codex\raytracing_modules\rt_cp_uwb\results\stage1\week4_prep.md`
