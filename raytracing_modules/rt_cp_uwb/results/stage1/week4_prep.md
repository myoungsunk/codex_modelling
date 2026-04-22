# Week 4 Prep Checklist

- [x] Stage 2 antenna policy: use `patch_ffd` as the primary antenna condition. Keep ideal CP only as a small sanity/control subset if needed.
- [x] `makeRoomABCScene.m` exists and instantiates A/B/C rooms.
- [x] Room dimensions: default `[5 4 3]` m are already implemented and remain reasonable for Stage 2.
- [x] Clutter levels:
  - Room A: 0 clutter objects (6 enclosure faces total)
  - Room B: sparse clutter, 3 added surfaces (9 total surfaces)
  - Room C: dense clutter, 7 added surfaces (13 total surfaces)
- [x] TX/RX grid draft: 1 ceiling anchor per room + 50 tag positions per room, split into 25 coarse baseline points + 25 targeted points.
  - Coarse layer: room-wide coverage for baseline regime sampling.
  - Targeted layer: wall-near and angle-selective points to emphasize Stage 1 strong regimes.
- [x] FFD antenna Stage 2 integration test basis: Stage 1 FFD path is already operational.
- [x] Estimated Stage 2 runtime from pilot: 6.94 s/case mean across A/B/C.

## Room Surface Counts

- Room A surfaces: 6
- Room B surfaces: 9
- Room C surfaces: 13

## Estimated Stage 2 Sweep Size

- Patch-only baseline: 50 positions x 3 rooms = 150 cases
- Estimated runtime at 6.94 s/case: 17.36 min for 150 cases
- Expanded patch-only run: 100 positions x 3 rooms = 300 cases, about 34.72 min total

## Label Definition Recommendation

- Keep the current ideal-reference path-strength label as the primary Stage 2 label.
- Current rule in code: `effective_nlos = ~has_los_path || (max_bounce_strength / los_strength >= 0.20)` using ideal reference antennas.
- Do not switch the primary label to observed-channel `k_factor_estimate`, because `k_factor_estimate` is already part of the modeling feature set and would introduce label leakage.
- If desired, add a secondary sensitivity analysis with a K-factor-based threshold, but keep it separate from the main reported label.

## Stage 3 HFSS SBR+ Selection Seed

- Group 1 (15 cases): 3 strongest Stage 1 patch-only Delta AUC regimes, 5 representative samples per regime.
- Group 2 (15 cases): 3 patch-weak regimes, 5 representative samples per regime.
- Group 3 (15 cases): deferred until Stage 2; use top room-regime Delta AUC cells once Week 4 results exist.
- Detailed Stage 3 criteria: `D:\codex\raytracing_modules\rt_cp_uwb\results\stage1\stage3_selection_criteria.md`
