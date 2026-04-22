# Stage 3 Validation Protocol After Deterministic Rerun

## Stage 1 Deterministic Recheck Boundary

Stage 1 was rerun deterministically before locking the Stage 3 protocol.

Summary:

- original `Delta AUC`: `+0.0410`
- deterministic `Delta AUC`: `+0.0463`
- shift: `+0.0053`
- stability gate `|shift| < 0.01`: `PASS`
- top-3 overlap: `1 / 3`
- top-10 overlap: `8 / 10`

Interpretation:

- Stage 1 is stable at the **global lift** level
- Stage 1 is **not** stable enough to treat the exact top-3 cells as fixed ground truth

Therefore:

1. Stage 1 supports the broad claim that CP is a moderate complement to CIR
2. Stage 1 should not be used as the sole source for locking exact Stage 3 regimes
3. Stage 2 deterministic relabeled selection remains the primary basis for the HFSS 30-case list

## Candidate Room Distribution

- Total HFSS candidates: `30`
- Room C: `20`
- Room A: `5`
- Room B: `5`

Room B is **not overrepresented** in the deterministic Stage 3 list.

Detailed group split:

- `GEO, C`: `15`
- `BOUNCE, C`: `5`
- `BOUNCE, A`: `5`
- `BOUNCE, B`: `5`

The Room B candidates are all from the same bounce-dominant regime:

- `room type=B, dominant wall material=brick`

## GEO Group Concentration

The deterministic `GEO` group is entirely from Room C:

- `GEO, C`: `15`
- `GEO, B`: `0`
- `GEO, A`: `0`

This is intentional in the current selection logic, not a coding bug.
The reason is support imbalance in the `geo_only` label:

- Room A geo positives: `0`
- Room B geo positives: `7`
- Room C geo positives: `60`

The `GEO` selection was driven by top `delta_auc` cells with minimum class-count support, so only Room C produced stable enough GEO regimes.

Interpretation:

- The current `GEO` group validates the **Room C geometric-blockage regime**
- It does **not** claim that GEO behavior is room-invariant

If room-diverse GEO coverage is later needed, add a supplemental set from the `7` Room B geo-positive cases, but keep it separate from the core top-regime validation list.

## Why Room B Needs Special Interpretation

For `mixed@0.33`, Room B had the smallest `delta_auc` shift after the deterministic rerun, but both `auc_cir` and `auc_joint` moved downward together:

- `delta_auc shift = +0.000313`
- `auc_cir shift = -0.018519`
- `auc_joint shift = -0.018205`

This means Room B is not "absolutely stable" in the sense of fixed baseline AUC values.
It is only stable in the **relative lift** sense.

Therefore, Room B should not be used as the room where absolute MATLAB-vs-HFSS AUC agreement is enforced most strongly.

## Stage 3 Comparison Rule

### Primary comparison for all rooms

Use **room-internal ordering and lift direction** as the primary validation target:

1. Sign of `Joint - CIR` lift
2. Relative ordering of selected regimes inside the same room
3. Case-level `CP rescue` tendency over `CIR-only` failures

### Absolute AUC comparison

- Room C: allowed as a secondary check
- Room A: allowed as a secondary check
- Room B: **descriptive only**, not a hard pass/fail criterion

For Room B, compare:

1. Whether the selected regime remains bounce-favorable under HFSS
2. Whether the room-internal ranking stays consistent with MATLAB
3. Whether the selected cases still show positive CP contribution at the case level

Do **not** reject the MATLAB model only because Room B absolute AUC differs from HFSS while `delta_auc` sign and ranking remain consistent.

## Practical Consequence

The deterministic Stage 3 list does not need reselection because of Room B.
The candidate count is only `5/30`, and the right fix is to adjust the validation protocol:

- keep Room B in the HFSS set
- score Room B by ranking / lift direction
- reserve strict absolute-AUC comparison for the more stable rooms and the all-room aggregate

## Locked Rule Set

The Stage 3 validation protocol is now locked as:

1. **Selection authority**: Stage 2 deterministic relabeled outputs
2. **Stage 1 role**: broad family support only, not exact top-cell fixing
3. **Primary validation target**: case-level feature behavior and room-internal ranking
4. **Secondary validation target**: sign and magnitude of `Joint - CIR` lift
5. **Absolute AUC comparison**: supplemental, with Room B treated descriptively
6. **GEO interpretation**: Room C geometric-blockage regime, not room-general GEO physics
