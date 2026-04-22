# Stage 1 Deterministic Recheck

## Runtime

- cases: `3000`
- elapsed: `17.18 min`
- per case: `343.557 ms`
- failed: `0`

## Global Stability

- original `CIR-only`: `0.6795`
- deterministic `CIR-only`: `0.6745`
- shift: `-0.0050`

- original `CP-only`: `0.6485`
- deterministic `CP-only`: `0.6519`
- shift: `+0.0034`

- original `Joint`: `0.7205`
- deterministic `Joint`: `0.7208`
- shift: `+0.0003`

- original `Delta AUC`: `+0.0410`
- deterministic `Delta AUC`: `+0.0463`
- shift: `+0.0053`

`Delta AUC` stability gate `|shift| < 0.01` is **PASS**.

## Top Regime Stability

- top-3 overlap: `1 / 3`
- top-10 overlap: `8 / 10`

This means Stage 1 is globally stable, but the exact ordering of the strongest conditional cells is not fully stable.

## Interpretation

1. Stage 1 still supports the same high-level conclusion: CP provides a moderate positive complement to CIR.
2. Stage 1 should **not** be used as the sole source for fixing exact top-3 regimes.
3. Stage 2 deterministic relabeled selection remains the stronger basis for Stage 3 HFSS case selection.
4. Any Stage 1 to Stage 2 persistence claim should be written at the level of:
   - stable direction of global lift
   - broad regime families
   - not exact top-cell identity
