# Plan 3 Results And Evaluation

## Core Verdict

The current data supports only the first half of the original claim.

- Supported under an epsilon margin: after `family_radiated_power` normalization, `P_NI(epsilon)` is `1.0` across the non-proxy indoor scenes considered here.
- Not supported under a strict zero margin: `P_NI(0)` remains in the range `0.365 ~ 0.702`, so the stronger statement "CP is always at least as good as LP" is not justified.
- Not supported for robustness: the current reflection-only campaign does not reveal a measurable lower-tail or outage advantage for dual-CP. `Delta R_5pct` stays at numerical-noise level and `Delta outage` is `0.0` across every non-proxy pose row.

The safest conclusion sentence is:

> After family-radiated-power normalization, dual-CP is epsilon-non-inferior to dual-linear slant across the non-proxy indoor scenarios considered here. However, the current reflection-only campaign does not yet reveal a measurable lower-tail or outage-robustness advantage for dual-CP.

## What The Current Results Actually Show

The raw CP gains are real in the present outputs, but they disappear after normalization.

- At `10 dB`, raw mean `Delta R_EP` for the selected scenes is:
  - `mixed_office`: `+0.00219`
  - `straight_corridor`: `+0.00629`
  - `factory_lite`: `+0.01357`
- The corresponding normalized mean `Delta R_EP` values are all effectively zero.

That means the apparent CP advantage in the current campaign is best interpreted as coming primarily from family-level radiated-power differences, not from a scene-driven robustness benefit that survives normalization.

## Important Limitations

- `C_WF` is not exported scene-wise in the current campaign output, so Table 5 leaves the capacity columns empty.
- The normalized absolute rate scale should be treated as a comparison unit, not as a physical throughput claim.
- The current setup is still reflection-only and field-only. It does not yet include accepted-power mismatch, corner diffraction, or realistic blocker shadowing, so it is not a strong environment for demonstrating a lower-tail robustness gain.

## Recommended Next Steps

1. Extend the campaign export to include scene-wise `C_WF`.
2. Add a roll-only pose stress protocol if the goal is to isolate polarization-rotation robustness.
3. Add blocker and diffraction modeling before making any stronger lower-tail or outage claim.

## Plot Script

Run:

```bash
python phase3_materials/plot_plan3_results.py --root D:/codex/ISAC_communication_verification --out D:/codex/ISAC_communication_verification/outputs/phase3_materials_report
```
