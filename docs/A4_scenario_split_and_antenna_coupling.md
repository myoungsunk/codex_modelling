# A4 Scenario Split and Antenna Coupling Gap — Design Notes

## A4 Scenario: A4_iso vs A4_bridge Split

### Current default (A4 as run)
```
include_late_panel = True       # secondary delayed panel ALWAYS present
material_dispersion = "off"     # all materials → constant eps_r, tan_delta
```

### Recommended split for paper

| Variant    | include_late_panel | material_dispersion | Role                              |
|------------|-------------------|---------------------|-----------------------------------|
| A4_iso     | False             | off                 | L2-M primary endpoint: primary reflection material effect only |
| A4_bridge  | True              | on (table/debye)    | L2-M secondary: does material change propagate into late/L_pol? |

### Why the split matters
- Current A4 default: `include_late_panel=True` means `XPD_late_ex` and `L_pol` receive contributions from **both** the primary panel AND the secondary delayed panel of the **same** material.
- This conflates "material changes early reflection" with "material changes late leakage."
- A4_iso isolates the primary endpoint (`XPD_early_ex`, `XPD_target_ex`).
- A4_bridge tests whether the late/leakage path changes with material.

### Code: no changes needed
The split is achieved purely through run-time parameters:
```python
# A4_iso
run_case(params, f_hz, include_late_panel=False, material_dispersion="off")

# A4_bridge
run_case(params, f_hz, include_late_panel=True, material_dispersion="on")
```

The `run_case()` signature in `scenarios/A4_dielectric_plane.py` already supports both arguments.

### Claim scope
- `A4_iso`: "Material affects the polarization of the reflected wave at the primary surface."
  → Safe claim; isolated single-surface effect.
- `A4_bridge`: "Material-dependent changes persist into the late window and affect L_pol."
  → Requires `dispersion=on` and `include_late_panel=True`; must note that late panel uses the same material.
- Do NOT claim "broadband dispersive material validation" from `dispersion="off"` runs.

---

## Antenna Coupling Floor vs C0 Simulation Floor

### Numbers
| Parameter              | Value       | Source                             |
|------------------------|-------------|-------------------------------------|
| Tx cross_pol_leakage   | 35.0 dB     | `scenarios/common.py` default      |
| Rx cross_pol_leakage   | 35.0 dB     | `scenarios/common.py` default      |
| C0 simulated XPD_floor | 24.93 dB    | `floor_reference_used.json`        |
| delta_floor            | 0.61 dB     | `floor_reference_used.json`        |

### Why the gap (35 dB → 24.93 dB) is expected
The per-antenna cross-pol leakage spec (35 dB) is the **isolation at one end**.
The combined system floor from two antennas is lower:
- Tx emits: main polarization + 35 dB isolation leakage
- Rx receives both; the received cross-pol from the channel is convolved with Rx's own leakage
- The effective system XPD floor (as measured in the co/cross channel) reflects the cascade of both antenna leakages plus any path geometry effects

The 24.93 dB system floor is self-consistent with the 35 dB per-antenna spec given the simulation's cascade model.

### What to document
In the calibration section:
1. State the per-antenna coupling parameters explicitly (`tx_cross_pol_leakage_db=35`, `rx_cross_pol_leakage_db=35`).
2. State the resulting simulated C0 floor: `XPD_floor = 24.93 dB, delta_floor = 0.61 dB (p5/p95 method, n=30)`.
3. Note: if matching to real measurement, the measurement-derived floor replaces this simulated value. The delta_floor tolerance (0.61 dB) is the uncertainty budget for all excess-domain claims.

### Consequence for excess-domain claims
Any `XPD_excess` claim is only valid if `|XPD_excess| > delta_ref = max(delta_floor, delta_repeat)`.
Currently `delta_floor ≈ 0.61 dB`, so claims are safe as long as effect sizes exceed ~1.2 dB.
The `claim_caution` flag in `metrics.apply_floor_excess()` marks points where this criterion is not met.

### Action if real measurement floor differs
If measured C0 floor differs from simulated 24.93 dB by more than ~1 dB, replace `xpd_floor_db` in `floor_reference_used.json` with the measured value and re-run the diagnostic pipeline to recompute all excess-domain metrics.
