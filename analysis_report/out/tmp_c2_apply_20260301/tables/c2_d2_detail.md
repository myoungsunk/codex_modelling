# C2/D2 Detailed Diagnostic Report

- c0_delta_floor_db: 1.2214848660738156

## C2 (A4 material / A5 stress)

### A4 set: A4_tuned3
- csv: /Users/kimmyoungsun/Documents/codex/outputs/step1_diag/A4_tuned3/link_metrics.csv
- n_links: 72
- material_shift_range_db: 8.189290571385877
- exceeds_delta_floor: True
- material medians:
  - glass: early=-9.378960553911483, late=-8.145146729919597, L_pol=-1.3194624882938246
  - gypsum: early=-5.115264947632366, late=-4.14703972865855, L_pol=-1.47015156102449
  - wood: early=-1.1896699825256054, late=0.0, L_pol=-1.0301840625651537
- material×incidence coverage:
  - glass: {'low': 2, 'mid': 10, 'high': 12, 'NA': 0}
  - gypsum: {'low': 2, 'mid': 10, 'high': 12, 'NA': 0}
  - wood: {'low': 2, 'mid': 10, 'high': 12, 'NA': 0}

### A5 base vs stress effect
- XPD_early_db: Δmedian=-5.8962649490633785, var_ratio=3.506373105940637, q10_shift=-7.1693281841592285, ks_p=1.6911233892144742e-17
- XPD_late_db: Δmedian=0.0, var_ratio=nan, q10_shift=-0.0478423437932698, ks_p=0.9578462903438838
- L_pol_db: Δmedian=-5.852639695953271, var_ratio=5.141924790886765, q10_shift=-7.1693281841592285, ks_p=1.6911233892144742e-17

## D2 Identifiability

- n_rows: 132
- design_rank / cols: 11 / 13
- condition_number: 2.5405077257822573e+17
- vif_threshold: 5.0
- vif: {'d_m': 6.5180744468003065, 'EL_proxy_db': 13.290922038506494}
- vif_warnings: {'d_m': 6.5180744468003065, 'EL_proxy_db': 13.290922038506494}
- stress_x_incidence_coverage: {'base': {'low': 0, 'mid': 0, 'high': 0, 'NA': 30}, 'stress': {'low': 0, 'mid': 0, 'high': 0, 'NA': 30}}
