# HFSS Stage 3 Launch Checklist

## Selection Lock

- [x] Use deterministic Stage 2 relabeled candidate list:
  [hfss_case_list_det.csv](/D:/codex/raytracing_modules/rt_cp_uwb/results/stage3/hfss_case_list_det.csv:1)
- [x] Use deterministic regime table:
  [stage3_reselection_cells_det.csv](/D:/codex/raytracing_modules/rt_cp_uwb/results/stage3/stage3_reselection_cells_det.csv:1)
- [x] Do not revert to pre-deterministic Stage 1 top-cell ranking

## Validation Rule Lock

- [x] Primary validation target:
  case-level feature agreement and room-internal regime ranking
- [x] Secondary validation target:
  sign and relative magnitude of `Joint - CIR` lift
- [x] Absolute AUC comparison is supplemental
- [x] Room B absolute AUC mismatch is descriptive only
- [x] GEO group is interpreted as Room C geometric-blockage validation

Reference:
- [stage3_validation_protocol.md](/D:/codex/raytracing_modules/rt_cp_uwb/results/stage3/stage3_validation_protocol.md:1)

## Upstream Stability Checks

- [x] Stage 2 deterministic rerun complete
- [x] Stage 2 `mixed@0.33` `Delta AUC` shift within `0.01`
  - original: `+0.0225`
  - deterministic: `+0.0248`
  - shift: `+0.0023`
- [x] Stage 1 deterministic rerun complete
- [x] Stage 1 `Delta AUC` shift within `0.01`
  - original: `+0.0410`
  - deterministic: `+0.0463`
  - shift: `+0.0053`
- [x] Determinism audit checks `6/7/8` pass

References:
- [deterministic_delta_compare.md](/D:/codex/raytracing_modules/rt_cp_uwb/results/stage2/deterministic_delta_compare.md:1)
- [stage1_det_compare.md](/D:/codex/raytracing_modules/rt_cp_uwb/results/stage1/stage1_det_compare.md:1)
- [week4_integrity_audit.md](/D:/codex/raytracing_modules/rt_cp_uwb/results/audit/week4_integrity_audit.md:1)

## Known Limitations To Preserve In Interpretation

- [x] `patch_ffd` odd-bounce reversal is moderate, not ideal-like
- [x] `negative xpd_db_at_los` remains supplemental only
- [x] `xpol_coupling_db` is a depolarization proxy, not a direct HFSS physical input
- [x] Stage 1 exact top-3 conditional cells are not stable enough to be treated as hard ground truth

## HFSS Output Expectations

- [ ] For each case, preserve MATLAB case_id mapping
- [ ] Compare HFSS vs MATLAB at case level before room/global aggregation
- [ ] Record feature ranking agreement inside each room/regime
- [ ] Record whether CP rescues CIR-fail cases in the same direction as MATLAB
- [ ] Treat absolute AUC as a secondary summary, not the primary pass/fail gate

## Go / No-Go

Current status: **GO**

Reason:

- deterministic blockers are cleared
- Stage 2 deterministic relabeled list is locked
- Stage 1 deterministic recheck does not invalidate the Stage 2/3 basis
- remaining ambiguity is interpretive, not pipeline-breaking
