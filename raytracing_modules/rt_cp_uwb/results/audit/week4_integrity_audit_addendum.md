# Week 4 Integrity Audit Addendum

- Item 15 (`MATLAB <-> CSV consistency`) should be evaluated against the relabeled export pair:
  - [stage2_900_ffd_relabel.mat](/D:/codex/raytracing_modules/rt_cp_uwb/results/stage2/stage2_900_ffd_relabel.mat:1)
  - [stage2_900_ffd_relabel.csv](/D:/codex/raytracing_modules/rt_cp_uwb/results/stage2/stage2_900_ffd_relabel.csv:1)

- Quick re-check on sampled case ids `{1, 123, 450, 777, 900}`:
  - `PASS = 1`
  - `DETAIL = ok`

- The original [week4_integrity_audit.md](/D:/codex/raytracing_modules/rt_cp_uwb/results/audit/week4_integrity_audit.md:1) compared the original Stage 2 MAT with the relabeled CSV, so item 15 there is stale.
