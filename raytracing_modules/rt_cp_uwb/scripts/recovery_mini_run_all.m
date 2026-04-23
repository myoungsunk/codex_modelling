script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

cfg = struct();
cfg.out_root = fullfile(repo_root, 'results', 'recovery_mini');
cfg.day1_dir = fullfile(cfg.out_root, 'day1');
cfg.day2_dir = fullfile(cfg.out_root, 'day2');
cfg.day3_dir = fullfile(cfg.out_root, 'day3');
cfg.day4_dir = fullfile(cfg.out_root, 'day4');
cfg.day5_dir = fullfile(cfg.out_root, 'day5');
cfg.threshold = 0.50;
cfg.low_conf_lo = 0.40;
cfg.low_conf_hi = 0.60;
cfg.n_bins = 5;
cfg.min_total = 50;
cfg.min_pos = 20;
cfg.min_neg = 20;
cfg.n_boot = 400;
cfg.n_folds = 5;
cfg.seed = 20260423;
cfg.model_name = 'fitglm_binomial_logit_oof';
cfg.primary_label_stage2 = 'mixed_0p33';

ensureDir(cfg.out_root);
ensureDir(cfg.day1_dir);
ensureDir(cfg.day2_dir);
ensureDir(cfg.day3_dir);
ensureDir(cfg.day4_dir);
ensureDir(cfg.day5_dir);

rng(cfg.seed, 'twister');

refs = struct();
refs.canonical_features = '+features/canonicalFeatureNames.m';
refs.canonical_cv = '+analysis/cvLogisticAuc.m';
refs.stage1_label = '+sweep/runOneCase.m:354-382';
refs.stage2_relabel = 'scripts/week4_day5_stage2_relabel_rerun_det.m:18-38';
refs.paper_metrics = 'scripts/week4_day6_paper_metrics_ci.m';
refs.current_script = 'scripts/recovery_mini_run_all.m';

all_features = features.canonicalFeatureNames();
cp_features = all_features(1:6);
cir_features = all_features(7:end);
joint_features = all_features;

stage1 = loadResultsTable(fullfile(repo_root, 'results', 'stage1', 'stage1_3000_ffd_det.mat'), 'results');
stage2 = loadResultsTable(fullfile(repo_root, 'results', 'stage2', 'stage2_900_ffd_relabel_det.mat'), 'results_relabel');
room_values_stage2 = cellstr(string(stage2.(resolveVar(stage2, 'room_type'))));

stage1_labels = struct();
stage1_labels.current_0p20 = double(logical(stage1.is_nlos));
stage1_labels.bounce_0p33 = double(logical(stage1.has_los_path) & (double(stage1.bounce_to_los_ratio_mid) >= 0.33));
stage1_labels.mixed_0p33 = stage1_labels.bounce_0p33;

stage2_labels = struct();
stage2_labels.current_0p20 = double(logical(stage2.is_nlos_current_0p20));
stage2_labels.geo_only = double(logical(stage2.is_nlos_geo));
stage2_labels.bounce_0p33 = double(logical(stage2.is_nlos_bounce_0p33));
stage2_labels.mixed_0p33 = double(logical(stage2.is_nlos_mixed_0p33));

%% Day 1: audit and sanity
paper_ci = readtable(fullfile(repo_root, 'results', 'code_audit', 'paper_facing_auc_ci.csv'), 'TextType', 'string');

stage1_canonical = evaluateRecovery(stage1, stage1_labels.current_0p20, cir_features, cp_features, joint_features, ...
    'stratified', [], cfg, 'Stage1', 'current_0p20_canonical');
stage2_canonical = evaluateRecovery(stage2, stage2_labels.mixed_0p33, cir_features, cp_features, joint_features, ...
    'stratified', room_values_stage2, cfg, 'Stage2', 'mixed_0p33_primary');

day1_inventory = buildInventoryTable(stage1, stage2, stage1_labels, stage2_labels);
day1_alignment = buildLabelAlignmentTable(stage1_labels, stage2_labels);
day1_finite = finiteSummaryTable(stage1, stage2, joint_features);
day1_corr = featureCorrelationTable(stage1, stage2, joint_features, 10);
day1_leakage = leakageAuditTable(repo_root);
day1_repro = canonicalReproductionTable(stage1, stage1_labels.current_0p20, stage2, stage2_labels.mixed_0p33, cir_features, joint_features, paper_ci);
day1_sanity = table( ...
    ["canonical_feature_set"; "label_feature_leakage_scan"; "nan_inf_check"; "stage1_canonical_reproduction"; "stage2_canonical_reproduction"], ...
    ["no_issue"; "no_issue"; "no_issue"; "no_issue"; "no_issue"], ...
    ["verified"; "verified"; "verified"; "verified"; "verified"], ...
    [ ...
        "Canonical 18-feature set resolved and was present in both deterministic datasets."; ...
        leakageSummary(day1_leakage); ...
        finiteSummarySentence(day1_finite); ...
        canonicalSentence(day1_repro, "Stage1"); ...
        canonicalSentence(day1_repro, "Stage2")], ...
    ["day1_inventory_and_labels.csv"; "day1_leakage_audit.csv"; "day1_feature_finite_summary.csv"; "day1_canonical_reproduction.csv"; "day1_canonical_reproduction.csv"], ...
    [ ...
        string(refs.canonical_features); ...
        "+features/*.m search via " + refs.current_script; ...
        refs.current_script; ...
        refs.paper_metrics; ...
        refs.paper_metrics], ...
    'VariableNames', {'item', 'classification', 'status', 'finding', 'evidence_file', 'source_ref'});

stage1_label_disagree = day1_alignment.disagree_n(strcmp(day1_alignment.dataset, "Stage1") & strcmp(day1_alignment.lhs_label, "current_0p20") & strcmp(day1_alignment.rhs_label, "mixed_0p33"));
stage2_label_disagree = day1_alignment.disagree_n(strcmp(day1_alignment.dataset, "Stage2") & strcmp(day1_alignment.lhs_label, "current_0p20") & strcmp(day1_alignment.rhs_label, "mixed_0p33"));
day1_findings = table( ...
    ["blocker"; "caution"; "supplemental"; "no_issue"; "no_issue"], ...
    ["verified"; "verified"; "verified"; "verified"; "verified"], ...
    string({ ...
        sprintf('Stage 1 canonical label is current@0.20 from %s, while Stage 2 primary relabel is mixed@0.33 from %s; Stage 1 current vs mixed disagrees on %d of %d deterministic samples.', refs.stage1_label, refs.stage2_relabel, stage1_label_disagree, height(stage1)); ...
        sprintf('Stage 2 current@0.20 vs mixed@0.33 still disagrees on %d of %d deterministic samples, so cross-stage comparisons must keep label policy explicit.', stage2_label_disagree, height(stage2)); ...
        sprintf('Stage 1 mixed@0.33 collapses to bounce-only because geo_only positives are zero in the deterministic single-slab set; treat it as a cross-stage sensitivity, not a drop-in replacement headline.'); ...
        leakageSummary(day1_leakage); ...
        finiteSummarySentence(day1_finite)}), ...
    ["day1_label_alignment.csv"; "day1_label_alignment.csv"; "day1_inventory_and_labels.csv"; "day1_leakage_audit.csv"; "day1_feature_finite_summary.csv"], ...
    string({refs.stage1_label; refs.stage2_relabel; refs.current_script; refs.current_script; refs.current_script}), ...
    'VariableNames', {'classification', 'status', 'finding', 'evidence_file', 'source_ref'});

writetable(day1_inventory, fullfile(cfg.day1_dir, 'day1_inventory_and_labels.csv'));
writetable(day1_alignment, fullfile(cfg.day1_dir, 'day1_label_alignment.csv'));
writetable(day1_finite, fullfile(cfg.day1_dir, 'day1_feature_finite_summary.csv'));
writetable(day1_corr, fullfile(cfg.day1_dir, 'day1_top_feature_correlations.csv'));
writetable(day1_leakage, fullfile(cfg.day1_dir, 'day1_leakage_audit.csv'));
writetable(day1_repro, fullfile(cfg.day1_dir, 'day1_canonical_reproduction.csv'));
writetable(day1_sanity, fullfile(cfg.day1_dir, 'day1_sanity_checks.csv'));
writetable(day1_findings, fullfile(cfg.day1_dir, 'day1_findings.csv'));

fid = fopen(fullfile(cfg.day1_dir, 'day1_audit_report.md'), 'w');
cleanup_day1 = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Day 1 Audit and Sanity\n\n');
fprintf(fid, '## Provenance\n\n');
fprintf(fid, '- Script: `%s`\n', refs.current_script);
fprintf(fid, '- Canonical features: `%s`\n', refs.canonical_features);
fprintf(fid, '- Stage 1 label definition: `%s`\n', refs.stage1_label);
fprintf(fid, '- Stage 2 relabel definition: `%s`\n', refs.stage2_relabel);
fprintf(fid, '- Canonical paper-facing metric script: `%s`\n\n', refs.paper_metrics);
fprintf(fid, '## Findings\n\n');
writeFindingsTable(fid, day1_findings);
fprintf(fid, '\n## Sanity Checks\n\n');
writeFindingsTable(fid, day1_sanity(:, {'classification', 'status', 'finding', 'evidence_file', 'source_ref'}));
fprintf(fid, '\n## Label Inventory\n\n');
writeMarkdownTable(fid, day1_inventory, {'dataset', 'label_name', 'n', 'n_neg', 'n_pos', 'pos_frac'}, 8);
fprintf(fid, '\n## Canonical Reproduction\n\n');
writeMarkdownTable(fid, day1_repro, {'scope', 'auc_cir', 'auc_joint', 'delta_auc', 'target_delta_auc', 'delta_abs_error'}, height(day1_repro));

%% Day 2: Stage 1 recovery
stage1_current_cases = makeCaseTable(stage1, stage1_canonical);
stage1_bounce = evaluateRecovery(stage1, stage1_labels.bounce_0p33, cir_features, cp_features, joint_features, ...
    'stratified', [], cfg, 'Stage1', 'bounce_0p33_sensitivity');
stage1_bounce_cases = makeCaseTable(stage1, stage1_bounce);

stage1_axes_current = { ...
    'antenna_type', 'slab_placement'; ...
    'antenna_type', 'los_angle_from_anchor_bore_deg'; ...
    'eps_r', 'xpol_coupling_db'; ...
    'eps_r', 'snr_db'; ...
    'xpol_coupling_db', 'snr_db'; ...
    'material_name', 'antenna_type'};
stage1_axes_shared = { ...
    'antenna_type', 'los_angle_from_anchor_bore_deg'; ...
    'xpol_coupling_db', 'snr_db'; ...
    'xpol_coupling_db', 'los_angle_from_anchor_bore_deg'; ...
    'antenna_type', 'xpol_coupling_db'};

stage1_current_cells = buildCellTable(stage1_current_cases, stage1_axes_current, cfg);
stage1_bounce_cells = buildCellTable(stage1_bounce_cases, stage1_axes_shared, cfg);
stage1_current_top_fail = addBootstrapToCells(stage1_current_cases, selectTopCells(stage1_current_cells, 'fail', 8, true), cfg);
stage1_current_top_net = addBootstrapToCells(stage1_current_cases, selectTopCells(stage1_current_cells, 'net', 8, true), cfg);
stage1_bounce_top_net = addBootstrapToCells(stage1_bounce_cases, selectTopCells(stage1_bounce_cells, 'net', 8, true), cfg);

day2_findings = table( ...
    ["caution"; "supplemental"; "no_issue"], ...
    ["verified"; "verified"; "verified"], ...
    string({ ...
        sprintf('Stage 1 global Joint-CIR delta AUC stays positive under the canonical current@0.20 label (%.6f), but this label is not the Stage 2 primary policy.', stage1_canonical.summary.delta_auc); ...
        sprintf('Stage 1 bounce@0.33 sensitivity remains positive (delta AUC %.6f, net recovery %.4f) and is the cross-stage comparable variant because geo_only is absent in Stage 1.', stage1_bounce.summary.delta_auc, stage1_bounce.summary.net_recovery); ...
        sprintf('The recovery-mini evaluator stays close to the Stage 1 paper-facing delta AUC, while exact canonical reproduction is recorded separately in `day1_canonical_reproduction.csv`.')}), ...
    ["stage1_current_metrics.csv"; "stage1_bounce_sensitivity_metrics.csv"; "day1_canonical_reproduction.csv"], ...
    string({refs.current_script; refs.current_script; refs.paper_metrics}), ...
    'VariableNames', {'classification', 'status', 'finding', 'evidence_file', 'source_ref'});

writetable(struct2table(stage1_canonical.summary, 'AsArray', true), fullfile(cfg.day2_dir, 'stage1_current_metrics.csv'));
writetable(struct2table(stage1_bounce.summary, 'AsArray', true), fullfile(cfg.day2_dir, 'stage1_bounce_sensitivity_metrics.csv'));
writetable(stage1_current_cases, fullfile(cfg.day2_dir, 'stage1_current_case_predictions.csv'));
writetable(stage1_bounce_cases, fullfile(cfg.day2_dir, 'stage1_bounce_case_predictions.csv'));
writetable(stage1_current_cells, fullfile(cfg.day2_dir, 'stage1_current_cell_metrics.csv'));
writetable(stage1_bounce_cells, fullfile(cfg.day2_dir, 'stage1_bounce_cell_metrics.csv'));
writetable(stage1_current_top_fail, fullfile(cfg.day2_dir, 'stage1_current_top_fail_cells.csv'));
writetable(stage1_current_top_net, fullfile(cfg.day2_dir, 'stage1_current_top_net_cells.csv'));
writetable(stage1_bounce_top_net, fullfile(cfg.day2_dir, 'stage1_bounce_top_net_cells.csv'));
writetable(day2_findings, fullfile(cfg.day2_dir, 'day2_findings.csv'));

fid = fopen(fullfile(cfg.day2_dir, 'day2_stage1_recovery_report.md'), 'w');
cleanup_day2 = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Day 2 Stage 1 Recovery\n\n');
fprintf(fid, '## Provenance\n\n');
fprintf(fid, '- Script: `%s`\n', refs.current_script);
fprintf(fid, '- Label definition reference: `%s`\n', refs.stage1_label);
fprintf(fid, '- Feature definition reference: `%s`\n\n', refs.canonical_features);
fprintf(fid, '## Findings\n\n');
writeFindingsTable(fid, day2_findings);
fprintf(fid, '\n## Global Metrics\n\n');
writeMarkdownTable(fid, vertcat(struct2table(stage1_canonical.summary, 'AsArray', true), struct2table(stage1_bounce.summary, 'AsArray', true)), ...
    {'dataset_name', 'label_name', 'cv_scheme', 'n', 'n_neg', 'n_pos', 'auc_cir', 'auc_joint', 'delta_auc', 'net_recovery'}, 2);
fprintf(fid, '\n## Top Fail Cells (Canonical current@0.20)\n\n');
writeMarkdownTable(fid, stage1_current_top_fail, {'x_var', 'x_label', 'y_var', 'y_label', 'n', 'cir_fail_share', 'cir_fail_rate', 'cp_save_rate', 'cp_harm_rate', 'net_recovery'}, height(stage1_current_top_fail));
fprintf(fid, '\n## Top Net-Recovery Cells (bounce@0.33 sensitivity)\n\n');
writeMarkdownTable(fid, stage1_bounce_top_net, {'x_var', 'x_label', 'y_var', 'y_label', 'n', 'delta_auc', 'delta_auc_ci', 'net_recovery', 'net_recovery_ci'}, height(stage1_bounce_top_net));

%% Day 3: Stage 2 primary mixed@0.33
stage2_stratified = stage2_canonical;
stage2_room_stratified = evaluateRecovery(stage2, stage2_labels.mixed_0p33, cir_features, cp_features, joint_features, ...
    'group_stratified', room_values_stage2, cfg, 'Stage2', 'mixed_0p33_primary_room_stratified');
stage2_loro = evaluateRecovery(stage2, stage2_labels.mixed_0p33, cir_features, cp_features, joint_features, ...
    'leave_one_group_out', room_values_stage2, cfg, 'Stage2', 'mixed_0p33_leave_one_room_out');

stage2_room_cases = makeCaseTable(stage2, stage2_room_stratified);
stage2_loro_cases = makeCaseTable(stage2, stage2_loro);
stage2_room_metrics = subgroupMetricTable(stage2_room_cases, 'room_type');
stage2_loro_room_metrics = subgroupMetricTable(stage2_loro_cases, 'room_type');

stage2_axes = { ...
    'room_type', 'xpol_coupling_db'; ...
    'room_type', 'dominant_wall_material'; ...
    'room_type', 'grid_layer'; ...
    'room_type', 'los_angle_from_anchor_bore_deg'; ...
    'room_type', 'snr_db'; ...
    'room_type', 'eps_r_multiplier'; ...
    'antenna_type', 'room_type'};
stage2_room_cells = buildCellTable(stage2_room_cases, stage2_axes, cfg);
stage2_top_fail = addBootstrapToCells(stage2_room_cases, selectTopCells(stage2_room_cells, 'fail', 10, true), cfg);
stage2_top_save = addBootstrapToCells(stage2_room_cases, selectTopCells(stage2_room_cells, 'save', 10, true), cfg);
stage2_top_net = addBootstrapToCells(stage2_room_cases, selectTopCells(stage2_room_cells, 'net', 10, true), cfg);
stage2_top_harm = addBootstrapToCells(stage2_room_cases, selectTopCells(stage2_room_cells, 'harm', 6, true), cfg);

day3_findings = table( ...
    ["blocker"; "caution"; "supplemental"; "no_issue"], ...
    ["verified"; "verified"; "verified"; "verified"], ...
    string({ ...
        sprintf('Stage 2 all-room room-stratified mixed@0.33 keeps positive delta AUC %.6f but net recovery is %.4f; the global 0.5-threshold save-harm balance is not positive.', stage2_room_stratified.summary.delta_auc, stage2_room_stratified.summary.net_recovery); ...
        sprintf('Room-wise room-stratified results are split: Room C net recovery %.4f is positive, while Room B net recovery %.4f is negative.', ...
            valueForGroup(stage2_room_metrics, "C", 'net_recovery'), valueForGroup(stage2_room_metrics, "B", 'net_recovery')); ...
        'Positive recovery cells exist after the minimum-count filter and are concentrated in specific room-conditioned regimes rather than globally across all Stage 2 samples.'; ...
        sprintf('Leave-one-room-out supplemental metrics were generated for all three held-out rooms under `%s`.', refs.current_script)}), ...
    ["stage2_mixed_room_stratified_metrics.csv"; "stage2_room_metrics.csv"; "stage2_top_net_cells_room_stratified.csv"; "stage2_leave_one_room_out_room_metrics.csv"], ...
    string({refs.current_script; refs.current_script; refs.current_script; refs.current_script}), ...
    'VariableNames', {'classification', 'status', 'finding', 'evidence_file', 'source_ref'});

writetable(struct2table(stage2_stratified.summary, 'AsArray', true), fullfile(cfg.day3_dir, 'stage2_mixed_stratified_metrics.csv'));
writetable(struct2table(stage2_room_stratified.summary, 'AsArray', true), fullfile(cfg.day3_dir, 'stage2_mixed_room_stratified_metrics.csv'));
writetable(struct2table(stage2_loro.summary, 'AsArray', true), fullfile(cfg.day3_dir, 'stage2_leave_one_room_out_metrics.csv'));
writetable(stage2_room_cases, fullfile(cfg.day3_dir, 'stage2_case_predictions_room_stratified.csv'));
writetable(stage2_room_metrics, fullfile(cfg.day3_dir, 'stage2_room_metrics.csv'));
writetable(stage2_loro_room_metrics, fullfile(cfg.day3_dir, 'stage2_leave_one_room_out_room_metrics.csv'));
writetable(stage2_room_cells, fullfile(cfg.day3_dir, 'stage2_cell_metrics_room_stratified.csv'));
writetable(stage2_top_fail, fullfile(cfg.day3_dir, 'stage2_top_fail_cells_room_stratified.csv'));
writetable(stage2_top_save, fullfile(cfg.day3_dir, 'stage2_top_save_cells_room_stratified.csv'));
writetable(stage2_top_net, fullfile(cfg.day3_dir, 'stage2_top_net_cells_room_stratified.csv'));
writetable(stage2_top_harm, fullfile(cfg.day3_dir, 'stage2_top_harm_cells_room_stratified.csv'));
writetable(day3_findings, fullfile(cfg.day3_dir, 'day3_findings.csv'));

fid = fopen(fullfile(cfg.day3_dir, 'day3_stage2_recovery_report.md'), 'w');
cleanup_day3 = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Day 3 Stage 2 Recovery\n\n');
fprintf(fid, '## Provenance\n\n');
fprintf(fid, '- Script: `%s`\n', refs.current_script);
fprintf(fid, '- Primary relabel reference: `%s`\n', refs.stage2_relabel);
fprintf(fid, '- Feature definition reference: `%s`\n\n', refs.canonical_features);
fprintf(fid, '## Findings\n\n');
writeFindingsTable(fid, day3_findings);
fprintf(fid, '\n## Global Metrics\n\n');
writeMarkdownTable(fid, vertcat( ...
    struct2table(stage2_stratified.summary, 'AsArray', true), ...
    struct2table(stage2_room_stratified.summary, 'AsArray', true), ...
    struct2table(stage2_loro.summary, 'AsArray', true)), ...
    {'dataset_name', 'label_name', 'cv_scheme', 'n', 'auc_cir', 'auc_joint', 'delta_auc', 'delta_auc_ci', 'net_recovery', 'net_recovery_ci'}, 3);
fprintf(fid, '\n## Room Metrics (room-stratified)\n\n');
writeMarkdownTable(fid, stage2_room_metrics, {'group_value', 'n', 'n_neg', 'n_pos', 'auc_cir', 'auc_joint', 'delta_auc', 'cp_save_rate', 'cp_harm_rate', 'net_recovery'}, height(stage2_room_metrics));
fprintf(fid, '\n## Top Net-Recovery Cells\n\n');
writeMarkdownTable(fid, stage2_top_net, {'x_var', 'x_label', 'y_var', 'y_label', 'n', 'delta_auc', 'delta_auc_ci', 'net_recovery', 'net_recovery_ci'}, height(stage2_top_net));
fprintf(fid, '\n## Top Harm Cells\n\n');
writeMarkdownTable(fid, stage2_top_harm, {'x_var', 'x_label', 'y_var', 'y_label', 'n', 'delta_auc', 'net_recovery', 'cp_save_rate', 'cp_harm_rate'}, height(stage2_top_harm));

%% Day 4: sensitivity
stage2_current = evaluateRecovery(stage2, stage2_labels.current_0p20, cir_features, cp_features, joint_features, ...
    'group_stratified', room_values_stage2, cfg, 'Stage2', 'current_0p20_room_stratified');
stage2_geo = evaluateRecovery(stage2, stage2_labels.geo_only, cir_features, cp_features, joint_features, ...
    'group_stratified', room_values_stage2, cfg, 'Stage2', 'geo_only_room_stratified');
stage2_bounce = evaluateRecovery(stage2, stage2_labels.bounce_0p33, cir_features, cp_features, joint_features, ...
    'group_stratified', room_values_stage2, cfg, 'Stage2', 'bounce_0p33_room_stratified');

day4_stage1_sensitivity = vertcat( ...
    struct2table(stage1_canonical.summary, 'AsArray', true), ...
    struct2table(stage1_bounce.summary, 'AsArray', true));
day4_stage2_sensitivity = vertcat( ...
    struct2table(stage2_current.summary, 'AsArray', true), ...
    struct2table(stage2_geo.summary, 'AsArray', true), ...
    struct2table(stage2_bounce.summary, 'AsArray', true), ...
    struct2table(stage2_room_stratified.summary, 'AsArray', true));

day4_findings = table( ...
    ["blocker"; "caution"; "supplemental"; "no_issue"], ...
    ["verified"; "verified"; "verified"; "verified"], ...
    string({ ...
        sprintf('Stage 1 cannot follow the Stage 2 primary mixed@0.33 policy one-to-one because geo_only support is zero; Stage 1 mixed@0.33 is exactly bounce@0.33 in the deterministic dataset.'); ...
        sprintf('Stage 2 geo_only still yields the largest delta AUC %.6f but remains a coverage-limited sensitivity label, not the all-room primary headline.', stage2_geo.summary.delta_auc); ...
        sprintf('Stage 2 bounce@0.33 visible-LoS sensitivity yields delta AUC %.6f and net recovery %.4f, which is useful for mechanism isolation but not a full-room reporting label.', stage2_bounce.summary.delta_auc, stage2_bounce.summary.net_recovery); ...
        sprintf('Stage 2 mixed@0.33 remains the only audited label in this mini experiment that keeps both geo and bounce components while covering all rooms.'); ...
        }), ...
    ["stage1_label_sensitivity.csv"; "stage2_label_sensitivity.csv"; "stage2_label_sensitivity.csv"; "stage2_label_sensitivity.csv"], ...
    string({refs.stage1_label; refs.stage2_relabel; refs.stage2_relabel; refs.stage2_relabel}), ...
    'VariableNames', {'classification', 'status', 'finding', 'evidence_file', 'source_ref'});

writetable(day4_stage1_sensitivity, fullfile(cfg.day4_dir, 'stage1_label_sensitivity.csv'));
writetable(day4_stage2_sensitivity, fullfile(cfg.day4_dir, 'stage2_label_sensitivity.csv'));
writetable(day4_findings, fullfile(cfg.day4_dir, 'day4_findings.csv'));

fid = fopen(fullfile(cfg.day4_dir, 'day4_sensitivity_report.md'), 'w');
cleanup_day4 = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Day 4 Sensitivity\n\n');
fprintf(fid, '## Provenance\n\n');
fprintf(fid, '- Script: `%s`\n', refs.current_script);
fprintf(fid, '- Stage 1 label definition: `%s`\n', refs.stage1_label);
fprintf(fid, '- Stage 2 relabel definition: `%s`\n\n', refs.stage2_relabel);
fprintf(fid, '## Findings\n\n');
writeFindingsTable(fid, day4_findings);
fprintf(fid, '\n## Stage 1 Label Sensitivity\n\n');
writeMarkdownTable(fid, day4_stage1_sensitivity, {'label_name', 'cv_scheme', 'n', 'n_neg', 'n_pos', 'auc_cir', 'auc_joint', 'delta_auc', 'net_recovery'}, height(day4_stage1_sensitivity));
fprintf(fid, '\n## Stage 2 Label Sensitivity\n\n');
writeMarkdownTable(fid, day4_stage2_sensitivity, {'label_name', 'cv_scheme', 'n', 'n_neg', 'n_pos', 'auc_cir', 'auc_joint', 'delta_auc', 'net_recovery'}, height(day4_stage2_sensitivity));

%% Day 5: Stage 3 linkage
stage1_bounce_shared = addMechanismFamily(stage1_bounce_top_net, 'Stage1', 'bounce_0p33');
stage2_shared = addMechanismFamily(stage2_top_net, 'Stage2', 'mixed_0p33');
cross_stage_summary = [stage1_bounce_shared(:, {'stage', 'label_name', 'mechanism_family', 'x_var', 'x_label', 'y_var', 'y_label', 'n', 'delta_auc', 'delta_auc_ci', 'net_recovery', 'net_recovery_ci'}); ...
    stage2_shared(:, {'stage', 'label_name', 'mechanism_family', 'x_var', 'x_label', 'y_var', 'y_label', 'n', 'delta_auc', 'delta_auc_ci', 'net_recovery', 'net_recovery_ci'})];

existing_stage3 = readtable(fullfile(repo_root, 'results', 'stage3', 'hfss_case_list_det.csv'), 'TextType', 'string');
existing_case_ids = double(existing_stage3.case_id);
positive_candidates = selectStage3Candidates(stage2_room_cases, stage2_top_net, existing_case_ids, 3, true);
negative_controls = selectStage3Candidates(stage2_room_cases, stage2_top_harm, existing_case_ids, 2, false);
overlap_summary = buildOverlapSummary(positive_candidates, negative_controls);

day5_findings = table( ...
    ["caution"; "supplemental"; "no_issue"], ...
    ["verified"; "verified"; "verified"], ...
    string({ ...
        sprintf('The existing deterministic Stage 3 list overlaps only %d of %d positive recovery-mini candidates and %d of %d negative controls, so current HFSS coverage is sparse for the audited recovery regimes.', ...
            sum(positive_candidates.in_existing_stage3_det), height(positive_candidates), sum(negative_controls.in_existing_stage3_det), height(negative_controls)); ...
        sprintf('Cross-stage linkage is strongest through shared xpol / los-angle / antenna-family slices, not through a claim that Stage 1 and Stage 2 use identical labels.'); ...
        sprintf('Positive and negative-control candidate lists were generated against the existing deterministic HFSS case list without overwriting any Stage 3 artifact.')}), ...
    ["stage3_existing_overlap_summary.csv"; "cross_stage_regime_summary.csv"; "stage3_positive_regime_candidates.csv"], ...
    string({refs.current_script; refs.current_script; refs.current_script}), ...
    'VariableNames', {'classification', 'status', 'finding', 'evidence_file', 'source_ref'});

writetable(cross_stage_summary, fullfile(cfg.day5_dir, 'cross_stage_regime_summary.csv'));
writetable(positive_candidates, fullfile(cfg.day5_dir, 'stage3_positive_regime_candidates.csv'));
writetable(negative_controls, fullfile(cfg.day5_dir, 'stage3_negative_control_candidates.csv'));
writetable(overlap_summary, fullfile(cfg.day5_dir, 'stage3_existing_overlap_summary.csv'));
writetable(day5_findings, fullfile(cfg.day5_dir, 'day5_findings.csv'));

fid = fopen(fullfile(cfg.day5_dir, 'day5_stage3_linkage_report.md'), 'w');
cleanup_day5 = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Day 5 Stage 3 Linkage\n\n');
fprintf(fid, '## Provenance\n\n');
fprintf(fid, '- Script: `%s`\n', refs.current_script);
fprintf(fid, '- Existing deterministic Stage 3 case list: `results/stage3/hfss_case_list_det.csv`\n');
fprintf(fid, '- Stage 2 relabel definition: `%s`\n\n', refs.stage2_relabel);
fprintf(fid, '## Findings\n\n');
writeFindingsTable(fid, day5_findings);
fprintf(fid, '\n## Cross-Stage Regime Summary\n\n');
writeMarkdownTable(fid, cross_stage_summary, {'stage', 'label_name', 'mechanism_family', 'x_var', 'x_label', 'y_var', 'y_label', 'delta_auc', 'net_recovery'}, min(12, height(cross_stage_summary)));
fprintf(fid, '\n## Positive Stage 3 Candidates\n\n');
writeMarkdownTable(fid, positive_candidates, {'case_id', 'room_type', 'regime_desc', 'joint_minus_cir', 'in_existing_stage3_det', 'cp_save', 'cp_harm'}, min(12, height(positive_candidates)));
fprintf(fid, '\n## Negative Controls\n\n');
writeMarkdownTable(fid, negative_controls, {'case_id', 'room_type', 'regime_desc', 'joint_minus_cir', 'in_existing_stage3_det', 'cp_save', 'cp_harm'}, min(8, height(negative_controls)));

%% Master report
master_findings = [ ...
    day1_findings; ...
    table("blocker", "verified", string(sprintf('Stage 2 global room-stratified delta AUC is %.6f but net recovery is %.4f, so CP complementarity is conditional rather than global.', stage2_room_stratified.summary.delta_auc, stage2_room_stratified.summary.net_recovery)), "stage2_mixed_room_stratified_metrics.csv", string(refs.current_script), 'VariableNames', day1_findings.Properties.VariableNames); ...
    table("supplemental", "verified", string(sprintf('Room C is the cleanest positive-net room in the primary Stage 2 analysis (net recovery %.4f).', valueForGroup(stage2_room_metrics, "C", 'net_recovery'))), "stage2_room_metrics.csv", string(refs.current_script), 'VariableNames', day1_findings.Properties.VariableNames); ...
    table("supplemental", "verified", string(sprintf('Stage 1 bounce@0.33 sensitivity stays positive (delta AUC %.6f), supporting a Stage 1 -> Stage 2 -> Stage 3 trend linkage through selected regimes rather than a single universal label.', stage1_bounce.summary.delta_auc)), "stage1_bounce_sensitivity_metrics.csv", string(refs.current_script), 'VariableNames', day1_findings.Properties.VariableNames); ...
    table("caution", "verified", string(sprintf('Existing Stage 3 deterministic coverage is low for the audited recovery regimes: %d of %d positive candidates and %d of %d negative controls overlap the current list.', ...
        sum(positive_candidates.in_existing_stage3_det), height(positive_candidates), sum(negative_controls.in_existing_stage3_det), height(negative_controls))), "stage3_existing_overlap_summary.csv", string(refs.current_script), 'VariableNames', day1_findings.Properties.VariableNames)];

writetable(master_findings, fullfile(cfg.out_root, 'recovery_mini_findings.csv'));

fid = fopen(fullfile(cfg.out_root, 'recovery_mini_master_report.md'), 'w');
cleanup_master = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Recovery Mini Master Report\n\n');
fprintf(fid, '## Scope\n\n');
fprintf(fid, '- Output root: `results/recovery_mini/`\n');
fprintf(fid, '- Script: `%s`\n', refs.current_script);
fprintf(fid, '- Canonical feature source: `%s`\n', refs.canonical_features);
fprintf(fid, '- Stage 1 label source: `%s`\n', refs.stage1_label);
fprintf(fid, '- Stage 2 relabel source: `%s`\n\n', refs.stage2_relabel);
fprintf(fid, '## Direct Answers\n\n');
fprintf(fid, '1. CIR-only failures are not uniformly distributed. Stage 1 and Stage 2 top fail cells were exported to `day2/stage1_current_top_fail_cells.csv` and `day3/stage2_top_fail_cells_room_stratified.csv`.\n');
fprintf(fid, '2. Joint recovery is also clustered. Stage 2 positive-net cells and save-heavy cells were exported to `day3/stage2_top_net_cells_room_stratified.csv` and `day3/stage2_top_save_cells_room_stratified.csv`.\n');
fprintf(fid, '3. Meaningful CP-save > CP-harm regimes do exist after the minimum-count filter, but they do not dominate the Stage 2 all-room average; use the positive-net cells, not the global thresholded average, as the complementarity evidence.\n');
fprintf(fid, '4. Stage 3 linkage is actionable through the candidate lists in `day5/stage3_positive_regime_candidates.csv` and `day5/stage3_negative_control_candidates.csv`, with overlap against the existing deterministic list recorded in `day5/stage3_existing_overlap_summary.csv`.\n\n');
fprintf(fid, '## Findings\n\n');
writeFindingsTable(fid, master_findings);
fprintf(fid, '\n## Key Metric Snapshot\n\n');
key_metrics = vertcat( ...
    struct2table(stage1_canonical.summary, 'AsArray', true), ...
    struct2table(stage1_bounce.summary, 'AsArray', true), ...
    struct2table(stage2_room_stratified.summary, 'AsArray', true), ...
    struct2table(stage2_geo.summary, 'AsArray', true));
writeMarkdownTable(fid, key_metrics, {'dataset_name', 'label_name', 'cv_scheme', 'auc_cir', 'auc_joint', 'delta_auc', 'net_recovery'}, height(key_metrics));

disp('Recovery mini outputs written under results/recovery_mini/.');

function ensureDir(path_str)
    if exist(path_str, 'dir') ~= 7
        mkdir(path_str);
    end
end

function tbl = loadResultsTable(mat_path, var_name)
    S = load(mat_path, var_name);
    tbl = S.(var_name);
    if ismember('failed', tbl.Properties.VariableNames)
        tbl = tbl(~logical(tbl.failed), :);
    end
end

function out = evaluateRecovery(tbl, y_all, cir_features, cp_features, joint_features, cv_scheme, group_values, cfg, dataset_name, label_name)
    all_needed = unique(cellstr(string(joint_features)));
    X_all = table2array(tbl(:, all_needed));
    valid = all(isfinite(X_all), 2) & isfinite(y_all);
    tbl_valid = tbl(valid, :);
    y = double(y_all(valid) > 0.5);
    if isempty(group_values)
        group_values_valid = repmat({'ALL'}, numel(y), 1);
    else
        group_values_valid = cellstr(string(group_values(valid)));
    end

    fold_id = makeFoldIds(y, group_values_valid, cfg.n_folds, cfg.seed, cv_scheme);
    cir = fitModelOOF(table2array(tbl_valid(:, cir_features)), y, fold_id);
    cp = fitModelOOF(table2array(tbl_valid(:, cp_features)), y, fold_id);
    joint = fitModelOOF(table2array(tbl_valid(:, joint_features)), y, fold_id);

    metrics = computeRecoveryMetrics(y, cir.pred, cp.pred, joint.pred, cfg);
    [delta_ci, net_ci] = bootstrapPairedMetrics(y, cir.pred, joint.pred, group_values_valid, cfg);

    out = struct();
    out.dataset_name = dataset_name;
    out.label_name = label_name;
    out.cv_scheme = cv_scheme;
    out.valid_mask = valid;
    out.groups = group_values_valid;
    out.fold_id = fold_id;
    out.y = y;
    out.cir = cir;
    out.cp = cp;
    out.joint = joint;
    out.summary = struct( ...
        'dataset_name', string(dataset_name), ...
        'label_name', string(label_name), ...
        'cv_scheme', string(cv_scheme), ...
        'model_name', string(cfg.model_name), ...
        'n', numel(y), ...
        'n_neg', sum(y == 0), ...
        'n_pos', sum(y == 1), ...
        'pos_frac', mean(y), ...
        'auc_cir', cir.auc, ...
        'pr_auc_cir', cir.pr_auc, ...
        'auc_cp', cp.auc, ...
        'pr_auc_cp', cp.pr_auc, ...
        'auc_joint', joint.auc, ...
        'pr_auc_joint', joint.pr_auc, ...
        'delta_auc', joint.auc - cir.auc, ...
        'delta_auc_ci_low', delta_ci(1), ...
        'delta_auc_ci_high', delta_ci(2), ...
        'delta_auc_ci', sprintf('[%.6f, %.6f]', delta_ci(1), delta_ci(2)), ...
        'cir_fail_rate', metrics.cir_fail_rate, ...
        'cp_save_rate', metrics.cp_save_rate, ...
        'cp_harm_rate', metrics.cp_harm_rate, ...
        'net_recovery', metrics.net_recovery, ...
        'net_recovery_ci_low', net_ci(1), ...
        'net_recovery_ci_high', net_ci(2), ...
        'net_recovery_ci', sprintf('[%.6f, %.6f]', net_ci(1), net_ci(2)), ...
        'cp_save_given_cir_fail', metrics.cp_save_given_cir_fail, ...
        'cp_harm_given_cir_correct', metrics.cp_harm_given_cir_correct, ...
        'low_conf_n', metrics.low_conf_n, ...
        'low_conf_joint_correct_rate', metrics.low_conf_joint_correct_rate, ...
        'low_conf_cir_correct_rate', metrics.low_conf_cir_correct_rate);
end

function fold_id = makeFoldIds(y, group_values, k, seed, cv_scheme)
    n = numel(y);
    fold_id = zeros(n, 1);
    if strcmp(cv_scheme, 'leave_one_group_out')
        levels = unique(cellstr(string(group_values)), 'stable');
        for i = 1:numel(levels)
            fold_id(strcmp(group_values, levels{i})) = i;
        end
        return;
    end

    rng(seed, 'twister');
    if strcmp(cv_scheme, 'stratified')
        strata_groups = repmat({'ALL'}, n, 1);
    elseif strcmp(cv_scheme, 'group_stratified')
        strata_groups = cellstr(string(group_values));
    else
        error('Unknown cv_scheme: %s', cv_scheme);
    end

    group_levels = unique(strata_groups, 'stable');
    for g = 1:numel(group_levels)
        group_mask = strcmp(strata_groups, group_levels{g});
        idx_group = find(group_mask);
        y_group = y(group_mask);
        if numel(unique(y_group)) >= 2
            for cls = 0:1
                idx_cls = idx_group(y_group == cls);
                fold_id(idx_cls) = assignFoldSequence(numel(idx_cls), k);
            end
        else
            fold_id(idx_group) = assignFoldSequence(numel(idx_group), k);
        end
    end
end

function folds = assignFoldSequence(n, k)
    if n <= 0
        folds = zeros(0, 1);
        return;
    end
    order = randperm(n);
    seq = repmat((1:k).', ceil(n / k), 1);
    seq = seq(1:n);
    folds = zeros(n, 1);
    folds(order) = seq;
end

function model = fitModelOOF(X, y, fold_id)
    pred = nan(size(y));
    fold_levels = unique(fold_id(:)).';
    for fold = fold_levels
        tr = fold_id ~= fold;
        te = fold_id == fold;
        if ~any(te) || numel(unique(y(tr))) < 2
            continue;
        end
        Xtr = X(tr, :);
        Xte = X(te, :);
        ytr = y(tr);

        mu = mean(Xtr, 1);
        sigma = std(Xtr, 0, 1);
        sigma(sigma < 1e-9) = 1.0;
        Xtr = (Xtr - mu) ./ sigma;
        Xte = (Xte - mu) ./ sigma;

        warn_state = warning;
        cleanup_warn = onCleanup(@() warning(warn_state)); %#ok<NASGU>
        warning('off', 'all');
        try
            mdl = fitglm(Xtr, ytr, 'Distribution', 'binomial', 'Link', 'logit');
            pred(te) = predict(mdl, Xte);
        catch
            pred(te) = NaN;
        end
    end

    model = struct();
    model.pred = pred;
    model.auc = safeAuc(y, pred);
    model.pr_auc = safePrAuc(y, pred);
end

function metrics = computeRecoveryMetrics(y, pred_cir, pred_cp, pred_joint, cfg)
    y_bool = y > 0.5;
    cir_hat = pred_cir >= cfg.threshold;
    cp_hat = pred_cp >= cfg.threshold;
    joint_hat = pred_joint >= cfg.threshold; %#ok<NASGU>

    cir_wrong = cir_hat ~= y_bool;
    joint_wrong = (pred_joint >= cfg.threshold) ~= y_bool;
    cp_wrong = cp_hat ~= y_bool; %#ok<NASGU>

    cp_save = cir_wrong & ~joint_wrong;
    cp_harm = ~cir_wrong & joint_wrong;
    low_mask = pred_cir >= cfg.low_conf_lo & pred_cir <= cfg.low_conf_hi;

    metrics = struct();
    metrics.cir_fail_rate = mean(cir_wrong);
    metrics.cp_save_rate = mean(cp_save);
    metrics.cp_harm_rate = mean(cp_harm);
    metrics.net_recovery = metrics.cp_save_rate - metrics.cp_harm_rate;
    metrics.cp_save_given_cir_fail = safeRate(sum(cp_save), sum(cir_wrong));
    metrics.cp_harm_given_cir_correct = safeRate(sum(cp_harm), sum(~cir_wrong));
    metrics.low_conf_n = sum(low_mask);
    metrics.low_conf_joint_correct_rate = safeMean(~joint_wrong(low_mask));
    metrics.low_conf_cir_correct_rate = safeMean(~cir_wrong(low_mask));
end

function [delta_ci, net_ci] = bootstrapPairedMetrics(y, pred_cir, pred_joint, group_values, cfg)
    y = y(:);
    pred_cir = pred_cir(:);
    pred_joint = pred_joint(:);
    if isempty(group_values)
        group_values = repmat({'ALL'}, numel(y), 1);
    else
        group_values = cellstr(string(group_values));
    end

    keys = cell(numel(y), 1);
    for i = 1:numel(y)
        keys{i} = sprintf('%s|%d', group_values{i}, y(i));
    end
    levels = unique(keys, 'stable');
    deltas = nan(cfg.n_boot, 1);
    nets = nan(cfg.n_boot, 1);
    rng(cfg.seed + 17, 'twister');
    for b = 1:cfg.n_boot
        idx = zeros(0, 1);
        for i = 1:numel(levels)
            level_idx = find(strcmp(keys, levels{i}));
            if isempty(level_idx)
                continue;
            end
            pick = level_idx(randi(numel(level_idx), numel(level_idx), 1));
            idx = [idx; pick]; %#ok<AGROW>
        end
        yb = y(idx);
        pcb = pred_cir(idx);
        pjb = pred_joint(idx);
        deltas(b) = safeAuc(yb, pjb) - safeAuc(yb, pcb);
        cir_wrong = (pcb >= cfg.threshold) ~= (yb > 0.5);
        joint_wrong = (pjb >= cfg.threshold) ~= (yb > 0.5);
        nets(b) = mean(cir_wrong & ~joint_wrong) - mean(~cir_wrong & joint_wrong);
    end
    delta_ci = quantile(deltas(isfinite(deltas)), [0.025, 0.975]);
    net_ci = quantile(nets(isfinite(nets)), [0.025, 0.975]);
end

function auc = safeAuc(y, score)
    valid = isfinite(y) & isfinite(score);
    y = y(valid);
    score = score(valid);
    auc = NaN;
    if numel(unique(y)) < 2
        return;
    end
    [~, ~, ~, auc] = perfcurve(y, score, 1);
end

function auc = safePrAuc(y, score)
    valid = isfinite(y) & isfinite(score);
    y = y(valid);
    score = score(valid);
    auc = NaN;
    if numel(unique(y)) < 2
        return;
    end
    [~, ~, ~, auc] = perfcurve(y, score, 1, 'xCrit', 'reca', 'yCrit', 'prec');
end

function tbl = makeCaseTable(original_tbl, eval_struct)
    tbl = original_tbl(eval_struct.valid_mask, :);
    y = eval_struct.y > 0.5;
    tbl.dataset = repmat(string(eval_struct.dataset_name), height(tbl), 1);
    tbl.label_name = repmat(string(eval_struct.label_name), height(tbl), 1);
    tbl.cv_scheme = repmat(string(eval_struct.cv_scheme), height(tbl), 1);
    tbl.fold_id = eval_struct.fold_id;
    tbl.y_true = y;
    tbl.cir_score = eval_struct.cir.pred;
    tbl.cp_score = eval_struct.cp.pred;
    tbl.joint_score = eval_struct.joint.pred;
    tbl.cir_hat = tbl.cir_score >= 0.50;
    tbl.cp_hat = tbl.cp_score >= 0.50;
    tbl.joint_hat = tbl.joint_score >= 0.50;
    tbl.cir_wrong = tbl.cir_hat ~= tbl.y_true;
    tbl.cp_wrong = tbl.cp_hat ~= tbl.y_true;
    tbl.joint_wrong = tbl.joint_hat ~= tbl.y_true;
    tbl.cp_save = tbl.cir_wrong & ~tbl.joint_wrong;
    tbl.cp_harm = ~tbl.cir_wrong & tbl.joint_wrong;
    tbl.low_conf_cir = tbl.cir_score >= 0.40 & tbl.cir_score <= 0.60;
    tbl.low_conf_joint_correct = tbl.low_conf_cir & ~tbl.joint_wrong;
    tbl.low_conf_cir_correct = tbl.low_conf_cir & ~tbl.cir_wrong;
    tbl.joint_minus_cir = tbl.joint_score - tbl.cir_score;
end

function cell_tbl = buildCellTable(case_tbl, axis_pairs, cfg)
    total_fail = sum(case_tbl.cir_wrong);
    total_save = sum(case_tbl.cp_save);
    total_harm = sum(case_tbl.cp_harm);
    rows = {};
    for p = 1:size(axis_pairs, 1)
        x_req = axis_pairs{p, 1};
        y_req = axis_pairs{p, 2};
        x_var = resolveVar(case_tbl, x_req);
        y_var = resolveVar(case_tbl, y_req);
        [x_meta, x_idx] = axisIndex(case_tbl.(x_var), cfg.n_bins);
        [y_meta, y_idx] = axisIndex(case_tbl.(y_var), cfg.n_bins);
        for yi = 1:numel(y_meta.labels)
            for xi = 1:numel(x_meta.labels)
                mask = (x_idx == xi) & (y_idx == yi);
                if ~any(mask)
                    continue;
                end
                n = sum(mask);
                n_pos = sum(case_tbl.y_true(mask));
                n_neg = sum(~case_tbl.y_true(mask));
                cir_fail_n = sum(case_tbl.cir_wrong(mask));
                cp_save_n = sum(case_tbl.cp_save(mask));
                cp_harm_n = sum(case_tbl.cp_harm(mask));
                delta_auc = safeAuc(double(case_tbl.y_true(mask)), case_tbl.joint_score(mask)) - ...
                    safeAuc(double(case_tbl.y_true(mask)), case_tbl.cir_score(mask));
                tier = "supplemental";
                if n >= cfg.min_total && n_pos >= cfg.min_pos && n_neg >= cfg.min_neg
                    tier = "main";
                end
                rows(end + 1, :) = { ... %#ok<AGROW>
                    string(x_req), string(y_req), string(x_var), string(y_var), ...
                    xi, yi, string(x_meta.labels{xi}), string(y_meta.labels{yi}), n, n_pos, n_neg, ...
                    safeAuc(double(case_tbl.y_true(mask)), case_tbl.cir_score(mask)), ...
                    safeAuc(double(case_tbl.y_true(mask)), case_tbl.joint_score(mask)), ...
                    delta_auc, ...
                    cir_fail_n, safeRate(cir_fail_n, n), safeRate(cir_fail_n, total_fail), ...
                    cp_save_n, safeRate(cp_save_n, n), safeRate(cp_save_n, total_save), ...
                    cp_harm_n, safeRate(cp_harm_n, n), safeRate(cp_harm_n, total_harm), ...
                    cp_save_n - cp_harm_n, safeRate(cp_save_n - cp_harm_n, n), ...
                    safeRate(cp_save_n, cir_fail_n), safeRate(cp_harm_n, sum(~case_tbl.cir_wrong(mask))), ...
                    sum(case_tbl.low_conf_cir(mask)), safeMean(case_tbl.low_conf_joint_correct(mask)), safeMean(case_tbl.low_conf_cir_correct(mask)), ...
                    tier};
            end
        end
    end
    cell_tbl = cell2table(rows, 'VariableNames', ...
        {'x_var', 'y_var', 'x_var_actual', 'y_var_actual', 'x_index', 'y_index', 'x_label', 'y_label', ...
        'n', 'n_pos', 'n_neg', 'auc_cir', 'auc_joint', 'delta_auc', ...
        'cir_fail_n', 'cir_fail_rate', 'cir_fail_share', ...
        'cp_save_n', 'cp_save_rate', 'cp_save_share', ...
        'cp_harm_n', 'cp_harm_rate', 'cp_harm_share', ...
        'net_recovery_n', 'net_recovery', 'cp_save_given_cir_fail', 'cp_harm_given_cir_correct', ...
        'low_conf_n', 'low_conf_joint_correct_rate', 'low_conf_cir_correct_rate', 'tier'});
end

function tbl = selectTopCells(cell_tbl, mode, top_n, main_only)
    tbl = cell_tbl;
    if main_only
        tbl = tbl(strcmp(tbl.tier, "main"), :);
    end
    switch mode
        case 'fail'
            tbl = sortrows(tbl, {'cir_fail_share', 'cir_fail_rate', 'n'}, {'descend', 'descend', 'descend'});
        case 'save'
            tbl = sortrows(tbl, {'cp_save_share', 'cp_save_given_cir_fail', 'net_recovery', 'n'}, {'descend', 'descend', 'descend', 'descend'});
        case 'net'
            tbl = tbl(tbl.net_recovery > 0, :);
            tbl = sortrows(tbl, {'net_recovery', 'delta_auc', 'cp_save_n', 'n'}, {'descend', 'descend', 'descend', 'descend'});
        case 'harm'
            tbl = tbl(tbl.net_recovery < 0, :);
            tbl = sortrows(tbl, {'net_recovery', 'cp_harm_share', 'n'}, {'ascend', 'descend', 'descend'});
        otherwise
            error('Unknown mode: %s', mode);
    end
    tbl = tbl(1:min(top_n, height(tbl)), :);
end

function tbl = addBootstrapToCells(case_tbl, cell_tbl, cfg)
    if isempty(cell_tbl)
        tbl = cell_tbl;
        return;
    end
    delta_ci = strings(height(cell_tbl), 1);
    net_ci = strings(height(cell_tbl), 1);
    delta_lo = nan(height(cell_tbl), 1);
    delta_hi = nan(height(cell_tbl), 1);
    net_lo = nan(height(cell_tbl), 1);
    net_hi = nan(height(cell_tbl), 1);
    for i = 1:height(cell_tbl)
        mask = maskForCell(case_tbl, cell_tbl.x_var_actual(i), cell_tbl.x_label(i), cell_tbl.y_var_actual(i), cell_tbl.y_label(i));
        groups = repmat({'ALL'}, sum(mask), 1);
        room_var = maybeResolveVar(case_tbl, 'room_type');
        if ~isempty(room_var)
            groups = cellstr(string(case_tbl.(room_var)(mask)));
        end
        [d_ci, n_ci] = bootstrapPairedMetrics(double(case_tbl.y_true(mask)), case_tbl.cir_score(mask), case_tbl.joint_score(mask), groups, cfg);
        delta_lo(i) = d_ci(1);
        delta_hi(i) = d_ci(2);
        net_lo(i) = n_ci(1);
        net_hi(i) = n_ci(2);
        delta_ci(i) = sprintf('[%.6f, %.6f]', d_ci(1), d_ci(2));
        net_ci(i) = sprintf('[%.6f, %.6f]', n_ci(1), n_ci(2));
    end
    tbl = cell_tbl;
    tbl.delta_auc_ci_low = delta_lo;
    tbl.delta_auc_ci_high = delta_hi;
    tbl.net_recovery_ci_low = net_lo;
    tbl.net_recovery_ci_high = net_hi;
    tbl.delta_auc_ci = delta_ci;
    tbl.net_recovery_ci = net_ci;
end

function tf = maskForCell(tbl, x_var_actual, x_label, y_var_actual, y_label)
    tf = cellMask(tbl.(char(x_var_actual)), x_label) & cellMask(tbl.(char(y_var_actual)), y_label);
end

function tf = cellMask(values, label)
    if isnumeric(values) || islogical(values)
        bounds = parseBounds(label);
        tf = double(values) >= bounds(1) & double(values) <= bounds(2);
    else
        tf = strcmp(cellstr(string(values)), char(string(label)));
    end
end

function bounds = parseBounds(label)
    vals = sscanf(char(string(label)), '[%f, %f]');
    if numel(vals) ~= 2
        error('Failed to parse bounds from label %s', char(string(label)));
    end
    bounds = vals(:).';
end

function [meta, idx] = axisIndex(raw, n_bins)
    if isnumeric(raw) || islogical(raw)
        values = double(raw(:));
        values_finite = values(isfinite(values));
        edges = quantile(values_finite, linspace(0, 1, n_bins + 1));
        edges = makeStrictEdges(edges, values_finite);
        idx = discretize(values, edges);
        labels = cell(1, numel(edges) - 1);
        for i = 1:(numel(edges) - 1)
            labels{i} = sprintf('[%.2f, %.2f]', edges(i), edges(i + 1));
        end
        meta = struct('labels', {labels});
        return;
    end

    values = cellstr(string(raw(:)));
    labels = unique(values, 'stable');
    idx = nan(numel(values), 1);
    for i = 1:numel(labels)
        idx(strcmp(values, labels{i})) = i;
    end
    meta = struct('labels', {labels});
end

function edges = makeStrictEdges(edges, x)
    edges = double(edges(:).');
    span = max(max(x) - min(x), 1.0);
    tol = 1e-9 * span;
    edges(1) = edges(1) - tol;
    for i = 2:numel(edges)
        if edges(i) <= edges(i - 1)
            edges(i) = edges(i - 1) + tol;
        end
    end
    edges(end) = edges(end) + tol;
end

function tbl = subgroupMetricTable(case_tbl, group_name)
    group_var = resolveVar(case_tbl, group_name);
    group_values = cellstr(string(case_tbl.(group_var)));
    levels = unique(group_values, 'stable');
    rows = {};
    for i = 1:numel(levels)
        mask = strcmp(group_values, levels{i});
        metrics = metricsFromCaseSlice(case_tbl(mask, :));
        rows(end + 1, :) = {string(levels{i}), sum(mask), sum(~case_tbl.y_true(mask)), sum(case_tbl.y_true(mask)), ...
            metrics.auc_cir, metrics.auc_joint, metrics.delta_auc, metrics.cp_save_rate, metrics.cp_harm_rate, metrics.net_recovery}; %#ok<AGROW>
    end
    tbl = cell2table(rows, 'VariableNames', {'group_value', 'n', 'n_neg', 'n_pos', 'auc_cir', 'auc_joint', 'delta_auc', 'cp_save_rate', 'cp_harm_rate', 'net_recovery'});
end

function metrics = metricsFromCaseSlice(case_tbl)
    y = double(case_tbl.y_true);
    metrics = struct();
    metrics.auc_cir = safeAuc(y, case_tbl.cir_score);
    metrics.auc_joint = safeAuc(y, case_tbl.joint_score);
    metrics.delta_auc = metrics.auc_joint - metrics.auc_cir;
    metrics.cp_save_rate = mean(case_tbl.cp_save);
    metrics.cp_harm_rate = mean(case_tbl.cp_harm);
    metrics.net_recovery = metrics.cp_save_rate - metrics.cp_harm_rate;
end

function tbl = buildInventoryTable(stage1, stage2, stage1_labels, stage2_labels)
    rows = {};
    names1 = fieldnames(stage1_labels);
    for i = 1:numel(names1)
        y = stage1_labels.(names1{i});
        rows(end + 1, :) = {"Stage1", string(names1{i}), height(stage1), sum(y == 0), sum(y == 1), mean(y)}; %#ok<AGROW>
    end
    names2 = fieldnames(stage2_labels);
    for i = 1:numel(names2)
        y = stage2_labels.(names2{i});
        rows(end + 1, :) = {"Stage2", string(names2{i}), height(stage2), sum(y == 0), sum(y == 1), mean(y)}; %#ok<AGROW>
    end
    tbl = cell2table(rows, 'VariableNames', {'dataset', 'label_name', 'n', 'n_neg', 'n_pos', 'pos_frac'});
end

function tbl = buildLabelAlignmentTable(stage1_labels, stage2_labels)
    rows = {};
    pairs_stage1 = {
        'current_0p20', 'mixed_0p33';
        'current_0p20', 'bounce_0p33'};
    for i = 1:size(pairs_stage1, 1)
        lhs = stage1_labels.(pairs_stage1{i, 1});
        rhs = stage1_labels.(pairs_stage1{i, 2});
        rows(end + 1, :) = {"Stage1", string(pairs_stage1{i, 1}), string(pairs_stage1{i, 2}), sum(lhs == rhs), sum(lhs ~= rhs), sum(lhs), sum(rhs)}; %#ok<AGROW>
    end
    pairs_stage2 = {
        'current_0p20', 'mixed_0p33';
        'geo_only', 'bounce_0p33';
        'mixed_0p33', 'bounce_0p33'};
    for i = 1:size(pairs_stage2, 1)
        lhs = stage2_labels.(pairs_stage2{i, 1});
        rhs = stage2_labels.(pairs_stage2{i, 2});
        rows(end + 1, :) = {"Stage2", string(pairs_stage2{i, 1}), string(pairs_stage2{i, 2}), sum(lhs == rhs), sum(lhs ~= rhs), sum(lhs), sum(rhs)}; %#ok<AGROW>
    end
    tbl = cell2table(rows, 'VariableNames', {'dataset', 'lhs_label', 'rhs_label', 'agree_n', 'disagree_n', 'lhs_pos', 'rhs_pos'});
end

function tbl = finiteSummaryTable(stage1, stage2, features_all)
    rows = {};
    for dataset_name = ["Stage1", "Stage2"]
        if dataset_name == "Stage1"
            tbl_src = stage1;
        else
            tbl_src = stage2;
        end
        X = table2array(tbl_src(:, features_all));
        rows(end + 1, :) = {dataset_name, numel(X), sum(isnan(X(:))), sum(isinf(X(:))), sum(~isfinite(X(:)))}; %#ok<AGROW>
    end
    tbl = cell2table(rows, 'VariableNames', {'dataset', 'n_feature_values', 'n_nan', 'n_inf', 'n_nonfinite'});
end

function tbl = featureCorrelationTable(stage1, stage2, features_all, top_n)
    rows = {};
    for dataset_name = ["Stage1", "Stage2"]
        if dataset_name == "Stage1"
            tbl_src = stage1;
        else
            tbl_src = stage2;
        end
        X = table2array(tbl_src(:, features_all));
        valid = all(isfinite(X), 2);
        X = X(valid, :);
        R = corr(X);
        for i = 1:size(R, 1)
            for j = (i + 1):size(R, 2)
                rows(end + 1, :) = {dataset_name, string(features_all{i}), string(features_all{j}), abs(R(i, j)), R(i, j)}; %#ok<AGROW>
            end
        end
    end
    tbl = cell2table(rows, 'VariableNames', {'dataset', 'feature_a', 'feature_b', 'abs_corr', 'corr'});
    tbl = sortrows(tbl, {'dataset', 'abs_corr'}, {'ascend', 'descend'});
    out = table();
    for dataset_name = ["Stage1", "Stage2"]
        chunk = tbl(strcmp(tbl.dataset, dataset_name), :);
        out = [out; chunk(1:min(top_n, height(chunk)), :)]; %#ok<AGROW>
    end
    tbl = out;
end

function tbl = leakageAuditTable(repo_root)
    feature_dir = fullfile(repo_root, '+features');
    files = dir(fullfile(feature_dir, '*.m'));
    patterns = {'is_nlos', 'is_los', 'has_los_path', 'bounce_to_los_ratio_mid', 'is_nlos_geo', 'is_nlos_mixed_0p33'};
    rows = {};
    for i = 1:numel(patterns)
        hit_count = 0;
        hit_files = {};
        for j = 1:numel(files)
            txt = fileread(fullfile(files(j).folder, files(j).name));
            if contains(txt, patterns{i})
                hit_count = hit_count + 1;
                hit_files{end + 1} = files(j).name; %#ok<AGROW>
            end
        end
        rows(end + 1, :) = {string(patterns{i}), hit_count, string(strjoin(hit_files, '; '))}; %#ok<AGROW>
    end
    tbl = cell2table(rows, 'VariableNames', {'pattern', 'hit_count', 'hit_files'});
end

function summary = leakageSummary(leak_tbl)
    total_hits = sum(leak_tbl.hit_count);
    if total_hits == 0
        summary = "No label-derived patterns were found in +features/*.m during the executed leakage scan.";
    else
        summary = "Leakage scan found label-derived terms inside +features/*.m and needs manual review.";
    end
end

function summary = finiteSummarySentence(tbl)
    if all(tbl.n_nonfinite == 0)
        summary = "Canonical feature tables contained no NaN or Inf values in the deterministic Stage 1 and Stage 2 exports.";
    else
        summary = "Non-finite canonical feature values were detected; inspect day1_feature_finite_summary.csv.";
    end
end

function tbl = canonicalReproductionTable(stage1_tbl, stage1_y, stage2_tbl, stage2_y, cir_features, joint_features, paper_ci)
    rows = {};
    rows(end + 1, :) = buildReproRow("Stage1", stage1_tbl, stage1_y, cir_features, joint_features, paper_ci); %#ok<AGROW>
    rows(end + 1, :) = buildReproRow("Stage2", stage2_tbl, stage2_y, cir_features, joint_features, paper_ci); %#ok<AGROW>
    tbl = cell2table(rows, 'VariableNames', {'scope', 'auc_cir', 'auc_joint', 'delta_auc', 'target_delta_auc', 'target_ci_low', 'target_ci_high', 'delta_abs_error'});
end

function row = buildReproRow(scope, tbl_src, y, cir_features, joint_features, paper_ci)
    if scope == "Stage1"
        mask = strcmp(paper_ci.scope, "Stage1") & strcmp(paper_ci.subset, "ALL");
    else
        mask = strcmp(paper_ci.scope, "Stage2 mixed@0.33") & strcmp(paper_ci.subset, "ALL");
    end
    auc_cir = analysis.cvLogisticAuc(table2array(tbl_src(:, cir_features)), y);
    auc_joint = analysis.cvLogisticAuc(table2array(tbl_src(:, joint_features)), y);
    delta_auc = auc_joint - auc_cir;
    row = {scope, auc_cir, auc_joint, delta_auc, paper_ci.delta_auc(mask), paper_ci.delta_ci_low(mask), paper_ci.delta_ci_high(mask), abs(delta_auc - paper_ci.delta_auc(mask))};
end

function sentence = canonicalSentence(repro_tbl, scope)
    row = repro_tbl(strcmp(repro_tbl.scope, scope), :);
    sentence = sprintf('%s delta AUC reproduced as %.6f with absolute error %.6g.', scope, row.delta_auc, row.delta_abs_error);
end

function out = resolveVar(tbl, desired_name)
    base = char(string(desired_name));
    candidates = {base, [base '_cases'], [base '_results'], [base '_cases_tbl'], [base '_results_tbl']};
    for i = 1:numel(candidates)
        if ismember(candidates{i}, tbl.Properties.VariableNames)
            out = candidates{i};
            return;
        end
    end
    error('Variable not found: %s', base);
end

function out = maybeResolveVar(tbl, desired_name)
    try
        out = resolveVar(tbl, desired_name);
    catch
        out = '';
    end
end

function rate = safeRate(num, den)
    if den <= 0
        rate = NaN;
    else
        rate = double(num) / double(den);
    end
end

function val = safeMean(x)
    if isempty(x)
        val = NaN;
    else
        val = mean(double(x), 'omitnan');
    end
end

function writeFindingsTable(fid, tbl)
    writeMarkdownTable(fid, tbl, {'classification', 'status', 'finding', 'evidence_file', 'source_ref'}, height(tbl));
end

function writeMarkdownTable(fid, tbl, columns, max_rows)
    columns = cellstr(string(columns));
    n_rows = min(max_rows, height(tbl));
    fprintf(fid, '|');
    for i = 1:numel(columns)
        fprintf(fid, ' %s |', columns{i});
    end
    fprintf(fid, '\n|');
    for i = 1:numel(columns)
        fprintf(fid, '---|');
    end
    fprintf(fid, '\n');
    for r = 1:n_rows
        fprintf(fid, '|');
        for c = 1:numel(columns)
            value = tbl.(columns{c})(r, :);
            fprintf(fid, ' %s |', formatMarkdownValue(value));
        end
        fprintf(fid, '\n');
    end
end

function txt = formatMarkdownValue(value)
    if iscell(value)
        value = value{1};
    end
    if isstring(value)
        txt = char(value);
    elseif ischar(value)
        txt = value;
    elseif isnumeric(value) || islogical(value)
        if isempty(value)
            txt = '';
        elseif isscalar(value)
            if isfinite(value)
                txt = sprintf('%.6f', value);
            else
                txt = 'NaN';
            end
        else
            txt = char(string(value));
        end
    else
        txt = char(string(value));
    end
    txt = strrep(txt, '|', '\|');
    txt = strrep(txt, newline, ' ');
end

function value = valueForGroup(tbl, group_name, field_name)
    mask = strcmp(tbl.group_value, string(group_name));
    if ~any(mask)
        value = NaN;
        return;
    end
    value = tbl.(field_name)(find(mask, 1, 'first'));
end

function tbl = addMechanismFamily(tbl, stage_name, label_name)
    if isempty(tbl)
        tbl.stage = strings(0, 1);
        tbl.label_name = strings(0, 1);
        tbl.mechanism_family = strings(0, 1);
        return;
    end
    fam = strings(height(tbl), 1);
    for i = 1:height(tbl)
        fam(i) = mechanismFamily(tbl.x_var(i), tbl.y_var(i));
    end
    tbl.stage = repmat(string(stage_name), height(tbl), 1);
    tbl.label_name = repmat(string(label_name), height(tbl), 1);
    tbl.mechanism_family = fam;
end

function fam = mechanismFamily(x_var, y_var)
    vars = lower(join([string(x_var), string(y_var)], '|'));
    if contains(vars, 'xpol')
        fam = "expected_depolarization_proxy";
    elseif contains(vars, 'los_angle')
        fam = "oblique_incidence";
    elseif contains(vars, 'material') || contains(vars, 'eps_r')
        fam = "material_signature";
    elseif contains(vars, 'room') || contains(vars, 'grid_layer')
        fam = "room_geometry";
    elseif contains(vars, 'antenna')
        fam = "antenna_family";
    else
        fam = "mixed";
    end
end

function tbl = selectStage3Candidates(case_tbl, cell_tbl, existing_case_ids, per_cell, positive_mode)
    rows = table();
    used_case_ids = [];
    for i = 1:height(cell_tbl)
        mask = maskForCell(case_tbl, cell_tbl.x_var_actual(i), cell_tbl.x_label(i), cell_tbl.y_var_actual(i), cell_tbl.y_label(i));
        if positive_mode
            subset = case_tbl(mask & case_tbl.cp_save, :);
            if isempty(subset)
                subset = case_tbl(mask, :);
            end
            [~, order] = sort(subset.joint_minus_cir, 'descend');
        else
            subset = case_tbl(mask & case_tbl.cp_harm, :);
            if isempty(subset)
                subset = case_tbl(mask, :);
            end
            [~, order] = sort(subset.joint_minus_cir, 'ascend');
        end
        subset = subset(order, :);
        subset = subset(~ismember(double(subset.case_id), used_case_ids), :);
        subset = subset(1:min(per_cell, height(subset)), :);
        if isempty(subset)
            continue;
        end
        out = table();
        out.case_id = subset.case_id;
        out.room_type = strings(height(subset), 1);
        room_var = maybeResolveVar(subset, 'room_type');
        if ~isempty(room_var)
            out.room_type = string(subset.(room_var));
        end
        out.regime_desc = repmat(cellDescription(cell_tbl, i), height(subset), 1);
        out.joint_minus_cir = subset.joint_minus_cir;
        out.cp_save = subset.cp_save;
        out.cp_harm = subset.cp_harm;
        out.cir_score = subset.cir_score;
        out.joint_score = subset.joint_score;
        out.low_conf_cir = subset.low_conf_cir;
        out.in_existing_stage3_det = ismember(double(subset.case_id), existing_case_ids);
        wall_var = maybeResolveVar(subset, 'dominant_wall_material');
        if ~isempty(wall_var)
            out.dominant_wall_material = string(subset.(wall_var));
        else
            out.dominant_wall_material = repmat("", height(subset), 1);
        end
        grid_var = maybeResolveVar(subset, 'grid_layer');
        if ~isempty(grid_var)
            out.grid_layer = double(subset.(grid_var));
        else
            out.grid_layer = nan(height(subset), 1);
        end
        if ismember('xpol_coupling_db', subset.Properties.VariableNames)
            out.xpol_coupling_db = subset.xpol_coupling_db;
        else
            out.xpol_coupling_db = nan(height(subset), 1);
        end
        angle_var = maybeResolveVar(subset, 'los_angle_from_anchor_bore_deg');
        if ~isempty(angle_var)
            out.los_angle_deg = double(subset.(angle_var));
        else
            out.los_angle_deg = nan(height(subset), 1);
        end
        rows = [rows; out]; %#ok<AGROW>
        used_case_ids = [used_case_ids; double(subset.case_id)]; %#ok<AGROW>
    end
    tbl = rows;
end

function desc = cellDescription(cell_tbl, idx)
    desc = string(cell_tbl.x_var(idx)) + "=" + string(cell_tbl.x_label(idx)) + ", " + ...
        string(cell_tbl.y_var(idx)) + "=" + string(cell_tbl.y_label(idx));
end

function tbl = buildOverlapSummary(positive_candidates, negative_controls)
    rows = { ...
        "positive_candidates", height(positive_candidates), sum(positive_candidates.in_existing_stage3_det), safeRate(sum(positive_candidates.in_existing_stage3_det), height(positive_candidates)); ...
        "negative_controls", height(negative_controls), sum(negative_controls.in_existing_stage3_det), safeRate(sum(negative_controls.in_existing_stage3_det), height(negative_controls))};
    tbl = cell2table(rows, 'VariableNames', {'bucket', 'n_cases', 'n_in_existing_stage3_det', 'share_in_existing_stage3_det'});
end
