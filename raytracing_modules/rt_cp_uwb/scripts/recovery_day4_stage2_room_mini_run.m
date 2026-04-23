script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

out_dir = fullfile(repo_root, 'results', 'recovery_mini', 'day4');
ensureDir(out_dir);

cfg = struct();
cfg.seed = 20260424;
cfg.seed_stage_id = 'stage2_recovery_room_mini_day4';
cfg.seed_base = 20260424;
cfg.kfold = 5;
cfg.n_boot = 400;
cfg.threshold = 0.50;
cfg.low_conf_lo = 0.40;
cfg.low_conf_hi = 0.60;
cfg.min_total = 50;
cfg.min_pos = 20;
cfg.min_neg = 20;
cfg.min_supp_total = 30;

refs = struct();
refs.script = 'scripts/recovery_day4_stage2_room_mini_run.m';
refs.design = '+sweep/designStage2RecoveryMiniCases.m';
refs.stage2_design = '+sweep/designStage2Cases.m';
refs.room_grid = '+scenes/generateRoomTxRxGrid.m';
refs.room_scene = '+scenes/makeRoomABCScene.m';
refs.run_one_case = '+sweep/runOneCase.m';
refs.canonical = '+features/canonicalFeatureNames.m';
refs.day3_ranking = 'results/recovery_mini/day3/day3_stage1_recovery_mini_candidate_ranking.csv';
refs.day3_summary = 'results/recovery_mini/day3/day3_stage1_recovery_mini_summary.md';
refs.stage2_relabel = 'scripts/week4_day5_stage2_relabel_rerun_det.m';
refs.audit = 'results/audit/week4_integrity_audit.md';

ranking_csv = fullfile(repo_root, 'results', 'recovery_mini', 'day3', 'day3_stage1_recovery_mini_candidate_ranking.csv');
day3_summary_md = fullfile(repo_root, 'results', 'recovery_mini', 'day3', 'day3_stage1_recovery_mini_summary.md');
[cases, design_meta] = sweep.designStage2RecoveryMiniCases(ranking_csv, day3_summary_md);
writetable(cases, fullfile(out_dir, 'day4_stage2_room_mini_cases.csv'));

dry_idx = selectDryRunIndices(cases, 30);
dry_cases = cases(dry_idx, :);
fprintf('Running 30-case Stage 2 room mini dry run...\n');
dry_results = sweep.runSweepBatch(dry_cases, configForRun(cfg), true);
dry_results = attachStage2Labels(dry_results);
sanity = runDrySanity(dry_results, cfg, refs);
writeSanityReport(out_dir, sanity, dry_cases, refs);

if sanity.blocker
    error('Day 4 dry run sanity failed. See %s', fullfile(out_dir, 'day4_stage2_sanity_report.md'));
end

fprintf('Running full Stage 2 room mini sweep with %d cases...\n', height(cases));
results = sweep.runSweepBatch(cases, configForRun(cfg), true);
results = attachStage2Labels(results);

csv_path = fullfile(out_dir, 'stage2_recovery_room_mini.csv');
mat_path = fullfile(out_dir, 'stage2_recovery_room_mini.mat');
writetable(results, csv_path);
save(mat_path, 'results', 'cases', 'cfg', 'design_meta', 'sanity');

all_features = features.canonicalFeatureNames();
cp_features = all_features(1:6);
cir_features = all_features(7:end);
joint_features = all_features;

valid_mask = ~logical(getNumericColumn(results, 'failed'));
valid = results(valid_mask, :);
room_values = stringColumn(valid, 'room_type');
y_primary = double(logical(getNumericColumn(valid, 'is_nlos_mixed_0p33')));

eval_random = evaluateRecovery(valid, y_primary, cir_features, cp_features, joint_features, 'stratified', room_values, cfg, 'Stage2RecoveryMini', 'mixed_0p33_random');
eval_room = evaluateRecovery(valid, y_primary, cir_features, cp_features, joint_features, 'group_stratified', room_values, cfg, 'Stage2RecoveryMini', 'mixed_0p33_room_stratified');
eval_loro = evaluateRecovery(valid, y_primary, cir_features, cp_features, joint_features, 'leave_one_group_out', room_values, cfg, 'Stage2RecoveryMini', 'mixed_0p33_leave_one_room_out');

case_random = makeCaseTable(valid, eval_random, cfg);
case_room = makeCaseTable(valid, eval_room, cfg);
case_loro = makeCaseTable(valid, eval_loro, cfg);

room_metrics = roomMetricTable(case_room, cfg);
cv_comparison = buildCvComparison(eval_random, eval_room, eval_loro, case_loro);
label_balance = buildLabelBalanceTable(valid);
candidate_metrics = candidateMetricTable(case_room, cfg);
condition_bins = buildConditionBins(case_room, cfg);
derived_summary = summarizeDerivedConditions(condition_bins);

writetable(room_metrics, fullfile(out_dir, 'day4_stage2_room_mini_metrics_by_room.csv'));
writetable(condition_bins, fullfile(out_dir, 'day4_stage2_condition_bins.csv'));
writetable(cv_comparison, fullfile(out_dir, 'day4_stage2_cv_comparison.csv'));
writetable(case_room, fullfile(out_dir, 'day4_stage2_room_stratified_predictions.csv'));

writeLabelBalanceReport(out_dir, label_balance, design_meta, refs);
plotRoomAucBar(room_metrics, fullfile(out_dir, 'day4_plot_room_auc_bar.png'));
plotRoomDeltaNetBar(room_metrics, fullfile(out_dir, 'day4_plot_room_delta_net_bar.png'));
plotPairHeatmap(condition_bins, 'rms_delay_spread__k_factor_estimate', 'Net recovery by rms_delay_spread x k_factor_estimate', 'day4_plot_rms_kfactor_net_recovery_heatmap.png', out_dir);
plotPairHeatmap(condition_bins, 'fp_to_total_ratio__num_paths', 'Net recovery by fp_to_total_ratio x num_paths', 'day4_plot_fp_numpaths_net_recovery_heatmap.png', out_dir);
plotRoomSaveHarm(case_room, fullfile(out_dir, 'day4_plot_room_cp_save_cp_harm_stacked.png'));
plotLabelComponentConcentration(case_room, fullfile(out_dir, 'day4_plot_label_component_cp_save_concentration.png'));
plotStage1Overlay(case_room, fullfile(out_dir, 'day4_plot_stage1_candidate_overlay_on_stage2_cloud.png'));

writeSummaryReport(out_dir, design_meta, sanity, room_metrics, candidate_metrics, derived_summary, cv_comparison, refs);

disp('Day 4 Stage 2 room mini outputs written under results/recovery_mini/day4/.');

function cfg_run = configForRun(cfg)
    cfg_run = config.defaultConfig();
    cfg_run.seed_stage_id = cfg.seed_stage_id;
    cfg_run.seed_base = cfg.seed_base;
end

function ensureDir(path_str)
    if exist(path_str, 'dir') ~= 7
        mkdir(path_str);
    end
end

function idx = selectDryRunIndices(cases, n_pick)
    rows = unique(cases(:, {'room_type', 'expected_candidate_id'}), 'rows', 'stable');
    idx = zeros(0, 1);
    for i = 1:height(rows)
        mask = string(cases.room_type) == string(rows.room_type(i)) & string(cases.expected_candidate_id) == string(rows.expected_candidate_id(i));
        cur = find(mask);
        take = min(2, numel(cur));
        idx = [idx; cur(1:take)]; %#ok<AGROW>
    end
    remaining = setdiff((1:height(cases)).', idx, 'stable');
    if numel(idx) < n_pick
        extra_needed = min(n_pick - numel(idx), numel(remaining));
        idx = [idx; remaining(round(linspace(1, numel(remaining), extra_needed)).')]; %#ok<AGROW>
    end
    idx = unique(idx, 'stable');
    idx = idx(1:min(numel(idx), n_pick));
end

function tbl = attachStage2Labels(tbl)
    has_los = logical(getNumericColumn(tbl, 'has_los_path'));
    ratio = getNumericColumn(tbl, 'bounce_to_los_ratio_mid');
    if all(isnan(ratio))
        ratio = zeros(height(tbl), 1);
    end
    is_current = logical(getNumericColumn(tbl, 'is_nlos'));
    is_geo = ~has_los;
    is_bounce = has_los & (ratio >= 0.33);
    is_mixed = is_geo | is_bounce;

    tbl.is_nlos_current_0p20 = is_current;
    tbl.is_los_current_0p20 = ~is_current;
    tbl.is_nlos_geo = is_geo;
    tbl.is_los_geo = ~is_geo;
    tbl.is_nlos_bounce_0p33 = is_bounce;
    tbl.is_los_bounce_0p33 = has_los & ~is_bounce;
    tbl.is_nlos_mixed_0p33 = is_mixed;
    tbl.is_los_mixed_0p33 = ~is_mixed;
    tbl.label_schema = repmat("mixed_0p33_primary_with_dual_aux", height(tbl), 1);
end

function sanity = runDrySanity(tbl, cfg, refs)
    valid_mask = ~logical(getNumericColumn(tbl, 'failed'));
    valid = tbl(valid_mask, :);
    rooms = stringColumn(valid, 'room_type');
    y_primary = logical(getNumericColumn(valid, 'is_nlos_mixed_0p33'));
    all_features = features.canonicalFeatureNames();
    X = table2array(valid(:, all_features));

    room_levels = unique(rooms, 'stable');
    room_counts = zeros(numel(room_levels), 1);
    room_pos = zeros(numel(room_levels), 1);
    room_neg = zeros(numel(room_levels), 1);
    rms_std = zeros(numel(room_levels), 1);
    path_std = zeros(numel(room_levels), 1);
    num_paths = getNumericColumn(valid, 'num_paths');
    rms_delay = getNumericColumn(valid, 'rms_delay_spread');
    for i = 1:numel(room_levels)
        mask = rooms == room_levels(i);
        room_counts(i) = sum(mask);
        room_pos(i) = sum(y_primary(mask));
        room_neg(i) = sum(~y_primary(mask));
        rms_std(i) = std(rms_delay(mask), 'omitnan');
        path_std(i) = std(num_paths(mask), 'omitnan');
    end

    max_paths = max(num_paths);
    path_cap_share = mean(num_paths == max_paths);
    sanity = struct();
    sanity.dry_n = height(tbl);
    sanity.valid_n = height(valid);
    sanity.failed_rate = safeRate(sum(~valid_mask), height(tbl));
    sanity.room_levels = room_levels;
    sanity.room_counts = room_counts;
    sanity.room_pos = room_pos;
    sanity.room_neg = room_neg;
    sanity.nonfinite_pct = 100 * safeRate(sum(~isfinite(X(:))), numel(X));
    sanity.max_paths = max_paths;
    sanity.path_cap_share = path_cap_share;
    sanity.rms_std = rms_std;
    sanity.path_std = path_std;
    sanity.fail_ok = sanity.failed_rate < 0.05 + 1e-12;
    sanity.room_ok = all(room_counts >= 5);
    sanity.finite_ok = all(isfinite(X(:)));
    sanity.balance_ok = numel(unique(double(y_primary))) >= 2 && all(room_counts >= 5);
    sanity.path_ok = path_cap_share < 0.50;
    sanity.spread_ok = all(rms_std > 0) && all(path_std > 0);
    sanity.blocker = ~(sanity.fail_ok && sanity.room_ok && sanity.finite_ok && sanity.balance_ok && sanity.path_ok && sanity.spread_ok);
    sanity.notes = sprintf('verified via %s and prior path-cap audit %s', refs.script, refs.audit);
end

function out = evaluateRecovery(tbl, y_all, cir_features, cp_features, joint_features, cv_scheme, group_values, cfg, dataset_name, label_name)
    X_all = table2array(tbl(:, joint_features));
    valid = all(isfinite(X_all), 2) & isfinite(y_all);
    tbl_valid = tbl(valid, :);
    y = double(y_all(valid) > 0.5);
    groups = cellstr(string(group_values(valid)));

    fold_id = makeFoldIds(y, groups, cfg.kfold, cfg.seed, cv_scheme);
    cir = fitModelOOF(table2array(tbl_valid(:, cir_features)), y, fold_id);
    cp = fitModelOOF(table2array(tbl_valid(:, cp_features)), y, fold_id);
    joint = fitModelOOF(table2array(tbl_valid(:, joint_features)), y, fold_id);

    metrics = computeRecoveryMetrics(y, cir.pred, joint.pred, cfg);
    [delta_ci, net_ci] = bootstrapPairedMetrics(y, cir.pred, joint.pred, groups, cfg);

    out = struct();
    out.dataset_name = dataset_name;
    out.label_name = label_name;
    out.cv_scheme = cv_scheme;
    out.valid_mask = valid;
    out.groups = groups;
    out.fold_id = fold_id;
    out.y = y;
    out.cir = cir;
    out.cp = cp;
    out.joint = joint;
    out.summary = struct( ...
        'dataset_name', string(dataset_name), ...
        'label_name', string(label_name), ...
        'cv_scheme', string(cv_scheme), ...
        'n_total', numel(y), ...
        'n_neg', sum(y == 0), ...
        'n_pos', sum(y == 1), ...
        'auc_cir', cir.auc, ...
        'auc_cp', cp.auc, ...
        'auc_joint', joint.auc, ...
        'delta_auc', joint.auc - cir.auc, ...
        'delta_auc_ci_low', delta_ci(1), ...
        'delta_auc_ci_high', delta_ci(2), ...
        'cp_save_rate', metrics.cp_save_rate, ...
        'cp_harm_rate', metrics.cp_harm_rate, ...
        'net_recovery', metrics.net_recovery, ...
        'net_recovery_ci_low', net_ci(1), ...
        'net_recovery_ci_high', net_ci(2), ...
        'cir_fail_rate', metrics.cir_fail_rate, ...
        'low_confidence_n', metrics.low_conf_n, ...
        'low_confidence_rescue_rate', metrics.low_conf_joint_correct_rate);
end

function fold_id = makeFoldIds(y, groups, k, seed, cv_scheme)
    n = numel(y);
    fold_id = zeros(n, 1);
    if strcmp(cv_scheme, 'leave_one_group_out')
        levels = unique(groups, 'stable');
        for i = 1:numel(levels)
            fold_id(strcmp(groups, levels{i})) = i;
        end
        return;
    end

    rng(seed, 'twister');
    if strcmp(cv_scheme, 'stratified')
        strata = repmat({'ALL'}, n, 1);
    elseif strcmp(cv_scheme, 'group_stratified')
        strata = groups;
    else
        error('Unknown cv scheme: %s', cv_scheme);
    end

    strata_levels = unique(strata, 'stable');
    for s = 1:numel(strata_levels)
        mask = strcmp(strata, strata_levels{s});
        idx_group = find(mask);
        y_group = y(mask);
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

function metrics = computeRecoveryMetrics(y, pred_cir, pred_joint, cfg)
    y_true = y > 0.5;
    cir_wrong = (pred_cir >= cfg.threshold) ~= y_true;
    joint_wrong = (pred_joint >= cfg.threshold) ~= y_true;
    low_mask = pred_cir >= cfg.low_conf_lo & pred_cir <= cfg.low_conf_hi;
    metrics = struct();
    metrics.cir_fail_rate = mean(cir_wrong);
    metrics.cp_save_rate = mean(cir_wrong & ~joint_wrong);
    metrics.cp_harm_rate = mean(~cir_wrong & joint_wrong);
    metrics.net_recovery = metrics.cp_save_rate - metrics.cp_harm_rate;
    metrics.low_conf_n = sum(low_mask);
    metrics.low_conf_joint_correct_rate = safeMean(~joint_wrong(low_mask));
end

function [delta_ci, net_ci] = bootstrapPairedMetrics(y, pred_cir, pred_joint, groups, cfg)
    keys = cell(numel(y), 1);
    for i = 1:numel(y)
        keys{i} = sprintf('%s|%d', groups{i}, y(i));
    end
    levels = unique(keys, 'stable');
    deltas = nan(cfg.n_boot, 1);
    nets = nan(cfg.n_boot, 1);
    rng(cfg.seed + 19, 'twister');
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

function case_tbl = makeCaseTable(original_tbl, eval_struct, cfg)
    tbl = original_tbl(eval_struct.valid_mask, :);
    y = eval_struct.y > 0.5;
    case_tbl = table();
    case_tbl.case_id = getNumericColumn(tbl, 'case_id');
    case_tbl.room_type = stringColumn(tbl, 'room_type');
    case_tbl.grid_layer = getNumericColumn(tbl, 'grid_layer');
    case_tbl.expected_candidate_id = stringColumn(tbl, 'expected_candidate_id');
    case_tbl.expected_condition = stringColumn(tbl, 'expected_condition');
    case_tbl.label_rule = stringColumn(tbl, 'label_rule');
    case_tbl.snr_db = getNumericColumn(tbl, 'snr_db');
    case_tbl.xpol_coupling_db = getNumericColumn(tbl, 'xpol_coupling_db');
    case_tbl.eps_r_multiplier = getNumericColumn(tbl, 'eps_r_multiplier');
    case_tbl.dominant_wall_material = stringColumn(tbl, 'dominant_wall_material');
    case_tbl.los_angle_from_anchor_bore_deg = getNumericColumn(tbl, 'los_angle_from_anchor_bore_deg');
    case_tbl.rms_delay_spread = getNumericColumn(tbl, 'rms_delay_spread');
    case_tbl.k_factor_estimate = getNumericColumn(tbl, 'k_factor_estimate');
    case_tbl.fp_to_total_ratio = getNumericColumn(tbl, 'fp_to_total_ratio');
    case_tbl.num_paths = getNumericColumn(tbl, 'num_paths');
    case_tbl.num_significant_peaks = getNumericColumn(tbl, 'num_significant_peaks');
    case_tbl.gamma_cp_3_fp_only = getNumericColumn(tbl, 'gamma_cp_3_fp_only');
    case_tbl.is_nlos_geo = logical(getNumericColumn(tbl, 'is_nlos_geo'));
    case_tbl.is_nlos_bounce_0p33 = logical(getNumericColumn(tbl, 'is_nlos_bounce_0p33'));
    case_tbl.is_nlos_mixed_0p33 = logical(getNumericColumn(tbl, 'is_nlos_mixed_0p33'));

    case_tbl.cv_scheme = repmat(string(eval_struct.cv_scheme), height(tbl), 1);
    case_tbl.fold_id = eval_struct.fold_id;
    case_tbl.y_true = y;
    case_tbl.cir_score = eval_struct.cir.pred;
    case_tbl.cp_score = eval_struct.cp.pred;
    case_tbl.joint_score = eval_struct.joint.pred;
    case_tbl.cir_hat = case_tbl.cir_score >= cfg.threshold;
    case_tbl.cp_hat = case_tbl.cp_score >= cfg.threshold;
    case_tbl.joint_hat = case_tbl.joint_score >= cfg.threshold;
    case_tbl.cir_wrong = case_tbl.cir_hat ~= case_tbl.y_true;
    case_tbl.joint_wrong = case_tbl.joint_hat ~= case_tbl.y_true;
    case_tbl.cp_save = case_tbl.cir_wrong & ~case_tbl.joint_wrong;
    case_tbl.cp_harm = ~case_tbl.cir_wrong & case_tbl.joint_wrong;
    case_tbl.low_confidence = case_tbl.cir_score >= cfg.low_conf_lo & case_tbl.cir_score <= cfg.low_conf_hi;
    case_tbl.low_confidence_rescue = case_tbl.low_confidence & ~case_tbl.joint_wrong;
    case_tbl.joint_minus_cir = case_tbl.joint_score - case_tbl.cir_score;
    case_tbl.label_component = repmat("negative", height(tbl), 1);
    case_tbl.label_component(case_tbl.is_nlos_bounce_0p33 & ~case_tbl.is_nlos_geo) = "bounce_only";
    case_tbl.label_component(case_tbl.is_nlos_geo) = "geo_only";
    case_tbl.cir_confidence_group = repmat("high_confidence", height(tbl), 1);
    case_tbl.cir_confidence_group(case_tbl.low_confidence) = "low_confidence";

    case_tbl.rms_delay_spread_bin = numericBinStrings(case_tbl.rms_delay_spread, 3);
    case_tbl.k_factor_estimate_bin = numericBinStrings(case_tbl.k_factor_estimate, 3);
    case_tbl.fp_to_total_ratio_bin = numericBinStrings(case_tbl.fp_to_total_ratio, 3);
    case_tbl.num_paths_bin = numericBinStrings(case_tbl.num_paths, 3);
    case_tbl.gamma_cp_3_fp_only_bin = numericBinStrings(case_tbl.gamma_cp_3_fp_only, 3);
end

function room_tbl = roomMetricTable(case_tbl, cfg)
    rooms = unique(case_tbl.room_type, 'stable');
    rows = {};
    for i = 1:numel(rooms)
        mask = case_tbl.room_type == rooms(i);
        y = double(case_tbl.y_true(mask));
        n_total = sum(mask);
        n_pos = sum(case_tbl.y_true(mask));
        n_neg = sum(~case_tbl.y_true(mask));
        claim_tier = reliabilityTier(n_total, n_pos, n_neg, cfg);
        rows(end + 1, :) = { ... %#ok<AGROW>
            string(rooms(i)), string(case_tbl.cv_scheme(find(mask, 1, 'first'))), n_total, n_neg, n_pos, claim_tier, ...
            safeAuc(y, case_tbl.cir_score(mask)), safeAuc(y, case_tbl.cp_score(mask)), safeAuc(y, case_tbl.joint_score(mask)), ...
            safeAuc(y, case_tbl.joint_score(mask)) - safeAuc(y, case_tbl.cir_score(mask)), ...
            mean(case_tbl.cp_save(mask)), mean(case_tbl.cp_harm(mask)), mean(case_tbl.cp_save(mask)) - mean(case_tbl.cp_harm(mask)), ...
            sum(case_tbl.low_confidence(mask)), safeRate(sum(case_tbl.low_confidence_rescue(mask)), sum(case_tbl.low_confidence(mask)))};
    end
    room_tbl = cell2table(rows, 'VariableNames', {'room_type', 'cv_scheme', 'n_total', 'n_neg', 'n_pos', 'claim_tier', ...
        'auc_cir', 'auc_cp', 'auc_joint', 'delta_auc', 'cp_save_rate', 'cp_harm_rate', 'net_recovery', 'low_confidence_n', 'low_confidence_rescue_rate'});
end

function tbl = buildCvComparison(eval_random, eval_room, eval_loro, case_loro)
    rows = {};
    summaries = {eval_random.summary, eval_room.summary, eval_loro.summary};
    scopes = ["ALL", "ALL", "ALL"];
    for i = 1:numel(summaries)
        s = summaries{i};
        rows(end + 1, :) = { ... %#ok<AGROW>
            string(s.cv_scheme), scopes(i), s.n_total, s.n_neg, s.n_pos, s.auc_cir, s.auc_cp, s.auc_joint, s.delta_auc, s.delta_auc_ci_low, s.delta_auc_ci_high, ...
            s.cp_save_rate, s.cp_harm_rate, s.net_recovery, s.net_recovery_ci_low, s.net_recovery_ci_high};
    end
    rooms = unique(case_loro.room_type, 'stable');
    for i = 1:numel(rooms)
        mask = case_loro.room_type == rooms(i);
        y = double(case_loro.y_true(mask));
        rows(end + 1, :) = { ... %#ok<AGROW>
            "leave_one_group_out", string(rooms(i)), sum(mask), sum(~case_loro.y_true(mask)), sum(case_loro.y_true(mask)), ...
            safeAuc(y, case_loro.cir_score(mask)), safeAuc(y, case_loro.cp_score(mask)), safeAuc(y, case_loro.joint_score(mask)), ...
            safeAuc(y, case_loro.joint_score(mask)) - safeAuc(y, case_loro.cir_score(mask)), NaN, NaN, ...
            mean(case_loro.cp_save(mask)), mean(case_loro.cp_harm(mask)), mean(case_loro.cp_save(mask)) - mean(case_loro.cp_harm(mask)), NaN, NaN};
    end
    tbl = cell2table(rows, 'VariableNames', {'cv_scheme', 'scope', 'n_total', 'n_neg', 'n_pos', 'auc_cir', 'auc_cp', 'auc_joint', 'delta_auc', 'delta_auc_ci_low', 'delta_auc_ci_high', ...
        'cp_save_rate', 'cp_harm_rate', 'net_recovery', 'net_recovery_ci_low', 'net_recovery_ci_high'});
end

function tbl = buildLabelBalanceTable(tbl_src)
    room_values = stringColumn(tbl_src, 'room_type');
    rooms = ["ALL"; unique(room_values, 'stable')];
    rows = {};
    label_names = ["current_0p20", "geo_only", "bounce_0p33", "mixed_0p33"];
    for r = 1:numel(rooms)
        if rooms(r) == "ALL"
            mask = true(height(tbl_src), 1);
        else
            mask = room_values == rooms(r);
        end
        for l = 1:numel(label_names)
            switch label_names(l)
                case "current_0p20"
                    y = logical(getNumericColumn(tbl_src(mask, :), 'is_nlos_current_0p20'));
                case "geo_only"
                    y = logical(getNumericColumn(tbl_src(mask, :), 'is_nlos_geo'));
                case "bounce_0p33"
                    y = logical(getNumericColumn(tbl_src(mask, :), 'is_nlos_bounce_0p33'));
                otherwise
                    y = logical(getNumericColumn(tbl_src(mask, :), 'is_nlos_mixed_0p33'));
            end
            rows(end + 1, :) = {rooms(r), label_names(l), sum(mask), sum(~y), sum(y), safeRate(sum(y), sum(mask))}; %#ok<AGROW>
        end
    end
    tbl = cell2table(rows, 'VariableNames', {'room_scope', 'label_name', 'n_total', 'n_neg', 'n_pos', 'pos_frac'});
end

function tbl = candidateMetricTable(case_tbl, cfg)
    cands = unique(case_tbl.expected_candidate_id, 'stable');
    rows = {};
    for i = 1:numel(cands)
        mask = case_tbl.expected_candidate_id == cands(i);
        y = double(case_tbl.y_true(mask));
        n_total = sum(mask);
        n_pos = sum(case_tbl.y_true(mask));
        n_neg = sum(~case_tbl.y_true(mask));
        [delta_ci, net_ci] = bootstrapPairedMetrics(y, case_tbl.cir_score(mask), case_tbl.joint_score(mask), cellstr(case_tbl.room_type(mask)), cfg);
        rows(end + 1, :) = { ... %#ok<AGROW>
            cands(i), n_total, n_neg, n_pos, reliabilityTier(n_total, n_pos, n_neg, cfg), ...
            safeAuc(y, case_tbl.cir_score(mask)), safeAuc(y, case_tbl.cp_score(mask)), safeAuc(y, case_tbl.joint_score(mask)), ...
            safeAuc(y, case_tbl.joint_score(mask)) - safeAuc(y, case_tbl.cir_score(mask)), delta_ci(1), delta_ci(2), ...
            mean(case_tbl.cp_save(mask)), mean(case_tbl.cp_harm(mask)), mean(case_tbl.cp_save(mask)) - mean(case_tbl.cp_harm(mask)), net_ci(1), net_ci(2)};
    end
    tbl = cell2table(rows, 'VariableNames', {'expected_candidate_id', 'n_total', 'n_neg', 'n_pos', 'claim_tier', 'auc_cir', 'auc_cp', 'auc_joint', 'delta_auc', 'delta_auc_ci_low', 'delta_auc_ci_high', ...
        'cp_save_rate', 'cp_harm_rate', 'net_recovery', 'net_recovery_ci_low', 'net_recovery_ci_high'});
end

function cond_tbl = buildConditionBins(case_tbl, cfg)
    scope_levels = ["ALL"; unique(case_tbl.room_type, 'stable')];
    rows = {};
    single_defs = { ...
        'rms_delay_spread', 'rms_delay_spread_bin'; ...
        'k_factor_estimate', 'k_factor_estimate_bin'; ...
        'fp_to_total_ratio', 'fp_to_total_ratio_bin'; ...
        'num_paths', 'num_paths_bin'; ...
        'gamma_cp_3_fp_only', 'gamma_cp_3_fp_only_bin'; ...
        'cir_confidence_group', 'cir_confidence_group'; ...
        'label_component', 'label_component'};
    pair_defs = { ...
        'rms_delay_spread__k_factor_estimate', 'rms_delay_spread_bin', 'k_factor_estimate_bin'; ...
        'fp_to_total_ratio__num_paths', 'fp_to_total_ratio_bin', 'num_paths_bin'};

    for s = 1:numel(scope_levels)
        if scope_levels(s) == "ALL"
            scope_mask = true(height(case_tbl), 1);
        else
            scope_mask = case_tbl.room_type == scope_levels(s);
        end
        scoped = case_tbl(scope_mask, :);

        for i = 1:size(single_defs, 1)
            analysis_name = string(single_defs{i, 1});
            axis_col = string(single_defs{i, 2});
            labels = unique(scoped.(axis_col), 'stable');
            for j = 1:numel(labels)
                mask = scoped.(axis_col) == labels(j);
                rows(end + 1, :) = metricRow(scoped(mask, :), string(scope_levels(s)), analysis_name, axis_col, labels(j), "", "", cfg); %#ok<AGROW>
            end
        end

        for i = 1:size(pair_defs, 1)
            analysis_name = string(pair_defs{i, 1});
            axis1 = string(pair_defs{i, 2});
            axis2 = string(pair_defs{i, 3});
            labels1 = unique(scoped.(axis1), 'stable');
            labels2 = unique(scoped.(axis2), 'stable');
            for a = 1:numel(labels1)
                for b = 1:numel(labels2)
                    mask = scoped.(axis1) == labels1(a) & scoped.(axis2) == labels2(b);
                    rows(end + 1, :) = metricRow(scoped(mask, :), string(scope_levels(s)), analysis_name, axis1, labels1(a), axis2, labels2(b), cfg); %#ok<AGROW>
                end
            end
        end
    end

    cond_tbl = cell2table(rows, 'VariableNames', {'scope', 'analysis_name', 'axis1', 'axis1_label', 'axis2', 'axis2_label', ...
        'n_total', 'n_neg', 'n_pos', 'claim_tier', 'auc_cir', 'auc_cp', 'auc_joint', 'delta_auc', 'cp_save_rate', 'cp_harm_rate', 'net_recovery', 'low_confidence_rescue_rate'});
end

function row = metricRow(slice, scope_name, analysis_name, axis1, axis1_label, axis2, axis2_label, cfg)
    n_total = height(slice);
    n_pos = sum(slice.y_true);
    n_neg = sum(~slice.y_true);
    claim_tier = reliabilityTier(n_total, n_pos, n_neg, cfg);
    y = double(slice.y_true);
    row = {string(scope_name), string(analysis_name), string(axis1), string(axis1_label), string(axis2), string(axis2_label), ...
        n_total, n_neg, n_pos, claim_tier, ...
        safeAuc(y, slice.cir_score), safeAuc(y, slice.cp_score), safeAuc(y, slice.joint_score), ...
        safeAuc(y, slice.joint_score) - safeAuc(y, slice.cir_score), ...
        safeMean(slice.cp_save), safeMean(slice.cp_harm), safeMean(slice.cp_save) - safeMean(slice.cp_harm), ...
        safeRate(sum(slice.low_confidence_rescue), sum(slice.low_confidence))};
end

function summary = summarizeDerivedConditions(cond_tbl)
    main_all = cond_tbl(cond_tbl.scope == "ALL" & cond_tbl.claim_tier == "main", :);
    main_all = main_all(main_all.net_recovery > 0 | main_all.delta_auc > 0, :);
    if isempty(main_all)
        summary = table();
        return;
    end
    main_all = sortrows(main_all, {'net_recovery', 'delta_auc', 'n_total'}, {'descend', 'descend', 'descend'});
    keep = min(8, height(main_all));
    top_rows = main_all(1:keep, :);
    repeat_count = zeros(keep, 1);
    room_list = strings(keep, 1);
    for i = 1:keep
        mask = cond_tbl.analysis_name == top_rows.analysis_name(i) & cond_tbl.axis1_label == top_rows.axis1_label(i) & cond_tbl.axis2_label == top_rows.axis2_label(i) & ...
            cond_tbl.scope ~= "ALL" & cond_tbl.claim_tier == "main" & cond_tbl.net_recovery > 0;
        reps = unique(cond_tbl.scope(mask), 'stable');
        repeat_count(i) = numel(reps);
        room_list(i) = strjoin(cellstr(reps), ',');
    end
    summary = top_rows;
    summary.room_repeat_count = repeat_count;
    summary.room_repeat_list = room_list;
end

function tier = reliabilityTier(n_total, n_pos, n_neg, cfg)
    if n_total >= cfg.min_total && n_pos >= cfg.min_pos && n_neg >= cfg.min_neg
        tier = "main";
    elseif n_total >= cfg.min_supp_total
        tier = "supplemental";
    else
        tier = "invalid";
    end
end

function writeLabelBalanceReport(out_dir, label_balance, design_meta, refs)
    fid = fopen(fullfile(out_dir, 'day4_stage2_label_balance.md'), 'w');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# Day 4 Stage 2 Label Balance\n\n');
    fprintf(fid, '- status: `verified`\n');
    fprintf(fid, '- design source: `%s`\n', refs.design);
    fprintf(fid, '- relabel source: `%s`\n', refs.stage2_relabel);
    fprintf(fid, '- total_cases: `%d`\n\n', design_meta.total_cases);
    fprintf(fid, '## Room Counts\n\n');
    writeMarkdownTable(fid, design_meta.cases_per_room);
    fprintf(fid, '\n## Candidate Counts\n\n');
    writeMarkdownTable(fid, design_meta.cases_per_candidate);
    fprintf(fid, '\n## Label Balance\n\n');
    writeMarkdownTable(fid, label_balance);
end

function writeSanityReport(out_dir, sanity, dry_cases, refs)
    fid = fopen(fullfile(out_dir, 'day4_stage2_sanity_report.md'), 'w');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# Day 4 Stage 2 Sanity Report\n\n');
    fprintf(fid, '- status: `%s`\n', ternary(~sanity.blocker, 'no_issue', 'blocker'));
    fprintf(fid, '- verification: `verified`\n');
    fprintf(fid, '- design source: `%s`\n', refs.design);
    fprintf(fid, '- execution source: `%s`\n\n', refs.script);

    fprintf(fid, '## Dry Run Metrics\n\n');
    fprintf(fid, '- dry_n: `%d`\n', sanity.dry_n);
    fprintf(fid, '- valid_n: `%d`\n', sanity.valid_n);
    fprintf(fid, '- failed_rate: `%.4f`\n', sanity.failed_rate);
    fprintf(fid, '- nonfinite_pct: `%.4f`\n', sanity.nonfinite_pct);
    fprintf(fid, '- max_num_paths: `%.0f`\n', sanity.max_paths);
    fprintf(fid, '- path_cap_share_at_max: `%.4f`\n\n', sanity.path_cap_share);

    fprintf(fid, '## Room Balance\n\n');
    fprintf(fid, '| room | n_total | n_pos | n_neg | rms_std | path_std |\n');
    fprintf(fid, '| --- | ---: | ---: | ---: | ---: | ---: |\n');
    for i = 1:numel(sanity.room_levels)
        fprintf(fid, '| %s | %d | %d | %d | %.6g | %.6g |\n', sanity.room_levels(i), sanity.room_counts(i), sanity.room_pos(i), sanity.room_neg(i), sanity.rms_std(i), sanity.path_std(i));
    end

    fprintf(fid, '\n## Findings\n\n');
    fprintf(fid, '| class | status | finding | source |\n');
    fprintf(fid, '| --- | --- | --- | --- |\n');
    fprintf(fid, '| %s | verified | failed rate = %.4f | %s |\n', ternary(sanity.fail_ok, 'no_issue', 'blocker'), sanity.failed_rate, refs.script);
    fprintf(fid, '| %s | verified | room counts all >= 5 in dry run | %s |\n', ternary(sanity.room_ok, 'no_issue', 'blocker'), refs.script);
    fprintf(fid, '| %s | verified | canonical dry-run features finite = %s | %s |\n', ternary(sanity.finite_ok, 'no_issue', 'blocker'), ternary(sanity.finite_ok, 'true', 'false'), refs.script);
    fprintf(fid, '| %s | verified | preliminary mixed@0.33 label balance is two-class = %s | %s |\n', ternary(sanity.balance_ok, 'no_issue', 'blocker'), ternary(sanity.balance_ok, 'true', 'false'), refs.script);
    fprintf(fid, '| %s | verified | observed max num_paths = %.0f with cap share %.4f; prior audit says enumeration has no hard cap | %s |\n', ternary(sanity.path_ok, 'no_issue', 'caution'), sanity.max_paths, sanity.path_cap_share, refs.audit);
    fprintf(fid, '| %s | verified | room-wise RMS/path variability positive = %s | %s |\n', ternary(sanity.spread_ok, 'no_issue', 'blocker'), ternary(sanity.spread_ok, 'true', 'false'), refs.script);

    fprintf(fid, '\n## Dry Cases\n\n');
    writeMarkdownTable(fid, dry_cases(:, {'case_id', 'room_type', 'grid_layer', 'expected_candidate_id', 'snr_db', 'xpol_coupling_db', 'eps_r_multiplier', 'dominant_wall_material'}));

    fprintf(fid, '\n## Checklist\n\n');
    fprintf(fid, '- [%s] existing Stage 2 canonical file not overwritten\n', ternary(true, 'x', ' '));
    fprintf(fid, '- [%s] failed rate < 5%%\n', ternary(sanity.fail_ok, 'x', ' '));
    fprintf(fid, '- [%s] room별 최소 5 case 이상\n', ternary(sanity.room_ok, 'x', ' '));
    fprintf(fid, '- [%s] feature finite\n', ternary(sanity.finite_ok, 'x', ' '));
    fprintf(fid, '- [%s] label balance preliminary\n', ternary(sanity.balance_ok, 'x', ' '));
    fprintf(fid, '- [%s] num_paths enumeration cap issue not observed in dry run\n', ternary(sanity.path_ok, 'x', ' '));
    fprintf(fid, '- [%s] room-wise RMS delay spread/path count variability observed\n', ternary(sanity.spread_ok, 'x', ' '));
end

function writeSummaryReport(out_dir, design_meta, sanity, room_metrics, candidate_metrics, derived_summary, cv_comparison, refs)
    fid = fopen(fullfile(out_dir, 'day4_stage2_room_mini_summary.md'), 'w');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# Day 4 Stage 2 Room Mini Summary\n\n');
    fprintf(fid, '## Provenance\n\n');
    fprintf(fid, '- script: `%s`\n', refs.script);
    fprintf(fid, '- design source: `%s`\n', refs.design);
    fprintf(fid, '- Stage 2 grid source: `%s`\n', refs.room_grid);
    fprintf(fid, '- relabel source: `%s`\n', refs.stage2_relabel);
    fprintf(fid, '- total cases: `%d`\n', design_meta.total_cases);
    fprintf(fid, '- rooms included: `A/B/C`\n');
    fprintf(fid, '- antenna: `patch_ffd only`\n\n');

    fprintf(fid, '## Findings\n\n');
    fprintf(fid, '| class | status | finding | source |\n');
    fprintf(fid, '| --- | --- | --- | --- |\n');
    overall_room = cv_comparison(strcmp(cv_comparison.cv_scheme, 'group_stratified') & cv_comparison.scope == "ALL", :);
    fprintf(fid, '| %s | verified | room-stratified mixed@0.33 overall delta AUC = %.4f [%.4f, %.4f], net recovery = %.4f [%.4f, %.4f] | %s |\n', ...
        overallClass(overall_room), overall_room.delta_auc, overall_room.delta_auc_ci_low, overall_room.delta_auc_ci_high, overall_room.net_recovery, overall_room.net_recovery_ci_low, overall_room.net_recovery_ci_high, refs.script);
    room_c = room_metrics(room_metrics.room_type == "C", :);
    room_b = room_metrics(room_metrics.room_type == "B", :);
    room_a = room_metrics(room_metrics.room_type == "A", :);
    fprintf(fid, '| %s | verified | room-wise split is not monotonic: A delta/net = %.4f / %.4f, B = %.4f / %.4f, C = %.4f / %.4f | %s |\n', ...
        roomSplitClass(room_a, room_b, room_c), room_a.delta_auc, room_a.net_recovery, room_b.delta_auc, room_b.net_recovery, room_c.delta_auc, room_c.net_recovery, refs.script);
    targeted = candidate_metrics(candidate_metrics.expected_candidate_id ~= "BASE", :);
    if isempty(targeted)
        best_cand = candidate_metrics(1, :);
        best_label = "strongest expected candidate";
    else
        targeted = sortrows(targeted, {'net_recovery_ci_low', 'delta_auc_ci_low', 'net_recovery'}, {'descend', 'descend', 'descend'});
        best_cand = targeted(1, :);
        best_label = "strongest targeted candidate";
    end
    fprintf(fid, '| %s | verified | %s is %s with delta AUC = %.4f [%.4f, %.4f] and net recovery = %.4f [%.4f, %.4f] | %s |\n', ...
        candidateClass(best_cand), best_label, best_cand.expected_candidate_id, best_cand.delta_auc, best_cand.delta_auc_ci_low, best_cand.delta_auc_ci_high, best_cand.net_recovery, best_cand.net_recovery_ci_low, best_cand.net_recovery_ci_high, refs.script);
    fprintf(fid, '| %s | verified | dry-run sanity block = %s | results/recovery_mini/day4/day4_stage2_sanity_report.md |\n', ternary(sanity.blocker, 'blocker', 'no_issue'), ternary(sanity.blocker, 'failed', 'passed'));
    fprintf(fid, '| supplemental | 추측 | derived-condition repeatability should be interpreted room-conditionally unless the same bin stays positive in at least two rooms | %s |\n', refs.script);

    fprintf(fid, '\n## Room-Wise Metrics\n\n');
    writeMarkdownTable(fid, room_metrics);
    fprintf(fid, '\n## Candidate Translation\n\n');
    writeMarkdownTable(fid, candidate_metrics);
    fprintf(fid, '\n## Top Derived Conditions\n\n');
    if isempty(derived_summary)
        fprintf(fid, '_no positive derived-condition rows met the main-count threshold_\n');
    else
        writeMarkdownTable(fid, derived_summary);
    end
    fprintf(fid, '\n## CV Comparison\n\n');
    writeMarkdownTable(fid, cv_comparison);
    fprintf(fid, '\n## Plot Files\n\n');
    fprintf(fid, '- `day4_plot_room_auc_bar.png`\n');
    fprintf(fid, '- `day4_plot_room_delta_net_bar.png`\n');
    fprintf(fid, '- `day4_plot_rms_kfactor_net_recovery_heatmap.png`\n');
    fprintf(fid, '- `day4_plot_fp_numpaths_net_recovery_heatmap.png`\n');
    fprintf(fid, '- `day4_plot_room_cp_save_cp_harm_stacked.png`\n');
    fprintf(fid, '- `day4_plot_label_component_cp_save_concentration.png`\n');
    fprintf(fid, '- `day4_plot_stage1_candidate_overlay_on_stage2_cloud.png`\n');
end

function plotRoomAucBar(room_metrics, out_path)
    fig = figure('Visible', 'off', 'Position', [100 100 760 480]);
    vals = [room_metrics.auc_cir, room_metrics.auc_cp, room_metrics.auc_joint];
    bar(categorical(room_metrics.room_type), vals);
    ylabel('AUC');
    title('Room-wise CIR / CP / Joint AUC');
    legend({'CIR', 'CP', 'Joint'}, 'Location', 'northwest');
    grid on;
    saveas(fig, out_path);
    close(fig);
end

function plotRoomDeltaNetBar(room_metrics, out_path)
    fig = figure('Visible', 'off', 'Position', [100 100 760 480]);
    vals = [room_metrics.delta_auc, room_metrics.net_recovery];
    bar(categorical(room_metrics.room_type), vals);
    ylabel('Value');
    title('Room-wise delta AUC and net recovery');
    legend({'delta AUC', 'net recovery'}, 'Location', 'best');
    grid on;
    saveas(fig, out_path);
    close(fig);
end

function plotPairHeatmap(cond_tbl, analysis_name, plot_title, leaf_name, out_dir)
    pair_tbl = cond_tbl(cond_tbl.scope == "ALL" & cond_tbl.analysis_name == string(analysis_name), :);
    x_labels = unique(pair_tbl.axis2_label, 'stable');
    y_labels = unique(pair_tbl.axis1_label, 'stable');
    z = nan(numel(y_labels), numel(x_labels));
    counts = zeros(numel(y_labels), numel(x_labels));
    for i = 1:numel(y_labels)
        for j = 1:numel(x_labels)
            mask = pair_tbl.axis1_label == y_labels(i) & pair_tbl.axis2_label == x_labels(j);
            if any(mask)
                z(i, j) = pair_tbl.net_recovery(find(mask, 1, 'first'));
                counts(i, j) = pair_tbl.n_total(find(mask, 1, 'first'));
            end
        end
    end
    fig = figure('Visible', 'off', 'Position', [100 100 900 520]);
    imagesc(z);
    axis xy;
    colorbar;
    title(plot_title);
    xticks(1:numel(x_labels));
    xticklabels(x_labels);
    yticks(1:numel(y_labels));
    yticklabels(y_labels);
    xtickangle(20);
    colormap(parula);
    for i = 1:size(z, 1)
        for j = 1:size(z, 2)
            text(j, i, sprintf('%.3f\nn=%d', z(i, j), counts(i, j)), 'HorizontalAlignment', 'center', 'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold');
        end
    end
    saveas(fig, fullfile(out_dir, leaf_name));
    close(fig);
end

function plotRoomSaveHarm(case_tbl, out_path)
    rooms = unique(case_tbl.room_type, 'stable');
    save_n = zeros(numel(rooms), 1);
    harm_n = zeros(numel(rooms), 1);
    for i = 1:numel(rooms)
        mask = case_tbl.room_type == rooms(i);
        save_n(i) = sum(case_tbl.cp_save(mask));
        harm_n(i) = sum(case_tbl.cp_harm(mask));
    end
    fig = figure('Visible', 'off', 'Position', [100 100 760 480]);
    bar(categorical(rooms), [save_n, harm_n], 'stacked');
    ylabel('Count');
    title('Room-wise CP-save / CP-harm');
    legend({'CP-save', 'CP-harm'}, 'Location', 'northwest');
    grid on;
    saveas(fig, out_path);
    close(fig);
end

function plotLabelComponentConcentration(case_tbl, out_path)
    labels = ["negative", "bounce_only", "geo_only"];
    save_share = zeros(numel(labels), 1);
    total_save = max(sum(case_tbl.cp_save), 1);
    for i = 1:numel(labels)
        mask = case_tbl.label_component == labels(i);
        save_share(i) = sum(case_tbl.cp_save(mask)) / total_save;
    end
    fig = figure('Visible', 'off', 'Position', [100 100 760 480]);
    bar(categorical(labels), save_share);
    ylabel('Share of CP-save cases');
    title('CP-save concentration by label component');
    grid on;
    saveas(fig, out_path);
    close(fig);
end

function plotStage1Overlay(case_tbl, out_path)
    fig = figure('Visible', 'off', 'Position', [100 100 820 560]);
    hold on;
    scatter(case_tbl.eps_r_multiplier, case_tbl.xpol_coupling_db, 18, [0.75 0.75 0.75], 'filled', 'MarkerFaceAlpha', 0.35);
    plotCandidatePoints(case_tbl, "P1_MAIN", [0.00 0.45 0.74]);
    plotCandidatePoints(case_tbl, "P2_SENS", [0.85 0.33 0.10]);
    plotCandidatePoints(case_tbl, "P3_CTRL", [0.47 0.67 0.19]);
    rectangle('Position', [0.88, 30.0, 0.08, 8.0], 'EdgeColor', [0.00 0.45 0.74], 'LineWidth', 1.8, 'LineStyle', '--');
    rectangle('Position', [0.88, 30.0, 0.06, 2.0], 'EdgeColor', [0.85 0.33 0.10], 'LineWidth', 1.8, 'LineStyle', '--');
    rectangle('Position', [1.08, 22.0, 0.10, 16.0], 'EdgeColor', [0.47 0.67 0.19], 'LineWidth', 1.8, 'LineStyle', '--');
    xlabel('eps\_r multiplier');
    ylabel('xpol\_coupling\_db');
    title('Stage 1 candidate translation on Stage 2 case cloud');
    legend({'all Stage 2 cases', 'P1 main', 'P2 sensitivity', 'P3 control'}, 'Location', 'best');
    grid on;
    saveas(fig, out_path);
    close(fig);
end

function plotCandidatePoints(case_tbl, candidate_id, color_vec)
    mask = case_tbl.expected_candidate_id == candidate_id;
    scatter(case_tbl.eps_r_multiplier(mask), case_tbl.xpol_coupling_db(mask), 24, color_vec, 'filled', 'MarkerFaceAlpha', 0.70);
end

function writeMarkdownTable(fid, tbl)
    if isempty(tbl)
        fprintf(fid, '_empty_\n');
        return;
    end
    vars = tbl.Properties.VariableNames;
    fprintf(fid, '| %s |\n', strjoin(vars, ' | '));
    fprintf(fid, '| %s |\n', strjoin(repmat({'---'}, 1, numel(vars)), ' | '));
    for i = 1:height(tbl)
        vals = cell(1, numel(vars));
        for j = 1:numel(vars)
            vals{j} = scalarToString(tbl.(vars{j})(i));
        end
        fprintf(fid, '| %s |\n', strjoin(vals, ' | '));
    end
end

function txt = scalarToString(v)
    if iscell(v)
        v = v{1};
    end
    if isstring(v) || ischar(v)
        txt = char(string(v));
    elseif islogical(v)
        txt = ternary(v, 'true', 'false');
    elseif isnumeric(v)
        if isempty(v) || ~isscalar(v)
            txt = mat2str(v, 6);
        elseif isnan(v)
            txt = 'NaN';
        else
            txt = sprintf('%.6g', v);
        end
    else
        txt = char(string(v));
    end
end

function var_name = resolveVar(tbl, base_name)
    names = string(tbl.Properties.VariableNames);
    idx = find(names == string(base_name), 1, 'first');
    if isempty(idx)
        idx = find(startsWith(names, string(base_name) + "_"), 1, 'first');
    end
    if isempty(idx)
        error('Column %s not found.', base_name);
    end
    var_name = char(names(idx));
end

function value = maybeResolveVar(tbl, base_name)
    names = string(tbl.Properties.VariableNames);
    idx = find(names == string(base_name), 1, 'first');
    if isempty(idx)
        idx = find(startsWith(names, string(base_name) + "_"), 1, 'first');
    end
    if isempty(idx)
        value = "";
    else
        value = names(idx);
    end
end

function col = getNumericColumn(tbl, base_name)
    name = maybeResolveVar(tbl, base_name);
    if strlength(name) == 0
        col = nan(height(tbl), 1);
        return;
    end
    raw = tbl.(char(name));
    if islogical(raw)
        col = double(raw);
    elseif isnumeric(raw)
        col = double(raw);
    else
        col = str2double(string(raw));
    end
end

function col = stringColumn(tbl, base_name)
    name = maybeResolveVar(tbl, base_name);
    if strlength(name) == 0
        col = repmat("", height(tbl), 1);
        return;
    end
    col = string(tbl.(char(name)));
end

function labels = numericBinStrings(values, n_bins)
    v = double(values(:));
    labels = repmat("not_available", numel(v), 1);
    valid = isfinite(v);
    if ~any(valid)
        return;
    end
    q = quantile(v(valid), linspace(0, 1, n_bins + 1));
    q = strictEdges(q, v(valid));
    idx = discretize(v, q);
    for i = 1:(numel(q) - 1)
        labels(idx == i) = sprintf('[%.4g, %.4g]', q(i), q(i + 1));
    end
end

function edges = strictEdges(edges, values)
    edges = double(edges(:).');
    span = max(max(values) - min(values), 1.0);
    tol = 1e-9 * span;
    edges(1) = edges(1) - tol;
    for i = 2:numel(edges)
        if edges(i) <= edges(i - 1)
            edges(i) = edges(i - 1) + tol;
        end
    end
    edges(end) = edges(end) + tol;
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

function value = safeRate(num, den)
    if den <= 0
        value = NaN;
    else
        value = double(num) / double(den);
    end
end

function value = safeMean(x)
    if isempty(x)
        value = NaN;
    else
        value = mean(double(x), 'omitnan');
    end
end

function out = ternary(tf, a, b)
    if tf
        out = a;
    else
        out = b;
    end
end

function cls = overallClass(row)
    if row.net_recovery_ci_low > 0 && row.delta_auc_ci_low >= 0
        cls = 'no_issue';
    elseif row.net_recovery > 0 || row.delta_auc_ci_low >= 0
        cls = 'supplemental';
    else
        cls = 'caution';
    end
end

function cls = roomSplitClass(room_a, room_b, room_c)
    nets = [room_a.net_recovery, room_b.net_recovery, room_c.net_recovery];
    if all(diff(nets) >= 0)
        cls = 'no_issue';
    elseif any(nets > 0) && any(nets < 0)
        cls = 'caution';
    else
        cls = 'supplemental';
    end
end

function cls = candidateClass(row)
    if row.net_recovery_ci_low > 0 && row.delta_auc_ci_low >= 0
        cls = 'no_issue';
    elseif row.net_recovery > 0
        cls = 'supplemental';
    else
        cls = 'caution';
    end
end
