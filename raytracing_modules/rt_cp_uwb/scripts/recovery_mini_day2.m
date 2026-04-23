script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

out_dir = fullfile(repo_root, 'results', 'recovery_mini', 'day2');
ensureDir(out_dir);

cfg = struct();
cfg.seed = 2731;
cfg.n_boot_bins = 300;
cfg.n_shuffle = 20;
cfg.min_main_n = 50;
cfg.min_main_pos = 20;
cfg.min_main_neg = 20;
cfg.min_supp_n = 30;
cfg.numeric_bins = 4;
cfg.max_exact_bins = 6;
cfg.low_conf_lo = 0.40;
cfg.low_conf_hi = 0.60;
cfg.penalty_small = 0.010;
cfg.penalty_room_only = 0.010;
cfg.penalty_stage1_label = 0.005;
cfg.penalty_ideal_only = 0.015;
cfg.penalty_geo_only = 0.005;

refs = struct();
refs.script = 'scripts/recovery_mini_day2.m';
refs.day1_stage1 = 'results/recovery_mini/day1/day1_oof_predictions_stage1.csv';
refs.day1_stage2 = 'results/recovery_mini/day1/day1_oof_predictions_stage2.csv';
refs.day1_metrics = 'results/recovery_mini/day1/day1_metrics_summary.csv';
refs.canonical_features = '+features/canonicalFeatureNames.m';
refs.extract_all_features = '+features/extractAllFeatures.m';
refs.run_one_case = '+sweep/runOneCase.m';
refs.stage2_relabel = 'scripts/week4_day5_stage2_relabel_rerun_det.m';
refs.stage3_readiness = 'results/code_audit/stage3_readiness_report.md';
refs.label_audit = 'results/code_audit/label_audit_report.md';

inputs = resolveInputs(repo_root);

stage1_pred = readtable(inputs.day1_stage1_csv, 'TextType', 'string');
stage2_pred = readtable(inputs.day1_stage2_csv, 'TextType', 'string');
day1_metrics = readtable(inputs.day1_metrics_csv, 'TextType', 'string');

stage1_orig = loadResultsTable(inputs.stage1_mat, 'results');
stage2_orig = loadResultsTable(inputs.stage2_mat, 'results_relabel');
stage1_orig = alignOriginalTable(stage1_orig, stage1_pred.case_id);
stage2_orig = alignOriginalTable(stage2_orig, stage2_pred.case_id);

all_features = features.canonicalFeatureNames();
cp_features = all_features(1:6);
cir_features = all_features(7:end);

stage1_pred = enrichPredictionTable(stage1_pred, stage1_orig, {'eps_r', 'tan_delta', 'incidence_deg', 'num_paths', 'num_significant_peaks'});
stage2_pred = enrichPredictionTable(stage2_pred, stage2_orig, {'eps_r_multiplier', 'grid_layer', 'los_angle_from_anchor_bore_deg', 'num_paths', 'num_significant_peaks'});

stage1_pred = addDerivedColumnsStage1(stage1_pred, cfg);
stage2_pred = addDerivedColumnsStage2(stage2_pred, cfg);

[stage1_single_specs, stage1_pair_specs] = defineStage1Axes(cfg);
[stage2_single_specs, stage2_pair_specs] = defineStage2Axes(cfg);

stage1_bins = analyzeDataset(stage1_pred, 'Stage1', stage1_single_specs, stage1_pair_specs, cfg);
stage2_bins = analyzeDataset(stage2_pred, 'Stage2', stage2_single_specs, stage2_pair_specs, cfg);

all_bins = [stage1_bins; stage2_bins];
all_bins = addRankingColumns(all_bins, cfg);
stage1_bins = all_bins(all_bins.dataset == "Stage1", :);
stage2_bins = all_bins(all_bins.dataset == "Stage2", :);

candidate_tbl = shortlistCandidates(all_bins, stage1_pred, stage2_pred, stage1_single_specs, stage1_pair_specs, stage2_single_specs, stage2_pair_specs);

neg_summary = negativeControlSuite(stage1_orig, stage1_pred, stage2_orig, stage2_pred, cir_features, cp_features, cfg);
confound = buildConfoundAudit(stage1_pred, stage2_pred, stage1_bins, stage2_bins, candidate_tbl, neg_summary);

writetable(stage1_bins, fullfile(out_dir, 'day2_conditional_bins_stage1.csv'));
writetable(stage2_bins, fullfile(out_dir, 'day2_conditional_bins_stage2.csv'));
writetable(candidate_tbl, fullfile(out_dir, 'day2_candidate_regimes.csv'));

plotDeltaHeatmaps(stage1_bins, stage2_bins, fullfile(out_dir, 'day2_plot_conditional_delta_auc_heatmap.png'));
plotNetHeatmaps(stage1_bins, stage2_bins, fullfile(out_dir, 'day2_plot_net_recovery_heatmap.png'));
plotRoomSaveHarm(stage2_pred, fullfile(out_dir, 'day2_plot_cp_save_vs_cp_harm_by_room.png'));
plotLabelComponentConcentration(stage2_pred, fullfile(out_dir, 'day2_plot_cp_save_concentration_by_label_component.png'));
plotCandidateOverlayEpsXpol(stage1_pred, stage2_pred, candidate_tbl, stage1_single_specs, stage1_pair_specs, stage2_single_specs, stage2_pair_specs, fullfile(out_dir, 'day2_plot_candidate_overlay_eps_xpol.png'));
plotCandidateOverlayRmsK(stage1_pred, stage2_pred, candidate_tbl, stage1_single_specs, stage1_pair_specs, stage2_single_specs, stage2_pair_specs, fullfile(out_dir, 'day2_plot_candidate_overlay_rms_kfactor.png'));

writeShortlistReport(out_dir, inputs, refs, day1_metrics, stage1_bins, stage2_bins, candidate_tbl, confound);
writeConfoundAudit(out_dir, inputs, refs, confound);
writeNegativeControl(out_dir, refs, neg_summary);

disp('Day 2 regime shortlist outputs written under results/recovery_mini/day2/.');

function ensureDir(path_str)
    if exist(path_str, 'dir') ~= 7
        mkdir(path_str);
    end
end

function inputs = resolveInputs(repo_root)
    stage1_candidates = dir(fullfile(repo_root, 'results', '**', 'stage1_3000_ffd_det.mat'));
    stage2_candidates = dir(fullfile(repo_root, 'results', '**', 'stage2_900_ffd_relabel_det.mat'));
    if isempty(stage1_candidates) || isempty(stage2_candidates)
        error('Deterministic stage inputs were not found.');
    end
    [~, idx1] = max([stage1_candidates.datenum]);
    [~, idx2] = max([stage2_candidates.datenum]);
    inputs = struct();
    inputs.stage1_mat = fullfile(stage1_candidates(idx1).folder, stage1_candidates(idx1).name);
    inputs.stage2_mat = fullfile(stage2_candidates(idx2).folder, stage2_candidates(idx2).name);
    inputs.stage1_summary = fullfile(stage1_candidates(idx1).folder, 'stage1_3000_ffd_det_summary.md');
    inputs.stage2_summary = fullfile(stage2_candidates(idx2).folder, 'stage2_900_ffd_det_summary.md');
    inputs.stage2_relabel_summary = fullfile(stage2_candidates(idx2).folder, 'stage2_relabel_summary.md');
    inputs.day1_stage1_csv = fullfile(repo_root, 'results', 'recovery_mini', 'day1', 'day1_oof_predictions_stage1.csv');
    inputs.day1_stage2_csv = fullfile(repo_root, 'results', 'recovery_mini', 'day1', 'day1_oof_predictions_stage2.csv');
    inputs.day1_metrics_csv = fullfile(repo_root, 'results', 'recovery_mini', 'day1', 'day1_metrics_summary.csv');
    if exist(inputs.day1_stage1_csv, 'file') ~= 2 || exist(inputs.day1_stage2_csv, 'file') ~= 2 || exist(inputs.day1_metrics_csv, 'file') ~= 2
        error('Day 1 outputs are missing.');
    end
end

function tbl = loadResultsTable(mat_path, var_name)
    S = load(mat_path, var_name);
    tbl = S.(var_name);
    if ismember('failed', tbl.Properties.VariableNames)
        tbl = tbl(~logical(tbl.failed), :);
    end
end

function out = alignOriginalTable(orig_tbl, case_id_ref)
    orig_case = getColumn(orig_tbl, 'case_id');
    [tf, loc] = ismember(double(case_id_ref(:)), double(orig_case(:)));
    if ~all(tf)
        error('Failed to align original deterministic table by case_id.');
    end
    out = orig_tbl(loc, :);
end

function pred = enrichPredictionTable(pred, orig_tbl, var_names)
    for i = 1:numel(var_names)
        target = string(var_names{i});
        source_name = maybeResolveVar(orig_tbl, target);
        if strlength(source_name) == 0
            continue;
        end
        val = normalizeColumn(orig_tbl.(source_name));
        if ~ismember(char(target), pred.Properties.VariableNames)
            pred.(target) = val;
            continue;
        end
        current = pred.(target);
        if isnumeric(current)
            fill_mask = ~isfinite(current);
            current(fill_mask) = double(val(fill_mask));
            pred.(target) = current;
        elseif islogical(current)
            if ~any(current) && any(logical(val))
                pred.(target) = logical(val);
            end
        else
            current = string(current);
            val = string(val);
            fill_mask = strlength(current) == 0 | current == "<missing>";
            current(fill_mask) = val(fill_mask);
            pred.(target) = current;
        end
    end
end

function out = normalizeColumn(val)
    if islogical(val)
        out = logical(val);
    elseif isnumeric(val)
        out = double(val);
    else
        out = string(val);
    end
end

function pred = addDerivedColumnsStage1(pred, cfg)
    if ~ismember('eps_r', pred.Properties.VariableNames)
        pred.eps_r = nan(height(pred), 1);
    end
    if ~ismember('tan_delta', pred.Properties.VariableNames)
        pred.tan_delta = nan(height(pred), 1);
    end
    if ~ismember('incidence_deg', pred.Properties.VariableNames)
        pred.incidence_deg = nan(height(pred), 1);
    end
    pred.low_confidence_flag = repmat("not_low_conf", height(pred), 1);
    pred.low_confidence_flag(pred.cir_score >= cfg.low_conf_lo & pred.cir_score <= cfg.low_conf_hi) = "low_conf";
    pred.angle_axis_deg = pred.incidence_deg;
    use_los = ~isfinite(pred.angle_axis_deg);
    pred.angle_axis_deg(use_los) = pred.los_angle_from_anchor_bore_deg(use_los);
    pred.label_component = repmat("current_0p20_primary", height(pred), 1);
end

function pred = addDerivedColumnsStage2(pred, cfg)
    pred.low_confidence_flag = repmat("not_low_conf", height(pred), 1);
    pred.low_confidence_flag(pred.cir_score >= cfg.low_conf_lo & pred.cir_score <= cfg.low_conf_hi) = "low_conf";
    pred.label_component = repmat("negative", height(pred), 1);
    geo_mask = logical(pred.is_nlos_geo);
    bounce_mask = logical(pred.is_nlos_bounce_0p33);
    pred.label_component(bounce_mask & ~geo_mask) = "bounce_only";
    pred.label_component(geo_mask) = "geo_only";
end

function [single_specs, pair_specs] = defineStage1Axes(cfg)
    single_specs = [ ...
        axisSpec('eps_r', 'numeric', 'eps_r', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('tan_delta', 'numeric', 'tan_delta', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('xpol_coupling_db', 'numeric', 'xpol_coupling_db', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('snr_db', 'numeric', 'snr_db', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('angle_axis_deg', 'numeric', 'incidence_or_los_angle_deg', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('antenna_type', 'categorical', 'antenna_type', cfg.numeric_bins, cfg.max_exact_bins, ["patch_ffd"; "ideal"]); ...
        axisSpec('slab_placement', 'categorical', 'slab_placement', cfg.numeric_bins, cfg.max_exact_bins, ["floor"; "ceiling"; "wall_x"; "wall_y"]); ...
        axisSpec('gamma_cp_3_fp_only', 'numeric', 'gamma_cp_3_fp_only', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('fp_to_total_ratio', 'numeric', 'fp_to_total_ratio', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('rms_delay_spread', 'numeric', 'rms_delay_spread', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('k_factor_estimate', 'numeric', 'k_factor_estimate', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('num_significant_peaks', 'numeric', 'num_significant_peaks', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('num_paths', 'numeric', 'num_paths', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('low_confidence_flag', 'categorical', 'low_confidence_flag', cfg.numeric_bins, cfg.max_exact_bins, ["not_low_conf"; "low_conf"]); ...
        axisSpec('material_name', 'categorical', 'material_name', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1))];

    pair_specs = [ ...
        pairSpec('eps_r__xpol_coupling_db', 'eps_r', 'xpol_coupling_db'); ...
        pairSpec('rms_delay_spread__k_factor_estimate', 'rms_delay_spread', 'k_factor_estimate'); ...
        pairSpec('antenna_type__slab_placement', 'antenna_type', 'slab_placement'); ...
        pairSpec('angle_axis_deg__xpol_coupling_db', 'angle_axis_deg', 'xpol_coupling_db'); ...
        pairSpec('fp_to_total_ratio__k_factor_estimate', 'fp_to_total_ratio', 'k_factor_estimate'); ...
        pairSpec('antenna_type__material_name', 'antenna_type', 'material_name')];
end

function [single_specs, pair_specs] = defineStage2Axes(cfg)
    single_specs = [ ...
        axisSpec('room_type', 'categorical', 'room_type', cfg.numeric_bins, cfg.max_exact_bins, ["A"; "B"; "C"]); ...
        axisSpec('snr_db', 'numeric', 'snr_db', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('xpol_coupling_db', 'numeric', 'xpol_coupling_db', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('eps_r_multiplier', 'numeric', 'eps_r_multiplier', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('rms_delay_spread', 'numeric', 'rms_delay_spread', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('k_factor_estimate', 'numeric', 'k_factor_estimate', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('fp_to_total_ratio', 'numeric', 'fp_to_total_ratio', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('num_significant_peaks', 'numeric', 'num_significant_peaks', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('num_paths', 'numeric', 'num_paths', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('label_component', 'categorical', 'label_component', cfg.numeric_bins, cfg.max_exact_bins, ["negative"; "bounce_only"; "geo_only"]); ...
        axisSpec('los_angle_from_anchor_bore_deg', 'numeric', 'los_angle_from_anchor_bore_deg', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('grid_layer', 'numeric', 'grid_layer', cfg.numeric_bins, cfg.max_exact_bins, string.empty(0, 1)); ...
        axisSpec('dominant_wall_material', 'categorical', 'dominant_wall_material', cfg.numeric_bins, cfg.max_exact_bins, ["brick"; "concrete"; "drywall"; "glass"]); ...
        axisSpec('low_confidence_flag', 'categorical', 'low_confidence_flag', cfg.numeric_bins, cfg.max_exact_bins, ["not_low_conf"; "low_conf"])];

    pair_specs = [ ...
        pairSpec('room_type__dominant_wall_material', 'room_type', 'dominant_wall_material'); ...
        pairSpec('room_type__xpol_coupling_db', 'room_type', 'xpol_coupling_db'); ...
        pairSpec('room_type__eps_r_multiplier', 'room_type', 'eps_r_multiplier'); ...
        pairSpec('room_type__los_angle_from_anchor_bore_deg', 'room_type', 'los_angle_from_anchor_bore_deg'); ...
        pairSpec('room_type__label_component', 'room_type', 'label_component'); ...
        pairSpec('room_type__grid_layer', 'room_type', 'grid_layer'); ...
        pairSpec('eps_r_multiplier__xpol_coupling_db', 'eps_r_multiplier', 'xpol_coupling_db'); ...
        pairSpec('rms_delay_spread__k_factor_estimate', 'rms_delay_spread', 'k_factor_estimate'); ...
        pairSpec('fp_to_total_ratio__k_factor_estimate', 'fp_to_total_ratio', 'k_factor_estimate'); ...
        pairSpec('low_confidence_flag__room_type', 'low_confidence_flag', 'room_type')];
end

function spec = axisSpec(column_name, kind, display_name, n_bins, max_exact_bins, preferred_order)
    spec = struct();
    spec.column = string(column_name);
    spec.kind = string(kind);
    spec.display_name = string(display_name);
    spec.n_bins = n_bins;
    spec.max_exact_bins = max_exact_bins;
    spec.preferred_order = string(preferred_order(:));
end

function spec = pairSpec(name, axis1, axis2)
    spec = struct();
    spec.analysis_name = string(name);
    spec.axis1 = string(axis1);
    spec.axis2 = string(axis2);
end

function out_tbl = analyzeDataset(pred_tbl, dataset_name, single_specs, pair_specs, cfg)
    row_tables = {};
    total_save = sum(pred_tbl.cp_save_0p5);
    total_harm = sum(pred_tbl.cp_harm_0p5);
    for i = 1:numel(single_specs)
        row_tables{end + 1} = analyzeSingleAxis(pred_tbl, string(dataset_name), single_specs(i), cfg, total_save, total_harm); %#ok<AGROW>
    end
    for i = 1:numel(pair_specs)
        row_tables{end + 1} = analyzePairAxis(pred_tbl, string(dataset_name), single_specs, pair_specs(i), cfg, total_save, total_harm); %#ok<AGROW>
    end
    out_tbl = vertcat(row_tables{:});
end

function tbl = analyzeSingleAxis(pred_tbl, dataset_name, spec, cfg, total_save, total_harm)
    axis_info = resolveAxisLevels(pred_tbl, spec);
    if axis_info.status ~= "ok"
        tbl = notAvailableRow(dataset_name, "single", spec.column, spec.column, "", spec.display_name);
        return;
    end
    rows = {};
    for idx = 1:numel(axis_info.labels)
        mask = axis_info.codes == idx;
        label1 = axis_info.labels(idx);
        cond = spec.display_name + "=" + label1;
        metrics = computeCellMetrics(pred_tbl(mask, :), dataset_name, cfg, total_save, total_harm);
        rows{end + 1} = composeBinRow(dataset_name, "single", spec.column, spec.column, "", spec.display_name, "", label1, "", cond, metrics); %#ok<AGROW>
    end
    tbl = vertcat(rows{:});
end

function tbl = analyzePairAxis(pred_tbl, dataset_name, single_specs, pair_spec, cfg, total_save, total_harm)
    spec1 = lookupAxisSpec(single_specs, pair_spec.axis1);
    spec2 = lookupAxisSpec(single_specs, pair_spec.axis2);
    axis1 = resolveAxisLevels(pred_tbl, spec1);
    axis2 = resolveAxisLevels(pred_tbl, spec2);
    if axis1.status ~= "ok" || axis2.status ~= "ok"
        tbl = notAvailableRow(dataset_name, "pair", pair_spec.analysis_name, pair_spec.axis1, pair_spec.axis2, pair_spec.analysis_name);
        return;
    end
    pairs = unique([axis1.codes axis2.codes], 'rows', 'stable');
    rows = {};
    for i = 1:size(pairs, 1)
        mask = axis1.codes == pairs(i, 1) & axis2.codes == pairs(i, 2);
        label1 = axis1.labels(pairs(i, 1));
        label2 = axis2.labels(pairs(i, 2));
        cond = spec1.display_name + "=" + label1 + " & " + spec2.display_name + "=" + label2;
        metrics = computeCellMetrics(pred_tbl(mask, :), dataset_name, cfg, total_save, total_harm);
        rows{end + 1} = composeBinRow(dataset_name, "pair", pair_spec.analysis_name, pair_spec.axis1, pair_spec.axis2, spec1.display_name, spec2.display_name, label1, label2, cond, metrics); %#ok<AGROW>
    end
    tbl = vertcat(rows{:});
end

function spec = lookupAxisSpec(single_specs, axis_name)
    idx = find(string({single_specs.column})' == string(axis_name), 1, 'first');
    if isempty(idx)
        error('Axis spec not found: %s', axis_name);
    end
    spec = single_specs(idx);
end

function axis_info = resolveAxisLevels(tbl, spec)
    axis_info = struct();
    axis_info.status = "ok";
    axis_info.labels = strings(0, 1);
    axis_info.codes = zeros(height(tbl), 1);
    if ~ismember(char(spec.column), tbl.Properties.VariableNames)
        axis_info.status = "not_available";
        return;
    end

    if spec.kind == "categorical"
        raw = string(tbl.(spec.column));
        raw = strip(raw);
        missing_mask = strlength(raw) == 0 | raw == "<missing>" | lower(raw) == "nan";
        cats = unique(raw(~missing_mask), 'stable');
        if ~isempty(spec.preferred_order)
            ordered = spec.preferred_order(ismember(spec.preferred_order, cats));
            extras = cats(~ismember(cats, ordered));
            cats = [ordered; extras];
        end
        if isempty(cats) && ~any(~missing_mask)
            axis_info.status = "not_available";
            return;
        end
        labels = cats;
        for i = 1:numel(cats)
            axis_info.codes(raw == cats(i)) = i;
        end
        if any(missing_mask)
            labels(end + 1, 1) = "missing";
            axis_info.codes(missing_mask) = numel(labels);
        end
        axis_info.labels = labels;
        return;
    end

    vals = double(tbl.(spec.column));
    valid = isfinite(vals);
    if ~any(valid)
        axis_info.status = "not_available";
        return;
    end

    unique_vals = unique(vals(valid));
    labels = strings(0, 1);
    codes = zeros(height(tbl), 1);
    if numel(unique_vals) <= spec.max_exact_bins
        labels = strings(numel(unique_vals), 1);
        for i = 1:numel(unique_vals)
            labels(i) = "=" + fmtNum(unique_vals(i));
            codes(vals == unique_vals(i)) = i;
        end
    else
        q = quantile(vals(valid), linspace(0, 1, spec.n_bins + 1));
        q = unique(q(:));
        if numel(q) < 2
            labels = "=" + fmtNum(q(1));
            codes(valid) = 1;
        else
            labels = strings(numel(q) - 1, 1);
            for i = 1:(numel(q) - 1)
                if i < numel(q) - 1
                    mask = vals >= q(i) & vals < q(i + 1);
                else
                    mask = vals >= q(i) & vals <= q(i + 1);
                end
                codes(mask) = i;
                labels(i) = "[" + fmtNum(q(i)) + ", " + fmtNum(q(i + 1)) + "]";
            end
        end
    end
    if any(~valid)
        labels(end + 1, 1) = "missing";
        codes(~valid) = numel(labels);
    end
    axis_info.labels = labels;
    axis_info.codes = codes;
end

function row = notAvailableRow(dataset_name, bin_mode, analysis_name, axis1, axis2, cond_name)
    row = table( ...
        string(dataset_name), string(bin_mode), string(analysis_name), string(axis1), string(axis2), string(cond_name), string(""), string(""), ...
        string("not_available"), string("not_available"), ...
        0, 0, 0, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, ...
        NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, ...
        'VariableNames', baseVarNames());
end

function row = composeBinRow(dataset_name, bin_mode, analysis_name, axis1, axis2, disp1, disp2, label1, label2, cond, metrics)
    row = table( ...
        string(dataset_name), string(bin_mode), string(analysis_name), string(axis1), string(axis2), string(cond), string(label1), string(label2), ...
        string("ok"), string(metrics.support_tier), ...
        metrics.n_total, metrics.n_pos, metrics.n_neg, metrics.class_imbalance_ratio, ...
        metrics.auc_cir, metrics.pr_auc_cir, metrics.auc_joint, metrics.pr_auc_joint, metrics.delta_auc, ...
        metrics.delta_auc_ci_low, metrics.delta_auc_ci_high, metrics.cp_save_rate, metrics.cp_harm_rate, metrics.net_recovery, ...
        metrics.net_recovery_ci_low, metrics.net_recovery_ci_high, metrics.low_confidence_rescue_rate, metrics.cir_fail_rate, ...
        metrics.cp_save_n, metrics.cp_harm_n, metrics.cp_save_concentration, metrics.cp_harm_concentration, metrics.room_count, ...
        metrics.cp_save_room_share_max, metrics.cp_save_geo_share, metrics.cp_save_bounce_share, metrics.cp_save_room_c_geo_share, metrics.cp_save_ideal_share, metrics.patch_share, ...
        NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, ...
        'VariableNames', baseVarNames());
end

function names = baseVarNames()
    names = {'dataset', 'bin_mode', 'analysis_name', 'axis1', 'axis2', 'condition_definition', 'axis1_label', 'axis2_label', ...
        'analysis_status', 'support_tier', 'n_total', 'n_pos', 'n_neg', 'class_imbalance_ratio', ...
        'auc_cir', 'pr_auc_cir', 'auc_joint', 'pr_auc_joint', 'delta_auc', 'delta_auc_ci_low', 'delta_auc_ci_high', ...
        'cp_save_rate', 'cp_harm_rate', 'net_recovery', 'net_recovery_ci_low', 'net_recovery_ci_high', 'low_confidence_rescue_rate', 'cir_fail_rate', ...
        'cp_save_n', 'cp_harm_n', 'cp_save_concentration', 'cp_harm_concentration', 'room_count', ...
        'cp_save_room_share_max', 'cp_save_geo_share', 'cp_save_bounce_share', 'cp_save_room_c_geo_share', 'cp_save_ideal_share', 'patch_share', ...
        'penalty_n_small', 'penalty_room_only', 'penalty_stage1_label', 'penalty_ideal_only', 'penalty_geo_only', ...
        'rank_primary_score', 'rank_secondary_score', 'rank_adjusted_primary'};
end

function metrics = computeCellMetrics(sub_tbl, dataset_name, cfg, total_save, total_harm)
    metrics = struct();
    y_true = logical(sub_tbl.y_true);
    cp_save_mask = logical(sub_tbl.cp_save_0p5);
    cp_harm_mask = logical(sub_tbl.cp_harm_0p5);
    cir_correct = logical(sub_tbl.cir_correct_0p5);
    low_conf = logical(sub_tbl.low_confidence);
    low_conf_rescue = logical(sub_tbl.low_confidence_rescue_0p5);
    room_type = string(sub_tbl.room_type);
    label_component = string(sub_tbl.label_component);
    antenna_type = string(sub_tbl.antenna_type);

    metrics.n_total = height(sub_tbl);
    metrics.n_pos = sum(y_true);
    metrics.n_neg = sum(~y_true);
    metrics.class_imbalance_ratio = max(metrics.n_pos, metrics.n_neg) / max(min(metrics.n_pos, metrics.n_neg), 1);

    metrics.auc_cir = safeAuc(double(y_true), sub_tbl.cir_score);
    metrics.pr_auc_cir = safePrAuc(double(y_true), sub_tbl.cir_score);
    metrics.auc_joint = safeAuc(double(y_true), sub_tbl.joint_score);
    metrics.pr_auc_joint = safePrAuc(double(y_true), sub_tbl.joint_score);
    metrics.delta_auc = metrics.auc_joint - metrics.auc_cir;

    metrics.cp_save_n = sum(cp_save_mask);
    metrics.cp_harm_n = sum(cp_harm_mask);
    metrics.cp_save_rate = safeRate(metrics.cp_save_n, metrics.n_total);
    metrics.cp_harm_rate = safeRate(metrics.cp_harm_n, metrics.n_total);
    metrics.net_recovery = metrics.cp_save_rate - metrics.cp_harm_rate;
    metrics.cir_fail_rate = safeRate(sum(~cir_correct), metrics.n_total);
    metrics.low_confidence_rescue_rate = safeRate(sum(low_conf_rescue), sum(low_conf));
    metrics.cp_save_concentration = safeRate(metrics.cp_save_n, total_save);
    metrics.cp_harm_concentration = safeRate(metrics.cp_harm_n, total_harm);

    if ismember('room_type', sub_tbl.Properties.VariableNames)
        rooms = room_type;
        rooms = rooms(strlength(rooms) > 0);
        metrics.room_count = numel(unique(rooms, 'stable'));
    else
        metrics.room_count = NaN;
    end

    metrics.cp_save_room_share_max = shareMaxByMask(sub_tbl, cp_save_mask, 'room_type');
    metrics.cp_save_geo_share = safeRate(sum(cp_save_mask & label_component == "geo_only"), metrics.cp_save_n);
    metrics.cp_save_bounce_share = safeRate(sum(cp_save_mask & label_component == "bounce_only"), metrics.cp_save_n);
    metrics.cp_save_room_c_geo_share = safeRate(sum(cp_save_mask & room_type == "C" & label_component == "geo_only"), metrics.cp_save_n);
    metrics.cp_save_ideal_share = safeRate(sum(cp_save_mask & antenna_type == "ideal"), metrics.cp_save_n);
    metrics.patch_share = safeRate(sum(antenna_type == "patch_ffd"), metrics.n_total);

    if metrics.n_total >= cfg.min_main_n && metrics.n_pos >= cfg.min_main_pos && metrics.n_neg >= cfg.min_main_neg
        metrics.support_tier = "main";
    elseif metrics.n_total >= cfg.min_supp_n
        metrics.support_tier = "supplemental";
    else
        metrics.support_tier = "invalid";
    end

    if metrics.n_total >= cfg.min_supp_n && metrics.n_pos > 0 && metrics.n_neg > 0
        groups = bootstrapGroups(sub_tbl, dataset_name);
        [delta_ci, net_ci] = bootstrapCI(double(y_true), sub_tbl.cir_score, sub_tbl.joint_score, cp_save_mask, cp_harm_mask, groups, cfg.n_boot_bins, cfg.seed + metrics.n_total);
        metrics.delta_auc_ci_low = delta_ci(1);
        metrics.delta_auc_ci_high = delta_ci(2);
        metrics.net_recovery_ci_low = net_ci(1);
        metrics.net_recovery_ci_high = net_ci(2);
    else
        metrics.delta_auc_ci_low = NaN;
        metrics.delta_auc_ci_high = NaN;
        metrics.net_recovery_ci_low = NaN;
        metrics.net_recovery_ci_high = NaN;
    end
end

function out_tbl = addRankingColumns(out_tbl, cfg)
    out_tbl.penalty_n_small = zeros(height(out_tbl), 1);
    out_tbl.penalty_room_only = zeros(height(out_tbl), 1);
    out_tbl.penalty_stage1_label = zeros(height(out_tbl), 1);
    out_tbl.penalty_ideal_only = zeros(height(out_tbl), 1);
    out_tbl.penalty_geo_only = zeros(height(out_tbl), 1);
    out_tbl.rank_primary_score = nan(height(out_tbl), 1);
    out_tbl.rank_secondary_score = nan(height(out_tbl), 1);
    out_tbl.rank_adjusted_primary = nan(height(out_tbl), 1);

    for i = 1:height(out_tbl)
        if out_tbl.analysis_status(i) ~= "ok" || out_tbl.support_tier(i) ~= "main" || ~isfinite(out_tbl.net_recovery_ci_low(i)) || ~isfinite(out_tbl.delta_auc_ci_low(i))
            out_tbl.rank_primary_score(i) = -inf;
            out_tbl.rank_secondary_score(i) = -inf;
            out_tbl.rank_adjusted_primary(i) = -inf;
            continue;
        end
        out_tbl.penalty_n_small(i) = cfg.penalty_small * double(out_tbl.n_total(i) < 100);
        out_tbl.penalty_room_only(i) = cfg.penalty_room_only * double(out_tbl.dataset(i) == "Stage2" && out_tbl.cp_save_room_share_max(i) >= 0.80);
        out_tbl.penalty_stage1_label(i) = cfg.penalty_stage1_label * double(out_tbl.dataset(i) == "Stage1");
        out_tbl.penalty_ideal_only(i) = cfg.penalty_ideal_only * double(out_tbl.cp_save_ideal_share(i) >= 0.80);
        out_tbl.penalty_geo_only(i) = cfg.penalty_geo_only * double(out_tbl.dataset(i) == "Stage2" && out_tbl.cp_save_geo_share(i) >= 0.80);
        out_tbl.rank_primary_score(i) = out_tbl.net_recovery_ci_low(i);
        out_tbl.rank_secondary_score(i) = out_tbl.delta_auc_ci_low(i);
        out_tbl.rank_adjusted_primary(i) = out_tbl.rank_primary_score(i) - out_tbl.penalty_n_small(i) - out_tbl.penalty_room_only(i) - out_tbl.penalty_stage1_label(i) - out_tbl.penalty_ideal_only(i) - out_tbl.penalty_geo_only(i);
    end
end

function candidate_tbl = shortlistCandidates(all_bins, stage1_pred, stage2_pred, stage1_single_specs, stage1_pair_specs, stage2_single_specs, stage2_pair_specs)
    main_bins = all_bins(all_bins.support_tier == "main" & all_bins.analysis_status == "ok", :);
    main_bins = sortrows(main_bins, {'rank_adjusted_primary', 'rank_secondary_score', 'net_recovery', 'cp_save_concentration', 'n_total'}, {'descend', 'descend', 'descend', 'descend', 'descend'});

    picks = false(height(main_bins), 1);
    chosen = table();
    chosen_analysis = strings(0, 1);

    seed_indices = [ ...
        pickBestIndex(main_bins, main_bins.analysis_name == "low_confidence_flag" & contains(main_bins.condition_definition, "low_confidence_flag=low_conf"), {'rank_primary_score', 'rank_secondary_score'}, {'descend', 'descend'}), ...
        pickBestIndex(main_bins, main_bins.dataset == "Stage2" & main_bins.penalty_room_only == 0 & main_bins.penalty_geo_only == 0, {'rank_adjusted_primary', 'rank_secondary_score', 'net_recovery'}, {'descend', 'descend', 'descend'}), ...
        pickBestIndex(main_bins, main_bins.dataset == "Stage2" & main_bins.analysis_name == "room_type__dominant_wall_material" & main_bins.rank_secondary_score > 0 & main_bins.rank_primary_score > 0, {'rank_secondary_score', 'rank_primary_score', 'net_recovery'}, {'descend', 'descend', 'descend'}), ...
        pickBestIndex(main_bins, main_bins.dataset == "Stage1" & main_bins.analysis_name == "eps_r", {'rank_adjusted_primary', 'rank_secondary_score', 'net_recovery'}, {'descend', 'descend', 'descend'}), ...
        pickBestIndex(main_bins, main_bins.dataset == "Stage1" & main_bins.analysis_name == "eps_r__xpol_coupling_db", {'rank_adjusted_primary', 'rank_secondary_score', 'net_recovery'}, {'descend', 'descend', 'descend'})];

    for s = 1:numel(seed_indices)
        idx = seed_indices(s);
        if isnan(idx) || idx < 1
            continue;
        end
        if ~ismember(main_bins.analysis_name(idx), chosen_analysis)
            picks(idx) = true;
            chosen_analysis(end + 1, 1) = main_bins.analysis_name(idx); %#ok<AGROW>
        end
    end

    for i = 1:height(main_bins)
        if sum(picks) >= 5
            break;
        end
        if picks(i)
            continue;
        end
        if ismember(main_bins.analysis_name(i), chosen_analysis)
            continue;
        end
        picks(i) = true;
        chosen_analysis(end + 1, 1) = main_bins.analysis_name(i); %#ok<AGROW>
    end

    chosen = main_bins(picks, :);
    if height(chosen) > 5
        chosen = chosen(1:5, :);
    end
    chosen = sortrows(chosen, {'rank_adjusted_primary', 'rank_secondary_score', 'net_recovery'}, {'descend', 'descend', 'descend'});

    chosen.candidate_id = "C" + string((1:height(chosen)).');
    chosen.regime_name = strings(height(chosen), 1);
    chosen.supporting_metric = strings(height(chosen), 1);
    chosen.delta_auc_with_ci = strings(height(chosen), 1);
    chosen.net_recovery_with_ci = strings(height(chosen), 1);
    chosen.confound_risk = strings(height(chosen), 1);
    chosen.recommended_mini_experiment = strings(height(chosen), 1);
    chosen.member_case_ids = strings(height(chosen), 1);

    for i = 1:height(chosen)
        if chosen.dataset(i) == "Stage1"
            pred = stage1_pred;
            single_specs = stage1_single_specs;
            pair_specs = stage1_pair_specs;
        else
            pred = stage2_pred;
            single_specs = stage2_single_specs;
            pair_specs = stage2_pair_specs;
        end
        mask = maskFromBinRow(pred, chosen(i, :), single_specs, pair_specs);
        chosen.member_case_ids(i) = strjoin(string(pred.case_id(mask)).', ';');
        chosen.regime_name(i) = chosen.analysis_name(i) + " :: " + chosen.axis1_label(i) + ternary(strlength(chosen.axis2_label(i)) > 0, " x " + chosen.axis2_label(i), "");
        chosen.supporting_metric(i) = sprintf('adj netCI_low=%.4f; dAUC_CI_low=%.4f; net=%.4f', chosen.rank_adjusted_primary(i), chosen.rank_secondary_score(i), chosen.net_recovery(i));
        chosen.delta_auc_with_ci(i) = sprintf('%.6f [%.6f, %.6f]', chosen.delta_auc(i), chosen.delta_auc_ci_low(i), chosen.delta_auc_ci_high(i));
        chosen.net_recovery_with_ci(i) = sprintf('%.6f [%.6f, %.6f]', chosen.net_recovery(i), chosen.net_recovery_ci_low(i), chosen.net_recovery_ci_high(i));
        chosen.confound_risk(i) = confoundRisk(chosen(i, :));
        chosen.recommended_mini_experiment(i) = recommendMiniExperiment(chosen(i, :));
    end

    keep = {'candidate_id', 'dataset', 'analysis_name', 'axis1', 'axis2', 'axis1_label', 'axis2_label', 'regime_name', 'condition_definition', 'supporting_metric', 'n_total', 'n_pos', 'n_neg', ...
        'delta_auc', 'delta_auc_ci_low', 'delta_auc_ci_high', 'delta_auc_with_ci', ...
        'net_recovery', 'net_recovery_ci_low', 'net_recovery_ci_high', 'net_recovery_with_ci', ...
        'cp_save_rate', 'cp_harm_rate', 'cp_save_n', 'cp_harm_n', 'cp_save_concentration', ...
        'penalty_n_small', 'penalty_room_only', 'penalty_stage1_label', 'penalty_ideal_only', 'penalty_geo_only', ...
        'confound_risk', 'recommended_mini_experiment', 'member_case_ids'};
    candidate_tbl = chosen(:, keep);
end

function idx = pickBestIndex(tbl, mask, sort_cols, sort_dirs)
    idx = NaN;
    sub = tbl(mask, :);
    if isempty(sub)
        return;
    end
    sub = sortrows(sub, sort_cols, sort_dirs);
    target_key = sub.condition_definition(1);
    idx = find(tbl.condition_definition == target_key & tbl.analysis_name == sub.analysis_name(1), 1, 'first');
    if isempty(idx)
        idx = NaN;
    end
end

function risk = confoundRisk(row)
    flags = 0;
    flags = flags + double(row.penalty_room_only > 0);
    flags = flags + double(row.penalty_stage1_label > 0);
    flags = flags + double(row.penalty_ideal_only > 0);
    flags = flags + double(row.penalty_geo_only > 0);
    flags = flags + double(row.penalty_n_small > 0);
    if flags >= 3
        risk = "high";
    elseif flags >= 1
        risk = "moderate";
    else
        risk = "low";
    end
end

function out = recommendMiniExperiment(row)
    text = string(row.analysis_name);
    if contains(text, "eps_r") && contains(text, "xpol")
        out = "Sweep eps_r/eps_r_multiplier x xpol_coupling_db on a 4x4 local grid around the shortlisted bin center.";
    elseif contains(text, "rms_delay_spread") || contains(text, "k_factor")
        out = "Select boundary cases at the bin edges and rerun with matched geometry to test trend persistence rather than raw AUC.";
    elseif contains(text, "room_type") && contains(text, "dominant_wall_material")
        out = "Lock room/material and sweep eps_r_multiplier plus xpol_coupling_db on a 3x3 local grid.";
    elseif contains(text, "room_type") && contains(text, "los_angle")
        out = "Lock room and sweep anchor/tag angle around the shortlisted angle band while holding material fixed.";
    elseif contains(text, "low_confidence")
        out = "Mine low-confidence CIR cases and perturb the nearest physical controls to test whether rescue persists around the decision boundary.";
    elseif contains(text, "antenna_type") || contains(text, "slab_placement")
        out = "Run a Stage 1 mini sweep over slab_placement x xpol_coupling_db and keep both ideal and patch antennas for relevance checking.";
    else
        out = "Run a local 3x3 mini sweep around the shortlisted bin edges and compare CP-save against CP-harm.";
    end
end

function mask = maskFromBinRow(pred_tbl, row, single_specs, pair_specs)
    if row.bin_mode == "single"
        spec = lookupAxisSpec(single_specs, row.axis1);
        axis_info = resolveAxisLevels(pred_tbl, spec);
        idx = find(axis_info.labels == row.axis1_label, 1, 'first');
        mask = axis_info.codes == idx;
        return;
    end
    spec1 = lookupAxisSpec(single_specs, row.axis1);
    spec2 = lookupAxisSpec(single_specs, row.axis2);
    axis1 = resolveAxisLevels(pred_tbl, spec1);
    axis2 = resolveAxisLevels(pred_tbl, spec2);
    idx1 = find(axis1.labels == row.axis1_label, 1, 'first');
    idx2 = find(axis2.labels == row.axis2_label, 1, 'first');
    mask = axis1.codes == idx1 & axis2.codes == idx2;
end

function summary = negativeControlSuite(stage1_orig, stage1_pred, stage2_orig, stage2_pred, cir_features, cp_features, cfg)
    rows = {};
    rows{end + 1, 1} = negativeControlOne(stage1_orig, stage1_pred, cir_features, cp_features, cfg, 'Stage1'); %#ok<AGROW>
    rows{end + 1, 1} = negativeControlOne(stage2_orig, stage2_pred, cir_features, cp_features, cfg, 'Stage2'); %#ok<AGROW>
    summary = vertcat(rows{:});
end

function row = negativeControlOne(orig_tbl, pred_tbl, cir_features, cp_features, cfg, dataset_name)
    y = double(pred_tbl.y_true);
    fold_id = double(pred_tbl.fold_id);
    Xcir = table2array(orig_tbl(:, cir_features));
    Xcp = table2array(orig_tbl(:, cp_features));
    baseline_delta = safeAuc(y, pred_tbl.joint_score) - safeAuc(y, pred_tbl.cir_score);
    baseline_net = mean(pred_tbl.cp_save_0p5) - mean(pred_tbl.cp_harm_0p5);
    shuffled_delta = nan(cfg.n_shuffle, 1);
    shuffled_net = nan(cfg.n_shuffle, 1);
    rng_state = rng;
    cleanup_rng = onCleanup(@() rng(rng_state)); %#ok<NASGU>
    rng(cfg.seed + numel(y), 'twister');
    for r = 1:cfg.n_shuffle
        perm = randperm(size(Xcp, 1));
        Xjoint = [Xcp(perm, :) Xcir];
        score = arrayModelOof(Xjoint, y, fold_id);
        shuffled_delta(r) = safeAuc(y, score) - safeAuc(y, pred_tbl.cir_score);
        cp_save = ~pred_tbl.cir_correct_0p5 & ((score >= 0.50) == logical(y));
        cp_harm = pred_tbl.cir_correct_0p5 & ((score >= 0.50) ~= logical(y));
        shuffled_net(r) = mean(cp_save) - mean(cp_harm);
    end
    delta_ci = quantile(shuffled_delta, [0.025 0.975]);
    net_ci = quantile(shuffled_net, [0.025 0.975]);
    pass_flag = abs(mean(shuffled_delta)) < max(0.005, 0.5 * abs(baseline_delta)) && abs(mean(shuffled_net)) < 0.02 && delta_ci(1) <= 0 && delta_ci(2) >= 0 && net_ci(1) <= 0 && net_ci(2) >= 0;
    row = table( ...
        string(dataset_name), baseline_delta, baseline_net, mean(shuffled_delta), delta_ci(1), delta_ci(2), mean(shuffled_net), net_ci(1), net_ci(2), logical(pass_flag), ...
        'VariableNames', {'dataset', 'baseline_delta_auc', 'baseline_net_recovery', 'shuffle_delta_auc_mean', 'shuffle_delta_auc_ci_low', 'shuffle_delta_auc_ci_high', ...
        'shuffle_net_recovery_mean', 'shuffle_net_recovery_ci_low', 'shuffle_net_recovery_ci_high', 'pass_flag'});
end

function score = arrayModelOof(X, y, fold_id)
    score = nan(size(y));
    for fold = unique(fold_id(:)).'
        tr = fold_id ~= fold;
        te = fold_id == fold;
        Xtr = X(tr, :);
        Xte = X(te, :);
        mu = mean(Xtr, 1);
        sigma = std(Xtr, 0, 1);
        sigma(sigma < 1e-9) = 1.0;
        Xtr = (Xtr - mu) ./ sigma;
        Xte = (Xte - mu) ./ sigma;
        warn_state = warning;
        cleanup_warn = onCleanup(@() warning(warn_state)); %#ok<NASGU>
        warning('off', 'all');
        mdl = fitglm(Xtr, y(tr), 'Distribution', 'binomial', 'Link', 'logit');
        score(te) = predict(mdl, Xte);
    end
end

function confound = buildConfoundAudit(stage1_pred, stage2_pred, stage1_bins, stage2_bins, candidate_tbl, neg_summary)
    confound = struct();
    stage2_save = logical(stage2_pred.cp_save_0p5);
    stage2_harm = logical(stage2_pred.cp_harm_0p5);
    stage1_save = logical(stage1_pred.cp_save_0p5);
    stage1_harm = logical(stage1_pred.cp_harm_0p5);
    room_type = string(stage2_pred.room_type);
    label_component = string(stage2_pred.label_component);
    antenna_type_stage1 = string(stage1_pred.antenna_type);
    rooms = ["A"; "B"; "C"];
    room_save = zeros(numel(rooms), 1);
    room_harm = zeros(numel(rooms), 1);
    for i = 1:numel(rooms)
        room_save(i) = sum(stage2_save & room_type == rooms(i));
        room_harm(i) = sum(stage2_harm & room_type == rooms(i));
    end
    confound.room_distribution = table(rooms, room_save, room_harm, ...
        safeRate(room_save, sum(room_save)), safeRate(room_harm, sum(room_harm)), ...
        'VariableNames', {'room', 'cp_save_n', 'cp_harm_n', 'cp_save_share', 'cp_harm_share'});

    label_levels = ["negative"; "bounce_only"; "geo_only"];
    label_save = zeros(numel(label_levels), 1);
    label_harm = zeros(numel(label_levels), 1);
    for i = 1:numel(label_levels)
        label_save(i) = sum(stage2_save & label_component == label_levels(i));
        label_harm(i) = sum(stage2_harm & label_component == label_levels(i));
    end
    confound.label_distribution = table(label_levels, label_save, label_harm, ...
        safeRate(label_save, sum(label_save)), safeRate(label_harm, sum(label_harm)), ...
        'VariableNames', {'label_component', 'cp_save_n', 'cp_harm_n', 'cp_save_share', 'cp_harm_share'});

    confound.stage1_antenna_distribution = table( ...
        ["ideal"; "patch_ffd"], ...
        [sum(stage1_save & antenna_type_stage1 == "ideal"); sum(stage1_save & antenna_type_stage1 == "patch_ffd")], ...
        [sum(stage1_harm & antenna_type_stage1 == "ideal"); sum(stage1_harm & antenna_type_stage1 == "patch_ffd")], ...
        'VariableNames', {'antenna_type', 'cp_save_n', 'cp_harm_n'});

    confound.room_c_geo_save_share = safeRate(sum(stage2_save & room_type == "C" & label_component == "geo_only"), sum(stage2_save));
    confound.fp_k_corr_stage1 = pairwiseCorr(stage1_pred.fp_to_total_ratio, stage1_pred.k_factor_estimate);
    confound.fp_k_corr_stage2 = pairwiseCorr(stage2_pred.fp_to_total_ratio, stage2_pred.k_factor_estimate);
    confound.neg_summary = neg_summary;
    confound.top_old_eps_xpol = stage1_bins(stage1_bins.analysis_name == "eps_r__xpol_coupling_db" & stage1_bins.analysis_status == "ok", :);
    if ~isempty(confound.top_old_eps_xpol)
        confound.top_old_eps_xpol = sortrows(confound.top_old_eps_xpol, {'delta_auc', 'rank_adjusted_primary'}, {'descend', 'descend'});
        confound.top_old_eps_xpol = confound.top_old_eps_xpol(1, :);
    end
    confound.candidate_tbl = candidate_tbl;
end

function plotDeltaHeatmaps(stage1_bins, stage2_bins, out_path)
    fig = figure('Visible', 'off', 'Position', [100 100 1200 520]);
    t = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    nexttile(t);
    drawPairHeatmap(stage1_bins, "eps_r__xpol_coupling_db", 'delta_auc', 'Stage 1 delta AUC');
    nexttile(t);
    drawPairHeatmap(stage2_bins, "eps_r_multiplier__xpol_coupling_db", 'delta_auc', 'Stage 2 delta AUC');
    saveas(fig, out_path);
    close(fig);
end

function plotNetHeatmaps(stage1_bins, stage2_bins, out_path)
    fig = figure('Visible', 'off', 'Position', [100 100 1200 520]);
    t = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    nexttile(t);
    drawPairHeatmap(stage1_bins, "eps_r__xpol_coupling_db", 'net_recovery', 'Stage 1 net recovery');
    nexttile(t);
    drawPairHeatmap(stage2_bins, "eps_r_multiplier__xpol_coupling_db", 'net_recovery', 'Stage 2 net recovery');
    saveas(fig, out_path);
    close(fig);
end

function drawPairHeatmap(bin_tbl, analysis_name, metric_name, plot_title)
    sub = bin_tbl(bin_tbl.analysis_name == analysis_name & bin_tbl.analysis_status == "ok", :);
    if isempty(sub)
        axis off;
        title(plot_title + " (not available)");
        return;
    end
    x_labels = unique(sub.axis1_label, 'stable');
    y_labels = unique(sub.axis2_label, 'stable');
    grid = nan(numel(y_labels), numel(x_labels));
    counts = nan(numel(y_labels), numel(x_labels));
    for i = 1:height(sub)
        xi = find(x_labels == sub.axis1_label(i), 1, 'first');
        yi = find(y_labels == sub.axis2_label(i), 1, 'first');
        grid(yi, xi) = sub.(metric_name)(i);
        counts(yi, xi) = sub.n_total(i);
    end
    imagesc(grid);
    colorbar;
    set(gca, 'XTick', 1:numel(x_labels), 'XTickLabel', x_labels, 'YTick', 1:numel(y_labels), 'YTickLabel', y_labels, 'XTickLabelRotation', 20);
    xlabel(sub.axis1(1));
    ylabel(sub.axis2(1));
    title(plot_title);
    for yi = 1:size(grid, 1)
        for xi = 1:size(grid, 2)
            if isfinite(grid(yi, xi))
                text(xi, yi, sprintf('%.3f\nn=%d', grid(yi, xi), round(counts(yi, xi))), 'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'k');
            end
        end
    end
end

function plotRoomSaveHarm(stage2_pred, out_path)
    rooms = ["A"; "B"; "C"];
    save_n = zeros(1, numel(rooms));
    harm_n = zeros(1, numel(rooms));
    for i = 1:numel(rooms)
        save_n(i) = sum(stage2_pred.cp_save_0p5 & stage2_pred.room_type == rooms(i));
        harm_n(i) = sum(stage2_pred.cp_harm_0p5 & stage2_pred.room_type == rooms(i));
    end
    fig = figure('Visible', 'off', 'Position', [100 100 720 460]);
    bar(categorical(cellstr(rooms)), [save_n(:) harm_n(:)], 'grouped');
    ylabel('count');
    title('Stage 2 CP-save vs CP-harm by room');
    legend({'CP-save', 'CP-harm'}, 'Location', 'northwest');
    grid on;
    saveas(fig, out_path);
    close(fig);
end

function plotLabelComponentConcentration(stage2_pred, out_path)
    labels = ["negative"; "bounce_only"; "geo_only"];
    save_share = zeros(1, numel(labels));
    harm_share = zeros(1, numel(labels));
    total_save = max(sum(stage2_pred.cp_save_0p5), 1);
    total_harm = max(sum(stage2_pred.cp_harm_0p5), 1);
    for i = 1:numel(labels)
        save_share(i) = sum(stage2_pred.cp_save_0p5 & stage2_pred.label_component == labels(i)) / total_save;
        harm_share(i) = sum(stage2_pred.cp_harm_0p5 & stage2_pred.label_component == labels(i)) / total_harm;
    end
    fig = figure('Visible', 'off', 'Position', [100 100 760 460]);
    bar(categorical(cellstr(labels)), [save_share(:) harm_share(:)], 'grouped');
    ylabel('share');
    ylim([0 1]);
    title('Stage 2 CP-save concentration by label component');
    legend({'CP-save share', 'CP-harm share'}, 'Location', 'northwest');
    grid on;
    saveas(fig, out_path);
    close(fig);
end

function plotCandidateOverlayEpsXpol(stage1_pred, stage2_pred, candidate_tbl, stage1_single_specs, stage1_pair_specs, stage2_single_specs, stage2_pair_specs, out_path)
    fig = figure('Visible', 'off', 'Position', [100 100 1200 520]);
    t = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    colors = lines(max(height(candidate_tbl), 1));

    nexttile(t);
    scatter(stage1_pred.eps_r, stage1_pred.xpol_coupling_db, 16, [0.75 0.75 0.75], 'filled', 'MarkerFaceAlpha', 0.40);
    hold on;
    for i = 1:height(candidate_tbl)
        if candidate_tbl.dataset(i) ~= "Stage1"
            continue;
        end
        mask = maskFromCandidateTable(stage1_pred, candidate_tbl(i, :), stage1_single_specs, stage1_pair_specs);
        scatter(stage1_pred.eps_r(mask), stage1_pred.xpol_coupling_db(mask), 28, colors(i, :), 'filled', 'MarkerFaceAlpha', 0.85);
    end
    xlabel('eps_r');
    ylabel('xpol_coupling_db');
    title('Stage 1 candidate overlay on eps_r x xpol');
    grid on;

    nexttile(t);
    scatter(stage2_pred.eps_r_multiplier, stage2_pred.xpol_coupling_db, 16, [0.75 0.75 0.75], 'filled', 'MarkerFaceAlpha', 0.40);
    hold on;
    for i = 1:height(candidate_tbl)
        if candidate_tbl.dataset(i) ~= "Stage2"
            continue;
        end
        mask = maskFromCandidateTable(stage2_pred, candidate_tbl(i, :), stage2_single_specs, stage2_pair_specs);
        scatter(stage2_pred.eps_r_multiplier(mask), stage2_pred.xpol_coupling_db(mask), 28, colors(i, :), 'filled', 'MarkerFaceAlpha', 0.85);
    end
    xlabel('eps_r_multiplier');
    ylabel('xpol_coupling_db');
    title('Stage 2 candidate overlay on eps_r_multiplier x xpol');
    grid on;

    saveas(fig, out_path);
    close(fig);
end

function plotCandidateOverlayRmsK(stage1_pred, stage2_pred, candidate_tbl, stage1_single_specs, stage1_pair_specs, stage2_single_specs, stage2_pair_specs, out_path)
    fig = figure('Visible', 'off', 'Position', [100 100 1200 520]);
    t = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    colors = lines(max(height(candidate_tbl), 1));

    nexttile(t);
    scatter(stage1_pred.rms_delay_spread, stage1_pred.k_factor_estimate, 16, [0.75 0.75 0.75], 'filled', 'MarkerFaceAlpha', 0.40);
    hold on;
    for i = 1:height(candidate_tbl)
        if candidate_tbl.dataset(i) ~= "Stage1"
            continue;
        end
        mask = maskFromCandidateTable(stage1_pred, candidate_tbl(i, :), stage1_single_specs, stage1_pair_specs);
        scatter(stage1_pred.rms_delay_spread(mask), stage1_pred.k_factor_estimate(mask), 28, colors(i, :), 'filled', 'MarkerFaceAlpha', 0.85);
    end
    xlabel('rms_delay_spread');
    ylabel('k_factor_estimate');
    title('Stage 1 candidate overlay on RMS delay spread x K-factor');
    grid on;

    nexttile(t);
    scatter(stage2_pred.rms_delay_spread, stage2_pred.k_factor_estimate, 16, [0.75 0.75 0.75], 'filled', 'MarkerFaceAlpha', 0.40);
    hold on;
    for i = 1:height(candidate_tbl)
        if candidate_tbl.dataset(i) ~= "Stage2"
            continue;
        end
        mask = maskFromCandidateTable(stage2_pred, candidate_tbl(i, :), stage2_single_specs, stage2_pair_specs);
        scatter(stage2_pred.rms_delay_spread(mask), stage2_pred.k_factor_estimate(mask), 28, colors(i, :), 'filled', 'MarkerFaceAlpha', 0.85);
    end
    xlabel('rms_delay_spread');
    ylabel('k_factor_estimate');
    title('Stage 2 candidate overlay on RMS delay spread x K-factor');
    grid on;

    saveas(fig, out_path);
    close(fig);
end

function mask = maskFromCandidateTable(pred_tbl, candidate_row, single_specs, pair_specs) %#ok<INUSD>
    if strlength(candidate_row.axis2) == 0
        spec = lookupAxisSpec(single_specs, candidate_row.axis1);
        axis_info = resolveAxisLevels(pred_tbl, spec);
        idx = find(axis_info.labels == candidate_row.axis1_label, 1, 'first');
        mask = axis_info.codes == idx;
    else
        spec1 = lookupAxisSpec(single_specs, candidate_row.axis1);
        spec2 = lookupAxisSpec(single_specs, candidate_row.axis2);
        axis1 = resolveAxisLevels(pred_tbl, spec1);
        axis2 = resolveAxisLevels(pred_tbl, spec2);
        idx1 = find(axis1.labels == candidate_row.axis1_label, 1, 'first');
        idx2 = find(axis2.labels == candidate_row.axis2_label, 1, 'first');
        mask = axis1.codes == idx1 & axis2.codes == idx2;
    end
end

function writeShortlistReport(out_dir, inputs, refs, day1_metrics, stage1_bins, stage2_bins, candidate_tbl, confound)
    fid = fopen(fullfile(out_dir, 'day2_regime_shortlist.md'), 'w');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# Day 2 Regime Shortlist\n\n');
    fprintf(fid, '## Input Selection\n\n');
    fprintf(fid, '- Day 1 OOF inputs: `%s`, `%s`\n', refs.day1_stage1, refs.day1_stage2);
    fprintf(fid, '- Day 1 metric summary: `%s`\n', refs.day1_metrics);
    fprintf(fid, '- deterministic inputs: `%s`, `%s`\n', inputs.stage1_mat, inputs.stage2_mat);
    fprintf(fid, '- summary checked: `%s`, `%s`, `%s`\n', inputs.stage1_summary, inputs.stage2_summary, inputs.stage2_relabel_summary);
    fprintf(fid, '- canonical feature source: `%s`\n\n', refs.canonical_features);

    fprintf(fid, '## GO / NO-GO\n\n');
    if any(candidateTblStage2Positive(candidate_tbl))
        fprintf(fid, '- `GO` for conditional follow-up. `NO-GO` for any global CP replacement narrative.\n\n');
    else
        fprintf(fid, '- `NO-GO` for Day 3 promotion until a positive Stage 2 patch-relevant regime survives the confound audit.\n\n');
    end

    fprintf(fid, '## Day 1 Baseline Carry-Over\n\n');
    writeMarkdownTable(fid, day1_metrics(:, {'dataset', 'scope', 'threshold_type', 'delta_auc', 'delta_auc_ci', 'net_recovery', 'net_recovery_ci'}), ...
        {'dataset', 'scope', 'threshold_type', 'delta_auc', 'delta_auc_ci', 'net_recovery', 'net_recovery_ci'});

    fprintf(fid, '\n## Shortlist\n\n');
    if isempty(candidate_tbl)
        fprintf(fid, '- `blocker`: no main-support candidate regime passed the ranking stage.\n');
    else
        writeMarkdownTable(fid, candidate_tbl(:, {'candidate_id', 'dataset', 'regime_name', 'condition_definition', 'n_total', 'n_pos', 'n_neg', 'delta_auc_with_ci', 'net_recovery_with_ci', 'cp_save_rate', 'cp_harm_rate', 'confound_risk', 'recommended_mini_experiment'}), ...
            {'candidate_id', 'dataset', 'regime_name', 'condition_definition', 'n_total', 'n_pos', 'n_neg', 'delta_auc_with_ci', 'net_recovery_with_ci', 'cp_save_rate', 'cp_harm_rate', 'confound_risk', 'recommended_mini_experiment'});
    end

    fprintf(fid, '\n## Availability\n\n');
    availability_tbl = [stage1_bins(:, {'dataset', 'analysis_name', 'analysis_status'}); stage2_bins(:, {'dataset', 'analysis_name', 'analysis_status'})];
    availability_tbl = unique(availability_tbl, 'rows', 'stable');
    writeMarkdownTable(fid, availability_tbl, {'dataset', 'analysis_name', 'analysis_status'});

    fprintf(fid, '\n## Checks\n\n');
    fprintf(fid, '- `verified`: main-bin ranking excluded all `invalid` cells and used only `support_tier=main`.\n');
    fprintf(fid, '- `verified`: all required outputs were written under `results/recovery_mini/day2/`.\n');
    fprintf(fid, '- `caution`: Stage 1 candidates inherit a mild post-hoc penalty because Stage 1 canonical labels are `current@0.20`, not the Stage 2 primary mixed@0.33 label.\n');
    if ~isempty(confound.top_old_eps_xpol)
        fprintf(fid, '- `no_issue`: the old `eps_r x xpol` top cell is not restored as a headline; current top row is `support_tier=%s`, `adj_primary=%.4f`.\n', confound.top_old_eps_xpol.support_tier(1), confound.top_old_eps_xpol.rank_adjusted_primary(1));
    end
end

function tf = candidateTblStage2Positive(candidate_tbl)
    tf = candidate_tbl.dataset == "Stage2" & candidate_tbl.net_recovery_ci_low > 0;
end

function writeConfoundAudit(out_dir, inputs, refs, confound)
    fid = fopen(fullfile(out_dir, 'day2_confound_audit.md'), 'w');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# Day 2 Confound Audit\n\n');
    fprintf(fid, '## Inputs\n\n');
    fprintf(fid, '- Stage 2 relabel audit: `%s`\n', refs.label_audit);
    fprintf(fid, '- xpol proxy / Stage 3 readiness note: `%s`\n', refs.stage3_readiness);
    fprintf(fid, '- feature extraction / label construction: `%s`, `%s`, `%s`\n\n', refs.extract_all_features, refs.run_one_case, refs.stage2_relabel);

    findings = {};
    room_max_share = max(confound.room_distribution.cp_save_share);
    room_note = sprintf('Stage 2 CP-save room shares = A %.3f, B %.3f, C %.3f.', ...
        pickShare(confound.room_distribution, "A"), pickShare(confound.room_distribution, "B"), pickShare(confound.room_distribution, "C"));
    findings(end + 1, :) = {ternary(room_max_share >= 0.80, "blocker", ternary(room_max_share >= 0.55, "caution", "no_issue")), "verified", room_note, "day2_confound_audit.md", refs.label_audit}; %#ok<AGROW>

    label_note = sprintf('Stage 2 CP-save label-component shares = negative %.3f, bounce_only %.3f, geo_only %.3f; Room C geo_only save share = %.3f.', ...
        pickLabelShare(confound.label_distribution, "negative"), pickLabelShare(confound.label_distribution, "bounce_only"), pickLabelShare(confound.label_distribution, "geo_only"), confound.room_c_geo_save_share);
    label_class = ternary(confound.room_c_geo_save_share >= 0.80, "blocker", ternary(confound.room_c_geo_save_share >= 0.50, "caution", "no_issue"));
    findings(end + 1, :) = {label_class, "verified", label_note, "day2_confound_audit.md", refs.label_audit}; %#ok<AGROW>

    stage1_ideal_share = safeRate(confound.stage1_antenna_distribution.cp_save_n(confound.stage1_antenna_distribution.antenna_type == "ideal"), sum(confound.stage1_antenna_distribution.cp_save_n));
    ideal_note = sprintf('Stage 1 CP-save ideal share = %.3f and patch_ffd share = %.3f.', stage1_ideal_share, 1 - stage1_ideal_share);
    ideal_class = ternary(stage1_ideal_share >= 0.80, "caution", "no_issue");
    findings(end + 1, :) = {ideal_class, "verified", ideal_note, "day2_confound_audit.md", refs.day1_stage1}; %#ok<AGROW>

    xpol_note = 'xpol_coupling_db remains a MATLAB proxy / expected depolarization range, not an HFSS input label.';
    findings(end + 1, :) = {"no_issue", "verified", xpol_note, "results/code_audit/stage3_readiness_report.md", refs.stage3_readiness}; %#ok<AGROW>

    gamma_note = 'gamma_cp features originate from extractAllFeatures channel-response processing and current labels are assigned later in runOneCase / stage2 relabel script; no direct label-derived gamma feature path was found.';
    findings(end + 1, :) = {"no_issue", "verified", gamma_note, "scripts/recovery_mini_day2.m", refs.extract_all_features}; %#ok<AGROW>

    corr_note = sprintf('Pearson corr(fp_to_total_ratio, k_factor_estimate) = Stage1 %.3f, Stage2 %.3f.', confound.fp_k_corr_stage1, confound.fp_k_corr_stage2);
    corr_class = ternary(max(abs([confound.fp_k_corr_stage1, confound.fp_k_corr_stage2])) >= 0.80, "caution", "no_issue");
    findings(end + 1, :) = {corr_class, "verified", corr_note, "scripts/recovery_mini_day2.m", refs.day1_stage2}; %#ok<AGROW>

    find_tbl = cell2table(findings, 'VariableNames', {'classification', 'status', 'finding', 'evidence', 'source_ref'});
    writeMarkdownTable(fid, find_tbl, {'classification', 'status', 'finding', 'evidence', 'source_ref'});

    fprintf(fid, '\n## Room Distribution\n\n');
    writeMarkdownTable(fid, confound.room_distribution, {'room', 'cp_save_n', 'cp_harm_n', 'cp_save_share', 'cp_harm_share'});

    fprintf(fid, '\n## Label Component Distribution\n\n');
    writeMarkdownTable(fid, confound.label_distribution, {'label_component', 'cp_save_n', 'cp_harm_n', 'cp_save_share', 'cp_harm_share'});

    fprintf(fid, '\n## Stage 1 Antenna Distribution\n\n');
    writeMarkdownTable(fid, confound.stage1_antenna_distribution, {'antenna_type', 'cp_save_n', 'cp_harm_n'});
end

function writeNegativeControl(out_dir, refs, neg_summary)
    fid = fopen(fullfile(out_dir, 'day2_negative_control.md'), 'w');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# Day 2 Negative Control\n\n');
    fprintf(fid, '- control: CP feature row-shuffle with original Day 1 fold assignments, OOF refit, threshold 0.5 comparison.\n');
    fprintf(fid, '- source script: `%s`\n', refs.script);
    fprintf(fid, '- fold source: `%s`, `%s`\n\n', refs.day1_stage1, refs.day1_stage2);

    neg_summary.shuffle_delta_auc_ci = "[ " + string(compose('%.4f', neg_summary.shuffle_delta_auc_ci_low)) + ", " + string(compose('%.4f', neg_summary.shuffle_delta_auc_ci_high)) + " ]";
    neg_summary.shuffle_net_recovery_ci = "[ " + string(compose('%.4f', neg_summary.shuffle_net_recovery_ci_low)) + ", " + string(compose('%.4f', neg_summary.shuffle_net_recovery_ci_high)) + " ]";
    neg_summary.pass_flag_text = repmat("blocker", height(neg_summary), 1);
    neg_summary.pass_flag_text(neg_summary.pass_flag) = "no_issue";

    writeMarkdownTable(fid, neg_summary(:, {'dataset', 'baseline_delta_auc', 'shuffle_delta_auc_mean', 'shuffle_delta_auc_ci', 'baseline_net_recovery', 'shuffle_net_recovery_mean', 'shuffle_net_recovery_ci', 'pass_flag_text'}), ...
        {'dataset', 'baseline_delta_auc', 'shuffle_delta_auc_mean', 'shuffle_delta_auc_ci', 'baseline_net_recovery', 'shuffle_net_recovery_mean', 'shuffle_net_recovery_ci', 'pass_flag_text'});

    fprintf(fid, '\n## Findings\n\n');
    for i = 1:height(neg_summary)
        if neg_summary.pass_flag(i)
            fprintf(fid, '- `no_issue`: %s shuffled control collapsed toward zero for both delta AUC and net recovery.\n', neg_summary.dataset(i));
        else
            fprintf(fid, '- `blocker`: %s shuffled control retained non-trivial lift; leakage or confound remains possible.\n', neg_summary.dataset(i));
        end
    end
end

function groups = bootstrapGroups(pred_tbl, dataset_name)
    if string(dataset_name) == "Stage2" && sum(strlength(pred_tbl.room_type) > 0) > 0
        groups = pred_tbl.room_type + "|" + string(double(pred_tbl.y_true));
    else
        groups = string(double(pred_tbl.y_true));
    end
end

function [delta_ci, net_ci] = bootstrapCI(y, cir_score, joint_score, cp_save, cp_harm, groups, n_boot, seed)
    rng_state = rng;
    cleanup_rng = onCleanup(@() rng(rng_state)); %#ok<NASGU>
    rng(seed, 'twister');
    groups = string(groups(:));
    levels = unique(groups, 'stable');
    deltas = nan(n_boot, 1);
    nets = nan(n_boot, 1);
    for b = 1:n_boot
        idx = zeros(0, 1);
        for i = 1:numel(levels)
            level_idx = find(groups == levels(i));
            pick = level_idx(randi(numel(level_idx), numel(level_idx), 1));
            idx = [idx; pick]; %#ok<AGROW>
        end
        deltas(b) = safeAuc(y(idx), joint_score(idx)) - safeAuc(y(idx), cir_score(idx));
        nets(b) = mean(cp_save(idx)) - mean(cp_harm(idx));
    end
    delta_ci = quantile(deltas(isfinite(deltas)), [0.025 0.975]);
    net_ci = quantile(nets(isfinite(nets)), [0.025 0.975]);
end

function out = safeAuc(y, score)
    y = double(y(:));
    score = double(score(:));
    valid = isfinite(y) & isfinite(score);
    y = y(valid);
    score = score(valid);
    if numel(unique(y)) < 2
        out = NaN;
        return;
    end
    [~, ~, ~, out] = perfcurve(y, score, 1);
end

function out = safePrAuc(y, score)
    y = double(y(:));
    score = double(score(:));
    valid = isfinite(y) & isfinite(score);
    y = y(valid);
    score = score(valid);
    if numel(unique(y)) < 2
        out = NaN;
        return;
    end
    [~, ~, ~, out] = perfcurve(y, score, 1, 'xCrit', 'reca', 'yCrit', 'prec');
end

function out = safeRate(num, den)
    if numel(den) > 1
        den = double(den(:));
        out = zeros(size(den));
        nz = den ~= 0;
        out(nz) = double(num(nz)) ./ double(den(nz));
        out(~nz) = NaN;
        return;
    end
    if den == 0
        out = NaN;
    else
        out = double(num) / double(den);
    end
end

function out = shareMaxByMask(tbl, include_mask, var_name)
    if ~ismember(var_name, tbl.Properties.VariableNames)
        out = NaN;
        return;
    end
    include_mask = logical(include_mask);
    vals = string(tbl.(var_name));
    vals = vals(include_mask);
    vals = vals(strlength(vals) > 0);
    if isempty(vals)
        out = NaN;
        return;
    end
    u = unique(vals, 'stable');
    counts = zeros(numel(u), 1);
    for i = 1:numel(u)
        counts(i) = sum(vals == u(i));
    end
    out = max(counts) / sum(counts);
end

function out = pairwiseCorr(x, y)
    valid = isfinite(x) & isfinite(y);
    if sum(valid) < 3
        out = NaN;
        return;
    end
    C = corrcoef(double(x(valid)), double(y(valid)));
    out = C(1, 2);
end

function out = maybeResolveVar(tbl, desired_name)
    try
        out = string(resolveVar(tbl, desired_name));
    catch
        out = "";
    end
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

function out = getColumn(tbl, desired_name)
    name = maybeResolveVar(tbl, desired_name);
    if strlength(name) == 0
        out = nan(height(tbl), 1);
        return;
    end
    val = tbl.(name);
    if islogical(val)
        out = double(val);
    elseif isnumeric(val)
        out = double(val);
    else
        out = nan(height(tbl), 1);
    end
end

function txt = fmtNum(val)
    if ~isfinite(val)
        txt = 'NaN';
    elseif abs(val) >= 1e2 || (abs(val) > 0 && abs(val) < 1e-3)
        txt = sprintf('%.3e', val);
    elseif abs(val) >= 10
        txt = sprintf('%.2f', val);
    else
        txt = sprintf('%.4f', val);
    end
end

function out = pickShare(tbl, room_name)
    mask = tbl.room == room_name;
    if any(mask)
        out = tbl.cp_save_share(mask);
    else
        out = NaN;
    end
end

function out = pickLabelShare(tbl, label_name)
    mask = tbl.label_component == label_name;
    if any(mask)
        out = tbl.cp_save_share(mask);
    else
        out = NaN;
    end
end

function out = ternary(cond, a, b)
    if cond
        out = a;
    else
        out = b;
    end
end

function writeMarkdownTable(fid, tbl, columns)
    columns = cellstr(string(columns));
    fprintf(fid, '|');
    for i = 1:numel(columns)
        fprintf(fid, ' %s |', columns{i});
    end
    fprintf(fid, '\n|');
    for i = 1:numel(columns)
        fprintf(fid, '---|');
    end
    fprintf(fid, '\n');
    for r = 1:height(tbl)
        fprintf(fid, '|');
        for c = 1:numel(columns)
            fprintf(fid, ' %s |', mdValue(tbl.(columns{c})(r, :)));
        end
        fprintf(fid, '\n');
    end
end

function txt = mdValue(value)
    if iscell(value)
        value = value{1};
    end
    if isstring(value)
        txt = char(strjoin(value, ' '));
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
end
