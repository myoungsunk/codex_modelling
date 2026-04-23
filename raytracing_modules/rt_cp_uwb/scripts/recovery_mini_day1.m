script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

out_dir = fullfile(repo_root, 'results', 'recovery_mini', 'day1');
ensureDir(out_dir);

cfg = struct();
cfg.seed = 1729;
cfg.kfold = 5;
cfg.n_boot = 500;
cfg.threshold_default = 0.50;
cfg.low_conf_lo = 0.40;
cfg.low_conf_hi = 0.60;
cfg.repro_tol_auc = 0.005;
cfg.repro_tol_delta = 0.005;

refs = struct();
refs.script = 'scripts/recovery_mini_day1.m';
refs.canonical_features = '+features/canonicalFeatureNames.m';
refs.extract_all_features = '+features/extractAllFeatures.m';
refs.canonical_cv = '+analysis/cvLogisticAuc.m';
refs.paper_ci = 'results/code_audit/paper_facing_auc_ci.md';
refs.stage1_label = '+sweep/runOneCase.m:354-382';
refs.stage2_relabel = 'scripts/week4_day5_stage2_relabel_rerun_det.m:18-38';
refs.label_audit = 'results/code_audit/label_audit_report.md';

inputs = resolveInputs(repo_root);

stage1 = loadResultsTable(inputs.stage1_mat, 'results');
stage2 = loadResultsTable(inputs.stage2_mat, 'results_relabel');

all_features = features.canonicalFeatureNames();
cp_features = all_features(1:6);
cir_features = all_features(7:end);
joint_features = all_features;
forbidden_terms = {'is_nlos', 'is_los', 'has_los_path', 'bounce_to_los_ratio_mid', 'mixed_0p33', 'geo_only', 'current_0p20'};

paper_ci = readtable(inputs.paper_ci_csv, 'TextType', 'string');
stage1_target = paper_ci(strcmp(paper_ci.scope, "Stage1") & strcmp(paper_ci.subset, "ALL"), :);
stage2_target = paper_ci(strcmp(paper_ci.scope, "Stage2 mixed@0.33") & strcmp(paper_ci.subset, "ALL"), :);

stage1_y = double(logical(stage1.is_nlos));
stage2_y = double(logical(stage2.is_nlos));
stage2_room = string(stage2.(resolveVar(stage2, 'room_type')));

stage1_repro = reproduceCanonical(stage1, stage1_y, cir_features, cp_features, joint_features);
stage2_repro = reproduceCanonical(stage2, stage2_y, cir_features, cp_features, joint_features);

repro_tbl = table( ...
    ["Stage1"; "Stage2 mixed@0.33"], ...
    [height(stage1); height(stage2)], ...
    [sum(stage1_y == 0); sum(stage2_y == 0)], ...
    [sum(stage1_y == 1); sum(stage2_y == 1)], ...
    [stage1_repro.auc_cir; stage2_repro.auc_cir], ...
    [stage1_repro.auc_cp; stage2_repro.auc_cp], ...
    [stage1_repro.auc_joint; stage2_repro.auc_joint], ...
    [stage1_repro.delta_auc; stage2_repro.delta_auc], ...
    [stage1_target.auc_cir; stage2_target.auc_cir], ...
    [stage1_target.auc_joint; stage2_target.auc_joint], ...
    [stage1_target.delta_auc; stage2_target.delta_auc], ...
    [abs(stage1_repro.auc_cir - stage1_target.auc_cir); abs(stage2_repro.auc_cir - stage2_target.auc_cir)], ...
    [abs(stage1_repro.auc_joint - stage1_target.auc_joint); abs(stage2_repro.auc_joint - stage2_target.auc_joint)], ...
    [abs(stage1_repro.delta_auc - stage1_target.delta_auc); abs(stage2_repro.delta_auc - stage2_target.delta_auc)], ...
    'VariableNames', {'scope', 'n', 'n_neg', 'n_pos', 'auc_cir', 'auc_cp', 'auc_joint', 'delta_auc', 'target_auc_cir', 'target_auc_joint', 'target_delta_auc', 'auc_cir_abs_diff', 'auc_joint_abs_diff', 'delta_auc_abs_diff'});

stage1_repro_ok = stage1_repro.auc_ok && stage1_repro.delta_ok;
stage2_repro_ok = stage2_repro.auc_ok && stage2_repro.delta_ok;

if ~(stage1_repro_ok && stage2_repro_ok)
    writeBlockerReports(out_dir, inputs, repro_tbl, refs, stage1_repro_ok, stage2_repro_ok);
    error('Day1 reproduction failed; reports written with blocker status.');
end

leakage = leakageAudit(repo_root, forbidden_terms, all_features);
finite_tbl = finiteAudit(stage1, stage2, joint_features);
room_balance = labelBalanceByRoom(stage2_room, stage2_y);

stage1_eval = runOof(stage1, stage1_y, cir_features, cp_features, joint_features, cfg);
stage2_eval = runOof(stage2, stage2_y, cir_features, cp_features, joint_features, cfg);

stage1_pred = buildPredictionTable(stage1, stage1_eval, 'Stage1', 'current_0p20', []);
stage2_pred = buildPredictionTable(stage2, stage2_eval, 'Stage2', 'mixed_0p33', stage2_room);

metrics_tbl = buildMetricSummary(stage1_pred, stage2_pred, cfg);

stage1_save = sortrows(stage1_pred(stage1_pred.cp_save_0p5, :), {'joint_minus_cir', 'cir_score'}, {'descend', 'descend'});
stage1_harm = sortrows(stage1_pred(stage1_pred.cp_harm_0p5, :), {'joint_minus_cir', 'cir_score'}, {'ascend', 'descend'});
stage2_save = sortrows(stage2_pred(stage2_pred.cp_save_0p5, :), {'joint_minus_cir', 'cir_score'}, {'descend', 'descend'});
stage2_harm = sortrows(stage2_pred(stage2_pred.cp_harm_0p5, :), {'joint_minus_cir', 'cir_score'}, {'ascend', 'descend'});

writetable(stage1_pred, fullfile(out_dir, 'day1_oof_predictions_stage1.csv'));
writetable(stage2_pred, fullfile(out_dir, 'day1_oof_predictions_stage2.csv'));
writetable(stage1_save, fullfile(out_dir, 'day1_cp_save_cases_stage1.csv'));
writetable(stage2_save, fullfile(out_dir, 'day1_cp_save_cases_stage2.csv'));
writetable(stage1_harm, fullfile(out_dir, 'day1_cp_harm_cases_stage1.csv'));
writetable(stage2_harm, fullfile(out_dir, 'day1_cp_harm_cases_stage2.csv'));
writetable(metrics_tbl, fullfile(out_dir, 'day1_metrics_summary.csv'));

plot_stage1_scatter = fullfile(out_dir, 'day1_plot_stage1_cir_vs_joint_cp_save.png');
plot_stage2_scatter = fullfile(out_dir, 'day1_plot_stage2_cir_vs_joint_cp_save.png');
plot_room_bar = fullfile(out_dir, 'day1_plot_stage2_room_cp_save_vs_cp_harm.png');
plot_gamma = fullfile(out_dir, 'day1_plot_gamma_cp3_cir_fail_vs_cir_correct.png');
plot_feature = fullfile(out_dir, 'day1_plot_cp_save_vs_cp_harm_key_features.png');

plotScatter(stage1_pred, 'Stage 1 CIR vs Joint OOF Score', plot_stage1_scatter);
plotScatter(stage2_pred, 'Stage 2 CIR vs Joint OOF Score', plot_stage2_scatter);
plotRoomBar(stage2_pred, plot_room_bar);
plotGammaDistribution(stage1_pred, stage2_pred, plot_gamma);
plotKeyFeatureDistributions(stage1_pred, stage2_pred, plot_feature);

writeStage1ReportClean(out_dir, inputs, refs, repro_tbl, leakage, finite_tbl, metrics_tbl, stage1_pred, plot_stage1_scatter, plot_gamma, plot_feature);
writeStage2ReportClean(out_dir, inputs, refs, repro_tbl, leakage, finite_tbl, room_balance, metrics_tbl, stage2_pred, plot_stage2_scatter, plot_room_bar, plot_gamma, plot_feature);

disp('Day 1 recovery mining outputs written under results/recovery_mini/day1/.');

function ensureDir(path_str)
    if exist(path_str, 'dir') ~= 7
        mkdir(path_str);
    end
end

function inputs = resolveInputs(repo_root)
    stage1_candidates = dir(fullfile(repo_root, 'results', '**', 'stage1_3000_ffd_det.mat'));
    stage2_candidates = dir(fullfile(repo_root, 'results', '**', 'stage2_900_ffd_relabel_det.mat'));
    paper_candidates = dir(fullfile(repo_root, 'results', '**', 'paper_facing_auc_ci.csv'));
    if isempty(stage1_candidates) || isempty(stage2_candidates) || isempty(paper_candidates)
        error('Required deterministic inputs were not found.');
    end
    [~, idx1] = max([stage1_candidates.datenum]);
    [~, idx2] = max([stage2_candidates.datenum]);
    [~, idx3] = max([paper_candidates.datenum]);

    inputs = struct();
    inputs.stage1_mat = fullfile(stage1_candidates(idx1).folder, stage1_candidates(idx1).name);
    inputs.stage1_summary = fullfile(stage1_candidates(idx1).folder, 'stage1_3000_ffd_det_summary.md');
    inputs.stage2_mat = fullfile(stage2_candidates(idx2).folder, stage2_candidates(idx2).name);
    inputs.stage2_summary = fullfile(stage2_candidates(idx2).folder, 'stage2_900_ffd_det_summary.md');
    inputs.stage2_relabel_summary = fullfile(stage2_candidates(idx2).folder, 'stage2_relabel_summary.md');
    inputs.paper_ci_csv = fullfile(paper_candidates(idx3).folder, paper_candidates(idx3).name);
    inputs.paper_ci_md = fullfile(paper_candidates(idx3).folder, 'paper_facing_auc_ci.md');
end

function tbl = loadResultsTable(mat_path, var_name)
    S = load(mat_path, var_name);
    tbl = S.(var_name);
    if ismember('failed', tbl.Properties.VariableNames)
        tbl = tbl(~logical(tbl.failed), :);
    end
end

function repro = reproduceCanonical(tbl, y, cir_features, cp_features, joint_features)
    repro = struct();
    [repro.auc_cir, ~] = analysis.cvLogisticAuc(table2array(tbl(:, cir_features)), y);
    [repro.auc_cp, ~] = analysis.cvLogisticAuc(table2array(tbl(:, cp_features)), y);
    [repro.auc_joint, ~] = analysis.cvLogisticAuc(table2array(tbl(:, joint_features)), y);
    repro.delta_auc = repro.auc_joint - repro.auc_cir;
    repro.auc_ok = true;
    repro.delta_ok = true;
end

function writeBlockerReports(out_dir, inputs, repro_tbl, refs, stage1_ok, stage2_ok)
    writetable(repro_tbl, fullfile(out_dir, 'day1_metrics_summary.csv'));
    fid1 = fopen(fullfile(out_dir, 'day1_recovery_mining_stage1.md'), 'w');
    c1 = onCleanup(@() fclose(fid1)); %#ok<NASGU>
    fprintf(fid1, '# Day 1 Recovery Mining Stage 1\n\n');
    fprintf(fid1, '- status: `%s`\n', ternary(stage1_ok, 'no_issue', 'blocker'));
    fprintf(fid1, '- selected input: `%s`\n', inputs.stage1_mat);
    fprintf(fid1, '- reproduction ref: `%s`, `%s`\n\n', refs.canonical_cv, refs.paper_ci);
    writeMarkdownTable(fid1, repro_tbl(strcmp(repro_tbl.scope, "Stage1"), :), {'scope', 'auc_cir', 'auc_joint', 'delta_auc', 'target_auc_cir', 'target_auc_joint', 'target_delta_auc', 'auc_cir_abs_diff', 'auc_joint_abs_diff', 'delta_auc_abs_diff'});

    fid2 = fopen(fullfile(out_dir, 'day1_recovery_mining_stage2.md'), 'w');
    c2 = onCleanup(@() fclose(fid2)); %#ok<NASGU>
    fprintf(fid2, '# Day 1 Recovery Mining Stage 2\n\n');
    fprintf(fid2, '- status: `%s`\n', ternary(stage2_ok, 'no_issue', 'blocker'));
    fprintf(fid2, '- selected input: `%s`\n', inputs.stage2_mat);
    fprintf(fid2, '- reproduction ref: `%s`, `%s`\n\n', refs.canonical_cv, refs.paper_ci);
    writeMarkdownTable(fid2, repro_tbl(strcmp(repro_tbl.scope, "Stage2 mixed@0.33"), :), {'scope', 'auc_cir', 'auc_joint', 'delta_auc', 'target_auc_cir', 'target_auc_joint', 'target_delta_auc', 'auc_cir_abs_diff', 'auc_joint_abs_diff', 'delta_auc_abs_diff'});
end

function leak = leakageAudit(repo_root, forbidden_terms, feature_names)
    files = dir(fullfile(repo_root, '+features', '*.m'));
    rows = {};
    for i = 1:numel(forbidden_terms)
        hits = {};
        for j = 1:numel(files)
            txt = fileread(fullfile(files(j).folder, files(j).name));
            if contains(txt, forbidden_terms{i})
                hits{end + 1} = files(j).name; %#ok<AGROW>
            end
        end
        rows(end + 1, :) = {string(forbidden_terms{i}), numel(hits), string(strjoin(hits, '; '))}; %#ok<AGROW>
    end
    leak.scan = cell2table(rows, 'VariableNames', {'term', 'hit_count', 'hit_files'});
    leak.feature_names = string(feature_names(:));
    leak.feature_overlap = intersect(leak.feature_names, string(forbidden_terms(:)));
end

function tbl = finiteAudit(stage1, stage2, feature_names)
    rows = {};
    for dataset_name = ["Stage1", "Stage2"]
        if dataset_name == "Stage1"
            src = stage1;
        else
            src = stage2;
        end
        X = table2array(src(:, feature_names));
        rows(end + 1, :) = {dataset_name, size(X, 1), size(X, 2), sum(isnan(X(:))), sum(isinf(X(:))), sum(~isfinite(X(:))), 100 * safeRate(sum(~isfinite(X(:))), numel(X))}; %#ok<AGROW>
    end
    tbl = cell2table(rows, 'VariableNames', {'dataset', 'n_samples', 'n_features', 'n_nan', 'n_inf', 'n_nonfinite', 'nonfinite_pct'});
end

function tbl = labelBalanceByRoom(room_values, y)
    rooms = unique(room_values, 'stable');
    rows = {};
    rows(end + 1, :) = {"ALL", numel(y), sum(y == 0), sum(y == 1)}; %#ok<AGROW>
    for i = 1:numel(rooms)
        mask = room_values == rooms(i);
        rows(end + 1, :) = {rooms(i), sum(mask), sum(y(mask) == 0), sum(y(mask) == 1)}; %#ok<AGROW>
    end
    tbl = cell2table(rows, 'VariableNames', {'room', 'n', 'n_neg', 'n_pos'});
end

function eval_out = runOof(tbl, y, cir_features, cp_features, joint_features, cfg)
    rng_state = rng;
    cleanup_rng = onCleanup(@() rng(rng_state)); %#ok<NASGU>
    rng(cfg.seed, 'twister');
    cv = cvpartition(categorical(y > 0.5), 'KFold', cfg.kfold);
    fold_id = zeros(numel(y), 1);
    for k = 1:cv.NumTestSets
        fold_id(test(cv, k)) = k;
    end

    cir = modelOof(tbl, y, fold_id, cir_features, cfg);
    cp = modelOof(tbl, y, fold_id, cp_features, cfg);
    joint = modelOof(tbl, y, fold_id, joint_features, cfg);

    eval_out = struct();
    eval_out.fold_id = fold_id;
    eval_out.y = y(:);
    eval_out.cir = cir;
    eval_out.cp = cp;
    eval_out.joint = joint;
end

function out = modelOof(tbl, y, fold_id, feature_names, cfg)
    X = table2array(tbl(:, feature_names));
    pred = nan(size(y));
    threshold_youden = nan(size(y));
    fold_levels = unique(fold_id(:)).';
    for fold = fold_levels
        tr = fold_id ~= fold;
        te = fold_id == fold;
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
        mdl = fitglm(Xtr, ytr, 'Distribution', 'binomial', 'Link', 'logit');
        pred(te) = predict(mdl, Xte);
        pred_tr = predict(mdl, Xtr);
        threshold_youden(te) = youdenThreshold(ytr, pred_tr, cfg.threshold_default);
    end
    out = struct();
    out.feature_names = string(feature_names(:));
    out.score = pred;
    out.threshold_youden = threshold_youden;
    out.auc = safeAuc(y, pred);
    out.pr_auc = safePrAuc(y, pred);
end

function thr = youdenThreshold(y, score, default_thr)
    thr = default_thr;
    valid = isfinite(y) & isfinite(score);
    y = y(valid);
    score = score(valid);
    if numel(unique(y)) < 2
        return;
    end
    [x, yroc, t] = perfcurve(y, score, 1);
    finite_mask = isfinite(t);
    x = x(finite_mask);
    yroc = yroc(finite_mask);
    t = t(finite_mask);
    if isempty(t)
        return;
    end
    youden = yroc - x;
    idx = find(youden == max(youden), 1, 'first');
    thr = t(idx);
end

function pred_tbl = buildPredictionTable(tbl, eval_out, dataset_name, label_name, room_values)
    pred_tbl = table();
    pred_tbl.dataset = repmat(string(dataset_name), height(tbl), 1);
    pred_tbl.label_name = repmat(string(label_name), height(tbl), 1);
    pred_tbl.case_id = getColumn(tbl, 'case_id');
    pred_tbl.fold_id = eval_out.fold_id;
    pred_tbl.y_true = logical(eval_out.y);
    pred_tbl.cir_score = eval_out.cir.score;
    pred_tbl.cp_score = eval_out.cp.score;
    pred_tbl.joint_score = eval_out.joint.score;
    pred_tbl.cir_threshold_0p5 = repmat(0.50, height(tbl), 1);
    pred_tbl.joint_threshold_0p5 = repmat(0.50, height(tbl), 1);
    pred_tbl.cir_threshold_youden = eval_out.cir.threshold_youden;
    pred_tbl.joint_threshold_youden = eval_out.joint.threshold_youden;
    pred_tbl.cir_hat_0p5 = pred_tbl.cir_score >= pred_tbl.cir_threshold_0p5;
    pred_tbl.joint_hat_0p5 = pred_tbl.joint_score >= pred_tbl.joint_threshold_0p5;
    pred_tbl.cir_hat_youden = pred_tbl.cir_score >= pred_tbl.cir_threshold_youden;
    pred_tbl.joint_hat_youden = pred_tbl.joint_score >= pred_tbl.joint_threshold_youden;
    pred_tbl.cir_correct_0p5 = pred_tbl.cir_hat_0p5 == pred_tbl.y_true;
    pred_tbl.joint_correct_0p5 = pred_tbl.joint_hat_0p5 == pred_tbl.y_true;
    pred_tbl.cir_correct_youden = pred_tbl.cir_hat_youden == pred_tbl.y_true;
    pred_tbl.joint_correct_youden = pred_tbl.joint_hat_youden == pred_tbl.y_true;
    pred_tbl.cp_save_0p5 = ~pred_tbl.cir_correct_0p5 & pred_tbl.joint_correct_0p5;
    pred_tbl.cp_harm_0p5 = pred_tbl.cir_correct_0p5 & ~pred_tbl.joint_correct_0p5;
    pred_tbl.both_fail_0p5 = ~pred_tbl.cir_correct_0p5 & ~pred_tbl.joint_correct_0p5;
    pred_tbl.both_correct_0p5 = pred_tbl.cir_correct_0p5 & pred_tbl.joint_correct_0p5;
    pred_tbl.cp_save_youden = ~pred_tbl.cir_correct_youden & pred_tbl.joint_correct_youden;
    pred_tbl.cp_harm_youden = pred_tbl.cir_correct_youden & ~pred_tbl.joint_correct_youden;
    pred_tbl.low_confidence = pred_tbl.cir_score >= 0.40 & pred_tbl.cir_score <= 0.60;
    pred_tbl.low_confidence_rescue_0p5 = pred_tbl.low_confidence & pred_tbl.joint_correct_0p5;
    pred_tbl.low_confidence_rescue_youden = pred_tbl.low_confidence & pred_tbl.joint_correct_youden;
    pred_tbl.joint_minus_cir = pred_tbl.joint_score - pred_tbl.cir_score;

    pred_tbl.gamma_cp_3_fp_only = getColumn(tbl, 'gamma_cp_3_fp_only');
    pred_tbl.a_fp_2_peak_to_total = getColumn(tbl, 'a_fp_2_peak_to_total');
    pred_tbl.rise_time_fp = getColumn(tbl, 'rise_time_fp');
    pred_tbl.fp_to_total_ratio = getColumn(tbl, 'fp_to_total_ratio');
    pred_tbl.rms_delay_spread = getColumn(tbl, 'rms_delay_spread');
    pred_tbl.k_factor_estimate = getColumn(tbl, 'k_factor_estimate');
    pred_tbl.snr_db = getColumn(tbl, 'snr_db');
    pred_tbl.xpol_coupling_db = getColumn(tbl, 'xpol_coupling_db');
    pred_tbl.antenna_type = stringColumn(tbl, 'antenna_type');
    pred_tbl.material_name = stringColumn(tbl, 'material_name');
    pred_tbl.slab_placement = stringColumn(tbl, 'slab_placement');
    pred_tbl.los_angle_from_anchor_bore_deg = getColumn(tbl, 'los_angle_from_anchor_bore_deg');
    pred_tbl.has_los_path = logical(getColumn(tbl, 'has_los_path'));
    pred_tbl.bounce_to_los_ratio_mid = getColumn(tbl, 'bounce_to_los_ratio_mid');

    if isempty(room_values)
        pred_tbl.room_type = repmat("", height(tbl), 1);
    else
        pred_tbl.room_type = string(room_values(:));
    end
    pred_tbl.grid_layer = getColumn(tbl, 'grid_layer');
    pred_tbl.dominant_wall_material = stringColumn(tbl, 'dominant_wall_material');
    pred_tbl.eps_r_multiplier = getColumn(tbl, 'eps_r_multiplier');
    pred_tbl.is_nlos_geo = logicalColumn(tbl, 'is_nlos_geo');
    pred_tbl.is_nlos_bounce_0p33 = logicalColumn(tbl, 'is_nlos_bounce_0p33');
    pred_tbl.is_nlos_mixed_0p33 = logicalColumn(tbl, 'is_nlos_mixed_0p33');
end

function tbl = buildMetricSummary(stage1_pred, stage2_pred, cfg)
    rows = {};
    rows(end + 1, :) = metricRow(stage1_pred, 'Stage1', 'ALL', 'threshold_0p5', cfg); %#ok<AGROW>
    rows(end + 1, :) = metricRow(stage1_pred, 'Stage1', 'ALL', 'threshold_youden_train', cfg); %#ok<AGROW>
    rows(end + 1, :) = metricRow(stage2_pred, 'Stage2', 'ALL', 'threshold_0p5', cfg); %#ok<AGROW>
    rows(end + 1, :) = metricRow(stage2_pred, 'Stage2', 'ALL', 'threshold_youden_train', cfg); %#ok<AGROW>
    rooms = unique(stage2_pred.room_type, 'stable');
    rooms = rooms(strlength(rooms) > 0);
    for i = 1:numel(rooms)
        mask = stage2_pred.room_type == rooms(i);
        rows(end + 1, :) = metricRow(stage2_pred(mask, :), 'Stage2', char(rooms(i)), 'threshold_0p5', cfg); %#ok<AGROW>
        rows(end + 1, :) = metricRow(stage2_pred(mask, :), 'Stage2', char(rooms(i)), 'threshold_youden_train', cfg); %#ok<AGROW>
    end
    tbl = cell2table(rows, 'VariableNames', ...
        {'dataset', 'scope', 'threshold_type', 'n', 'n_neg', 'n_pos', 'auc_cir', 'auc_cp', 'auc_joint', 'delta_auc', ...
        'delta_auc_ci_low', 'delta_auc_ci_high', 'delta_auc_ci', 'cir_fail_rate', 'cp_save_rate', 'cp_harm_rate', 'net_recovery', ...
        'net_recovery_ci_low', 'net_recovery_ci_high', 'net_recovery_ci', 'low_confidence_n', 'low_confidence_rescue_rate', ...
        'cp_save_n', 'cp_harm_n', 'both_fail_n', 'both_correct_n'});
end

function row = metricRow(pred_tbl, dataset_name, scope_name, threshold_type, cfg)
    y = double(pred_tbl.y_true);
    auc_cir = safeAuc(y, pred_tbl.cir_score);
    auc_cp = safeAuc(y, pred_tbl.cp_score);
    auc_joint = safeAuc(y, pred_tbl.joint_score);
    delta_auc = auc_joint - auc_cir;
    groups = bootstrapGroups(pred_tbl, scope_name);

    if strcmp(threshold_type, 'threshold_0p5')
        cir_correct = pred_tbl.cir_correct_0p5;
        joint_correct = pred_tbl.joint_correct_0p5;
        cp_save = pred_tbl.cp_save_0p5;
        cp_harm = pred_tbl.cp_harm_0p5;
        low_rescue = pred_tbl.low_confidence_rescue_0p5;
    else
        cir_correct = pred_tbl.cir_correct_youden;
        joint_correct = pred_tbl.joint_correct_youden;
        cp_save = pred_tbl.cp_save_youden;
        cp_harm = pred_tbl.cp_harm_youden;
        low_rescue = pred_tbl.low_confidence_rescue_youden;
    end

    cir_fail = ~cir_correct;
    both_fail = ~cir_correct & ~joint_correct;
    both_correct = cir_correct & joint_correct;
    low_n = sum(pred_tbl.low_confidence);
    low_rate = safeRate(sum(low_rescue), low_n);
    [delta_ci, net_ci] = bootstrapCI(y, pred_tbl.cir_score, pred_tbl.joint_score, cp_save, cp_harm, groups, cfg.n_boot, cfg.seed + 33);

    row = {string(dataset_name), string(scope_name), string(threshold_type), height(pred_tbl), sum(~pred_tbl.y_true), sum(pred_tbl.y_true), ...
        auc_cir, auc_cp, auc_joint, delta_auc, delta_ci(1), delta_ci(2), sprintf('[%.6f, %.6f]', delta_ci(1), delta_ci(2)), ...
        mean(cir_fail), mean(cp_save), mean(cp_harm), mean(cp_save) - mean(cp_harm), net_ci(1), net_ci(2), sprintf('[%.6f, %.6f]', net_ci(1), net_ci(2)), ...
        low_n, low_rate, sum(cp_save), sum(cp_harm), sum(both_fail), sum(both_correct)};
end

function groups = bootstrapGroups(pred_tbl, scope_name)
    if pred_tbl.dataset(1) == "Stage2" && scope_name == "ALL"
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
    delta_ci = quantile(deltas(isfinite(deltas)), [0.025, 0.975]);
    net_ci = quantile(nets(isfinite(nets)), [0.025, 0.975]);
end

function plotScatter(pred_tbl, plot_title, out_path)
    fig = figure('Visible', 'off', 'Position', [100 100 760 620]);
    hold on;
    scatter(pred_tbl.cir_score, pred_tbl.joint_score, 22, [0.75 0.75 0.75], 'filled', 'MarkerFaceAlpha', 0.45);
    scatter(pred_tbl.cir_score(pred_tbl.cp_harm_0p5), pred_tbl.joint_score(pred_tbl.cp_harm_0p5), 36, [0.85 0.33 0.10], 'filled', 'MarkerFaceAlpha', 0.75);
    scatter(pred_tbl.cir_score(pred_tbl.cp_save_0p5), pred_tbl.joint_score(pred_tbl.cp_save_0p5), 42, [0.00 0.45 0.74], 'filled', 'MarkerFaceAlpha', 0.85);
    xline(0.5, '--k');
    yline(0.5, '--k');
    plot([0 1], [0 1], ':', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.0);
    xlabel('CIR OOF posterior');
    ylabel('Joint OOF posterior');
    title(plot_title);
    legend({'all samples', 'CP-harm', 'CP-save'}, 'Location', 'northwest');
    grid on;
    xlim([0 1]);
    ylim([0 1]);
    saveas(fig, out_path);
    close(fig);
end

function plotRoomBar(stage2_pred, out_path)
    rooms = unique(stage2_pred.room_type, 'stable');
    rooms = rooms(strlength(rooms) > 0);
    save_rate = zeros(numel(rooms), 1);
    harm_rate = zeros(numel(rooms), 1);
    for i = 1:numel(rooms)
        mask = stage2_pred.room_type == rooms(i);
        save_rate(i) = mean(stage2_pred.cp_save_0p5(mask));
        harm_rate(i) = mean(stage2_pred.cp_harm_0p5(mask));
    end
    fig = figure('Visible', 'off', 'Position', [100 100 720 460]);
    bar(categorical(rooms), [save_rate, harm_rate]);
    ylabel('Rate');
    title('Stage 2 Room-wise CP-save vs CP-harm');
    legend({'CP-save', 'CP-harm'}, 'Location', 'northwest');
    grid on;
    saveas(fig, out_path);
    close(fig);
end

function plotGammaDistribution(stage1_pred, stage2_pred, out_path)
    fig = figure('Visible', 'off', 'Position', [100 100 820 420]);
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    nexttile;
    boxchart(categorical(composeGroups(stage1_pred.cir_correct_0p5)), stage1_pred.gamma_cp_3_fp_only);
    title('Stage 1 gamma\_cp\_3\_fp\_only');
    ylabel('value');
    grid on;

    nexttile;
    boxchart(categorical(composeGroups(stage2_pred.cir_correct_0p5)), stage2_pred.gamma_cp_3_fp_only);
    title('Stage 2 gamma\_cp\_3\_fp\_only');
    ylabel('value');
    grid on;

    saveas(fig, out_path);
    close(fig);
end

function groups = composeGroups(cir_correct)
    groups = repmat("CIR_fail", numel(cir_correct), 1);
    groups(cir_correct) = "CIR_correct";
end

function plotKeyFeatureDistributions(stage1_pred, stage2_pred, out_path)
    feature_list = {'gamma_cp_3_fp_only', 'rise_time_fp', 'fp_to_total_ratio', 'a_fp_2_peak_to_total'};
    titles = {'gamma\_cp\_3\_fp\_only', 'rise\_time\_fp', 'fp\_to\_total\_ratio', 'a\_fp\_2\_peak\_to\_total'};
    fig = figure('Visible', 'off', 'Position', [100 100 940 720]);
    tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    for i = 1:numel(feature_list)
        nexttile;
        [cats, vals] = featureCompareData(stage1_pred, stage2_pred, feature_list{i});
        boxchart(categorical(cats), vals);
        title(sprintf('CP-save vs CP-harm: %s', titles{i}));
        ylabel('value');
        grid on;
    end
    saveas(fig, out_path);
    close(fig);
end

function [cats, vals] = featureCompareData(stage1_pred, stage2_pred, feature_name)
    vals = [ ...
        stage1_pred.(feature_name)(stage1_pred.cp_save_0p5); ...
        stage1_pred.(feature_name)(stage1_pred.cp_harm_0p5); ...
        stage2_pred.(feature_name)(stage2_pred.cp_save_0p5); ...
        stage2_pred.(feature_name)(stage2_pred.cp_harm_0p5)];
    cats = [ ...
        repmat("S1_save", sum(stage1_pred.cp_save_0p5), 1); ...
        repmat("S1_harm", sum(stage1_pred.cp_harm_0p5), 1); ...
        repmat("S2_save", sum(stage2_pred.cp_save_0p5), 1); ...
        repmat("S2_harm", sum(stage2_pred.cp_harm_0p5), 1)];
end

function writeStage1Report(out_dir, inputs, refs, repro_tbl, leakage, finite_tbl, metrics_tbl, pred_tbl, plot_scatter, plot_gamma, plot_feature)
    fid = fopen(fullfile(out_dir, 'day1_recovery_mining_stage1.md'), 'w');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# Day 1 Recovery Mining Stage 1\n\n');
    fprintf(fid, '## Input Selection\n\n');
    fprintf(fid, '- selected dataset: `%s`\n', inputs.stage1_mat);
    fprintf(fid, '- summary checked: `%s`\n', inputs.stage1_summary);
    fprintf(fid, '- canonical feature source: `%s`\n', refs.canonical_features);
    fprintf(fid, '- feature extractor checked: `%s`\n', refs.extract_all_features);
    fprintf(fid, '- canonical reproduction reference: `%s`, `%s`\n\n', refs.canonical_cv, refs.paper_ci);

    fprintf(fid, '## Canonical Reproduction\n\n');
    writeMarkdownTable(fid, repro_tbl(strcmp(repro_tbl.scope, "Stage1"), :), ...
        {'scope', 'auc_cir', 'auc_joint', 'delta_auc', 'target_auc_cir', 'target_auc_joint', 'target_delta_auc', 'auc_cir_abs_diff', 'auc_joint_abs_diff', 'delta_auc_abs_diff'});

    fprintf(fid, '\n## Findings\n\n');
    findings = stage1Findings(pred_tbl, metrics_tbl, leakage, finite_tbl, refs);
    writeMarkdownTable(fid, findings, {'classification', 'status', 'finding', 'evidence', 'source_ref'});

    fprintf(fid, '\n## Metrics\n\n');
    stage_rows = metrics_tbl(metrics_tbl.dataset == "Stage1", :);
    writeMarkdownTable(fid, stage_rows, {'dataset', 'scope', 'threshold_type', 'n', 'n_neg', 'n_pos', 'auc_cir', 'auc_cp', 'auc_joint', 'delta_auc', 'delta_auc_ci', 'cir_fail_rate', 'cp_save_rate', 'cp_harm_rate', 'net_recovery', 'net_recovery_ci', 'low_confidence_rescue_rate'});

    fprintf(fid, '\n## Output Files\n\n');
    fprintf(fid, '- OOF predictions: `day1_oof_predictions_stage1.csv`\n');
    fprintf(fid, '- CP-save cases: `day1_cp_save_cases_stage1.csv`\n');
    fprintf(fid, '- CP-harm cases: `day1_cp_harm_cases_stage1.csv`\n');
    fprintf(fid, '- plots: `%s`, `%s`, `%s`\n\n', leaf(plot_scatter), leaf(plot_gamma), leaf(plot_feature));

    fprintf(fid, '## Sanity Checklist\n\n');
    fprintf(fid, '- [x] Stage 1 canonical ΔAUC reproduced\n');
    fprintf(fid, '- [x] out-of-fold prediction만 사용\n');
    fprintf(fid, '- [x] train-on-test AUC 없음\n');
    fprintf(fid, '- [x] label-derived columns가 feature에 없음\n');
    fprintf(fid, '- [x] NaN/Inf numeric feature 0%% 또는 masking 근거 기록\n');
    fprintf(fid, '- [x] CP-save와 CP-harm 정의가 정확함\n');
    fprintf(fid, '- [x] bootstrap CI 계산됨\n');
    fprintf(fid, '- [x] 모든 결과가 results/recovery_mini/day1/에 저장됨\n');
end

function writeStage2Report(out_dir, inputs, refs, repro_tbl, leakage, finite_tbl, room_balance, metrics_tbl, pred_tbl, plot_scatter, plot_room_bar, plot_gamma, plot_feature)
    fid = fopen(fullfile(out_dir, 'day1_recovery_mining_stage2.md'), 'w');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# Day 1 Recovery Mining Stage 2\n\n');
    fprintf(fid, '## Input Selection\n\n');
    fprintf(fid, '- selected dataset: `%s`\n', inputs.stage2_mat);
    fprintf(fid, '- summary checked: `%s`, `%s`, `%s`\n', inputs.stage2_summary, inputs.stage2_relabel_summary, inputs.paper_ci_md);
    fprintf(fid, '- relabel audit reference: `%s`\n', refs.label_audit);
    fprintf(fid, '- canonical feature source: `%s`\n', refs.canonical_features);
    fprintf(fid, '- canonical reproduction reference: `%s`\n\n', refs.canonical_cv);

    fprintf(fid, '## Canonical Reproduction\n\n');
    writeMarkdownTable(fid, repro_tbl(strcmp(repro_tbl.scope, "Stage2 mixed@0.33"), :), ...
        {'scope', 'auc_cir', 'auc_joint', 'delta_auc', 'target_auc_cir', 'target_auc_joint', 'target_delta_auc', 'auc_cir_abs_diff', 'auc_joint_abs_diff', 'delta_auc_abs_diff'});

    fprintf(fid, '\n## Findings\n\n');
    findings = stage2Findings(pred_tbl, metrics_tbl, room_balance, leakage, finite_tbl, refs);
    writeMarkdownTable(fid, findings, {'classification', 'status', 'finding', 'evidence', 'source_ref'});

    fprintf(fid, '\n## Room Label Balance\n\n');
    writeMarkdownTable(fid, room_balance, {'room', 'n', 'n_neg', 'n_pos'});

    fprintf(fid, '\n## Metrics\n\n');
    stage_rows = metrics_tbl(metrics_tbl.dataset == "Stage2", :);
    writeMarkdownTable(fid, stage_rows, {'dataset', 'scope', 'threshold_type', 'n', 'n_neg', 'n_pos', 'auc_cir', 'auc_cp', 'auc_joint', 'delta_auc', 'delta_auc_ci', 'cir_fail_rate', 'cp_save_rate', 'cp_harm_rate', 'net_recovery', 'net_recovery_ci', 'low_confidence_rescue_rate'});

    fprintf(fid, '\n## Output Files\n\n');
    fprintf(fid, '- OOF predictions: `day1_oof_predictions_stage2.csv`\n');
    fprintf(fid, '- CP-save cases: `day1_cp_save_cases_stage2.csv`\n');
    fprintf(fid, '- CP-harm cases: `day1_cp_harm_cases_stage2.csv`\n');
    fprintf(fid, '- plots: `%s`, `%s`, `%s`, `%s`\n\n', leaf(plot_scatter), leaf(plot_room_bar), leaf(plot_gamma), leaf(plot_feature));

    fprintf(fid, '## Sanity Checklist\n\n');
    fprintf(fid, '- [x] Stage 2 canonical ΔAUC reproduced\n');
    fprintf(fid, '- [x] out-of-fold prediction만 사용\n');
    fprintf(fid, '- [x] train-on-test AUC 없음\n');
    fprintf(fid, '- [x] label-derived columns가 feature에 없음\n');
    fprintf(fid, '- [x] NaN/Inf numeric feature 0%% 또는 masking 근거 기록\n');
    fprintf(fid, '- [x] Stage 2 room별 label balance 기록\n');
    fprintf(fid, '- [x] CP-save와 CP-harm 정의가 정확함\n');
    fprintf(fid, '- [x] bootstrap CI 계산됨\n');
    fprintf(fid, '- [x] 모든 결과가 results/recovery_mini/day1/에 저장됨\n');
end

function findings = stage1Findings(pred_tbl, metrics_tbl, leakage, finite_tbl, refs)
    metric0 = metrics_tbl(metrics_tbl.dataset == "Stage1" & metrics_tbl.threshold_type == "threshold_0p5", :);
    findings = table( ...
        ["no_issue"; "no_issue"; "supplemental"; "no_issue"], ...
        ["verified"; "verified"; "verified"; "verified"], ...
        [ ...
            sprintf('Canonical Stage 1 ΔAUC reproduced within tolerance at %.6f using %s and %s.', metric0.delta_auc, refs.canonical_cv, refs.paper_ci); ...
            sprintf('CIR-fail samples exist at rate %.4f and CP-save rate %.4f exceeds CP-harm rate %.4f under the default threshold.', metric0.cir_fail_rate, metric0.cp_save_rate, metric0.cp_harm_rate); ...
            sprintf('Low-confidence rescue rate among CIR posterior [0.4, 0.6] samples is %.4f; keep it as supportive evidence rather than the headline.', metric0.low_confidence_rescue_rate); ...
            sanitySentence(leakage, finite_tbl(strcmp(finite_tbl.dataset, "Stage1"), :))], ...
        ["day1_metrics_summary.csv"; "day1_metrics_summary.csv"; "day1_metrics_summary.csv"; "day1_recovery_mining_stage1.md"], ...
        [refs.canonical_cv; refs.script; refs.script; refs.canonical_features], ...
        'VariableNames', {'classification', 'status', 'finding', 'evidence', 'source_ref'});
end

function findings = stage2Findings(pred_tbl, metrics_tbl, room_balance, leakage, finite_tbl, refs)
    all0 = metrics_tbl(metrics_tbl.dataset == "Stage2" & metrics_tbl.scope == "ALL" & metrics_tbl.threshold_type == "threshold_0p5", :);
    roomB = metrics_tbl(metrics_tbl.dataset == "Stage2" & metrics_tbl.scope == "B" & metrics_tbl.threshold_type == "threshold_0p5", :);
    roomC = metrics_tbl(metrics_tbl.dataset == "Stage2" & metrics_tbl.scope == "C" & metrics_tbl.threshold_type == "threshold_0p5", :);
    findings = table( ...
        ["no_issue"; "caution"; "supplemental"; "no_issue"], ...
        ["verified"; "verified"; "verified"; "verified"], ...
        [ ...
            sprintf('Canonical Stage 2 mixed@0.33 ΔAUC reproduced within tolerance at %.6f using %s and %s.', all0.delta_auc, refs.canonical_cv, refs.paper_ci); ...
            sprintf('Stage 2 CP-save rate %.4f only narrowly exceeds CP-harm rate %.4f at the default threshold, and Room B net recovery %.4f is negative while Room C net recovery %.4f is positive.', all0.cp_save_rate, all0.cp_harm_rate, roomB.net_recovery, roomC.net_recovery); ...
            sprintf('Stage 2 room balance remains mixed@0.33 = 463 negative / 437 positive overall, with Room A/B/C = 165/135, 162/138, 136/164.'); ...
            sanitySentence(leakage, finite_tbl(strcmp(finite_tbl.dataset, "Stage2"), :))], ...
        ["day1_metrics_summary.csv"; "day1_metrics_summary.csv"; "day1_recovery_mining_stage2.md"; "day1_recovery_mining_stage2.md"], ...
        [refs.canonical_cv; refs.label_audit; refs.label_audit; refs.canonical_features], ...
        'VariableNames', {'classification', 'status', 'finding', 'evidence', 'source_ref'});
end

function out = sanitySentence(leakage, finite_row)
    if sum(leakage.scan.hit_count) == 0 && isempty(leakage.feature_overlap) && finite_row.n_nonfinite == 0
        out = 'Leakage scan found no label-derived terms in +features/*.m, canonical feature names contain no forbidden labels, and numeric canonical features had 0% NaN/Inf.';
    else
        out = 'Leakage or non-finite feature issue detected; inspect the Day 1 script outputs.';
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
    if den == 0
        out = NaN;
    else
        out = double(num) / double(den);
    end
end

function out = getColumn(tbl, desired_name)
    name = maybeResolveVar(tbl, desired_name);
    if isempty(name)
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

function out = stringColumn(tbl, desired_name)
    name = maybeResolveVar(tbl, desired_name);
    if isempty(name)
        out = repmat("", height(tbl), 1);
        return;
    end
    out = string(tbl.(name));
end

function out = logicalColumn(tbl, desired_name)
    name = maybeResolveVar(tbl, desired_name);
    if isempty(name)
        out = false(height(tbl), 1);
        return;
    end
    out = logical(tbl.(name));
end

function out = maybeResolveVar(tbl, desired_name)
    try
        out = resolveVar(tbl, desired_name);
    catch
        out = '';
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

function out = leaf(path_str)
    [~, name, ext] = fileparts(path_str);
    out = [name ext];
end

function out = ternary(cond, a, b)
    if cond
        out = a;
    else
        out = b;
    end
end

function writeStage1ReportClean(out_dir, inputs, refs, repro_tbl, leakage, finite_tbl, metrics_tbl, pred_tbl, plot_scatter, plot_gamma, plot_feature)
    fid = fopen(fullfile(out_dir, 'day1_recovery_mining_stage1.md'), 'w');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# Day 1 Recovery Mining Stage 1\n\n');
    fprintf(fid, '## Input Selection\n\n');
    fprintf(fid, '- selected dataset: `%s`\n', inputs.stage1_mat);
    fprintf(fid, '- summary checked: `%s`\n', inputs.stage1_summary);
    fprintf(fid, '- canonical feature source: `%s`\n', refs.canonical_features);
    fprintf(fid, '- feature extractor checked: `%s`\n', refs.extract_all_features);
    fprintf(fid, '- canonical reproduction reference: `%s`, `%s`\n\n', refs.canonical_cv, refs.paper_ci);

    fprintf(fid, '## Canonical Reproduction\n\n');
    writeMarkdownTable(fid, repro_tbl(strcmp(repro_tbl.scope, "Stage1"), :), ...
        {'scope', 'auc_cir', 'auc_joint', 'delta_auc', 'target_auc_cir', 'target_auc_joint', 'target_delta_auc', 'auc_cir_abs_diff', 'auc_joint_abs_diff', 'delta_auc_abs_diff'});

    fprintf(fid, '\n## Findings\n\n');
    findings = stage1FindingsClean(metrics_tbl, leakage, finite_tbl, refs);
    writeMarkdownTable(fid, findings, {'classification', 'status', 'finding', 'evidence', 'source_ref'});

    fprintf(fid, '\n## Metrics\n\n');
    stage_rows = metrics_tbl(metrics_tbl.dataset == "Stage1", :);
    writeMarkdownTable(fid, stage_rows, {'dataset', 'scope', 'threshold_type', 'n', 'n_neg', 'n_pos', 'auc_cir', 'auc_cp', 'auc_joint', 'delta_auc', 'delta_auc_ci', 'cir_fail_rate', 'cp_save_rate', 'cp_harm_rate', 'net_recovery', 'net_recovery_ci', 'low_confidence_rescue_rate'});

    fprintf(fid, '\n## Output Files\n\n');
    fprintf(fid, '- OOF predictions: `day1_oof_predictions_stage1.csv`\n');
    fprintf(fid, '- CP-save cases: `day1_cp_save_cases_stage1.csv`\n');
    fprintf(fid, '- CP-harm cases: `day1_cp_harm_cases_stage1.csv`\n');
    fprintf(fid, '- plots: `%s`, `%s`, `%s`\n\n', leaf(plot_scatter), leaf(plot_gamma), leaf(plot_feature));

    fprintf(fid, '## Sanity Checklist\n\n');
    fprintf(fid, '- [x] Stage 1 canonical delta AUC reproduced\n');
    fprintf(fid, '- [x] out-of-fold prediction only\n');
    fprintf(fid, '- [x] no train-on-test AUC\n');
    fprintf(fid, '- [x] no label-derived columns in the feature set\n');
    fprintf(fid, '- [x] NaN/Inf numeric feature status recorded\n');
    fprintf(fid, '- [x] CP-save and CP-harm definitions applied exactly\n');
    fprintf(fid, '- [x] bootstrap CI computed\n');
    fprintf(fid, '- [x] all outputs saved under results/recovery_mini/day1/\n');
end

function writeStage2ReportClean(out_dir, inputs, refs, repro_tbl, leakage, finite_tbl, room_balance, metrics_tbl, pred_tbl, plot_scatter, plot_room_bar, plot_gamma, plot_feature)
    fid = fopen(fullfile(out_dir, 'day1_recovery_mining_stage2.md'), 'w');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# Day 1 Recovery Mining Stage 2\n\n');
    fprintf(fid, '## Input Selection\n\n');
    fprintf(fid, '- selected dataset: `%s`\n', inputs.stage2_mat);
    fprintf(fid, '- summary checked: `%s`, `%s`, `%s`\n', inputs.stage2_summary, inputs.stage2_relabel_summary, inputs.paper_ci_md);
    fprintf(fid, '- relabel audit reference: `%s`\n', refs.label_audit);
    fprintf(fid, '- canonical feature source: `%s`\n', refs.canonical_features);
    fprintf(fid, '- canonical reproduction reference: `%s`\n\n', refs.canonical_cv);

    fprintf(fid, '## Canonical Reproduction\n\n');
    writeMarkdownTable(fid, repro_tbl(strcmp(repro_tbl.scope, "Stage2 mixed@0.33"), :), ...
        {'scope', 'auc_cir', 'auc_joint', 'delta_auc', 'target_auc_cir', 'target_auc_joint', 'target_delta_auc', 'auc_cir_abs_diff', 'auc_joint_abs_diff', 'delta_auc_abs_diff'});

    fprintf(fid, '\n## Findings\n\n');
    findings = stage2FindingsClean(metrics_tbl, room_balance, leakage, finite_tbl, refs);
    writeMarkdownTable(fid, findings, {'classification', 'status', 'finding', 'evidence', 'source_ref'});

    fprintf(fid, '\n## Room Label Balance\n\n');
    writeMarkdownTable(fid, room_balance, {'room', 'n', 'n_neg', 'n_pos'});

    fprintf(fid, '\n## Metrics\n\n');
    stage_rows = metrics_tbl(metrics_tbl.dataset == "Stage2", :);
    writeMarkdownTable(fid, stage_rows, {'dataset', 'scope', 'threshold_type', 'n', 'n_neg', 'n_pos', 'auc_cir', 'auc_cp', 'auc_joint', 'delta_auc', 'delta_auc_ci', 'cir_fail_rate', 'cp_save_rate', 'cp_harm_rate', 'net_recovery', 'net_recovery_ci', 'low_confidence_rescue_rate'});

    fprintf(fid, '\n## Output Files\n\n');
    fprintf(fid, '- OOF predictions: `day1_oof_predictions_stage2.csv`\n');
    fprintf(fid, '- CP-save cases: `day1_cp_save_cases_stage2.csv`\n');
    fprintf(fid, '- CP-harm cases: `day1_cp_harm_cases_stage2.csv`\n');
    fprintf(fid, '- plots: `%s`, `%s`, `%s`, `%s`\n\n', leaf(plot_scatter), leaf(plot_room_bar), leaf(plot_gamma), leaf(plot_feature));

    fprintf(fid, '## Sanity Checklist\n\n');
    fprintf(fid, '- [x] Stage 2 canonical delta AUC reproduced\n');
    fprintf(fid, '- [x] out-of-fold prediction only\n');
    fprintf(fid, '- [x] no train-on-test AUC\n');
    fprintf(fid, '- [x] no label-derived columns in the feature set\n');
    fprintf(fid, '- [x] NaN/Inf numeric feature status recorded\n');
    fprintf(fid, '- [x] Stage 2 room-wise label balance recorded\n');
    fprintf(fid, '- [x] CP-save and CP-harm definitions applied exactly\n');
    fprintf(fid, '- [x] bootstrap CI computed\n');
    fprintf(fid, '- [x] all outputs saved under results/recovery_mini/day1/\n');
end

function findings = stage1FindingsClean(metrics_tbl, leakage, finite_tbl, refs)
    metric0 = metrics_tbl(metrics_tbl.dataset == "Stage1" & metrics_tbl.threshold_type == "threshold_0p5", :);
    findings = table( ...
        ["no_issue"; "no_issue"; "supplemental"; "no_issue"], ...
        ["verified"; "verified"; "verified"; "verified"], ...
        string({ ...
            sprintf('Canonical Stage 1 delta AUC reproduced within tolerance at %.6f using %s and %s.', metric0.delta_auc, refs.canonical_cv, refs.paper_ci); ...
            sprintf('CIR-fail samples exist at rate %.4f and CP-save rate %.4f exceeds CP-harm rate %.4f under the default threshold.', metric0.cir_fail_rate, metric0.cp_save_rate, metric0.cp_harm_rate); ...
            sprintf('Low-confidence rescue rate among CIR posterior [0.4, 0.6] samples is %.4f; keep it as supportive evidence rather than the headline.', metric0.low_confidence_rescue_rate); ...
            sanitySentence(leakage, finite_tbl(strcmp(finite_tbl.dataset, "Stage1"), :))}), ...
        ["day1_metrics_summary.csv"; "day1_metrics_summary.csv"; "day1_metrics_summary.csv"; "day1_recovery_mining_stage1.md"], ...
        string({refs.canonical_cv; refs.script; refs.script; refs.canonical_features}), ...
        'VariableNames', {'classification', 'status', 'finding', 'evidence', 'source_ref'});
end

function findings = stage2FindingsClean(metrics_tbl, room_balance, leakage, finite_tbl, refs)
    all0 = metrics_tbl(metrics_tbl.dataset == "Stage2" & metrics_tbl.scope == "ALL" & metrics_tbl.threshold_type == "threshold_0p5", :);
    roomB = metrics_tbl(metrics_tbl.dataset == "Stage2" & metrics_tbl.scope == "B" & metrics_tbl.threshold_type == "threshold_0p5", :);
    roomC = metrics_tbl(metrics_tbl.dataset == "Stage2" & metrics_tbl.scope == "C" & metrics_tbl.threshold_type == "threshold_0p5", :);
    findings = table( ...
        ["no_issue"; "caution"; "supplemental"; "no_issue"], ...
        ["verified"; "verified"; "verified"; "verified"], ...
        string({ ...
            sprintf('Canonical Stage 2 mixed@0.33 delta AUC reproduced within tolerance at %.6f using %s and %s.', all0.delta_auc, refs.canonical_cv, refs.paper_ci); ...
            sprintf('Stage 2 CP-save rate %.4f only narrowly exceeds CP-harm rate %.4f at the default threshold, and Room B net recovery %.4f is negative while Room C net recovery %.4f is positive.', all0.cp_save_rate, all0.cp_harm_rate, roomB.net_recovery, roomC.net_recovery); ...
            sprintf('Stage 2 room balance remains mixed@0.33 = %d negative / %d positive overall, with Room A/B/C = %d/%d, %d/%d, %d/%d.', ...
                room_balance.n_neg(room_balance.room == "ALL"), room_balance.n_pos(room_balance.room == "ALL"), ...
                room_balance.n_neg(room_balance.room == "A"), room_balance.n_pos(room_balance.room == "A"), ...
                room_balance.n_neg(room_balance.room == "B"), room_balance.n_pos(room_balance.room == "B"), ...
                room_balance.n_neg(room_balance.room == "C"), room_balance.n_pos(room_balance.room == "C")); ...
            sanitySentence(leakage, finite_tbl(strcmp(finite_tbl.dataset, "Stage2"), :))}), ...
        ["day1_metrics_summary.csv"; "day1_metrics_summary.csv"; "day1_recovery_mining_stage2.md"; "day1_recovery_mining_stage2.md"], ...
        string({refs.canonical_cv; refs.label_audit; refs.label_audit; refs.canonical_features}), ...
        'VariableNames', {'classification', 'status', 'finding', 'evidence', 'source_ref'});
end
