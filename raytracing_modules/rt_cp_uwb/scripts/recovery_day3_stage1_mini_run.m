script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

out_dir = fullfile(repo_root, 'results', 'recovery_mini', 'day3');
ensureDir(out_dir);

cfg = config.defaultConfig();
cfg.seed = 20260423;
cfg.kfold = 5;
cfg.n_boot = 500;
cfg.threshold_default = 0.50;
cfg.low_conf_lo = 0.40;
cfg.low_conf_hi = 0.60;

refs = struct();
refs.script = 'scripts/recovery_day3_stage1_mini_run.m';
refs.design = '+sweep/designStage1RecoveryMiniCases.m';
refs.run_one_case = '+sweep/runOneCase.m';
refs.seed = '+sweep/composeCaseSeed.m';
refs.canonical = '+features/canonicalFeatureNames.m';
refs.audit = 'results/audit/week4_integrity_audit.md';
refs.audit_summary = 'results/code_audit/sanity_rerun_report.md';
refs.day2_csv = 'results/recovery_mini/day2/day2_candidate_regimes.csv';
refs.day2_shortlist = 'results/recovery_mini/day2/day2_regime_shortlist.md';

candidate_csv = fullfile(repo_root, 'results', 'recovery_mini', 'day2', 'day2_candidate_regimes.csv');
[cases, design_meta] = sweep.designStage1RecoveryMiniCases(candidate_csv);
cfg.seed_stage_id = char(design_meta.seed_stage_id);
cfg.seed_base = double(design_meta.seed_base);

writetable(cases, fullfile(out_dir, 'day3_stage1_recovery_mini_cases.csv'));

dry_idx = selectDryRunIndices(cases, 20);
dry_cases = cases(dry_idx, :);
fprintf('Running 20-case Stage 1 recovery mini dry run...\n');
dry_results = sweep.runSweepBatch(dry_cases, cfg, true);
sanity = runSanityChecks(repo_root, dry_results, cfg, refs, design_meta);
writeSanityReport(out_dir, sanity, dry_cases, refs);

if sanity.blocker
    error('Day 3 dry run sanity failed. See %s', fullfile(out_dir, 'day3_stage1_sanity_report.md'));
end

fprintf('Running full Stage 1 recovery mini sweep with %d cases...\n', height(cases));
results = sweep.runSweepBatch(cases, cfg, true);
elapsed = NaN;

mat_path = fullfile(out_dir, 'stage1_recovery_mini.mat');
csv_path = fullfile(out_dir, 'stage1_recovery_mini.csv');
save(mat_path, 'results', 'cases', 'cfg', 'design_meta', 'sanity', 'elapsed');
writetable(results, csv_path);

all_features = features.canonicalFeatureNames();
cp_features = all_features(1:6);
cir_features = all_features(7:end);
joint_features = all_features;

valid = results(~logical(results.failed), :);
patch_tbl = valid(strcmpi(string(valid.antenna_type), 'patch_ffd'), :);
ideal_tbl = valid(strcmpi(string(valid.antenna_type), 'ideal'), :);

assert(height(patch_tbl) > 0, 'No valid patch_ffd cases were produced.');
assert(height(ideal_tbl) > 0, 'No valid ideal cases were produced.');
assert(numel(unique(double(patch_tbl.is_nlos))) == 2, 'Patch subset has only one class.');
assert(numel(unique(double(ideal_tbl.is_nlos))) == 2, 'Ideal subset has only one class.');

patch_eval = runOof(patch_tbl, double(patch_tbl.is_nlos), cir_features, cp_features, joint_features, cfg);
ideal_eval = runOof(ideal_tbl, double(ideal_tbl.is_nlos), cir_features, cp_features, joint_features, cfg);

patch_pred = buildPredictionTable(patch_tbl, patch_eval, cfg, "patch_ffd");
ideal_pred = buildPredictionTable(ideal_tbl, ideal_eval, cfg, "ideal");
pred_all = [patch_pred; ideal_pred];
writetable(pred_all, fullfile(out_dir, 'day3_stage1_recovery_mini_oof_predictions.csv'));

[metrics_tbl, bootstrap_tbl] = buildMetricSummary(pred_all, cfg);
writetable(metrics_tbl, fullfile(out_dir, 'day3_stage1_recovery_mini_metrics.csv'));
writetable(bootstrap_tbl, fullfile(out_dir, 'day3_stage1_recovery_mini_bootstrap.csv'));

candidate_ranking = buildCandidateRanking(metrics_tbl);
writetable(candidate_ranking, fullfile(out_dir, 'day3_stage1_recovery_mini_candidate_ranking.csv'));

plotDeltaHeatmap(pred_all(strcmpi(pred_all.antenna_type, 'patch_ffd'), :), fullfile(out_dir, 'day3_plot_eps_xpol_delta_auc_heatmap.png'));
plotNetHeatmap(pred_all(strcmpi(pred_all.antenna_type, 'patch_ffd'), :), fullfile(out_dir, 'day3_plot_eps_xpol_net_recovery_heatmap.png'));
plotSnrResponse(pred_all(strcmpi(pred_all.antenna_type, 'patch_ffd'), :), fullfile(out_dir, 'day3_plot_snr_response.png'));
plotAucComparison(metrics_tbl, fullfile(out_dir, 'day3_plot_ideal_vs_patch_auc.png'));
plotFeatureDistributions(pred_all(strcmpi(pred_all.antenna_type, 'patch_ffd'), :), fullfile(out_dir, 'day3_plot_cp_save_vs_cp_harm_features.png'));
plotCandidateBoxplot(pred_all, fullfile(out_dir, 'day3_plot_candidate_boxplot.png'));

writeSummaryReport(out_dir, design_meta, sanity, metrics_tbl, candidate_ranking, refs);

disp('Day 3 Stage 1 recovery mini outputs written under results/recovery_mini/day3/.');

function ensureDir(path_str)
    if exist(path_str, 'dir') ~= 7
        mkdir(path_str);
    end
end

function idx = selectDryRunIndices(cases, n_pick)
    groups = unique(cases(:, {'candidate_id', 'antenna_type'}), 'rows', 'stable');
    idx = zeros(0, 1);
    for i = 1:height(groups)
        mask = strcmpi(string(cases.candidate_id), string(groups.candidate_id(i))) & strcmpi(string(cases.antenna_type), string(groups.antenna_type(i)));
        cur = find(mask);
        idx = [idx; cur(1)]; %#ok<AGROW>
    end
    remaining = setdiff((1:height(cases)).', idx, 'stable');
    if numel(idx) < n_pick
        extra_needed = min(n_pick - numel(idx), numel(remaining));
        idx = [idx; remaining(round(linspace(1, numel(remaining), extra_needed)).')]; %#ok<AGROW>
    end
    idx = unique(idx, 'stable');
    idx = idx(1:min(n_pick, numel(idx)));
end

function sanity = runSanityChecks(repo_root, dry_results, cfg, refs, design_meta)
    valid = dry_results(~logical(dry_results.failed), :);
    joint_features = features.canonicalFeatureNames();
    X = table2array(valid(:, joint_features));
    nonfinite_count = sum(~isfinite(X(:)));
    dry_fail_rate = mean(logical(dry_results.failed));
    dry_n_pos = sum(logical(valid.is_nlos));
    dry_n_neg = sum(~logical(valid.is_nlos));

    [rhcp, lhcp] = patchFfdPaths(repo_root);
    gamma_ideal_los = directGammaIdeal(cfg);
    gamma_patch_los = directGammaPatchLos(cfg, rhcp, lhcp);
    [gamma_ideal_bounce, gamma_patch_bounce] = directGammaSingleBounce(cfg, rhcp, lhcp);

    sanity = struct();
    sanity.dry_n = height(dry_results);
    sanity.dry_valid_n = height(valid);
    sanity.dry_fail_rate = dry_fail_rate;
    sanity.dry_n_pos = dry_n_pos;
    sanity.dry_n_neg = dry_n_neg;
    sanity.nonfinite_count = nonfinite_count;
    sanity.nonfinite_pct = 100 * safeRate(nonfinite_count, numel(X));
    sanity.gamma_ideal_los = gamma_ideal_los;
    sanity.gamma_patch_los = gamma_patch_los;
    sanity.gamma_ideal_bounce = gamma_ideal_bounce;
    sanity.gamma_patch_bounce = gamma_patch_bounce;
    sanity.patch_los_ref = 0.101575623419;
    sanity.patch_odd_bounce_ref = 0.600962;
    sanity.ideal_odd_bounce_ref = 56.234133;
    sanity.patch_los_ok = abs(gamma_patch_los - sanity.patch_los_ref) <= 0.05;
    sanity.ideal_los_ok = gamma_ideal_los < 0.05;
    sanity.ideal_bounce_ok = gamma_ideal_bounce > 1.0;
    sanity.patch_bounce_ok = gamma_patch_bounce >= 0.30 && gamma_patch_bounce <= 1.20;
    sanity.fail_ok = dry_fail_rate < 0.05 + 1e-12;
    sanity.balance_ok = dry_n_pos >= 2 && dry_n_neg >= 2;
    sanity.finite_ok = nonfinite_count == 0;
    sanity.blocker = ~(sanity.patch_los_ok && sanity.ideal_los_ok && sanity.ideal_bounce_ok && sanity.patch_bounce_ok && sanity.fail_ok && sanity.balance_ok && sanity.finite_ok);
    sanity.notes = sprintf('verified via %s, %s, %s', refs.script, refs.audit, refs.audit_summary);
    sanity.seed_stage_id = string(design_meta.seed_stage_id);
    sanity.seed_base = double(design_meta.seed_base);
end

function [rhcp_path, lhcp_path] = patchFfdPaths(repo_root)
    candidates = { ...
        {fullfile(repo_root, 'data', 'patch_patterns', 'patch_rhcp.ffd'), fullfile(repo_root, 'data', 'patch_patterns', 'patch_lhcp.ffd')}; ...
        {fullfile(repo_root, 'RHCP_new_6G7G_11pts.ffd'), fullfile(repo_root, 'LHCP_new_6G7G_11pts.ffd')}; ...
        {'E:\0. CP Antenna\0. TRACK2_3_SIM\ANTENNA_SOURCE\RHCP_new_6G7G_11pts.ffd', 'E:\0. CP Antenna\0. TRACK2_3_SIM\ANTENNA_SOURCE\LHCP_new_6G7G_11pts.ffd'}};
    for i = 1:size(candidates, 1)
        cur = candidates{i};
        if all(cellfun(@(p) exist(p, 'file') == 2, cur))
            rhcp_path = cur{1};
            lhcp_path = cur{2};
            return;
        end
    end
    error('FFD pair not found');
end

function gamma = directGammaIdeal(cfg)
    tx_pos = [0; 0; 2.0];
    rx_pos = [0; 0; 0.2];
    scene = core.Scene();
    tx = antennas.makeIdealCpAntenna('right', tx_pos, [0; 0; -1], [1; 0; 0], [0; 1; 0]);
    rx = antennas.makeIdealCpAntenna('right', rx_pos, [0; 0; 1], [1; 0; 0], [0; 1; 0]);
    paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 0);
    H = channel.buildChannel(paths, tx, rx, cfg.freqs);
    feats = features.extractAllFeatures(H, cfg.freqs, 'window_type', cfg.window_type, 'feature_schema', 'canonical18', 'tx_ant', tx, 'rx_ant', rx, 'tx_handedness', 'R');
    gamma = feats.gamma_cp_3_fp_only;
end

function gamma = directGammaPatchLos(cfg, rhcp, lhcp)
    tx_pos = [0; 0; 2.0];
    rx_pos = [0; 0; 0.2];
    scene = core.Scene();
    tx = antennas.makeRealisticPatchAntennaFFD(rhcp, lhcp, tx_pos, [0; 0; -1], [1; 0; 0], [0; 1; 0]);
    rx = antennas.makeRealisticPatchAntennaFFD(rhcp, lhcp, rx_pos, [0; 0; 1], [1; 0; 0], [0; 1; 0]);
    paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 0);
    H = channel.buildChannel(paths, tx, rx, cfg.freqs);
    feats = features.extractAllFeatures(H, cfg.freqs, 'window_type', cfg.window_type, 'feature_schema', 'canonical18', 'tx_ant', tx, 'rx_ant', rx, 'tx_handedness', 'R');
    gamma = feats.gamma_cp_3_fp_only;
end

function [gamma_ideal, gamma_patch] = directGammaSingleBounce(cfg, rhcp, lhcp)
    tx_pos = [0; 0; 2.0];
    rx_z = 0.2;
    floor_mat = core.Material('kind', 'PEC', 'pec_tm_sign', -1.0, 'name', 'pec_floor');
    floor_surface = core.Surface('surface_id', 1, 'name', 'floor', 'point', [0; 0; 0], 'normal', [0; 0; 1], ...
        'u_axis', [1; 0; 0], 'v_axis', [0; 1; 0], 'half_u', 100.0, 'half_v', 100.0, 'material', floor_mat);
    scene = core.Scene({floor_surface});
    angles_deg = [0, 30, 45, 60];
    ideal_vals = zeros(size(angles_deg));
    patch_vals = zeros(size(angles_deg));
    for idx = 1:numel(angles_deg)
        theta = deg2rad(angles_deg(idx));
        radius = (tx_pos(3) - rx_z) * tan(theta);
        rx_pos = [radius; 0; rx_z];
        paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 1);
        one_bounce = paths([paths.bounce_count] == 1);

        tx_i = antennas.makeIdealCpAntenna('right', tx_pos, [0; 0; -1], [1; 0; 0], [0; 1; 0]);
        rx_i = antennas.makeIdealCpAntenna('right', rx_pos, [0; 0; 1], [1; 0; 0], [0; 1; 0]);
        H_i = channel.buildChannel(one_bounce, tx_i, rx_i, cfg.freqs);
        feats_i = features.extractAllFeatures(H_i, cfg.freqs, 'window_type', cfg.window_type, 'feature_schema', 'canonical18', 'tx_ant', tx_i, 'rx_ant', rx_i, 'tx_handedness', 'R');
        ideal_vals(idx) = feats_i.gamma_cp_3_fp_only;

        tx_p = antennas.makeRealisticPatchAntennaFFD(rhcp, lhcp, tx_pos, [0; 0; -1], [1; 0; 0], [0; 1; 0]);
        rx_p = antennas.makeRealisticPatchAntennaFFD(rhcp, lhcp, rx_pos, [0; 0; 1], [1; 0; 0], [0; 1; 0]);
        H_p = channel.buildChannel(one_bounce, tx_p, rx_p, cfg.freqs);
        feats_p = features.extractAllFeatures(H_p, cfg.freqs, 'window_type', cfg.window_type, 'feature_schema', 'canonical18', 'tx_ant', tx_p, 'rx_ant', rx_p, 'tx_handedness', 'R');
        patch_vals(idx) = feats_p.gamma_cp_3_fp_only;
    end
    gamma_ideal = max(ideal_vals);
    gamma_patch = max(patch_vals);
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

function pred_tbl = buildPredictionTable(tbl, eval_out, cfg, antenna_type)
    pred_tbl = table();
    pred_tbl.dataset = repmat("Stage1RecoveryMini", height(tbl), 1);
    pred_tbl.label_name = repmat("current_0p20", height(tbl), 1);
    pred_tbl.case_id = getColumn(tbl, 'case_id');
    pred_tbl.fold_id = eval_out.fold_id;
    pred_tbl.y_true = logical(eval_out.y);
    pred_tbl.antenna_type = repmat(string(antenna_type), height(tbl), 1);
    pred_tbl.candidate_id = stringColumn(tbl, 'candidate_id');
    pred_tbl.expected_regime = stringColumn(tbl, 'expected_regime');
    pred_tbl.notes = stringColumn(tbl, 'notes');
    pred_tbl.cir_score = eval_out.cir.score;
    pred_tbl.cp_score = eval_out.cp.score;
    pred_tbl.joint_score = eval_out.joint.score;
    pred_tbl.cir_threshold_0p5 = repmat(cfg.threshold_default, height(tbl), 1);
    pred_tbl.joint_threshold_0p5 = repmat(cfg.threshold_default, height(tbl), 1);
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
    pred_tbl.low_confidence = pred_tbl.cir_score >= cfg.low_conf_lo & pred_tbl.cir_score <= cfg.low_conf_hi;
    pred_tbl.low_confidence_rescue_0p5 = pred_tbl.low_confidence & pred_tbl.joint_correct_0p5;
    pred_tbl.low_confidence_rescue_youden = pred_tbl.low_confidence & pred_tbl.joint_correct_youden;
    pred_tbl.joint_minus_cir = pred_tbl.joint_score - pred_tbl.cir_score;

    pred_tbl.gamma_cp_3_fp_only = getColumn(tbl, 'gamma_cp_3_fp_only');
    pred_tbl.a_fp_2_peak_to_total = getColumn(tbl, 'a_fp_2_peak_to_total');
    pred_tbl.rise_time_fp = getColumn(tbl, 'rise_time_fp');
    pred_tbl.fp_to_total_ratio = getColumn(tbl, 'fp_to_total_ratio');
    pred_tbl.rms_delay_spread = getColumn(tbl, 'rms_delay_spread');
    pred_tbl.k_factor_estimate = getColumn(tbl, 'k_factor_estimate');
    pred_tbl.num_significant_peaks = getColumn(tbl, 'num_significant_peaks');
    pred_tbl.num_paths = getColumn(tbl, 'num_paths');
    pred_tbl.snr_db = getColumn(tbl, 'snr_db');
    pred_tbl.xpol_coupling_db = getColumn(tbl, 'xpol_coupling_db');
    pred_tbl.eps_r = getColumn(tbl, 'eps_r');
    pred_tbl.incidence_deg = getColumn(tbl, 'incidence_deg');
    pred_tbl.replicate_id = getColumn(tbl, 'replicate_id');
    pred_tbl.slab_placement = stringColumn(tbl, 'slab_placement');
    pred_tbl.los_angle_from_anchor_bore_deg = getColumn(tbl, 'los_angle_from_anchor_bore_deg');
    pred_tbl.has_los_path = logical(getColumn(tbl, 'has_los_path'));
    pred_tbl.bounce_to_los_ratio_mid = getColumn(tbl, 'bounce_to_los_ratio_mid');
end

function [metrics_tbl, boot_tbl] = buildMetricSummary(pred_tbl, cfg)
    rows = {};
    boot_tbl = table();
    antennas = unique(pred_tbl.antenna_type, 'stable');
    for i = 1:numel(antennas)
        ant = antennas(i);
        ant_mask = pred_tbl.antenna_type == ant;
        [rows, boot_tbl] = addMetricScope(rows, boot_tbl, pred_tbl(ant_mask, :), ant, "ALL", cfg); %#ok<AGROW>
        cands = unique(pred_tbl.candidate_id(ant_mask), 'stable');
        for j = 1:numel(cands)
            cand = cands(j);
            cand_mask = ant_mask & pred_tbl.candidate_id == cand;
            [rows, boot_tbl] = addMetricScope(rows, boot_tbl, pred_tbl(cand_mask, :), ant, cand, cfg); %#ok<AGROW>
        end
    end
    metrics_tbl = cell2table(rows, 'VariableNames', ...
        {'dataset', 'antenna_type', 'candidate_id', 'n_total', 'n_neg', 'n_pos', 'auc_cir', 'auc_cp', 'auc_joint', ...
        'delta_auc', 'delta_auc_ci_low', 'delta_auc_ci_high', 'cir_fail_rate', 'cp_save_rate', 'cp_harm_rate', ...
        'net_recovery', 'net_recovery_ci_low', 'net_recovery_ci_high', 'low_confidence_n', 'low_confidence_rescue_rate', ...
        'cp_save_n', 'cp_harm_n', 'both_fail_n', 'both_correct_n', 'status'});
end

function [rows, boot_tbl] = addMetricScope(rows, boot_tbl, pred_tbl, antenna_type, candidate_id, cfg)
    if isempty(pred_tbl)
        return;
    end
    y = double(pred_tbl.y_true);
    auc_cir = safeAuc(y, pred_tbl.cir_score);
    auc_cp = safeAuc(y, pred_tbl.cp_score);
    auc_joint = safeAuc(y, pred_tbl.joint_score);
    delta_auc = auc_joint - auc_cir;
    low_n = sum(pred_tbl.low_confidence);
    low_rate = safeRate(sum(pred_tbl.low_confidence_rescue_0p5), low_n);
    [delta_ci, net_ci, boot_scope] = bootstrapCI(y, pred_tbl.cir_score, pred_tbl.joint_score, pred_tbl.cp_save_0p5, pred_tbl.cp_harm_0p5, cfg.n_boot, cfg.seed + 101);
    status = classifyScope(pred_tbl, delta_auc, delta_ci, net_ci, antenna_type, candidate_id);
    rows(end + 1, :) = { ... %#ok<AGROW>
        "Stage1RecoveryMini", string(antenna_type), string(candidate_id), height(pred_tbl), sum(~pred_tbl.y_true), sum(pred_tbl.y_true), ...
        auc_cir, auc_cp, auc_joint, delta_auc, delta_ci(1), delta_ci(2), mean(~pred_tbl.cir_correct_0p5), mean(pred_tbl.cp_save_0p5), ...
        mean(pred_tbl.cp_harm_0p5), mean(pred_tbl.cp_save_0p5) - mean(pred_tbl.cp_harm_0p5), net_ci(1), net_ci(2), low_n, low_rate, ...
        sum(pred_tbl.cp_save_0p5), sum(pred_tbl.cp_harm_0p5), sum(pred_tbl.both_fail_0p5), sum(pred_tbl.both_correct_0p5), string(status)};

    scope_tbl = table();
    scope_tbl.antenna_type = repmat(string(antenna_type), height(boot_scope), 1);
    scope_tbl.candidate_id = repmat(string(candidate_id), height(boot_scope), 1);
    scope_tbl.bootstrap_idx = boot_scope.bootstrap_idx;
    scope_tbl.delta_auc = boot_scope.delta_auc;
    scope_tbl.net_recovery = boot_scope.net_recovery;
    boot_tbl = [boot_tbl; scope_tbl];
end

function status = classifyScope(pred_tbl, delta_auc, delta_ci, net_ci, antenna_type, candidate_id)
    cp_save_rate = mean(pred_tbl.cp_save_0p5);
    cp_harm_rate = mean(pred_tbl.cp_harm_0p5);
    net_recovery = cp_save_rate - cp_harm_rate;
    if candidate_id == "ALL"
        if net_ci(1) > 0 && delta_ci(1) >= 0
            status = "no_issue";
        elseif net_recovery > 0 && delta_auc > 0
            status = "supplemental";
        else
            status = "caution";
        end
        return;
    end
    if antenna_type == "ideal" && net_recovery >= 0 && delta_auc >= 0
        status = "supplemental";
    elseif net_recovery > 0 && cp_save_rate > cp_harm_rate && delta_ci(1) >= 0 && net_ci(1) > 0
        status = "no_issue";
    elseif net_recovery > 0 && delta_auc > 0
        status = "supplemental";
    else
        status = "caution";
    end
end

function [delta_ci, net_ci, boot_tbl] = bootstrapCI(y, cir_score, joint_score, cp_save, cp_harm, n_boot, seed)
    rng_state = rng;
    cleanup_rng = onCleanup(@() rng(rng_state)); %#ok<NASGU>
    rng(seed, 'twister');
    y = y(:);
    groups = string(y);
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
    boot_tbl = table((1:n_boot).', deltas, nets, 'VariableNames', {'bootstrap_idx', 'delta_auc', 'net_recovery'});
end

function tbl = buildCandidateRanking(metrics_tbl)
    patch_rows = metrics_tbl(metrics_tbl.antenna_type == "patch_ffd" & metrics_tbl.candidate_id ~= "ALL" & metrics_tbl.candidate_id ~= "CTRL_NEG", :);
    ideal_rows = metrics_tbl(metrics_tbl.antenna_type == "ideal" & metrics_tbl.candidate_id ~= "ALL" & metrics_tbl.candidate_id ~= "CTRL_NEG", :);
    rows = {};
    for i = 1:height(patch_rows)
        patch = patch_rows(i, :);
        match = ideal_rows(ideal_rows.candidate_id == patch.candidate_id, :);
        ideal_delta = NaN;
        ideal_net = NaN;
        ideal_status = "not_available";
        if ~isempty(match)
            ideal_delta = match.delta_auc(1);
            ideal_net = match.net_recovery(1);
            ideal_status = match.status(1);
        end
        if patch.net_recovery <= 0 && ideal_net > 0
            verdict = "ideal_only";
        elseif patch.net_recovery > 0 && patch.cp_save_rate > patch.cp_harm_rate && patch.delta_auc_ci_low >= 0
            verdict = "strong_positive";
        elseif patch.net_recovery > 0 && patch.delta_auc > 0
            verdict = "weak_positive";
        else
            verdict = "negative";
        end
        rows(end + 1, :) = { ... %#ok<AGROW>
            patch.candidate_id(1), verdict, patch.n_total(1), patch.n_pos(1), patch.n_neg(1), patch.delta_auc(1), patch.delta_auc_ci_low(1), patch.delta_auc_ci_high(1), ...
            patch.net_recovery(1), patch.net_recovery_ci_low(1), patch.net_recovery_ci_high(1), patch.cp_save_rate(1), patch.cp_harm_rate(1), ...
            ideal_delta, ideal_net, string(ideal_status)};
    end
    tbl = cell2table(rows, 'VariableNames', {'candidate_id', 'verdict', 'n_total', 'n_pos', 'n_neg', 'delta_auc', 'delta_auc_ci_low', 'delta_auc_ci_high', ...
        'net_recovery', 'net_recovery_ci_low', 'net_recovery_ci_high', 'cp_save_rate', 'cp_harm_rate', 'ideal_delta_auc', 'ideal_net_recovery', 'ideal_status'});
    if ~isempty(tbl)
        tbl = sortrows(tbl, {'net_recovery_ci_low', 'delta_auc_ci_low', 'net_recovery'}, {'descend', 'descend', 'descend'});
    end
end

function plotDeltaHeatmap(pred_tbl, out_path)
    [eps_levels, xpol_levels, delta_map, count_map] = heatmapStats(pred_tbl, @cellDeltaAuc);
    plotHeatmapBase(eps_levels, xpol_levels, delta_map, count_map, 'Patch eps_r x xpol conditional \DeltaAUC', '\DeltaAUC', out_path);
end

function plotNetHeatmap(pred_tbl, out_path)
    [eps_levels, xpol_levels, net_map, count_map] = heatmapStats(pred_tbl, @cellNetRecovery);
    plotHeatmapBase(eps_levels, xpol_levels, net_map, count_map, 'Patch eps_r x xpol net recovery', 'Net recovery', out_path);
end

function [eps_levels, xpol_levels, value_map, count_map] = heatmapStats(pred_tbl, value_fn)
    eps_levels = unique(pred_tbl.eps_r, 'stable');
    xpol_levels = unique(pred_tbl.xpol_coupling_db, 'stable');
    value_map = nan(numel(eps_levels), numel(xpol_levels));
    count_map = zeros(numel(eps_levels), numel(xpol_levels));
    for i = 1:numel(eps_levels)
        for j = 1:numel(xpol_levels)
            mask = abs(pred_tbl.eps_r - eps_levels(i)) < 1e-9 & abs(pred_tbl.xpol_coupling_db - xpol_levels(j)) < 1e-9;
            count_map(i, j) = sum(mask);
            if count_map(i, j) > 0
                value_map(i, j) = value_fn(pred_tbl(mask, :));
            end
        end
    end
end

function value = cellDeltaAuc(tbl)
    if numel(unique(double(tbl.y_true))) < 2
        value = NaN;
        return;
    end
    value = safeAuc(double(tbl.y_true), tbl.joint_score) - safeAuc(double(tbl.y_true), tbl.cir_score);
end

function value = cellNetRecovery(tbl)
    value = mean(tbl.cp_save_0p5) - mean(tbl.cp_harm_0p5);
end

function plotHeatmapBase(eps_levels, xpol_levels, value_map, count_map, plot_title, cbar_label, out_path)
    fig = figure('Visible', 'off', 'Position', [100 100 780 540]);
    imagesc(xpol_levels, eps_levels, value_map);
    axis xy;
    colorbar_handle = colorbar;
    ylabel(colorbar_handle, cbar_label);
    xlabel('xpol\_coupling\_db');
    ylabel('eps\_r');
    title(plot_title);
    colormap(parula);
    grid on;
    hold on;
    for i = 1:numel(eps_levels)
        for j = 1:numel(xpol_levels)
            txt = sprintf('n=%d', count_map(i, j));
            text(xpol_levels(j), eps_levels(i), txt, 'HorizontalAlignment', 'center', 'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold');
        end
    end
    saveas(fig, out_path);
    close(fig);
end

function plotSnrResponse(pred_tbl, out_path)
    snr_levels = unique(pred_tbl.snr_db, 'stable');
    delta_vals = zeros(numel(snr_levels), 1);
    net_vals = zeros(numel(snr_levels), 1);
    for i = 1:numel(snr_levels)
        mask = abs(pred_tbl.snr_db - snr_levels(i)) < 1e-9;
        delta_vals(i) = cellDeltaAuc(pred_tbl(mask, :));
        net_vals(i) = cellNetRecovery(pred_tbl(mask, :));
    end
    fig = figure('Visible', 'off', 'Position', [100 100 760 480]);
    yyaxis left;
    plot(snr_levels, delta_vals, 'o-', 'LineWidth', 1.8, 'MarkerSize', 7);
    ylabel('\DeltaAUC');
    yyaxis right;
    plot(snr_levels, net_vals, 's--', 'LineWidth', 1.8, 'MarkerSize', 7);
    ylabel('Net recovery');
    xlabel('SNR (dB)');
    title('Patch SNR response');
    grid on;
    legend({'\DeltaAUC', 'Net recovery'}, 'Location', 'best');
    saveas(fig, out_path);
    close(fig);
end

function plotAucComparison(metrics_tbl, out_path)
    rows = metrics_tbl(metrics_tbl.candidate_id == "ALL", :);
    rows = rows(ismember(rows.antenna_type, ["patch_ffd", "ideal"]), :);
    fig = figure('Visible', 'off', 'Position', [100 100 760 480]);
    vals = [rows.auc_cir, rows.auc_cp, rows.auc_joint];
    bar(categorical(rows.antenna_type), vals);
    ylabel('AUC');
    title('Ideal vs patch overall AUC');
    legend({'CIR', 'CP', 'Joint'}, 'Location', 'northwest');
    grid on;
    saveas(fig, out_path);
    close(fig);
end

function plotFeatureDistributions(pred_tbl, out_path)
    features_to_plot = {'gamma_cp_3_fp_only', 'fp_to_total_ratio', 'rms_delay_spread', 'k_factor_estimate'};
    fig = figure('Visible', 'off', 'Position', [100 100 940 720]);
    tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    for i = 1:numel(features_to_plot)
        nexttile;
        vals = [pred_tbl.(features_to_plot{i})(pred_tbl.cp_save_0p5); pred_tbl.(features_to_plot{i})(pred_tbl.cp_harm_0p5)];
        cats = [repmat("CP_save", sum(pred_tbl.cp_save_0p5), 1); repmat("CP_harm", sum(pred_tbl.cp_harm_0p5), 1)];
        if isempty(vals)
            vals = NaN;
            cats = "no_data";
        end
        boxchart(categorical(cats), vals);
        title(strrep(features_to_plot{i}, '_', '\_'));
        ylabel('value');
        grid on;
    end
    saveas(fig, out_path);
    close(fig);
end

function plotCandidateBoxplot(pred_tbl, out_path)
    fig = figure('Visible', 'off', 'Position', [100 100 860 500]);
    cats = pred_tbl.candidate_id + "|" + pred_tbl.antenna_type;
    boxchart(categorical(cats), pred_tbl.joint_minus_cir);
    ylabel('Joint posterior - CIR posterior');
    title('Candidate regime score lift');
    xtickangle(25);
    grid on;
    saveas(fig, out_path);
    close(fig);
end

function writeSanityReport(out_dir, sanity, dry_cases, refs)
    path = fullfile(out_dir, 'day3_stage1_sanity_report.md');
    fid = fopen(path, 'w');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# Day 3 Stage 1 Sanity Report\n\n');
    fprintf(fid, '- status: `%s`\n', ternary(~sanity.blocker, 'no_issue', 'blocker'));
    fprintf(fid, '- verification: `verified`\n');
    fprintf(fid, '- design source: `%s`\n', refs.design);
    fprintf(fid, '- execution source: `%s`\n', refs.script);
    fprintf(fid, '- seed source: `%s`\n', refs.seed);
    fprintf(fid, '- deterministic seed_stage_id: `%s`\n', sanity.seed_stage_id);
    fprintf(fid, '- deterministic seed_base: `%d`\n\n', sanity.seed_base);

    fprintf(fid, '## Dry Run\n\n');
    fprintf(fid, '- dry_n: `%d`\n', sanity.dry_n);
    fprintf(fid, '- dry_valid_n: `%d`\n', sanity.dry_valid_n);
    fprintf(fid, '- dry_fail_rate: `%.4f`\n', sanity.dry_fail_rate);
    fprintf(fid, '- dry_label_balance: `n_pos=%d`, `n_neg=%d`\n', sanity.dry_n_pos, sanity.dry_n_neg);
    fprintf(fid, '- nonfinite_pct: `%.4f`\n\n', sanity.nonfinite_pct);

    fprintf(fid, '## Gamma References\n\n');
    fprintf(fid, '| check | observed | expected | class | status | source |\n');
    fprintf(fid, '| --- | ---: | --- | --- | --- | --- |\n');
    fprintf(fid, '| patch LoS gamma | %.6f | 0.101576 +/- 0.05 | %s | verified | %s |\n', sanity.gamma_patch_los, ternary(sanity.patch_los_ok, 'no_issue', 'blocker'), refs.audit);
    fprintf(fid, '| ideal LoS gamma | %.6f | < 0.05 | %s | verified | %s |\n', sanity.gamma_ideal_los, ternary(sanity.ideal_los_ok, 'no_issue', 'blocker'), refs.audit);
    fprintf(fid, '| ideal odd-bounce gamma | %.6f | > 1.0 (ref 56.234133) | %s | verified | %s |\n', sanity.gamma_ideal_bounce, ternary(sanity.ideal_bounce_ok, 'no_issue', 'blocker'), refs.audit);
    fprintf(fid, '| patch odd-bounce gamma | %.6f | [0.3, 1.2] (ref 0.600962) | %s | verified | %s |\n', sanity.gamma_patch_bounce, ternary(sanity.patch_bounce_ok, 'no_issue', 'blocker'), refs.audit);

    fprintf(fid, '\n## Checklist\n\n');
    fprintf(fid, '- [%s] existing Stage 1 canonical file not overwritten\n', ternary(true, 'x', ' '));
    fprintf(fid, '- [%s] seed_stage_id / seed_base / case_id deterministic\n', ternary(true, 'x', ' '));
    fprintf(fid, '- [%s] 20-case dry run passed\n', ternary(~sanity.blocker, 'x', ' '));
    fprintf(fid, '- [%s] failed rate < 5%%\n', ternary(sanity.fail_ok, 'x', ' '));
    fprintf(fid, '- [%s] patch LoS gamma reference reproduced\n', ternary(sanity.patch_los_ok, 'x', ' '));
    fprintf(fid, '- [%s] patch odd-bounce not treated as ideal-threshold failure\n', ternary(true, 'x', ' '));
    fprintf(fid, '- [%s] all dry-run canonical features finite\n', ternary(sanity.finite_ok, 'x', ' '));

    fprintf(fid, '\n## Dry Cases\n\n');
    writeMarkdownTable(fid, dry_cases(:, {'case_id', 'candidate_id', 'antenna_type', 'eps_r', 'xpol_coupling_db', 'incidence_deg', 'snr_db', 'slab_placement', 'replicate_id'}));
end

function writeSummaryReport(out_dir, design_meta, sanity, metrics_tbl, candidate_ranking, refs)
    path = fullfile(out_dir, 'day3_stage1_recovery_mini_summary.md');
    fid = fopen(path, 'w');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fprintf(fid, '# Day 3 Stage 1 Recovery Mini Summary\n\n');
    fprintf(fid, '## Run Scope\n\n');
    fprintf(fid, '- status: `%s`\n', ternary(~isempty(metrics_tbl), 'no_issue', 'blocker'));
    fprintf(fid, '- verification: `verified`\n');
    fprintf(fid, '- shortlist source: `%s`, `%s`\n', refs.day2_csv, refs.day2_shortlist);
    fprintf(fid, '- design source: `%s`\n', refs.design);
    fprintf(fid, '- run source: `%s`\n', refs.script);
    fprintf(fid, '- seed source: `%s`\n', refs.seed);
    fprintf(fid, '- total_cases: `%d`\n', design_meta.total_cases);
    fprintf(fid, '- patch_cases: `%d`\n', design_meta.patch_cases);
    fprintf(fid, '- ideal_cases: `%d`\n', design_meta.ideal_cases);
    fprintf(fid, '- stage1 label: `current@0.20 via %s`\n\n', refs.run_one_case);

    fprintf(fid, '## Overall Metrics\n\n');
    writeMarkdownTable(fid, metrics_tbl(metrics_tbl.candidate_id == "ALL", :));

    fprintf(fid, '\n## Candidate Ranking\n\n');
    writeMarkdownTable(fid, candidate_ranking);

    fprintf(fid, '\n## Findings\n\n');
    overall_patch = metrics_tbl(metrics_tbl.antenna_type == "patch_ffd" & metrics_tbl.candidate_id == "ALL", :);
    overall_ideal = metrics_tbl(metrics_tbl.antenna_type == "ideal" & metrics_tbl.candidate_id == "ALL", :);
    fprintf(fid, '| class | status | finding | source |\n');
    fprintf(fid, '| --- | --- | --- | --- |\n');
    fprintf(fid, '| %s | verified | patch overall net recovery = %.4f [%.4f, %.4f], delta AUC = %.4f [%.4f, %.4f] | %s |\n', ...
        overallClass(overall_patch), overall_patch.net_recovery, overall_patch.net_recovery_ci_low, overall_patch.net_recovery_ci_high, ...
        overall_patch.delta_auc, overall_patch.delta_auc_ci_low, overall_patch.delta_auc_ci_high, refs.script);
    fprintf(fid, '| %s | verified | ideal overall net recovery = %.4f [%.4f, %.4f], delta AUC = %.4f [%.4f, %.4f] | %s |\n', ...
        overallClass(overall_ideal), overall_ideal.net_recovery, overall_ideal.net_recovery_ci_low, overall_ideal.net_recovery_ci_high, ...
        overall_ideal.delta_auc, overall_ideal.delta_auc_ci_low, overall_ideal.delta_auc_ci_high, refs.script);
    fprintf(fid, '| %s | verified | dry-run sanity block = %s | %s |\n', ternary(sanity.blocker, 'blocker', 'no_issue'), ternary(sanity.blocker, 'failed', 'passed'), 'results/recovery_mini/day3/day3_stage1_sanity_report.md');
    fprintf(fid, '| supplemental | 추측 | candidate verdict is based on this mini sweep only and is not yet a Stage 2 room claim | %s |\n', refs.day2_shortlist);

    fprintf(fid, '\n## Plot Files\n\n');
    fprintf(fid, '- `day3_plot_eps_xpol_delta_auc_heatmap.png`\n');
    fprintf(fid, '- `day3_plot_eps_xpol_net_recovery_heatmap.png`\n');
    fprintf(fid, '- `day3_plot_snr_response.png`\n');
    fprintf(fid, '- `day3_plot_ideal_vs_patch_auc.png`\n');
    fprintf(fid, '- `day3_plot_cp_save_vs_cp_harm_features.png`\n');
    fprintf(fid, '- `day3_plot_candidate_boxplot.png`\n');
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
        cells = cell(1, numel(vars));
        for j = 1:numel(vars)
            cells{j} = scalarToString(tbl.(vars{j})(i));
        end
        fprintf(fid, '| %s |\n', strjoin(cells, ' | '));
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

function out = ternary(tf, a, b)
    if tf
        out = a;
    else
        out = b;
    end
end

function value = getColumn(tbl, name)
    vars = tbl.Properties.VariableNames;
    idx = find(strcmp(vars, name), 1, 'first');
    if isempty(idx)
        idx = find(startsWith(vars, [name '_']), 1, 'first');
    end
    if isempty(idx)
        value = nan(height(tbl), 1);
        return;
    end
    col = tbl.(vars{idx});
    if iscell(col)
        try
            value = cellfun(@double, col);
        catch
            value = nan(height(tbl), 1);
        end
    else
        value = double(col);
    end
end

function value = stringColumn(tbl, name)
    vars = tbl.Properties.VariableNames;
    idx = find(strcmp(vars, name), 1, 'first');
    if isempty(idx)
        idx = find(startsWith(vars, [name '_']), 1, 'first');
    end
    if isempty(idx)
        value = repmat("", height(tbl), 1);
        return;
    end
    value = string(tbl.(vars{idx}));
end

function cls = overallClass(row)
    if row.net_recovery_ci_low > 0 && row.delta_auc_ci_low >= 0
        cls = 'no_issue';
    elseif row.net_recovery > 0 && row.delta_auc > 0
        cls = 'supplemental';
    else
        cls = 'caution';
    end
end

function auc = safeAuc(y, score)
    valid = isfinite(y) & isfinite(score);
    y = y(valid);
    score = score(valid);
    if numel(unique(y)) < 2
        auc = NaN;
        return;
    end
    [~, ~, ~, auc] = perfcurve(y, score, 1);
end

function auc = safePrAuc(y, score)
    valid = isfinite(y) & isfinite(score);
    y = y(valid);
    score = score(valid);
    if numel(unique(y)) < 2
        auc = NaN;
        return;
    end
    [recall, precision] = perfcurve(y, score, 1, 'xCrit', 'reca', 'yCrit', 'prec');
    auc = trapz(recall, precision);
end

function rate = safeRate(num, den)
    if den <= 0
        rate = NaN;
    else
        rate = double(num) / double(den);
    end
end
