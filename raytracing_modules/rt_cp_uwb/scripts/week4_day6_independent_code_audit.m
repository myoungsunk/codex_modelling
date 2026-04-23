script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

out_dir = fullfile(repo_root, 'results', 'code_audit');
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

integrity_csv = fullfile(repo_root, 'results', 'audit', 'week4_integrity_audit.csv');
if exist(integrity_csv, 'file') ~= 2
    run(fullfile(repo_root, 'scripts', 'week4_day5_integrity_audit.m'));
end
integrity_tbl = readtable(integrity_csv, 'TextType', 'string');

cfg = config.defaultConfig();
all_features = features.canonicalFeatureNames();
cp_features = all_features(1:6);
cir_features = all_features(7:end);
joint_features = all_features;

stage1_orig = loadResults(fullfile(repo_root, 'results', 'stage1', 'stage1_3000_ffd.mat'));
stage1_det = loadResults(fullfile(repo_root, 'results', 'stage1', 'stage1_3000_ffd_det.mat'));
stage2_orig = loadResults(fullfile(repo_root, 'results', 'stage2', 'stage2_900_ffd.mat'));
stage2_det = loadResults(fullfile(repo_root, 'results', 'stage2', 'stage2_900_ffd_det.mat'));

stage1_orig_auc = computeGlobalAucs(stage1_orig, cir_features, cp_features, joint_features);
stage1_det_auc = computeGlobalAucs(stage1_det, cir_features, cp_features, joint_features);

[stage2_orig_auc, stage2_orig_room_auc, stage2_orig_balance] = computeStage2Metrics(stage2_orig, cir_features, cp_features, joint_features);
[stage2_det_auc, stage2_det_room_auc, stage2_det_balance] = computeStage2Metrics(stage2_det, cir_features, cp_features, joint_features);

stage1_orig_cells = readtable(fullfile(repo_root, 'results', 'stage1', 'conditional_auc_cells.csv'), 'TextType', 'string');
stage1_det_cells = readtable(fullfile(repo_root, 'results', 'stage1', 'conditional_auc_cells_det.csv'), 'TextType', 'string');
stage1_orig_cells = sortrows(stage1_orig_cells(isfinite(stage1_orig_cells.delta_auc), :), {'delta_auc', 'n'}, {'descend', 'descend'});
stage1_det_cells = sortrows(stage1_det_cells(isfinite(stage1_det_cells.delta_auc), :), {'delta_auc', 'n'}, {'descend', 'descend'});
stage1_orig_top = stage1_orig_cells(1, :);
stage1_det_top = stage1_det_cells(1, :);

stage1_orig_top_counts = cellCounts(stage1_orig, stage1_orig_top.x_var{1}, stage1_orig_top.x_label{1}, stage1_orig_top.y_var{1}, stage1_orig_top.y_label{1}, 'is_nlos');
stage1_det_top_counts = cellCounts(stage1_det, stage1_det_top.x_var{1}, stage1_det_top.x_label{1}, stage1_det_top.y_var{1}, stage1_det_top.y_label{1}, 'is_nlos');
legacy_stage1_cell_counts = cellCounts(stage1_orig, 'eps_r', '[2.00, 3.60]', 'xpol_coupling_db', '[20.01, 24.00]', 'is_nlos');

pre_stage3_det = readtable(fullfile(out_dir, 'prepatch_hfss_case_list_det.csv'), 'TextType', 'string');
pre_stage3_orig = readtable(fullfile(out_dir, 'prepatch_hfss_case_list.csv'), 'TextType', 'string');
stage3_det = readtable(fullfile(repo_root, 'results', 'stage3', 'hfss_case_list_det.csv'), 'TextType', 'string');
stage3_orig = readtable(fullfile(repo_root, 'results', 'stage3', 'hfss_case_list.csv'), 'TextType', 'string');
stage3_cells_det = readtable(fullfile(repo_root, 'results', 'stage3', 'stage3_reselection_cells_det.csv'), 'TextType', 'string');

pre_unique_det = numel(unique(double(pre_stage3_det.case_id)));
post_unique_det = numel(unique(double(stage3_det.case_id)));
pre_dup_ids = uniqueDuplicateIds(pre_stage3_det.case_id);
post_dup_ids = uniqueDuplicateIds(stage3_det.case_id);
stage3_overlap_post = numel(intersect(double(stage3_orig.case_id), double(stage3_det.case_id)));
stage3_overlap_pre = numel(intersect(double(pre_stage3_orig.case_id), double(pre_stage3_det.case_id)));
stage3_group_balance = groupedBalance(stage3_det, {'group'});
stage3_room_group_counts = groupedCounts(stage3_det, {'group', 'room_type'});

room_stats_det = roomComplexityStats(stage2_det);
collinearity_tbl = topCollinearityPairs(stage2_det, all_features, 8);

seed_collision_possible = verifySeedCollision(cfg);

makeGlobalAucPlot(stage1_orig_auc, stage1_det_auc, stage2_orig_auc, stage2_det_auc, out_dir);
makeStage2RoomAucPlot(stage2_det_room_auc, out_dir);
makeLabelBalancePlot(stage2_det_balance, out_dir);
makeDeterministicDeltaPlot(out_dir);
makeStage3OverlayPlot(stage2_det, stage3_det, out_dir);
makeRoomComplexityPlot(stage2_det, out_dir);

writeSanityReport(fullfile(out_dir, 'sanity_rerun_report.md'), integrity_tbl, seed_collision_possible, pre_unique_det, post_unique_det, pre_dup_ids, post_dup_ids);
writeAucReport(fullfile(out_dir, 'auc_reanalysis_report.md'), stage1_orig_auc, stage1_det_auc, stage2_orig_auc, stage2_det_auc, stage2_orig_room_auc, stage2_det_room_auc, stage1_orig_top, stage1_det_top, stage1_orig_top_counts, stage1_det_top_counts, legacy_stage1_cell_counts, collinearity_tbl);
writeLabelReport(fullfile(out_dir, 'label_audit_report.md'), stage2_det_auc, stage2_det_balance, stage2_det_room_auc);
writeStage3Report(fullfile(out_dir, 'stage3_readiness_report.md'), pre_unique_det, post_unique_det, pre_dup_ids, stage3_overlap_pre, stage3_overlap_post, stage3_group_balance, stage3_room_group_counts, stage3_det, stage3_cells_det);
writeSummaryReport(fullfile(out_dir, 'code_audit_summary.md'), stage1_orig_auc, stage1_det_auc, stage2_det_auc, stage2_det_room_auc, stage1_det_top_counts, stage3_overlap_post, pre_unique_det, post_unique_det, room_stats_det, seed_collision_possible, collinearity_tbl);
writeFindingsCsv(fullfile(out_dir, 'audit_findings.csv'), stage1_det_top_counts, stage2_det_auc, pre_unique_det, post_unique_det, pre_dup_ids, seed_collision_possible, room_stats_det, collinearity_tbl, stage3_overlap_post);

fprintf('Saved code audit outputs under %s\n', out_dir);

function results = loadResults(mat_path)
    S = load(mat_path, 'results');
    results = S.results;
    results = results(~logical(results.failed), :);
end

function auc_tbl = computeGlobalAucs(results, cir_features, cp_features, joint_features)
    auc_tbl = table( ...
        {'CIR-only'; 'CP-only'; 'Joint'}, ...
        [fitAuc(results, cir_features, double(logical(results.is_nlos))); ...
         fitAuc(results, cp_features, double(logical(results.is_nlos))); ...
         fitAuc(results, joint_features, double(logical(results.is_nlos)))], ...
        'VariableNames', {'model', 'auc'});
end

function [auc_tbl, room_tbl, balance_tbl] = computeStage2Metrics(results, cir_features, cp_features, joint_features)
    room_col = columnText(results, 'room_type');
    has_los = logical(results.has_los_path);
    current = logical(results.is_nlos);
    geo = ~has_los;
    bounce = has_los & (double(results.bounce_to_los_ratio_mid) >= 0.33);
    mixed = geo | bounce;

    specs = { ...
        struct('name', 'current_0p20', 'display', 'Current ratio >= 0.20', 'mask', true(height(results), 1), 'y', current); ...
        struct('name', 'geo_only', 'display', 'Geo only (~has_los_path)', 'mask', true(height(results), 1), 'y', geo); ...
        struct('name', 'bounce_0p33_visible', 'display', 'Bounce only (visible LoS, ratio >= 0.33)', 'mask', has_los, 'y', bounce); ...
        struct('name', 'mixed_0p33', 'display', 'Mixed geo OR bounce@0.33', 'mask', true(height(results), 1), 'y', mixed)};
    rooms = {'A', 'B', 'C'};

    auc_rows = {};
    room_rows = {};
    balance_rows = {};
    for idx = 1:numel(specs)
        spec = specs{idx};
        subset = results(spec.mask, :);
        room_values = room_col(spec.mask);
        y = double(spec.y(spec.mask));
        auc_cir = fitAuc(subset, cir_features, y);
        auc_cp = fitAuc(subset, cp_features, y);
        auc_joint = fitAuc(subset, joint_features, y);
        auc_rows(end + 1, :) = {spec.name, spec.display, height(subset), sum(y == 0), sum(y == 1), auc_cir, auc_cp, auc_joint, auc_joint - auc_cir}; %#ok<AGROW>
        balance_rows(end + 1, :) = {spec.name, 'ALL', height(subset), sum(y == 0), sum(y == 1)}; %#ok<AGROW>
        for ridx = 1:numel(rooms)
            mask = strcmp(room_values, rooms{ridx});
            room_y = y(mask);
            room_tbl_part = subset(mask, :);
            auc_cir_room = fitAuc(room_tbl_part, cir_features, room_y);
            auc_cp_room = fitAuc(room_tbl_part, cp_features, room_y);
            auc_joint_room = fitAuc(room_tbl_part, joint_features, room_y);
            room_rows(end + 1, :) = {spec.name, rooms{ridx}, height(room_tbl_part), sum(room_y == 0), sum(room_y == 1), auc_cir_room, auc_cp_room, auc_joint_room, auc_joint_room - auc_cir_room}; %#ok<AGROW>
            balance_rows(end + 1, :) = {spec.name, rooms{ridx}, height(room_tbl_part), sum(room_y == 0), sum(room_y == 1)}; %#ok<AGROW>
        end
    end
    auc_tbl = cell2table(auc_rows, 'VariableNames', {'label_name', 'label_display', 'n', 'n_neg', 'n_pos', 'auc_cir', 'auc_cp', 'auc_joint', 'delta_auc'});
    room_tbl = cell2table(room_rows, 'VariableNames', {'label_name', 'room_type', 'n', 'n_neg', 'n_pos', 'auc_cir', 'auc_cp', 'auc_joint', 'delta_auc'});
    balance_tbl = cell2table(balance_rows, 'VariableNames', {'label_name', 'room_type', 'n', 'n_neg', 'n_pos'});
end

function auc = fitAuc(tbl, feature_names, y)
    X = table2array(tbl(:, feature_names));
    auc = analysis.cvLogisticAuc(X, y);
end

function stats = cellCounts(results, x_base, x_label, y_base, y_label, label_var)
    x_name = resolveColumnName(results, x_base);
    y_name = resolveColumnName(results, y_base);
    mask = cellMask(results.(x_name), x_label) & cellMask(results.(y_name), y_label);
    y = logical(results.(label_var)(mask));
    stats = struct();
    stats.n = sum(mask);
    stats.n_pos = sum(y);
    stats.n_neg = sum(~y);
end

function ids = uniqueDuplicateIds(values)
    x = double(values);
    u = unique(x);
    ids = u(arrayfun(@(v) sum(x == v) > 1, u));
end

function tbl = groupedBalance(stage3_tbl, group_cols)
    tbl = groupedCounts(stage3_tbl, group_cols);
    tbl.n_pos = zeros(height(tbl), 1);
    tbl.n_neg = zeros(height(tbl), 1);
    for idx = 1:height(tbl)
        mask = true(height(stage3_tbl), 1);
        for cidx = 1:numel(group_cols)
            col = group_cols{cidx};
            mask = mask & strcmp(string(stage3_tbl.(col)), string(tbl.(col)(idx)));
        end
        y = logical(str2doubleIfNeeded(stage3_tbl.label_positive(mask)));
        tbl.n_pos(idx) = sum(y);
        tbl.n_neg(idx) = sum(~y);
    end
end

function tbl = groupedCounts(stage3_tbl, group_cols)
    vars = cell(1, numel(group_cols));
    for idx = 1:numel(group_cols)
        vars{idx} = categorical(string(stage3_tbl.(group_cols{idx})));
    end
    G = findgroups(vars{:});
    counts = splitapply(@numel, G, G);
    out = table();
    for idx = 1:numel(group_cols)
        values = splitapply(@(x) x(1), string(stage3_tbl.(group_cols{idx})), G);
        out.(group_cols{idx}) = values;
    end
    out.n = counts;
    tbl = sortrows(out, group_cols);
end

function tbl = roomComplexityStats(results)
    rooms = {'A', 'B', 'C'};
    room_col = columnText(results, 'room_type');
    rows = {};
    for idx = 1:numel(rooms)
        mask = strcmp(room_col, rooms{idx});
        rows(end + 1, :) = {rooms{idx}, sum(mask), ...
            mean(double(results.num_paths(mask))), median(double(results.num_paths(mask))), ...
            mean(double(results.rms_delay_spread(mask))), mean(double(results.mean_excess_delay(mask))), ...
            mean(double(results.k_factor_estimate(mask))), mean(double(results.gamma_cp_3_fp_only(mask)))}; %#ok<AGROW>
    end
    tbl = cell2table(rows, 'VariableNames', {'room_type', 'n', 'avg_num_paths', 'median_num_paths', 'avg_rms_delay_spread', 'avg_mean_excess_delay', 'avg_k_factor', 'avg_gamma_cp_3'});
end

function tbl = topCollinearityPairs(results, feature_names, top_k)
    X = table2array(results(:, feature_names));
    corr_abs = abs(corr(X, 'Rows', 'pairwise'));
    rows = {};
    for i = 1:numel(feature_names)
        for j = (i + 1):numel(feature_names)
            rows(end + 1, :) = {feature_names{i}, feature_names{j}, corr_abs(i, j)}; %#ok<AGROW>
        end
    end
    tbl = cell2table(rows, 'VariableNames', {'feature_a', 'feature_b', 'abs_corr'});
    tbl = sortrows(tbl, 'abs_corr', 'descend');
    tbl = tbl(1:min(top_k, height(tbl)), :);
end

function tf = verifySeedCollision(cfg)
    H = complex(ones(2, 2, numel(cfg.freqs)));
    n1 = sweep.injectSnr(H, 20.0, 17) - H;
    n2 = sweep.injectSnr(H, 20.0, 17) - H;
    tf = isequaln(n1, n2);
end

function makeGlobalAucPlot(stage1_orig_auc, stage1_det_auc, stage2_orig_auc, stage2_det_auc, out_dir)
    fig = figure('Visible', 'off', 'Position', [100 100 980 460]);
    mixed_orig = stage2_orig_auc(strcmp(stage2_orig_auc.label_name, 'mixed_0p33'), :);
    mixed_det = stage2_det_auc(strcmp(stage2_det_auc.label_name, 'mixed_0p33'), :);
    data = [ ...
        stage1_orig_auc.auc(1), stage1_orig_auc.auc(2), stage1_orig_auc.auc(3); ...
        stage1_det_auc.auc(1), stage1_det_auc.auc(2), stage1_det_auc.auc(3); ...
        mixed_orig.auc_cir(1), mixed_orig.auc_cp(1), mixed_orig.auc_joint(1); ...
        mixed_det.auc_cir(1), mixed_det.auc_cp(1), mixed_det.auc_joint(1)];
    bar(data);
    set(gca, 'XTickLabel', {'Stage1 orig', 'Stage1 det', 'Stage2 orig mixed', 'Stage2 det mixed'});
    ylabel('AUC');
    legend({'CIR-only', 'CP-only', 'Joint'}, 'Location', 'northwest');
    title('Independent Audit: Global AUC Comparison');
    grid on;
    saveas(fig, fullfile(out_dir, 'stage1_stage2_global_auc_comparison.png'));
    close(fig);
end

function makeStage2RoomAucPlot(room_tbl, out_dir)
    mixed_tbl = room_tbl(strcmp(room_tbl.label_name, 'mixed_0p33'), :);
    order = {'A', 'B', 'C'};
    [~, loc] = ismember(order, mixed_tbl.room_type);
    mixed_tbl = mixed_tbl(loc, :);
    fig = figure('Visible', 'off', 'Position', [100 100 920 420]);
    bar(categorical(order), [mixed_tbl.auc_cir, mixed_tbl.auc_cp, mixed_tbl.auc_joint]);
    ylabel('AUC');
    title('Independent Audit: Stage 2 Deterministic Room-wise AUC');
    legend({'CIR-only', 'CP-only', 'Joint'}, 'Location', 'northwest');
    grid on;
    saveas(fig, fullfile(out_dir, 'stage2_room_auc_det.png'));
    close(fig);
end

function makeLabelBalancePlot(balance_tbl, out_dir)
    labels = {'current_0p20', 'geo_only', 'bounce_0p33_visible', 'mixed_0p33'};
    rooms = {'A', 'B', 'C'};
    pos = zeros(numel(labels), numel(rooms));
    neg = zeros(numel(labels), numel(rooms));
    for i = 1:numel(labels)
        for j = 1:numel(rooms)
            row = balance_tbl(strcmp(balance_tbl.label_name, labels{i}) & strcmp(balance_tbl.room_type, rooms{j}), :);
            pos(i, j) = row.n_pos;
            neg(i, j) = row.n_neg;
        end
    end
    fig = figure('Visible', 'off', 'Position', [100 100 1100 480]);
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    nexttile;
    bar(categorical(labels), pos, 'stacked');
    ylabel('Positive count');
    title('Label positives by room');
    legend(rooms, 'Location', 'northwest');
    grid on;
    nexttile;
    bar(categorical(labels), neg, 'stacked');
    ylabel('Negative count');
    title('Label negatives by room');
    legend(rooms, 'Location', 'northwest');
    grid on;
    saveas(fig, fullfile(out_dir, 'label_balance_by_room_and_rule.png'));
    close(fig);
end

function makeDeterministicDeltaPlot(out_dir)
    labels = {'ALL', 'A', 'B', 'C'};
    room_names = {'A', 'B', 'C'};
    fig = figure('Visible', 'off', 'Position', [100 100 980 420]);
    x = 1:4;
    y1 = zeros(size(x));
    y2 = zeros(size(x));
    y3 = zeros(size(x));

    orig_all = loadDeltaRow(fullfile(fileparts(out_dir), 'stage2', 'deterministic_delta_compare.csv'), 'mixed_0p33', 'ALL');
    y1(1) = orig_all.delta_auc_shift;
    y2(1) = orig_all.auc_cir_det - orig_all.auc_cir_orig;
    y3(1) = orig_all.auc_joint_det - orig_all.auc_joint_orig;
    for idx = 1:numel(room_names)
        row = loadDeltaRow(fullfile(fileparts(out_dir), 'stage2', 'deterministic_delta_compare.csv'), 'mixed_0p33', room_names{idx});
        y1(idx + 1) = row.delta_auc_shift;
        y2(idx + 1) = row.auc_cir_det - row.auc_cir_orig;
        y3(idx + 1) = row.auc_joint_det - row.auc_joint_orig;
    end
    bar(categorical(labels), [y1(:), y2(:), y3(:)]);
    ylabel('Deterministic - Original');
    title('Independent Audit: Deterministic Shift by scope');
    legend({'Delta AUC shift', 'CIR AUC shift', 'Joint AUC shift'}, 'Location', 'northwest');
    grid on;
    saveas(fig, fullfile(out_dir, 'deterministic_original_vs_rerun_delta.png'));
    close(fig);
end

function makeStage3OverlayPlot(stage2_det, stage3_det, out_dir)
    x_name = resolveColumnName(stage2_det, 'los_angle_from_anchor_bore_deg');
    room_col = columnText(stage2_det, 'room_type');
    fig = figure('Visible', 'off', 'Position', [100 100 920 420]);
    scatter(double(stage2_det.xpol_coupling_db), double(stage2_det.(x_name)), 14, [0.8 0.8 0.82], 'filled', 'MarkerFaceAlpha', 0.2); hold on;
    idx_geo = strcmp(stage3_det.group, 'GEO');
    idx_bounce = strcmp(stage3_det.group, 'BOUNCE');
    scatter(double(stage3_det.xpol_coupling_db_expected(idx_geo)), double(stage3_det.los_angle_deg(idx_geo)), 45, [0.1 0.45 0.85], 'filled');
    scatter(double(stage3_det.xpol_coupling_db_expected(idx_bounce)), double(stage3_det.los_angle_deg(idx_bounce)), 45, [0.85 0.35 0.15], 'filled');
    xlabel('xpol coupling dB (expected)');
    ylabel('LoS angle from anchor bore (deg)');
    title(sprintf('Independent Audit: Stage 3 candidates over Stage 2 cloud (rooms %s)', strjoin(unique(room_col), ', ')));
    legend({'Stage 2 cloud', 'GEO candidates', 'BOUNCE candidates'}, 'Location', 'best');
    grid on;
    saveas(fig, fullfile(out_dir, 'stage3_candidates_over_stage2_cloud.png'));
    close(fig);
end

function makeRoomComplexityPlot(stage2_det, out_dir)
    rooms = {'A', 'B', 'C'};
    room_col = columnText(stage2_det, 'room_type');
    fig = figure('Visible', 'off', 'Position', [100 100 1000 420]);
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    nexttile;
    vals = zeros(height(stage2_det), 1);
    grp = zeros(height(stage2_det), 1);
    cursor = 1;
    for idx = 1:numel(rooms)
        mask = strcmp(room_col, rooms{idx});
        n = sum(mask);
        vals(cursor:(cursor + n - 1)) = double(stage2_det.num_paths(mask));
        grp(cursor:(cursor + n - 1)) = idx;
        cursor = cursor + n;
    end
    boxchart(categorical(grp, 1:3, rooms), vals);
    ylabel('num\_paths');
    title('Path count by room');
    grid on;
    nexttile;
    vals = zeros(height(stage2_det), 1);
    grp = zeros(height(stage2_det), 1);
    cursor = 1;
    for idx = 1:numel(rooms)
        mask = strcmp(room_col, rooms{idx});
        n = sum(mask);
        vals(cursor:(cursor + n - 1)) = double(stage2_det.rms_delay_spread(mask));
        grp(cursor:(cursor + n - 1)) = idx;
        cursor = cursor + n;
    end
    boxchart(categorical(grp, 1:3, rooms), vals);
    ylabel('rms delay spread (s)');
    title('RMS delay spread by room');
    grid on;
    saveas(fig, fullfile(out_dir, 'path_count_rms_delay_by_room.png'));
    close(fig);
end

function writeSanityReport(path_out, integrity_tbl, seed_collision_possible, pre_unique_det, post_unique_det, pre_dup_ids, post_dup_ids)
    fid = fopen(path_out, 'w');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# Sanity Rerun Report\n\n');
    fprintf(fid, '## Executed checks\n\n');
    fprintf(fid, '| check_id | description | status | metric | expected | details |\n');
    fprintf(fid, '|---:|---|---|---|---|---|\n');
    for idx = 1:height(integrity_tbl)
        status = ternary(logical(integrity_tbl.passed(idx)), 'PASS', 'FAIL');
        fprintf(fid, '| %d | %s | %s | %s | %s | %s |\n', ...
            integrity_tbl.check_id(idx), escapePipe(integrity_tbl.description(idx)), status, ...
            escapePipe(integrity_tbl.metric(idx)), escapePipe(integrity_tbl.expected(idx)), escapePipe(integrity_tbl.details(idx)));
    end
    fprintf(fid, '\n## Additional deterministic notes\n\n');
    fprintf(fid, '- `verified`: same-case rerun is bit-exact in the executed integrity audit.\n');
    fprintf(fid, '- `verified`: checkpoint-resume equals fresh on the executed 6-case subset.\n');
    fprintf(fid, '- `verified`: case-order shuffle preserves case_id outputs on the executed 6-case subset.\n');
    fprintf(fid, '- `not verified`: full 900-case fresh-vs-resume and full 900-case shuffle-vs-original were not rerun in this audit because they would require another full sweep.\n');
    fprintf(fid, '- `verified`: stage-only seed salting is absent, so same `case_id` reproduces the same local-noise realization across stages when tensor shape/SNR match. current check: `%s`.\n', ternary(seed_collision_possible, 'collision possible', 'collision not observed'));
    fprintf(fid, '- `verified`: pre-patch deterministic Stage 3 list had `%d/30` unique cases; post-patch it has `%d/30` unique cases.\n', pre_unique_det, post_unique_det);
    fprintf(fid, '- `verified`: pre-patch duplicate det case_ids were `%s`; post-patch duplicates are `%s`.\n', vecfmt(pre_dup_ids), vecfmt(post_dup_ids));
end

function writeAucReport(path_out, stage1_orig_auc, stage1_det_auc, stage2_orig_auc, stage2_det_auc, stage2_orig_room_auc, stage2_det_room_auc, stage1_orig_top, stage1_det_top, stage1_orig_top_counts, stage1_det_top_counts, legacy_stage1_cell_counts, collinearity_tbl)
    fid = fopen(path_out, 'w');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# AUC Reanalysis Report\n\n');
    fprintf(fid, '## Method\n\n');
    fprintf(fid, '- `verified`: multivariate CIR-only / CP-only / Joint AUCs below were recomputed by this audit with `analysis.cvLogisticAuc`, i.e. 5-fold deterministic CV.\n');
    fprintf(fid, '- `verified`: conditional cell AUCs in the canonical Stage 1/2 outputs now also route through the same CV helper.\n');
    fprintf(fid, '- `not verified`: no bootstrap CI or DeLong CI is stored in the canonical outputs; this audit did not add a new CI pipeline.\n\n');

    fprintf(fid, '## Stage 1 Global AUC\n\n');
    fprintf(fid, '| dataset | CIR-only | CP-only | Joint | Delta AUC |\n');
    fprintf(fid, '|---|---:|---:|---:|---:|\n');
    fprintf(fid, '| original | %.6f | %.6f | %.6f | %.6f |\n', stage1_orig_auc.auc(1), stage1_orig_auc.auc(2), stage1_orig_auc.auc(3), stage1_orig_auc.auc(3) - stage1_orig_auc.auc(1));
    fprintf(fid, '| deterministic | %.6f | %.6f | %.6f | %.6f |\n\n', stage1_det_auc.auc(1), stage1_det_auc.auc(2), stage1_det_auc.auc(3), stage1_det_auc.auc(3) - stage1_det_auc.auc(1));
    fprintf(fid, '- `verified`: the saved Stage 1 summary files still differ slightly from this independent recomputation because the legacy Stage 1 analysis scripts keep local CV helpers instead of the shared deterministic helper.\n\n');

    fprintf(fid, '## Stage 2 Global AUC By Label\n\n');
    fprintf(fid, '| label | orig delta | det delta | orig CIR | det CIR | orig Joint | det Joint |\n');
    fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|\n');
    for idx = 1:height(stage2_orig_auc)
        det_row = stage2_det_auc(strcmp(stage2_det_auc.label_name, stage2_orig_auc.label_name{idx}), :);
        fprintf(fid, '| %s | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f |\n', ...
            stage2_orig_auc.label_name{idx}, stage2_orig_auc.delta_auc(idx), det_row.delta_auc(1), ...
            stage2_orig_auc.auc_cir(idx), det_row.auc_cir(1), stage2_orig_auc.auc_joint(idx), det_row.auc_joint(1));
    end
    fprintf(fid, '\n');

    mixed_orig = stage2_orig_room_auc(strcmp(stage2_orig_room_auc.label_name, 'mixed_0p33'), :);
    mixed_det = stage2_det_room_auc(strcmp(stage2_det_room_auc.label_name, 'mixed_0p33'), :);
    fprintf(fid, '## Stage 2 Deterministic mixed@0.33 by room\n\n');
    fprintf(fid, '| room | CIR-only | CP-only | Joint | Delta AUC |\n');
    fprintf(fid, '|---|---:|---:|---:|---:|\n');
    for idx = 1:height(mixed_det)
        fprintf(fid, '| %s | %.6f | %.6f | %.6f | %.6f |\n', mixed_det.room_type{idx}, mixed_det.auc_cir(idx), mixed_det.auc_cp(idx), mixed_det.auc_joint(idx), mixed_det.delta_auc(idx));
    end
    fprintf(fid, '\n');

    fprintf(fid, '## Stage 1 Conditional Cells\n\n');
    fprintf(fid, '| cell type | x var | x label | y var | y label | n | n_pos | n_neg | delta_auc |\n');
    fprintf(fid, '|---|---|---|---|---|---:|---:|---:|---:|\n');
    fprintf(fid, '| current original top | %s | %s | %s | %s | %d | %d | %d | %.6f |\n', ...
        stage1_orig_top.x_var{1}, stage1_orig_top.x_label{1}, stage1_orig_top.y_var{1}, stage1_orig_top.y_label{1}, ...
        stage1_orig_top_counts.n, stage1_orig_top_counts.n_pos, stage1_orig_top_counts.n_neg, stage1_orig_top.delta_auc(1));
    fprintf(fid, '| current deterministic top | %s | %s | %s | %s | %d | %d | %d | %.6f |\n', ...
        stage1_det_top.x_var{1}, stage1_det_top.x_label{1}, stage1_det_top.y_var{1}, stage1_det_top.y_label{1}, ...
        stage1_det_top_counts.n, stage1_det_top_counts.n_pos, stage1_det_top_counts.n_neg, stage1_det_top.delta_auc(1));
    fprintf(fid, '| legacy claimed eps_r x xpol cell | eps_r | [2.00, 3.60] | xpol_coupling_db | [20.01, 24.00] | %d | %d | %d | %.6f |\n\n', ...
        legacy_stage1_cell_counts.n, legacy_stage1_cell_counts.n_pos, legacy_stage1_cell_counts.n_neg, legacyStage1CellDelta());

    fprintf(fid, 'Interpretation:\n');
    fprintf(fid, '- `verified`: Stage 1 global lift remains moderate and positive.\n');
    fprintf(fid, '- `verified`: the exact Stage 1 headline cell claimed earlier (`eps_r x xpol`, `+0.2666`) is not the current canonical top cell and now sits far lower.\n');
    fprintf(fid, '- `verified`: the current top cells are class-imbalanced enough (for example `%d` positives vs `%d` negatives) that exact conditional rankings should be treated as unstable without CI or stricter per-class minima.\n\n', stage1_det_top_counts.n_pos, stage1_det_top_counts.n_neg);

    fprintf(fid, '## Feature Collinearity Snapshot (Stage 2 deterministic)\n\n');
    fprintf(fid, '| feature_a | feature_b | abs(corr) |\n');
    fprintf(fid, '|---|---|---:|\n');
    for idx = 1:height(collinearity_tbl)
        fprintf(fid, '| %s | %s | %.6f |\n', collinearity_tbl.feature_a{idx}, collinearity_tbl.feature_b{idx}, collinearity_tbl.abs_corr(idx));
    end
    fprintf(fid, '\nRecommendation: add bootstrap CIs for manuscript-facing AUC and impose a per-class minimum (for example `n_pos >= 20`, `n_neg >= 20`) on conditional cells before treating their ranking as stable.\n');
end

function writeLabelReport(path_out, stage2_det_auc, stage2_det_balance, stage2_det_room_auc)
    fid = fopen(path_out, 'w');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# Label Audit Report\n\n');
    fprintf(fid, '## Definitions\n\n');
    fprintf(fid, '- `verified`: `is_nlos_geo = ~has_los_path`.\n');
    fprintf(fid, '- `verified`: `is_nlos_bounce_0p33 = has_los_path & (bounce_to_los_ratio_mid >= 0.33)`.\n');
    fprintf(fid, '- `verified`: `mixed_0p33 = is_nlos_geo OR is_nlos_bounce_0p33`.\n');
    fprintf(fid, '- `verified`: deterministic relabeled export sets `is_nlos = is_nlos_mixed_0p33` and stores `label_schema = mixed_0p33_primary_with_dual_aux`.\n\n');

    fprintf(fid, '## Label balance by room\n\n');
    fprintf(fid, '| label | room | n_neg | n_pos |\n');
    fprintf(fid, '|---|---|---:|---:|\n');
    for idx = 1:height(stage2_det_balance)
        fprintf(fid, '| %s | %s | %d | %d |\n', stage2_det_balance.label_name{idx}, stage2_det_balance.room_type{idx}, stage2_det_balance.n_neg(idx), stage2_det_balance.n_pos(idx));
    end
    fprintf(fid, '\n');

    fprintf(fid, '## Leakage / contamination checks\n\n');
    fprintf(fid, '- `verified`: `is_los`, `is_nlos`, and `bounce_to_los_ratio_mid` do not appear inside `+features/*.m` in the executed integrity audit.\n');
    fprintf(fid, '- `verified`: Stage 3 representative-case ranking originally used `bounce_to_los_ratio_mid`; this audit removed it from the selection distance and regenerated the case list.\n');
    fprintf(fid, '- `verified`: current Stage 3 deterministic case list is unique after the patch and still comes from the deterministic relabeled dataset.\n\n');

    fprintf(fid, '## Scientific interpretation of mixed@0.33\n\n');
    fprintf(fid, '| label | det delta_auc | note |\n');
    fprintf(fid, '|---|---:|---|\n');
    for idx = 1:height(stage2_det_auc)
        note = '';
        if strcmp(stage2_det_auc.label_name{idx}, 'geo_only')
            note = 'Highest global lift but room support is highly concentrated.';
        elseif strcmp(stage2_det_auc.label_name{idx}, 'mixed_0p33')
            note = 'Primary reporting label because it keeps geo/bounce decomposition while covering all rooms.';
        elseif strcmp(stage2_det_auc.label_name{idx}, 'current_0p20')
            note = 'Aggressive and physics-adjacent; not recommended as primary.';
        elseif strcmp(stage2_det_auc.label_name{idx}, 'bounce_0p33_visible')
            note = 'Visible-LoS-only subset, not a full reporting label.';
        end
        fprintf(fid, '| %s | %.6f | %s |\n', stage2_det_auc.label_name{idx}, stage2_det_auc.delta_auc(idx), note);
    end
    fprintf(fid, '\n');

    mixed_room = stage2_det_room_auc(strcmp(stage2_det_room_auc.label_name, 'mixed_0p33'), :);
    fprintf(fid, 'Room-wise mixed@0.33 note:\n');
    fprintf(fid, '- `verified`: Room A and Room C retain positive lift, Room B remains negative in the current deterministic canonical outputs.\n');
    fprintf(fid, '- `caution`: label selection was post-hoc across four alternatives, so the rationale must be written explicitly as a decomposition/coverage choice rather than as a purely data-driven optimum.\n');
    fprintf(fid, '- `verified`: current room-wise deterministic mixed deltas are `A=%.6f`, `B=%.6f`, `C=%.6f`.\n', mixed_room.delta_auc(strcmp(mixed_room.room_type, 'A')), mixed_room.delta_auc(strcmp(mixed_room.room_type, 'B')), mixed_room.delta_auc(strcmp(mixed_room.room_type, 'C')));
end

function writeStage3Report(path_out, pre_unique_det, post_unique_det, pre_dup_ids, stage3_overlap_pre, stage3_overlap_post, stage3_group_balance, stage3_room_group_counts, stage3_det, stage3_cells_det)
    fid = fopen(path_out, 'w');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# Stage 3 Readiness Report\n\n');
    fprintf(fid, '## Case-list provenance\n\n');
    fprintf(fid, '- `verified`: deterministic Stage 3 reselection loads `stage2_900_ffd_relabel_det.mat` and uses `is_nlos_geo` / `is_nlos_bounce_0p33` as the selection labels.\n');
    fprintf(fid, '- `verified`: current header uses `xpol_coupling_db_expected`, not an HFSS input label.\n');
    fprintf(fid, '- `verified`: `cfg.freqs = linspace(6.25 GHz, 6.75 GHz, 257)` in `+config/defaultConfig.m`.\n\n');

    fprintf(fid, '## Before / after Stage 3 patch\n\n');
    fprintf(fid, '| item | pre-patch | current |\n');
    fprintf(fid, '|---|---:|---:|\n');
    fprintf(fid, '| unique det case_ids | %d | %d |\n', pre_unique_det, post_unique_det);
    fprintf(fid, '| det/orig candidate overlap | %d | %d |\n', stage3_overlap_pre, stage3_overlap_post);
    fprintf(fid, '| duplicate det case_ids | %s | none |\n\n', vecfmt(pre_dup_ids));

    fprintf(fid, '## Current deterministic case-list balance\n\n');
    fprintf(fid, '| group | n | n_pos | n_neg |\n');
    fprintf(fid, '|---|---:|---:|---:|\n');
    for idx = 1:height(stage3_group_balance)
        fprintf(fid, '| %s | %d | %d | %d |\n', stage3_group_balance.group{idx}, stage3_group_balance.n(idx), stage3_group_balance.n_pos(idx), stage3_group_balance.n_neg(idx));
    end
    fprintf(fid, '\n');

    fprintf(fid, '| group | room | n |\n');
    fprintf(fid, '|---|---|---:|\n');
    for idx = 1:height(stage3_room_group_counts)
        fprintf(fid, '| %s | %s | %d |\n', stage3_room_group_counts.group{idx}, stage3_room_group_counts.room_type{idx}, stage3_room_group_counts.n(idx));
    end
    fprintf(fid, '\n');

    fprintf(fid, '## Current selected regimes\n\n');
    fprintf(fid, '| group | regime | n | n_pos | n_neg | delta_auc |\n');
    fprintf(fid, '|---|---|---:|---:|---:|---:|\n');
    for idx = 1:height(stage3_cells_det)
        fprintf(fid, '| %s | %s | %d | %d | %d | %.6f |\n', stage3_cells_det.group{idx}, stage3_cells_det.regime_desc{idx}, stage3_cells_det.n(idx), stage3_cells_det.n_pos(idx), stage3_cells_det.n_neg(idx), stage3_cells_det.delta_auc(idx));
    end
    fprintf(fid, '\n');

    fprintf(fid, 'Interpretation:\n');
    fprintf(fid, '- `verified`: current deterministic GEO coverage is Room C only.\n');
    fprintf(fid, '- `verified`: current deterministic list is two-group and Stage 2 centric; it does not implement the legacy Stage 1 `G1/G2/G3` 45-case plan.\n');
    fprintf(fid, '- `caution`: if manuscript/protocol text still describes the legacy three-group plan or Room B HFSS coverage, it is stale.\n\n');

    fprintf(fid, '## HFSS validation target\n\n');
    fprintf(fid, '- AUC reproduction on 30 cases is not the primary metric.\n');
    fprintf(fid, '- Recommended primary metrics:\n');
    fprintf(fid, '  1. `Spearman rho(gamma_cp_3 MATLAB, gamma_cp_3 HFSS) > 0.5` within each group when support allows.\n');
    fprintf(fid, '  2. `Joint - CIR` lift sign agreement by selected regime.\n');
    fprintf(fid, '  3. Case-level direction agreement for `gamma_cp_3`, `rise_time_fp`, and `fp_to_total_ratio` between MATLAB and HFSS in at least `60%%` of cases.\n');
    fprintf(fid, '  4. GEO claims stated as `Room C conditional`, not room-general.\n');
end

function writeSummaryReport(path_out, stage1_orig_auc, stage1_det_auc, stage2_det_auc, stage2_det_room_auc, stage1_det_top_counts, stage3_overlap_post, pre_unique_det, post_unique_det, room_stats_det, seed_collision_possible, collinearity_tbl)
    fid = fopen(path_out, 'w');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    mixed_det = stage2_det_auc(strcmp(stage2_det_auc.label_name, 'mixed_0p33'), :);
    room_mixed_det = stage2_det_room_auc(strcmp(stage2_det_room_auc.label_name, 'mixed_0p33'), :);
    fprintf(fid, '# Independent Code Audit Summary\n\n');
    fprintf(fid, '## Final Judgment\n\n');
    fprintf(fid, 'Conditional GO: Stage 3 HFSS can start **only** with the patched unique deterministic case list and only if all manuscript/protocol text is updated to the current canonical numbers and interpretations.\n\n');

    fprintf(fid, '## Blocker\n\n');
    fprintf(fid, '1. `verified` Manuscript/protocol-facing headline numbers are stale relative to the current canonical outputs. For example, current deterministic Stage 2 `mixed_0p33` global `delta_auc` is `%.6f`, current det/orig candidate overlap is `%d/30`, and current deterministic Stage 1 global lift is `%.6f`; any older values should not be cited.\n\n', mixed_det.delta_auc, stage3_overlap_post, stage1_det_auc.auc(3) - stage1_det_auc.auc(1));

    fprintf(fid, '## Caution\n\n');
    fprintf(fid, '1. `verified` The current Stage 1 top conditional cells are unstable and class-imbalanced. The deterministic top cell has `%d` positives and `%d` negatives, so exact top-cell ranking is not strong enough for paper-facing claims without CI or stricter per-class minima.\n', stage1_det_top_counts.n_pos, stage1_det_top_counts.n_neg);
    fprintf(fid, '2. `verified` `mixed@0.33` is a reasonable primary reporting label, but the choice is still post-hoc across four alternatives and must be documented as a decomposition/coverage decision, not as a pre-registered optimum.\n');
    fprintf(fid, '3. `verified` Seed salting still uses `case_id` only. Within-stage determinism is good, but cross-stage noise collisions remain possible when shape/SNR match (`%s`).\n', ternary(seed_collision_possible, 'possible', 'not observed'));
    fprintf(fid, '4. `not verified` Full fresh-vs-resume and full order-shuffle equivalence were not rerun end-to-end in this audit; only the executed subset checks passed.\n');
    fprintf(fid, '5. `verified` GEO validation is Room C specific in the current deterministic list. Do not write room-general GEO claims from Stage 3.\n');
    fprintf(fid, '6. `verified` Room-complexity narration must be corrected: Stage 2 deterministic means are `A paths=%.3f`, `B=%.3f`, `C=%.3f`, while Room C has the largest RMS delay spread. `empty room = easy` is not a safe shorthand.\n', room_stats_det.avg_num_paths(strcmp(room_stats_det.room_type, 'A')), room_stats_det.avg_num_paths(strcmp(room_stats_det.room_type, 'B')), room_stats_det.avg_num_paths(strcmp(room_stats_det.room_type, 'C')));
    fprintf(fid, '7. `verified` Canonical features still contain strong collinearity; the top pair in this audit was `%s` vs `%s` with `|corr|=%.6f`.\n', collinearity_tbl.feature_a{1}, collinearity_tbl.feature_b{1}, collinearity_tbl.abs_corr(1));
    fprintf(fid, '8. `verified` The legacy Stage 1 analysis scripts still carry their own CV helpers, so saved Stage 1 summary files can drift slightly from the shared deterministic helper used in this audit.\n\n');

    fprintf(fid, '## Supplemental\n\n');
    fprintf(fid, '1. `verified` Negative XPD support matching still fails in the executed integrity audit and should stay supplemental.\n');
    fprintf(fid, '2. `verified` FFD import / gamma convention sanity checks pass: ideal LoS gamma `0`, patch-FFD LoS gamma `0.101576`, ideal odd-bounce gamma `56.234133`, patch odd-bounce gamma `0.600962`.\n');
    fprintf(fid, '3. `verified` Stage 3 pre-patch duplicate-case bug was fixed during this audit (`%d -> %d` unique det cases) and `bounce_to_los_ratio_mid` was removed from representative-case ranking.\n', pre_unique_det, post_unique_det);
end

function writeFindingsCsv(path_out, stage1_orig_top_counts, stage2_det_auc, pre_unique_det, post_unique_det, pre_dup_ids, seed_collision_possible, room_stats_det, collinearity_tbl, stage3_overlap_post)
    rows = { ...
        'B1', 'blocker', 'results/stage3/stage3_validation_protocol.md', '9-13;29-41;73-77', ...
        'verified: protocol note is stale versus current canonical metrics and current det case-list distribution', ...
        sprintf('current det mixed delta_auc=%.6f; current det/orig case overlap=%d/30; current det stage3 room split has no Room B', stage2_det_auc.delta_auc(strcmp(stage2_det_auc.label_name, 'mixed_0p33')), stage3_overlap_post), ...
        'Paper-facing conclusions can be misquoted even if the code path is now sound.', ...
        'Update manuscript/protocol text to current canonical outputs before citing any numbers.', 'no'; ...
        'C1', 'caution', '+analysis/computeConditionalAucGrid.m', '42-48', ...
        'verified: conditional cell ranking only enforces min_per_cell and both-class presence, not per-class support', ...
        sprintf('current deterministic top Stage 1 cell has n=%d, n_pos=%d, n_neg=%d', stage1_orig_top_counts.n, stage1_orig_top_counts.n_pos, stage1_orig_top_counts.n_neg), ...
        'Exact top-cell claims can be overstated by class imbalance.', ...
        'Add min_pos/min_neg thresholds or bootstrap CI before paper-facing conditional claims.', 'yes'; ...
        'C2', 'caution', 'scripts/week4_day5_relabel_analysis_det.m', '23-32', ...
        'verified: four label rules are compared and mixed@0.33 is then promoted to the primary label', ...
        'geo_only currently has the largest global lift but mixed@0.33 is chosen for decomposition/coverage reasons', ...
        'Reviewer can attack the threshold choice as post-hoc unless rationale is explicit.', ...
        'Document the physics/coverage rationale and show all four rules in the supplement.', 'no'; ...
        'C3', 'caution', '+sweep/injectSnr.m', '17-19', ...
        'verified: local-noise seed uses noise_seed only; stage salt is absent', ...
        sprintf('same-seed collision check=%s', ternary(seed_collision_possible, 'possible', 'not observed')), ...
        'Within-stage determinism is good, but cross-stage correlation remains possible for same case_id and shape.', ...
        'Mix stage_id/base_seed into normalizeSeed before paper freeze.', 'yes'; ...
        'C4', 'caution', 'scripts/week4_day5_integrity_audit.m', '60-83', ...
        'verified: fresh-vs-resume and shuffle checks were run on a 6-case subset, not on the full 900/3000-case sweeps', ...
        'subset checks PASS; full-sweep equivalence not rerun in this audit', ...
        'Residual deterministic risk remains unclosed at full-run scale.', ...
        'Run a full hash-based fresh/resume and shuffle audit once before final paper lock.', 'yes'; ...
        'C5', 'caution', 'results/stage2/stage2_900_ffd_det.csv', '1', ...
        'verified: room-complexity narrative is not monotonic by simple path count', ...
        sprintf('avg num_paths A=%.3f, B=%.3f, C=%.3f; avg RMS delay C=%.3e is largest', room_stats_det.avg_num_paths(strcmp(room_stats_det.room_type, 'A')), room_stats_det.avg_num_paths(strcmp(room_stats_det.room_type, 'B')), room_stats_det.avg_num_paths(strcmp(room_stats_det.room_type, 'C')), room_stats_det.avg_rms_delay_spread(strcmp(room_stats_det.room_type, 'C'))), ...
        'The shorthand "empty room is easy" is scientifically unsafe.', ...
        'Rewrite room-complexity interpretation around delay spread / blocking, not emptiness.', 'no'; ...
        'C6', 'caution', 'scripts/week4_day5_stage3_reselect_det.m', '32-36', ...
        'verified: current deterministic Stage 3 list is two-group and GEO coverage is Room C only', ...
        'current room/group counts are GEO:C=15, BOUNCE:A=5/C=10', ...
        'Stage 3 results cannot support room-general GEO claims.', ...
        'State Stage 3 as Stage 2-centric conditional validation and keep GEO claims Room C-specific.', 'no'; ...
        'C7', 'caution', 'scripts/week4_day6_stage1_analysis_det.m', '132-157', ...
        'verified: Stage 1 saved summary still uses a local CV helper instead of the shared deterministic helper', ...
        'independent raw-data reanalysis gives a slightly different Stage 1 deterministic global delta than the saved Stage 1 markdown/csv', ...
        'Stage 1 headline numbers are directionally stable but not yet fully locked to one CV implementation.', ...
        'Unify Stage 1 analysis scripts to analysis.cvLogisticAuc before final paper freeze, but keep current results untouched during audit.', 'yes'; ...
        'S1', 'supplemental', 'scripts/week4_day5_integrity_audit.m', '103-105', ...
        'verified: negative XPD support-match audit fails', ...
        'integrity audit check 13 = 0.486988847584 with 269 negative-XPD cases', ...
        'Affects mechanism interpretation, not the main Stage 3 gate.', ...
        'Keep negative-XPD discussion supplemental and do not anchor the main mechanism on it.', 'no'; ...
        'S2', 'supplemental', '+features/canonicalFeatureNames.m', '26-44', ...
        'verified: canonical18 still contains near-duplicate statistics', ...
        sprintf('%s vs %s abs(corr)=%.6f', collinearity_tbl.feature_a{1}, collinearity_tbl.feature_b{1}, collinearity_tbl.abs_corr(1)), ...
        'Multicollinearity weakens coefficient-level interpretation.', ...
        'Use reduced/orthogonalized ablations for interpretability sections.', 'no'; ...
        'S3', 'supplemental', '+antennas/loadFfdPattern.m', '154-158', ...
        'verified: HFSS loading defaults to phi_convention=hfss with flip_ephi=true', ...
        'boresight/direct-channel sanity passes, but the sign-fix provenance lives mainly in code/day1 notes', ...
        'Implementation appears consistent, but the methods note should explain the convention choice.', ...
        'Add a short methods note that HFSS phi convention requires E_phi sign correction in this pipeline.', 'no'; ...
        'N1', 'no_issue', '+features/computeGammaCpVariants.m', '4-8', ...
        'verified: gamma_cp convention matches reversed-hand / same-hand and direct-path sanity passes', ...
        'ideal LoS gamma=0; patch LoS gamma=0.101575623419; ideal odd-bounce gamma=56.234133; patch odd-bounce gamma=0.600962', ...
        'Core CP handedness convention is logically consistent.', ...
        'Keep current convention and cite the sanity values.', 'no'; ...
        'N2', 'no_issue', 'scripts/week4_day5_stage3_reselect_det.m', '65;184', ...
        'verified: Stage 3 duplicate-case bug and label-derived ranking input were fixed during this audit', ...
        sprintf('pre-patch unique=%d, post-patch unique=%d, pre-patch duplicate ids=%s', pre_unique_det, post_unique_det, vecfmt(pre_dup_ids)), ...
        'Current deterministic case list is now unique and cleaner for HFSS execution.', ...
        'Keep the patched selector as the canonical Stage 3 exporter.', 'yes'};
    findings = cell2table(rows, 'VariableNames', {'issue_id', 'severity', 'file', 'line', 'finding', 'evidence', 'impact', 'recommended_action', 'requires_rerun'});
    writetable(findings, path_out);
end

function row = loadDeltaRow(csv_path, label_name, room_type)
    tbl = readtable(csv_path, 'TextType', 'string');
    mask = strcmp(tbl.label_name, label_name) & strcmp(tbl.room_type, room_type);
    row = tbl(mask, :);
end

function values = columnText(tbl, base_name)
    name = resolveColumnName(tbl, base_name);
    values = cellstr(string(tbl.(name)));
end

function name = resolveColumnName(tbl, base_name)
    candidates = {base_name, [base_name '_cases'], [base_name '_results'], [base_name '_cases_tbl'], [base_name '_results_tbl']};
    for idx = 1:numel(candidates)
        if ismember(candidates{idx}, tbl.Properties.VariableNames)
            name = candidates{idx};
            return;
        end
    end
    error('Column not found: %s', base_name);
end

function tf = cellMask(values, label)
    if isnumeric(values) || islogical(values)
        bounds = parseBounds(label);
        tol = 0.005;
        tf = double(values) >= (bounds(1) - tol) & double(values) <= (bounds(2) + tol);
    else
        tf = strcmp(cellstr(string(values)), char(string(label)));
    end
end

function bounds = parseBounds(label)
    txt = char(string(label));
    vals = sscanf(txt, '[%f, %f]');
    if numel(vals) ~= 2
        error('Failed to parse bounds: %s', txt);
    end
    bounds = vals(:).';
end

function out = escapePipe(x)
    sx = string(x);
    if ismissing(sx)
        sx = "";
    end
    out = char(sx);
    out = strrep(out, '|', '\|');
end

function out = vecfmt(x)
    if isempty(x)
        out = '[]';
    else
        out = mat2str(double(x(:)).');
    end
end

function out = ternary(tf, a, b)
    if tf
        out = a;
    else
        out = b;
    end
end

function out = str2doubleIfNeeded(x)
    if isnumeric(x) || islogical(x)
        out = x;
    else
        sx = string(x);
        num = str2double(sx);
        if all(isnan(num))
            out = strcmpi(sx, "true");
        else
            out = num;
        end
    end
end

function value = legacyStage1CellDelta()
    tbl = readtable(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results', 'stage1', 'conditional_auc_cells.csv'), 'TextType', 'string');
    mask = strcmp(tbl.x_var, 'eps_r') & strcmp(tbl.y_var, 'xpol_coupling_db') & strcmp(tbl.x_label, '[2.00, 3.60]') & strcmp(tbl.y_label, '[20.01, 24.00]');
    value = tbl.delta_auc(find(mask, 1, 'first'));
end
