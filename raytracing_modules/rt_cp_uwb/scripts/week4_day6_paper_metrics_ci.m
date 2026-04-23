script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

out_dir = fullfile(repo_root, 'results', 'code_audit');
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

rng(20260422, 'twister');
n_boot = 120;
all_features = features.canonicalFeatureNames();
cp_features = all_features(1:6);
cir_features = all_features(7:end);
joint_features = all_features;

stage1 = loadResults(fullfile(repo_root, 'results', 'stage1', 'stage1_3000_ffd_det.mat'));
stage2 = loadResults(fullfile(repo_root, 'results', 'stage2', 'stage2_900_ffd_det.mat'));
room_col = columnText(stage2, 'room_type');
has_los = logical(stage2.has_los_path);
mixed = (~has_los) | (has_los & (double(stage2.bounce_to_los_ratio_mid) >= 0.33));

global_rows = {};
global_rows(end + 1, :) = metricRow('Stage1', 'ALL', stage1, double(logical(stage1.is_nlos)), cir_features, joint_features, n_boot); %#ok<AGROW>
global_rows(end + 1, :) = metricRow('Stage2 mixed@0.33', 'ALL', stage2, double(mixed), cir_features, joint_features, n_boot); %#ok<AGROW>
for room = ["A", "B", "C"]
    mask = strcmp(room_col, room);
    global_rows(end + 1, :) = metricRow('Stage2 mixed@0.33', char(room), stage2(mask, :), double(mixed(mask)), cir_features, joint_features, n_boot); %#ok<AGROW>
end
global_tbl = cell2table(global_rows, 'VariableNames', ...
    {'scope', 'subset', 'n', 'n_neg', 'n_pos', 'auc_cir', 'auc_joint', 'delta_auc', 'delta_ci_low', 'delta_ci_high'});
writetable(global_tbl, fullfile(out_dir, 'paper_facing_auc_ci.csv'));

cells = readtable(fullfile(repo_root, 'results', 'stage1', 'conditional_auc_cells_det.csv'), 'TextType', 'string');
cells = sortrows(cells(isfinite(cells.delta_auc), :), {'delta_auc', 'n'}, {'descend', 'descend'});
qualified_rows = {};
count = 0;
for idx = 1:height(cells)
    stats = cellCounts(stage1, cells.x_var{idx}, cells.x_label{idx}, cells.y_var{idx}, cells.y_label{idx}, 'is_nlos');
    if stats.n_pos < 20 || stats.n_neg < 20
        continue;
    end
    subset = subsetForCell(stage1, cells.x_var{idx}, cells.x_label{idx}, cells.y_var{idx}, cells.y_label{idx});
    metrics = bootstrapMetrics(subset, double(logical(subset.is_nlos)), cir_features, joint_features, n_boot);
    qualified_rows(end + 1, :) = {cells.x_var{idx}, cells.x_label{idx}, cells.y_var{idx}, cells.y_label{idx}, stats.n, stats.n_pos, stats.n_neg, metrics.delta_auc, metrics.delta_ci(1), metrics.delta_ci(2)}; %#ok<AGROW>
    count = count + 1;
    if count >= 3
        break;
    end
end
legacy_subset = subsetForCell(stage1, 'eps_r', '[2.00, 3.60]', 'xpol_coupling_db', '[20.01, 24.00]');
legacy_stats = cellCounts(stage1, 'eps_r', '[2.00, 3.60]', 'xpol_coupling_db', '[20.01, 24.00]', 'is_nlos');
legacy_metrics = bootstrapMetrics(legacy_subset, double(logical(legacy_subset.is_nlos)), cir_features, joint_features, n_boot);
qualified_rows(end + 1, :) = {'eps_r', '[2.00, 3.60]', 'xpol_coupling_db', '[20.01, 24.00]', legacy_stats.n, legacy_stats.n_pos, legacy_stats.n_neg, legacy_metrics.delta_auc, legacy_metrics.delta_ci(1), legacy_metrics.delta_ci(2)}; %#ok<AGROW>
qualified_tbl = cell2table(qualified_rows, 'VariableNames', ...
    {'x_var', 'x_label', 'y_var', 'y_label', 'n', 'n_pos', 'n_neg', 'delta_auc', 'delta_ci_low', 'delta_ci_high'});
writetable(qualified_tbl, fullfile(out_dir, 'paper_facing_conditional_cells_ci.csv'));

fid = fopen(fullfile(out_dir, 'paper_facing_auc_ci.md'), 'w');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Paper-Facing AUC and CI Report\n\n');
fprintf(fid, '## Policy\n\n');
fprintf(fid, '- Main-text statistics should stay at the global / room level.\n');
fprintf(fid, '- Conditional cells should be treated as supplemental unless they satisfy `n_pos >= 20` and `n_neg >= 20` and carry a bootstrap CI.\n');
fprintf(fid, '- The earlier `eps_r x xpol +0.2666` headline cell should not be used as a paper-facing headline.\n\n');

fprintf(fid, '## Global and Room-Level Delta AUC\n\n');
fprintf(fid, '| scope | subset | n | n_neg | n_pos | auc_cir | auc_joint | delta_auc | bootstrap 95%% CI |\n');
fprintf(fid, '|---|---|---:|---:|---:|---:|---:|---:|---|\n');
for idx = 1:height(global_tbl)
    fprintf(fid, '| %s | %s | %d | %d | %d | %.6f | %.6f | %.6f | [%.6f, %.6f] |\n', ...
        global_tbl.scope{idx}, global_tbl.subset{idx}, global_tbl.n(idx), global_tbl.n_neg(idx), global_tbl.n_pos(idx), ...
        global_tbl.auc_cir(idx), global_tbl.auc_joint(idx), global_tbl.delta_auc(idx), ...
        global_tbl.delta_ci_low(idx), global_tbl.delta_ci_high(idx));
end
fprintf(fid, '\n');

fprintf(fid, '## Count-Qualified Conditional Cells (Supplemental)\n\n');
fprintf(fid, '| x var | x label | y var | y label | n | n_pos | n_neg | delta_auc | bootstrap 95%% CI |\n');
fprintf(fid, '|---|---|---|---|---:|---:|---:|---:|---|\n');
for idx = 1:height(qualified_tbl)
    fprintf(fid, '| %s | %s | %s | %s | %d | %d | %d | %.6f | [%.6f, %.6f] |\n', ...
        qualified_tbl.x_var{idx}, qualified_tbl.x_label{idx}, qualified_tbl.y_var{idx}, qualified_tbl.y_label{idx}, ...
        qualified_tbl.n(idx), qualified_tbl.n_pos(idx), qualified_tbl.n_neg(idx), qualified_tbl.delta_auc(idx), ...
        qualified_tbl.delta_ci_low(idx), qualified_tbl.delta_ci_high(idx));
end
fprintf(fid, '\n');

fprintf(fid, 'Interpretation:\n');
fprintf(fid, '- Stage 1 and Stage 2 paper-facing claims should cite the global / room-level tables above.\n');
fprintf(fid, '- Conditional cells are support-sensitive even after CV, so they stay supplemental.\n');
fprintf(fid, '- The legacy `eps_r x xpol` cell is reported here only to show that the old headline does not survive under the current CV + CI framing.\n');

function results = loadResults(mat_path)
    S = load(mat_path, 'results');
    results = S.results;
    results = results(~logical(results.failed), :);
end

function row = metricRow(scope, subset, tbl, y, cir_features, joint_features, n_boot)
    metrics = bootstrapMetrics(tbl, y, cir_features, joint_features, n_boot);
    row = {scope, subset, height(tbl), sum(y == 0), sum(y == 1), metrics.auc_cir, metrics.auc_joint, metrics.delta_auc, metrics.delta_ci(1), metrics.delta_ci(2)};
end

function metrics = bootstrapMetrics(tbl, y, cir_features, joint_features, n_boot)
    X_cir = table2array(tbl(:, cir_features));
    X_joint = table2array(tbl(:, joint_features));
    metrics = struct();
    metrics.auc_cir = analysis.cvLogisticAuc(X_cir, y);
    metrics.auc_joint = analysis.cvLogisticAuc(X_joint, y);
    metrics.delta_auc = metrics.auc_joint - metrics.auc_cir;

    deltas = nan(n_boot, 1);
    pos_idx = find(y == 1);
    neg_idx = find(y == 0);
    for b = 1:n_boot
        sample_idx = [ ...
            pos_idx(randi(numel(pos_idx), numel(pos_idx), 1)); ...
            neg_idx(randi(numel(neg_idx), numel(neg_idx), 1))];
        Xc = X_cir(sample_idx, :);
        Xj = X_joint(sample_idx, :);
        yb = y(sample_idx);
        auc_c = analysis.cvLogisticAuc(Xc, yb);
        auc_j = analysis.cvLogisticAuc(Xj, yb);
        deltas(b) = auc_j - auc_c;
    end
    metrics.delta_ci = quantile(deltas, [0.025, 0.975]);
end

function stats = cellCounts(results, x_base, x_label, y_base, y_label, label_var)
    subset = subsetForCell(results, x_base, x_label, y_base, y_label);
    y = logical(subset.(label_var));
    stats = struct('n', height(subset), 'n_pos', sum(y), 'n_neg', sum(~y));
end

function subset = subsetForCell(results, x_base, x_label, y_base, y_label)
    x_name = resolveColumnName(results, x_base);
    y_name = resolveColumnName(results, y_base);
    mask = cellMask(results.(x_name), x_label) & cellMask(results.(y_name), y_label);
    subset = results(mask, :);
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

function values = columnText(tbl, base_name)
    name = resolveColumnName(tbl, base_name);
    values = cellstr(string(tbl.(name)));
end
