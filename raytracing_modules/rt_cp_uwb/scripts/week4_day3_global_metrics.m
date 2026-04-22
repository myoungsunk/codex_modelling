script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

stage2_dir = fullfile(repo_root, 'results', 'stage2');
mat_path = fullfile(stage2_dir, 'stage2_900_ffd.mat');
if exist(mat_path, 'file') ~= 2
    run(fullfile(repo_root, 'scripts', 'week4_day3_stage2_full.m'));
end

S2 = load(mat_path, 'results');
results = S2.results;
results = results(~logical(results.failed), :);

all_features = features.canonicalFeatureNames();
cp_features = all_features(1:6);
cir_features = all_features(7:end);
joint_features = all_features;
room_col = columnText(results, 'room_type');
rooms = {'A', 'B', 'C'};

summary = table();
for idx = 1:numel(rooms)
    room_tbl = results(strcmp(room_col, rooms{idx}), :);
    summary = [summary; metricRow(room_tbl, rooms{idx}, cir_features, cp_features, joint_features)]; %#ok<AGROW>
end
summary = [summary; metricRow(results, 'ALL', cir_features, cp_features, joint_features)]; %#ok<AGROW>

stage1_auc = readtable(fullfile(repo_root, 'results', 'stage1', 'global_auc_summary.csv'));
stage1_summary = table( ...
    {'Stage1_ALL'}', ...
    NaN, NaN, NaN, ...
    stage1_auc.auc(strcmp(stage1_auc.model, 'CIR-only')), ...
    stage1_auc.auc(strcmp(stage1_auc.model, 'CP-only')), ...
    stage1_auc.auc(strcmp(stage1_auc.model, 'Joint')), ...
    stage1_auc.auc(strcmp(stage1_auc.model, 'Joint')) - stage1_auc.auc(strcmp(stage1_auc.model, 'CIR-only')), ...
    'VariableNames', {'room_type', 'n', 'n_los', 'n_nlos', 'auc_cir', 'auc_cp', 'auc_joint', 'delta_auc'});

compare_tbl = [stage1_summary; summary];

csv_path = fullfile(stage2_dir, 'global_metrics_stage2.csv');
compare_csv = fullfile(stage2_dir, 'stage1_vs_stage2_global_metrics.csv');
md_path = fullfile(stage2_dir, 'global_metrics_stage2.md');
plot_path = fullfile(stage2_dir, 'global_metrics_stage2.png');
writetable(summary, csv_path);
writetable(compare_tbl, compare_csv);

fig = figure('Visible', 'off', 'Position', [100 100 1100 450]);
auc_mat = [summary.auc_cir, summary.auc_cp, summary.auc_joint];
bar(categorical(summary.room_type), auc_mat);
ylabel('AUC');
legend({'CIR-only', 'CP-only', 'Joint'}, 'Location', 'northwest');
title('Stage 2 Global Metrics');
grid on;
saveas(fig, plot_path);
close(fig);

fid = fopen(md_path, 'w');
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Stage 2 Global Metrics\n\n');
fprintf(fid, '| room | n | n_los | n_nlos | auc_cir | auc_cp | auc_joint | delta_auc |\n');
fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|---:|\n');
for idx = 1:height(summary)
    fprintf(fid, '| %s | %d | %d | %d | %s | %s | %s | %s |\n', ...
        summary.room_type{idx}, summary.n(idx), summary.n_los(idx), summary.n_nlos(idx), ...
        fmt(summary.auc_cir(idx)), fmt(summary.auc_cp(idx)), fmt(summary.auc_joint(idx)), fmt(summary.delta_auc(idx)));
end
fprintf(fid, '\n## Stage 1 vs Stage 2\n\n');
fprintf(fid, '| set | auc_cir | auc_cp | auc_joint | delta_auc |\n');
fprintf(fid, '|---|---:|---:|---:|---:|\n');
for idx = 1:height(compare_tbl)
    fprintf(fid, '| %s | %s | %s | %s | %s |\n', ...
        compare_tbl.room_type{idx}, ...
        fmt(compare_tbl.auc_cir(idx)), fmt(compare_tbl.auc_cp(idx)), fmt(compare_tbl.auc_joint(idx)), fmt(compare_tbl.delta_auc(idx)));
end

function row = metricRow(tbl, label, cir_features, cp_features, joint_features)
    row = table({label}, height(tbl), sum(logical(tbl.is_los)), sum(logical(tbl.is_nlos)), ...
        fitAndAuc(tbl, cir_features), fitAndAuc(tbl, cp_features), fitAndAuc(tbl, joint_features), ...
        fitAndAuc(tbl, joint_features) - fitAndAuc(tbl, cir_features), ...
        'VariableNames', {'room_type', 'n', 'n_los', 'n_nlos', 'auc_cir', 'auc_cp', 'auc_joint', 'delta_auc'});
end

function auc = fitAndAuc(tbl, feature_names)
    X = table2array(tbl(:, feature_names));
    y = double(logical(tbl.is_nlos));
    valid = all(isfinite(X), 2) & isfinite(y);
    X = X(valid, :);
    y = y(valid);
    if size(X, 1) < 10 || numel(unique(y)) < 2
        auc = NaN;
        return;
    end
    mu = mean(X, 1);
    sigma = std(X, 0, 1);
    sigma(sigma < 1e-9) = 1.0;
    X = (X - mu) ./ sigma;
    warn_state = warning;
    cleanup = onCleanup(@() warning(warn_state)); %#ok<NASGU>
    warning('off', 'all');
    mdl = fitglm(X, y, 'Distribution', 'binomial', 'Link', 'logit');
    pred = predict(mdl, X);
    [~, ~, ~, auc] = perfcurve(y, pred, 1);
end

function values = columnText(tbl, base_name)
    names = resolveColumnNames(tbl, base_name);
    values = cellstr(string(tbl.(names{1})));
end

function names = resolveColumnNames(tbl, base_name)
    direct = tbl.Properties.VariableNames(strcmp(tbl.Properties.VariableNames, base_name));
    if ~isempty(direct)
        names = direct;
        return;
    end
    prefixed = tbl.Properties.VariableNames(startsWith(tbl.Properties.VariableNames, [base_name '_']));
    if ~isempty(prefixed)
        names = prefixed;
        return;
    end
    error('Column %s not found', base_name);
end

function out = fmt(x)
    if isfinite(x)
        out = sprintf('%.4f', x);
    else
        out = 'NaN';
    end
end
