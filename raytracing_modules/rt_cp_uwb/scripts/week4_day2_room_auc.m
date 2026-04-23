script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

results_dir = fullfile(repo_root, 'results', 'week4');
mat_path = fullfile(results_dir, 'smoke_stage2_225.mat');
if exist(mat_path, 'file') ~= 2
    run(fullfile(repo_root, 'scripts', 'week4_day2_stage2_smoke.m'));
end

S = load(mat_path, 'results');
results = S.results;
results = results(~logical(results.failed), :);

all_features = features.canonicalFeatureNames();
cp_features = all_features(1:6);
cir_features = all_features(7:end);
joint_features = all_features;

room_col = columnText(results, 'room_type');
rooms = {'A', 'B', 'C'};
summary = table();
feature_rows = {};

for idx = 1:numel(rooms)
    room_mask = strcmp(room_col, rooms{idx});
    room_tbl = results(room_mask, :);
    y = double(logical(room_tbl.is_nlos));

    auc_cir = fitAndAuc(room_tbl, cir_features);
    auc_cp = fitAndAuc(room_tbl, cp_features);
    auc_joint = fitAndAuc(room_tbl, joint_features);

    summary = [summary; table({rooms{idx}}, height(room_tbl), sum(room_tbl.is_los), sum(room_tbl.is_nlos), auc_cir, auc_cp, auc_joint, auc_joint - auc_cir, ... %#ok<AGROW>
        'VariableNames', {'room_type', 'n', 'n_los', 'n_nlos', 'auc_cir', 'auc_cp', 'auc_joint', 'delta_auc'})];

    for f_idx = 1:numel(joint_features)
        feature_name = joint_features{f_idx};
        auc_feat = singleFeatureAuc(room_tbl, feature_name, y);
        feature_rows(end + 1, :) = {rooms{idx}, feature_name, auc_feat}; %#ok<AGROW>
    end
end

feature_tbl = cell2table(feature_rows, 'VariableNames', {'room_type', 'feature', 'auc'});
feature_tbl = sortrows(feature_tbl, {'room_type', 'auc'}, {'ascend', 'descend'});

trend_flag = isStrictIncrease(summary.auc_cp);
if trend_flag
    trend_text = 'A < B < C confirmed for CP-only AUC';
else
    trend_text = 'CP-only AUC monotonic increase A < B < C not observed';
end

csv_summary = fullfile(results_dir, 'room_auc_summary.csv');
csv_feature = fullfile(results_dir, 'room_feature_auc.csv');
md_path = fullfile(results_dir, 'room_auc_report.md');
plot_path = fullfile(results_dir, 'room_auc_bars.png');
writetable(summary, csv_summary);
writetable(feature_tbl, csv_feature);

fig = figure('Visible', 'off', 'Position', [100 100 1000 480]);
auc_mat = [summary.auc_cir, summary.auc_cp, summary.auc_joint];
bar(categorical(summary.room_type), auc_mat);
ylabel('AUC');
legend({'CIR-only', 'CP-only', 'Joint'}, 'Location', 'northwest');
title('Stage 2 Smoke AUC by Room');
grid on;
saveas(fig, plot_path);
close(fig);

fid = fopen(md_path, 'w');
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Week 4 Day 2 Room-wise AUC\n\n');
fprintf(fid, '- verdict: **%s**\n\n', trend_text);
fprintf(fid, '| room | n | n_los | n_nlos | auc_cir | auc_cp | auc_joint | delta_auc |\n');
fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|---:|\n');
for idx = 1:height(summary)
    fprintf(fid, '| %s | %d | %d | %d | %s | %s | %s | %s |\n', ...
        summary.room_type{idx}, summary.n(idx), summary.n_los(idx), summary.n_nlos(idx), ...
        fmt(summary.auc_cir(idx)), fmt(summary.auc_cp(idx)), fmt(summary.auc_joint(idx)), fmt(summary.delta_auc(idx)));
end
fprintf(fid, '\n## Top Features Per Room\n\n');
for idx = 1:numel(rooms)
    room_rows = feature_tbl(strcmp(feature_tbl.room_type, rooms{idx}), :);
    room_rows = room_rows(1:min(5, height(room_rows)), :);
    fprintf(fid, '### Room %s\n\n', rooms{idx});
    fprintf(fid, '| feature | auc |\n');
    fprintf(fid, '|---|---:|\n');
    for row_idx = 1:height(room_rows)
        fprintf(fid, '| %s | %s |\n', room_rows.feature{row_idx}, fmt(room_rows.auc(row_idx)));
    end
    fprintf(fid, '\n');
end

function auc = fitAndAuc(tbl, feature_names)
    X = table2array(tbl(:, feature_names));
    y = double(logical(tbl.is_nlos));
    auc = analysis.cvLogisticAuc(X, y);
end

function auc = singleFeatureAuc(tbl, feature_name, y)
    x = double(tbl.(feature_name));
    valid = isfinite(x) & isfinite(y);
    x = x(valid);
    y = y(valid);
    if numel(unique(y)) < 2
        auc = NaN;
        return;
    end
    [~, ~, ~, auc] = perfcurve(y, x, 1);
end

function values = columnText(tbl, base_name)
    names = {base_name, [base_name '_cases_tbl'], [base_name '_results_tbl']};
    for i = 1:numel(names)
        if ismember(names{i}, tbl.Properties.VariableNames)
            values = cellstr(string(tbl.(names{i})));
            return;
        end
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

function tf = isStrictIncrease(x)
    x = double(x(:));
    tf = all(isfinite(x)) && all(diff(x) > 0);
end
