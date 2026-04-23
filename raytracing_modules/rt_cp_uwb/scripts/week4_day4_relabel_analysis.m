script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

stage2_dir = fullfile(repo_root, 'results', 'stage2');
mat_path = fullfile(stage2_dir, 'stage2_900_ffd.mat');
if exist(mat_path, 'file') ~= 2
    run(fullfile(repo_root, 'scripts', 'week4_day3_stage2_full.m'));
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

label_defs = {
    struct('name', 'current_0p20', 'display', 'Current ratio >= 0.20', 'mask', true(height(results), 1), ...
        'y', logical(results.bounce_to_los_ratio_mid >= 0.20));
    struct('name', 'geo_only', 'display', 'Geo only (has_los_path == 0)', 'mask', true(height(results), 1), ...
        'y', ~logical(results.has_los_path));
    struct('name', 'bounce_0p33_visible', 'display', 'Bounce only (visible LoS, ratio >= 0.33)', ...
        'mask', logical(results.has_los_path), ...
        'y', logical(results.has_los_path) & logical(results.bounce_to_los_ratio_mid >= 0.33));
    struct('name', 'mixed_0p33', 'display', 'Mixed geo OR bounce@0.33', 'mask', true(height(results), 1), ...
        'y', (~logical(results.has_los_path)) | (logical(results.has_los_path) & logical(results.bounce_to_los_ratio_mid >= 0.33)))};

auc_rows = {};
room_rows = {};
disagreement_rows = {};
label_rows = {};

for idx = 1:numel(label_defs)
    spec = label_defs{idx};
    tbl = results(spec.mask, :);
    room_values = room_col(spec.mask);
    y = double(spec.y(spec.mask));

    [auc_cir, pred_cir] = fitAndPredict(tbl, cir_features, y);
    [auc_cp, pred_cp] = fitAndPredict(tbl, cp_features, y);
    [auc_joint, pred_joint] = fitAndPredict(tbl, joint_features, y);
    auc_rise = singleFeatureAuc(tbl, 'rise_time_fp', y);
    auc_gamma = singleFeatureAuc(tbl, 'gamma_cp_3_fp_only', y);

    auc_rows(end + 1, :) = {spec.name, spec.display, height(tbl), sum(y == 0), sum(y == 1), auc_cir, auc_cp, auc_joint, auc_joint - auc_cir, auc_rise, auc_gamma}; %#ok<AGROW>
    label_rows(end + 1, :) = {spec.name, spec.display, 'ALL', height(tbl), sum(y == 0), sum(y == 1)}; %#ok<AGROW>

    [cir_fail_cp_save, cp_fail_cir_save] = disagreementCounts(y, pred_cir, pred_cp);
    disagreement_rows(end + 1, :) = {spec.name, 'ALL', cir_fail_cp_save, cp_fail_cir_save}; %#ok<AGROW>

    for room_idx = 1:numel(rooms)
        room_mask = strcmp(room_values, rooms{room_idx});
        y_room = y(room_mask);
        tbl_room = tbl(room_mask, :);
        auc_cir_room = fitOnly(tbl_room, cir_features, y_room);
        auc_cp_room = fitOnly(tbl_room, cp_features, y_room);
        auc_joint_room = fitOnly(tbl_room, joint_features, y_room);
        room_rows(end + 1, :) = {spec.name, rooms{room_idx}, height(tbl_room), sum(y_room == 0), sum(y_room == 1), ...
            auc_cir_room, auc_cp_room, auc_joint_room, auc_joint_room - auc_cir_room}; %#ok<AGROW>
        label_rows(end + 1, :) = {spec.name, spec.display, rooms{room_idx}, height(tbl_room), sum(y_room == 0), sum(y_room == 1)}; %#ok<AGROW>
    end
end

auc_tbl = cell2table(auc_rows, 'VariableNames', {'label_name', 'label_display', 'n', 'n_neg', 'n_pos', 'auc_cir', 'auc_cp', 'auc_joint', 'delta_auc', 'auc_rise_time_fp', 'auc_gamma_cp_3'});
room_tbl = cell2table(room_rows, 'VariableNames', {'label_name', 'room_type', 'n', 'n_neg', 'n_pos', 'auc_cir', 'auc_cp', 'auc_joint', 'delta_auc'});
dis_tbl = cell2table(disagreement_rows, 'VariableNames', {'label_name', 'room_type', 'cir_fail_cp_save', 'cp_fail_cir_save'});
label_tbl = cell2table(label_rows, 'VariableNames', {'label_name', 'label_display', 'room_type', 'n', 'n_neg', 'n_pos'});

auc_csv = fullfile(stage2_dir, 'label_reanalysis_auc.csv');
room_csv = fullfile(stage2_dir, 'label_reanalysis_room_auc.csv');
dis_csv = fullfile(stage2_dir, 'label_reanalysis_disagreement.csv');
label_csv = fullfile(stage2_dir, 'label_reanalysis_balance.csv');
md_path = fullfile(stage2_dir, 'label_reanalysis_report.md');
plot_path = fullfile(stage2_dir, 'label_reanalysis_auc.png');
writetable(auc_tbl, auc_csv);
writetable(room_tbl, room_csv);
writetable(dis_tbl, dis_csv);
writetable(label_tbl, label_csv);

fig = figure('Visible', 'off', 'Position', [100 100 1200 500]);
subplot(1, 2, 1);
bar(categorical(auc_tbl.label_name), [auc_tbl.auc_cir, auc_tbl.auc_cp, auc_tbl.auc_joint]);
ylabel('AUC');
legend({'CIR-only', 'CP-only', 'Joint'}, 'Location', 'northwest');
title('Label Redefinition AUC Comparison');
grid on;

subplot(1, 2, 2);
bar(categorical(auc_tbl.label_name), [auc_tbl.auc_rise_time_fp, auc_tbl.auc_gamma_cp_3]);
ylabel('Univariate AUC');
legend({'rise\_time\_fp', 'gamma\_cp\_3'}, 'Location', 'northwest');
title('Single-Feature Leakage Check');
grid on;
saveas(fig, plot_path);
close(fig);

fid = fopen(md_path, 'w');
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Stage 2 Label Reanalysis\n\n');
fprintf(fid, '## Overall Comparison\n\n');
fprintf(fid, '| label | n | n_neg | n_pos | auc_cir | auc_cp | auc_joint | delta_auc | auc_rise_time_fp | auc_gamma_cp_3 |\n');
fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n');
for idx = 1:height(auc_tbl)
    fprintf(fid, '| %s | %d | %d | %d | %s | %s | %s | %s | %s | %s |\n', ...
        auc_tbl.label_display{idx}, auc_tbl.n(idx), auc_tbl.n_neg(idx), auc_tbl.n_pos(idx), ...
        fmt(auc_tbl.auc_cir(idx)), fmt(auc_tbl.auc_cp(idx)), fmt(auc_tbl.auc_joint(idx)), fmt(auc_tbl.delta_auc(idx)), ...
        fmt(auc_tbl.auc_rise_time_fp(idx)), fmt(auc_tbl.auc_gamma_cp_3(idx)));
end

fprintf(fid, '\n## Room-wise Mixed@0.33\n\n');
fprintf(fid, '| room | n | n_neg | n_pos | auc_cir | auc_cp | auc_joint | delta_auc |\n');
fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|---:|\n');
mixed_rows = strcmp(room_tbl.label_name, 'mixed_0p33');
mixed_tbl = room_tbl(mixed_rows, :);
for idx = 1:height(mixed_tbl)
    fprintf(fid, '| %s | %d | %d | %d | %s | %s | %s | %s |\n', ...
        mixed_tbl.room_type{idx}, mixed_tbl.n(idx), mixed_tbl.n_neg(idx), mixed_tbl.n_pos(idx), ...
        fmt(mixed_tbl.auc_cir(idx)), fmt(mixed_tbl.auc_cp(idx)), fmt(mixed_tbl.auc_joint(idx)), fmt(mixed_tbl.delta_auc(idx)));
end

fprintf(fid, '\n## Disagreement Asymmetry\n\n');
fprintf(fid, '| label | cir_fail_cp_save | cp_fail_cir_save |\n');
fprintf(fid, '|---|---:|---:|\n');
for idx = 1:height(dis_tbl)
    fprintf(fid, '| %s | %d | %d |\n', dis_tbl.label_name{idx}, dis_tbl.cir_fail_cp_save(idx), dis_tbl.cp_fail_cir_save(idx));
end

fprintf(fid, '\n## Recommendation\n\n');
fprintf(fid, '- Preferred decomposition: `is_nlos_geo = ~has_los_path` and `is_nlos_bounce = has_los_path & (bounce_to_los_ratio_mid >= 0.33)`.\n');
fprintf(fid, '- Reporting aggregate label: `mixed_0p33 = is_nlos_geo OR is_nlos_bounce`.\n');
fprintf(fid, '- Avoid keeping `current_0p20` as the main Stage 2 label because `rise_time_fp` remains too close to the label-defining physics.\n');

fprintf('Saved:\n');
fprintf('  %s\n', auc_csv);
fprintf('  %s\n', room_csv);
fprintf('  %s\n', dis_csv);
fprintf('  %s\n', label_csv);
fprintf('  %s\n', md_path);

function [auc, pred] = fitAndPredict(tbl, feature_names, y)
    X = table2array(tbl(:, feature_names));
    [auc, pred] = analysis.cvLogisticAuc(X, y);
end

function auc = fitOnly(tbl, feature_names, y)
    [auc, ~] = fitAndPredict(tbl, feature_names, y);
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

function [cir_fail_cp_save, cp_fail_cir_save] = disagreementCounts(y, pred_cir, pred_cp)
    valid = isfinite(pred_cir) & isfinite(pred_cp) & isfinite(y);
    y = y(valid) > 0.5;
    cir_hat = pred_cir(valid) > 0.5;
    cp_hat = pred_cp(valid) > 0.5;
    cir_fail_cp_save = sum((cir_hat ~= y) & (cp_hat == y));
    cp_fail_cir_save = sum((cp_hat ~= y) & (cir_hat == y));
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
