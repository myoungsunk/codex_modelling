script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

stage2_dir = fullfile(repo_root, 'results', 'stage2');
src_mat = fullfile(stage2_dir, 'stage2_900_ffd.mat');
if exist(src_mat, 'file') ~= 2
    run(fullfile(repo_root, 'scripts', 'week4_day3_stage2_full.m'));
end

S = load(src_mat, 'results', 'cases', 'cfg', 'elapsed');
results = S.results;
cases = S.cases;
cfg = S.cfg;
elapsed = S.elapsed;

assert(ismember('has_los_path', results.Properties.VariableNames), 'results must contain has_los_path');
assert(ismember('bounce_to_los_ratio_mid', results.Properties.VariableNames), 'results must contain bounce_to_los_ratio_mid');

valid = ~logical(results.failed);
has_los = logical(results.has_los_path);
ratio = double(results.bounce_to_los_ratio_mid);

is_nlos_current = logical(results.is_nlos);
is_nlos_geo = ~has_los;
is_nlos_bounce_033 = has_los & (ratio >= 0.33);
is_nlos_mixed_033 = is_nlos_geo | is_nlos_bounce_033;

results_relabel = results;
results_relabel.is_nlos_current_0p20 = is_nlos_current;
results_relabel.is_los_current_0p20 = logical(results.is_los);
results_relabel.is_nlos_geo = is_nlos_geo;
results_relabel.is_los_geo = ~is_nlos_geo;
results_relabel.is_nlos_bounce_0p33 = is_nlos_bounce_033;
results_relabel.is_los_bounce_0p33 = has_los & ~is_nlos_bounce_033;
results_relabel.is_nlos_mixed_0p33 = is_nlos_mixed_033;
results_relabel.is_los_mixed_0p33 = ~is_nlos_mixed_033;

% Make mixed@0.33 the primary Stage 2 label in the relabeled export.
results_relabel.is_nlos = is_nlos_mixed_033;
results_relabel.is_los = ~is_nlos_mixed_033;
results_relabel.label_schema = repmat({'mixed_0p33_primary_with_dual_aux'}, height(results_relabel), 1);

out_mat = fullfile(stage2_dir, 'stage2_900_ffd_relabel.mat');
out_csv = fullfile(stage2_dir, 'stage2_900_ffd_relabel.csv');
summary_md = fullfile(stage2_dir, 'stage2_relabel_summary.md');
global_csv = fullfile(stage2_dir, 'stage2_relabel_global_metrics.csv');
room_csv = fullfile(stage2_dir, 'stage2_relabel_room_metrics.csv');

save(out_mat, 'results_relabel', 'cases', 'cfg', 'elapsed');
writetable(results_relabel, out_csv);

all_features = features.canonicalFeatureNames();
cp_features = all_features(1:6);
cir_features = all_features(7:end);
joint_features = all_features;

valid_tbl = results_relabel(valid, :);
room_col = columnText(valid_tbl, 'room_type');
rooms = {'A', 'B', 'C'};

global_tbl = table();
global_tbl = [global_tbl; metricRow(valid_tbl, 'ALL_mixed_0p33', cir_features, cp_features, joint_features, valid_tbl.is_nlos)]; %#ok<AGROW>
global_tbl = [global_tbl; metricRow(valid_tbl, 'ALL_geo_only', cir_features, cp_features, joint_features, valid_tbl.is_nlos_geo)]; %#ok<AGROW>

room_tbl = table();
for idx = 1:numel(rooms)
    room_mask = strcmp(room_col, rooms{idx});
    room_data = valid_tbl(room_mask, :);
    room_tbl = [room_tbl; metricRow(room_data, sprintf('%s_mixed_0p33', rooms{idx}), cir_features, cp_features, joint_features, room_data.is_nlos)]; %#ok<AGROW>
    room_tbl = [room_tbl; metricRow(room_data, sprintf('%s_geo_only', rooms{idx}), cir_features, cp_features, joint_features, room_data.is_nlos_geo)]; %#ok<AGROW>
end

writetable(global_tbl, global_csv);
writetable(room_tbl, room_csv);

fid = fopen(summary_md, 'w');
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Stage 2 Relabel Rerun\n\n');
fprintf(fid, '- source dataset: `%s`\n', src_mat);
fprintf(fid, '- relabeled dataset: `%s`\n', out_mat);
fprintf(fid, '- primary label in relabeled export: `mixed_0p33`\n');
fprintf(fid, '- auxiliary labels retained: `current_0p20`, `geo_only`, `bounce_0p33`\n\n');

fprintf(fid, '## Label Balance (valid cases only)\n\n');
fprintf(fid, '| label | n_neg | n_pos |\n');
fprintf(fid, '|---|---:|---:|\n');
fprintf(fid, '| current_0p20 | %d | %d |\n', sum(valid & ~is_nlos_current), sum(valid & is_nlos_current));
fprintf(fid, '| geo_only | %d | %d |\n', sum(valid & ~is_nlos_geo), sum(valid & is_nlos_geo));
fprintf(fid, '| bounce_0p33 | %d | %d |\n', sum(valid & ~(is_nlos_bounce_033)), sum(valid & is_nlos_bounce_033));
fprintf(fid, '| mixed_0p33 | %d | %d |\n\n', sum(valid & ~is_nlos_mixed_033), sum(valid & is_nlos_mixed_033));

fprintf(fid, '## Global Metrics\n\n');
fprintf(fid, '| set | n | n_los | n_nlos | auc_cir | auc_cp | auc_joint | delta_auc |\n');
fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|---:|\n');
for idx = 1:height(global_tbl)
    fprintf(fid, '| %s | %d | %d | %d | %s | %s | %s | %s |\n', ...
        global_tbl.label_name{idx}, global_tbl.n(idx), global_tbl.n_los(idx), global_tbl.n_nlos(idx), ...
        fmt(global_tbl.auc_cir(idx)), fmt(global_tbl.auc_cp(idx)), fmt(global_tbl.auc_joint(idx)), fmt(global_tbl.delta_auc(idx)));
end

fprintf(fid, '\n## Room Metrics\n\n');
fprintf(fid, '| set | n | n_los | n_nlos | auc_cir | auc_cp | auc_joint | delta_auc |\n');
fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|---:|\n');
for idx = 1:height(room_tbl)
    fprintf(fid, '| %s | %d | %d | %d | %s | %s | %s | %s |\n', ...
        room_tbl.label_name{idx}, room_tbl.n(idx), room_tbl.n_los(idx), room_tbl.n_nlos(idx), ...
        fmt(room_tbl.auc_cir(idx)), fmt(room_tbl.auc_cp(idx)), fmt(room_tbl.auc_joint(idx)), fmt(room_tbl.delta_auc(idx)));
end

fprintf('Saved:\n');
fprintf('  %s\n', out_mat);
fprintf('  %s\n', out_csv);
fprintf('  %s\n', summary_md);
fprintf('  %s\n', global_csv);
fprintf('  %s\n', room_csv);

function row = metricRow(tbl, label_name, cir_features, cp_features, joint_features, y_raw)
    y = double(logical(y_raw));
    auc_cir = fitAndAuc(tbl, cir_features, y);
    auc_cp = fitAndAuc(tbl, cp_features, y);
    auc_joint = fitAndAuc(tbl, joint_features, y);
    row = table({label_name}, height(tbl), sum(~y), sum(y), ...
        auc_cir, auc_cp, auc_joint, auc_joint - auc_cir, ...
        'VariableNames', {'label_name', 'n', 'n_los', 'n_nlos', 'auc_cir', 'auc_cp', 'auc_joint', 'delta_auc'});
end

function auc = fitAndAuc(tbl, feature_names, y)
    X = table2array(tbl(:, feature_names));
    auc = analysis.cvLogisticAuc(X, y);
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
