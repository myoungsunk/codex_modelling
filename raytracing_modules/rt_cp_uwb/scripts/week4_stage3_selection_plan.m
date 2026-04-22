projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
stage1Dir = fullfile(cfg.results_dir, 'stage1');
matPath = fullfile(stage1Dir, 'stage1_3000_ffd.mat');
if exist(matPath, 'file') ~= 2
    run(fullfile(projectRoot, 'scripts', 'week3_day5_stage1_full.m'));
end

S = load(matPath, 'results');
results = S.results;
results = results(~logical(results.failed), :);
patch = results(strcmp(cellstr(string(results.antenna_type)), 'patch_ffd'), :);

cir = { ...
    'rms_delay_spread', 'mean_excess_delay', 'max_excess_delay', ...
    'fp_to_total_ratio', 'rise_time_fp', 'fp_kurtosis', ...
    'kurtosis_total', 'skewness_total', 'energy_concentration_50ns', ...
    'num_significant_peaks', 'peak_to_avg_ratio', 'k_factor_estimate'};
joint = [cir, { ...
    'gamma_cp_1_freq_avg', 'gamma_cp_2_freq_db', 'gamma_cp_3_fp_only', ...
    'gamma_cp_6_phase_circvar', 'a_fp_2_peak_to_total', 'a_fp_6_fp_to_2nd_peak'}];

strong_specs = {
    struct('x_var', 'los_angle_from_anchor_bore_deg_results_tbl', 'y_var', 'slab_placement_cases_tbl', 'x_label', '[38.69, 45.86]', 'y_label', 'floor', 'group', 'G1');
    struct('x_var', 'los_angle_from_anchor_bore_deg_results_tbl', 'y_var', 'xpol_coupling_db', 'x_label', '[38.69, 45.86]', 'y_label', '[31.80, 35.90]', 'group', 'G1');
    struct('x_var', 'los_angle_from_anchor_bore_deg_results_tbl', 'y_var', 'xpol_coupling_db', 'x_label', '[1.27, 29.65]', 'y_label', '[23.97, 27.71]', 'group', 'G1')};

weak_specs = {
    struct('x_var', 'slab_placement_cases_tbl', 'y_var', 'snr_db', 'x_label', 'ceiling', 'y_label', '[34.34, 39.99]', 'group', 'G2');
    struct('x_var', 'los_angle_from_anchor_bore_deg_results_tbl', 'y_var', 'slab_placement_cases_tbl', 'x_label', '[38.69, 45.86]', 'y_label', 'ceiling', 'group', 'G2');
    struct('x_var', 'slab_placement_cases_tbl', 'y_var', 'snr_db', 'x_label', 'wall_y', 'y_label', '[28.54, 34.34]', 'group', 'G2')};

[strong_cells, strong_cases] = buildGroup(patch, cir, joint, strong_specs, 'top_delta_auc_repro');
[weak_cells, weak_cases] = buildGroup(patch, cir, joint, weak_specs, 'cp_weak_regime');

group3_notes = { ...
    'Reserve 15 Stage 3 slots for top Stage 2 room-regime Delta AUC cells once Week 4 results exist', ...
    'Prioritize room/position regimes where patch_ffd Joint - CIR lift is strongest under richer multipath', ...
    'Use 5 representative cases per selected Stage 2 regime (3 regimes x 5 cases)'};

cellsPath = fullfile(stage1Dir, 'stage3_selection_cells.csv');
casesPath = fullfile(stage1Dir, 'stage3_selection_candidates.csv');
writetable([strong_cells; weak_cells], cellsPath);
writetable([strong_cases; weak_cases], casesPath);

mdPath = fullfile(stage1Dir, 'stage3_selection_criteria.md');
fid = fopen(mdPath, 'w');
assert(fid ~= -1, 'Failed to open %s', mdPath);
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Stage 3 HFSS SBR+ Selection Criteria\n\n');
fprintf(fid, '## Principle\n\n');
fprintf(fid, '- Total Stage 3 target: 45 cases\n');
fprintf(fid, '- Group 1: 15 cases from Stage 1 top patch-only Delta AUC regimes\n');
fprintf(fid, '- Group 2: 15 cases from Stage 1 patch-weak regimes\n');
fprintf(fid, '- Group 3: 15 cases reserved for Stage 2 top room-regime Delta AUC cells\n\n');

fprintf(fid, '## Group 1: Stage 1 Top Delta AUC Regimes\n\n');
fprintf(fid, '| Regime | n | AUC CIR | AUC Joint | Delta AUC |\n');
fprintf(fid, '| --- | ---: | ---: | ---: | ---: |\n');
for i = 1:height(strong_cells)
    fprintf(fid, '| %s | %d | %.4f | %.4f | %.4f |\n', ...
        char(string(strong_cells.regime_desc{i})), strong_cells.n(i), strong_cells.auc_cir(i), strong_cells.auc_joint(i), strong_cells.delta_auc(i));
end
fprintf(fid, '\nSelection rule: choose 5 representative samples per regime by nearest-to-medoid distance within the cell.\n\n');

fprintf(fid, '## Group 2: Patch-Weak Regimes\n\n');
fprintf(fid, '| Regime | n | AUC CIR | AUC Joint | Delta AUC |\n');
fprintf(fid, '| --- | ---: | ---: | ---: | ---: |\n');
for i = 1:height(weak_cells)
    fprintf(fid, '| %s | %d | %.4f | %.4f | %.4f |\n', ...
        char(string(weak_cells.regime_desc{i})), weak_cells.n(i), weak_cells.auc_cir(i), weak_cells.auc_joint(i), weak_cells.delta_auc(i));
end
fprintf(fid, '\nSelection rule: choose 5 representative samples per weak regime to test whether weakness is scene-driven or antenna-driven.\n\n');

fprintf(fid, '## Group 3: Stage 2 Deferred Group\n\n');
for i = 1:numel(group3_notes)
    fprintf(fid, '- %s\n', group3_notes{i});
end
fprintf(fid, '\n');
fprintf(fid, '## Outputs\n\n');
fprintf(fid, '- Cell summary CSV: `%s`\n', cellsPath);
fprintf(fid, '- Candidate case CSV: `%s`\n', casesPath);

fprintf('Saved:\n');
fprintf('  %s\n', mdPath);
fprintf('  %s\n', cellsPath);
fprintf('  %s\n', casesPath);

function [cell_tbl, case_tbl] = buildGroup(patch, cir, joint, specs, reason)
    cell_rows = {};
    case_tables = cell(numel(specs), 1);
    for i = 1:numel(specs)
        spec = specs{i};
        cell_info = describeCell(patch, cir, joint, spec);
        cell_rows(end + 1, :) = { ... %#ok<AGROW>
            spec.group, spec.x_var, spec.y_var, spec.x_label, spec.y_label, ...
            cell_info.regime_desc, cell_info.n, cell_info.auc_cir, cell_info.auc_joint, cell_info.delta_auc};
        case_tables{i} = selectRepresentativeCases(cell_info.rows, spec.group, reason, cell_info.regime_desc);
    end
    cell_tbl = cell2table(cell_rows, 'VariableNames', ...
        {'group', 'x_var', 'y_var', 'x_label', 'y_label', 'regime_desc', 'n', 'auc_cir', 'auc_joint', 'delta_auc'});
    case_tbl = vertcat(case_tables{:});
end

function cell_info = describeCell(patch, cir, joint, spec)
    mask = matchCell(patch, spec.x_var, spec.x_label) & matchCell(patch, spec.y_var, spec.y_label);
    tbl = patch(mask, :);
    y = double(logical(tbl.is_nlos));
    X_cir = table2array(tbl(:, cir));
    X_joint = table2array(tbl(:, joint));
    auc_cir = fitAndAucLocal(X_cir, y);
    auc_joint = fitAndAucLocal(X_joint, y);
    cell_info = struct();
    cell_info.rows = tbl;
    cell_info.n = height(tbl);
    cell_info.auc_cir = auc_cir;
    cell_info.auc_joint = auc_joint;
    cell_info.delta_auc = auc_joint - auc_cir;
    cell_info.regime_desc = sprintf('%s=%s, %s=%s', prettyName(spec.x_var), spec.x_label, prettyName(spec.y_var), spec.y_label);
end

function tf = matchCell(tbl, var_name, label)
    values = tbl.(var_name);
    if isnumeric(values)
        bounds = parseBounds(label);
        tf = values >= bounds(1) & values <= bounds(2);
    else
        tf = strcmp(cellstr(string(values)), char(string(label)));
    end
end

function bounds = parseBounds(label)
    tokens = sscanf(char(string(label)), '[%f, %f]');
    if numel(tokens) ~= 2
        error('Failed to parse numeric bounds: %s', char(string(label)));
    end
    bounds = double(tokens(:)).';
end

function case_tbl = selectRepresentativeCases(tbl, group_name, reason, regime_desc)
    rep_count = min(5, height(tbl));
    cont_names = {'los_angle_from_anchor_bore_deg_results_tbl', 'eps_r', 'xpol_coupling_db', 'snr_db', 'gamma_cp_3_fp_only'};
    cont_names = cont_names(ismember(cont_names, tbl.Properties.VariableNames));
    X = table2array(tbl(:, cont_names));
    mu = median(X, 1, 'omitnan');
    sigma = std(X, 0, 1, 'omitnan');
    sigma(sigma < 1e-9) = 1.0;
    d = sqrt(sum(((X - mu) ./ sigma) .^ 2, 2));
    [~, order] = sort(d, 'ascend');
    selected = tbl(order(1:rep_count), :);
    case_tbl = table();
    case_tbl.group = repmat({group_name}, rep_count, 1);
    case_tbl.reason = repmat({reason}, rep_count, 1);
    case_tbl.regime_desc = repmat({regime_desc}, rep_count, 1);
    case_tbl.case_id = selected.case_id;
    case_tbl.slab_placement = cellstr(string(selected.slab_placement_cases_tbl));
    case_tbl.los_angle_deg = selected.los_angle_from_anchor_bore_deg_results_tbl;
    case_tbl.eps_r = selected.eps_r;
    case_tbl.xpol_coupling_db = selected.xpol_coupling_db;
    case_tbl.snr_db = selected.snr_db;
    case_tbl.gamma_cp_3_fp_only = selected.gamma_cp_3_fp_only;
    case_tbl.is_nlos = logical(selected.is_nlos);
end

function auc = fitAndAucLocal(X, y)
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

function text = prettyName(name)
    text = strrep(char(string(name)), '_results_tbl', '');
    text = strrep(text, '_cases_tbl', '');
    text = strrep(text, '_deg', '');
    text = strrep(text, '_', ' ');
end
