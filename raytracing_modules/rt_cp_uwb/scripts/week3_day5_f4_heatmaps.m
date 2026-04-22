projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'stage1');
matPath = fullfile(outDir, 'stage1_3000_ffd.mat');
if exist(matPath, 'file') ~= 2
    run(fullfile(projectRoot, 'scripts', 'week3_day5_stage1_full.m'));
end

S = load(matPath, 'results');
results = S.results;
results = results(~logical(results.failed), :);

cp_features = { ...
    'gamma_cp_1_freq_avg', ...
    'gamma_cp_2_freq_db', ...
    'gamma_cp_3_fp_only', ...
    'gamma_cp_6_phase_circvar', ...
    'a_fp_2_peak_to_total', ...
    'a_fp_6_fp_to_2nd_peak'};
cir_features = { ...
    'rms_delay_spread', ...
    'mean_excess_delay', ...
    'max_excess_delay', ...
    'fp_to_total_ratio', ...
    'rise_time_fp', ...
    'fp_kurtosis', ...
    'kurtosis_total', ...
    'skewness_total', ...
    'energy_concentration_50ns', ...
    'num_significant_peaks', ...
    'peak_to_avg_ratio', ...
    'k_factor_estimate'};
joint_features = [cir_features, cp_features];

global_auc = table( ...
    {'CIR-only'; 'CP-only'; 'Joint'}, ...
    [cvAuc(results, cir_features); cvAuc(results, cp_features); cvAuc(results, joint_features)], ...
    'VariableNames', {'model', 'auc'});
writetable(global_auc, fullfile(outDir, 'global_auc_summary.csv'));

axis_pairs = { ...
    'los_angle_from_anchor_bore_deg', 'eps_r'; ...
    'los_angle_from_anchor_bore_deg', 'xpol_coupling_db'; ...
    'los_angle_from_anchor_bore_deg', 'snr_db'; ...
    'los_angle_from_anchor_bore_deg', 'slab_placement'; ...
    'eps_r', 'xpol_coupling_db'; ...
    'eps_r', 'snr_db'; ...
    'xpol_coupling_db', 'snr_db'; ...
    'antenna_type', 'los_angle_from_anchor_bore_deg'; ...
    'antenna_type', 'slab_placement'; ...
    'slab_placement', 'snr_db'};

pair_results = cell(size(axis_pairs, 1), 1);
cell_tables = cell(size(axis_pairs, 1), 1);

fig = figure('Visible', 'off', 'Position', [100 100 1800 900]);
for i = 1:size(axis_pairs, 1)
    x_requested = axis_pairs{i, 1};
    y_requested = axis_pairs{i, 2};
    x_var = resolveVar(results, x_requested);
    y_var = resolveVar(results, y_requested);
    [delta_grid, cell_table, meta] = analysis.computeConditionalAucGrid(results, x_var, y_var, cir_features, joint_features, 'n_bins', 5, 'min_per_cell', 20);
    cell_table.x_var(:) = {x_requested};
    cell_table.y_var(:) = {y_requested};
    pair_results{i} = struct( ...
        'x_var', x_var, ...
        'y_var', y_var, ...
        'display_x_var', x_requested, ...
        'display_y_var', y_requested, ...
        'delta_grid', delta_grid, ...
        'meta', meta);
    cell_tables{i} = cell_table;

    subplot(2, 5, i);
    imagesc(delta_grid, [-0.15 0.15]);
    axis image;
    colormap(gca, parula(256));
    colorbar;
    title(sprintf('%s vs %s', prettyName(x_requested), prettyName(y_requested)), 'Interpreter', 'none');
    xticks(1:numel(meta.x.labels));
    xticklabels(trimLabels(meta.x.labels));
    xtickangle(45);
    yticks(1:numel(meta.y.labels));
    yticklabels(trimLabels(meta.y.labels));
end

plotPath = fullfile(outDir, 'conditional_auc_heatmaps.png');
saveas(fig, plotPath);
close(fig);

cell_table_all = vertcat(cell_tables{:});
writetable(cell_table_all, fullfile(outDir, 'conditional_auc_cells.csv'), 'WriteVariableNames', true);
save(fullfile(outDir, 'conditional_auc_heatmaps.mat'), 'pair_results', 'global_auc', 'cp_features', 'cir_features', 'joint_features', 'cell_table_all');

disagreement = analysis.computeDisagreementAnalysis(results, cir_features, cp_features);
save(fullfile(outDir, 'disagreement_analysis.mat'), 'disagreement');

mdPath = fullfile(outDir, 'conditional_auc_summary.md');
fid = fopen(mdPath, 'w');
assert(fid ~= -1, 'Failed to open %s', mdPath);
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Stage 1 Conditional AUC Summary\n\n');
fprintf(fid, '## Global AUC\n\n');
fprintf(fid, '| Model | AUC |\n');
fprintf(fid, '| --- | ---: |\n');
for i = 1:height(global_auc)
    fprintf(fid, '| %s | %.4f |\n', global_auc.model{i}, global_auc.auc(i));
end
fprintf(fid, '\n');
fprintf(fid, '## Top Conditional Cells\n\n');
top_cells = topConditionalCells(cell_table_all, 10);
fprintf(fid, '| Pair | X | Y | n | Delta AUC |\n');
fprintf(fid, '| --- | --- | --- | ---: | ---: |\n');
for i = 1:min(10, height(top_cells))
    fprintf(fid, '| %s x %s | %s | %s | %d | %.4f |\n', ...
        prettyName(top_cells.x_var{i}), prettyName(top_cells.y_var{i}), ...
        char(string(top_cells.x_label{i})), char(string(top_cells.y_label{i})), ...
        top_cells.n(i), top_cells.delta_auc(i));
end
fprintf(fid, '\n');
fprintf(fid, '## Disagreement\n\n');
fprintf(fid, '- CIR-only CV AUC: %.4f\n', disagreement.auc_cir_cv);
fprintf(fid, '- Misclassification rate: %.4f\n', disagreement.misc_rate);

fprintf('Saved:\n');
fprintf('  %s\n', plotPath);
fprintf('  %s\n', fullfile(outDir, 'conditional_auc_cells.csv'));
fprintf('  %s\n', mdPath);

function auc = cvAuc(results, feature_names)
    y_all = double(logical(results.is_nlos));
    X_all = table2array(results(:, feature_names));
    valid = all(isfinite(X_all), 2) & isfinite(y_all);
    X = X_all(valid, :);
    y = y_all(valid);
    cv = cvpartition(y, 'KFold', 5);
    pred = nan(size(y));
    for k = 1:cv.NumTestSets
        tr = training(cv, k);
        te = test(cv, k);
        Xtr = X(tr, :);
        Xte = X(te, :);
        ytr = y(tr);
        mu = mean(Xtr, 1);
        sigma = std(Xtr, 0, 1);
        sigma(sigma < 1e-9) = 1.0;
        Xtr = (Xtr - mu) ./ sigma;
        Xte = (Xte - mu) ./ sigma;
        warn_state = warning;
        cleanup = onCleanup(@() warning(warn_state)); %#ok<NASGU>
        warning('off', 'all');
        mdl = fitglm(Xtr, ytr, 'Distribution', 'binomial', 'Link', 'logit');
        pred(te) = predict(mdl, Xte);
    end
    [~, ~, ~, auc] = perfcurve(y, pred, 1);
end

function text = prettyName(name)
    text = strrep(char(string(name)), '_deg', '');
    text = strrep(text, '_', ' ');
end

function labels = trimLabels(labels)
    labels = cellfun(@(x) char(string(x)), labels, 'UniformOutput', false);
    for i = 1:numel(labels)
        if strlength(string(labels{i})) > 18
            labels{i} = char(extractBefore(string(labels{i}), 19) + "...");
        end
    end
end

function tbl = topConditionalCells(cell_table_all, top_n)
    mask = isfinite(cell_table_all.delta_auc);
    tbl = cell_table_all(mask, :);
    tbl = sortrows(tbl, {'delta_auc', 'n'}, {'descend', 'descend'});
    tbl = tbl(1:min(top_n, height(tbl)), :);
end

function name = resolveVar(results, desired_name)
    base = char(string(desired_name));
    candidates = {base, [base '_cases_tbl'], [base '_results_tbl']};
    for i = 1:numel(candidates)
        if ismember(candidates{i}, results.Properties.VariableNames)
            name = candidates{i};
            return;
        end
    end
    error('Variable not found: %s', desired_name);
end
