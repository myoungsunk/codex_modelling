projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
week2Dir = fullfile(cfg.results_dir, 'week2');
csvPath = fullfile(week2Dir, 'smoke_sweep_200.csv');
assert(exist(csvPath, 'file') == 2, 'Missing smoke sweep CSV: %s', csvPath);

results = readtable(csvPath);
if ismember('failed', results.Properties.VariableNames)
    results = results(~logical(results.failed), :);
end

y = double(logical(results.is_nlos));
feature_cols = {'gamma_cp_1_freq_avg', 'gamma_cp_2_freq_db', ...
    'gamma_cp_3_fp_only', 'gamma_cp_4_total_energy', 'gamma_cp_5_post_fp', ...
    'gamma_cp_6_phase_circvar', 'a_fp_1_norm_energy', 'a_fp_2_peak_to_total', ...
    'a_fp_3_peak_to_max', 'a_fp_4_kurt_local', 'a_fp_5_rise_time', ...
    'a_fp_6_fp_to_2nd_peak', 'rms_delay_spread', 'kurtosis_total', ...
    'k_factor_estimate', 'peak_to_avg_ratio'};

feature_out = cell(numel(feature_cols), 1);
auc_out = NaN(numel(feature_cols), 1);
n_valid_out = zeros(numel(feature_cols), 1);

for i = 1:numel(feature_cols)
    feature_out{i} = feature_cols{i};
    x = double(results.(feature_cols{i}));
    valid = isfinite(x) & isfinite(y);
    x = x(valid);
    y_valid = y(valid);
    n_valid_out(i) = numel(x);
    if numel(unique(y_valid)) < 2 || numel(x) < 2
        auc_out(i) = NaN;
        continue;
    end
    [~, ~, ~, auc_out(i)] = perfcurve(y_valid, x, 1);
end

auc_table = table(feature_out, auc_out, n_valid_out, ...
    'VariableNames', {'feature', 'AUC', 'n_valid'});
auc_table = sortrows(auc_table, 'AUC', 'descend');
disp(auc_table);

csvOut = fullfile(week2Dir, 'initial_auc.csv');
writetable(auc_table, csvOut);

fprintf('Saved:\n');
fprintf('  %s\n', csvOut);
