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

is_los = logical(results.is_los);
is_nlos = logical(results.is_nlos);

fig = figure('Visible', 'off', 'Position', [100 100 1200 900]);

subplot(2, 2, 1);
plotFeatureHist(results.gamma_cp_2_freq_db, is_los, is_nlos, '\gamma_{CP,2} (dB)', 'γ_CP distribution');

subplot(2, 2, 2);
plotFeatureHist(results.a_fp_2_peak_to_total, is_los, is_nlos, 'a_{FP,2}', 'a_FP distribution');

subplot(2, 2, 3);
plotFeatureHist(results.kurtosis_total, is_los, is_nlos, 'kurtosis', 'CIR kurtosis');

subplot(2, 2, 4);
numeric_cols = {'gamma_cp_2_freq_db', 'a_fp_2_peak_to_total', ...
    'kurtosis_total', 'rms_delay_spread', 'k_factor_estimate'};
numeric_data = table2array(results(:, numeric_cols));
corr_mat = corr(numeric_data, 'rows', 'pairwise');
imagesc(corr_mat);
axis square;
colorbar;
xticks(1:numel(numeric_cols));
xticklabels(numeric_cols);
xtickangle(45);
yticks(1:numel(numeric_cols));
yticklabels(numeric_cols);
title('Feature correlation');

pngPath = fullfile(week2Dir, 'smoke_diagnostics.png');
saveas(fig, pngPath);
close(fig);

fprintf('Saved:\n');
fprintf('  %s\n', pngPath);

function plotFeatureHist(x, is_los_mask, is_nlos_mask, xlabel_text, title_text)
    x = double(x);
    los_vals = x(is_los_mask & isfinite(x));
    nlos_vals = x(is_nlos_mask & isfinite(x));
    histogram(los_vals, 'FaceAlpha', 0.5); hold on;
    histogram(nlos_vals, 'FaceAlpha', 0.5);
    legend('LoS', 'NLoS', 'Location', 'best');
    xlabel(xlabel_text);
    title(title_text);
    grid on;
end
