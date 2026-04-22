projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'week25');
csvPath = fullfile(outDir, 'smoke_sweep_200.csv');
if exist(csvPath, 'file') ~= 2
    run(fullfile(projectRoot, 'scripts', 'week25_smoke_sweep.m'));
end

results = readtable(csvPath);
results = results(~logical(results.failed), :);
y = double(logical(results.is_nlos));
feature_cols = features.canonicalFeatureNames();

feature_out = cell(numel(feature_cols), 1);
auc_out = nan(numel(feature_cols), 1);
n_valid_out = zeros(numel(feature_cols), 1);

for i = 1:numel(feature_cols)
    feature_out{i} = feature_cols{i};
    x = double(results.(feature_cols{i}));
    valid = isfinite(x) & isfinite(y);
    x = x(valid);
    y_valid = y(valid);
    n_valid_out(i) = numel(x);
    if numel(x) < 2 || numel(unique(y_valid)) < 2
        continue;
    end
    [~, ~, ~, auc_out(i)] = perfcurve(y_valid, x, 1);
end

auc_table = table(feature_out, auc_out, n_valid_out, 'VariableNames', {'feature', 'AUC', 'n_valid'});
auc_table = sortrows(auc_table, 'AUC', 'descend');
disp(auc_table);

outCsv = fullfile(outDir, 'initial_auc.csv');
writetable(auc_table, outCsv);
fprintf('Saved:\n');
fprintf('  %s\n', outCsv);
