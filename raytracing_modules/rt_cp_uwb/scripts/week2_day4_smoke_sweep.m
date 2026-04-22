projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
week2Dir = fullfile(cfg.results_dir, 'week2');
if ~exist(week2Dir, 'dir')
    mkdir(week2Dir);
end

cases = sweep.designLhsSweep(200, 42);
results = sweep.runSweepBatch(cases, cfg);

feature_cols = results.Properties.VariableNames;
for idx = 1:numel(feature_cols)
    name = feature_cols{idx};
    v = results.(name);
    if isnumeric(v)
        n_bad = sum(~isfinite(v));
        if n_bad > 0
            fprintf('WARN: %s has %d non-finite values\n', name, n_bad);
        end
    end
end

fprintf('LoS: %d, NLoS: %d\n', sum(results.is_los), sum(results.is_nlos));

csvPath = fullfile(week2Dir, 'smoke_sweep_200.csv');
matPath = fullfile(week2Dir, 'smoke_sweep_200.mat');
writetable(results, csvPath);
save(matPath, 'results');

fprintf('Saved:\n');
fprintf('  %s\n', csvPath);
fprintf('  %s\n', matPath);
