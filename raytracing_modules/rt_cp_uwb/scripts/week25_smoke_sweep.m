projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'week25');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

cases = sweep.designLhsSweep(200, 42);
results = sweep.runSweepBatch(cases, cfg);
valid = ~logical(results.failed);

for i = 1:width(results)
    name = results.Properties.VariableNames{i};
    v = results.(name);
    if isnumeric(v)
        if islogical(v) || strcmp(name, 'case_id')
            continue;
        end
        n_bad = sum(~isfinite(v(valid)));
        if n_bad > 0
            fprintf('WARN: %s has %d non-finite values among non-failed rows\n', name, n_bad);
        end
    end
end

fprintf('LoS: %d, NLoS: %d, Failed: %d\n', sum(results.is_los), sum(results.is_nlos), sum(results.failed));

csvPath = fullfile(outDir, 'smoke_sweep_200.csv');
matPath = fullfile(outDir, 'smoke_sweep_200.mat');
writetable(results, csvPath);
save(matPath, 'results', 'cases');

fprintf('Saved:\n');
fprintf('  %s\n', csvPath);
fprintf('  %s\n', matPath);
