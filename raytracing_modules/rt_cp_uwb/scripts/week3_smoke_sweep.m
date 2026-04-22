projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'stage1');
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

cases = sweep.designLhsSweep(200, 42);
tStart = tic;
results = sweep.runSweepBatch(cases, cfg, true);
elapsed = toc(tStart);

csvPath = fullfile(outDir, 'smoke_sweep_200.csv');
matPath = fullfile(outDir, 'smoke_sweep_200.mat');
writetable(results, csvPath);
save(matPath, 'results', 'cases', 'elapsed');

valid = ~logical(results.failed);
fprintf('Saved:\n');
fprintf('  %s\n', csvPath);
fprintf('  %s\n', matPath);
fprintf('Elapsed: %.3f s (%.3f ms/case)\n', elapsed, elapsed / height(cases) * 1000.0);
fprintf('Failed: %d / %d\n', sum(logical(results.failed)), height(results));
fprintf('LoS: %d, NLoS: %d\n', sum(logical(results.is_los) & valid), sum(logical(results.is_nlos) & valid));
