projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'week3');
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

cases = sweep.designLhsSweep(200, 42);
tStart = tic;
results = sweep.runSweepBatch(cases, cfg, true);
elapsed = toc(tStart);

csvPath = fullfile(outDir, 'smoke_200_ffd.csv');
matPath = fullfile(outDir, 'smoke_200_ffd.mat');
writetable(results, csvPath);
save(matPath, 'results', 'cases', 'elapsed');

valid = ~logical(results.failed);
numericVars = results.Properties.VariableNames(varfun(@isnumeric, results, 'OutputFormat', 'uniform'));
warnLines = strings(0, 1);
for i = 1:numel(numericVars)
    name = numericVars{i};
    values = results.(name);
    nBad = sum(~isfinite(values));
    if nBad > 0
        warnLines(end + 1, 1) = sprintf('%s has %d non-finite values', name, nBad); %#ok<AGROW>
    end
end

mdPath = fullfile(outDir, 'smoke_200_ffd_summary.md');
fid = fopen(mdPath, 'w');
assert(fid ~= -1, 'Failed to open %s', mdPath);
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Week 3 Day 4 Smoke FFD\n\n');
fprintf(fid, '- cases: %d\n', height(cases));
fprintf(fid, '- elapsed_s: %.6f\n', elapsed);
fprintf(fid, '- per_case_ms: %.6f\n', elapsed / height(cases) * 1000.0);
fprintf(fid, '- failed: %d\n', sum(logical(results.failed)));
fprintf(fid, '- valid_los: %d\n', sum(logical(results.is_los) & valid));
fprintf(fid, '- valid_nlos: %d\n', sum(logical(results.is_nlos) & valid));
fprintf(fid, '\n');
if isempty(warnLines)
    fprintf(fid, '- non-finite warning: none\n');
else
    fprintf(fid, '- non-finite warnings:\n');
    for i = 1:numel(warnLines)
        fprintf(fid, '  - %s\n', warnLines(i));
    end
end

fprintf('Saved:\n');
fprintf('  %s\n', csvPath);
fprintf('  %s\n', matPath);
fprintf('  %s\n', mdPath);
fprintf('Elapsed %.3f s, per case %.3f ms\n', elapsed, elapsed / height(cases) * 1000.0);
fprintf('Failed %d / %d\n', sum(logical(results.failed)), height(results));
fprintf('LoS %d, NLoS %d\n', sum(logical(results.is_los) & valid), sum(logical(results.is_nlos) & valid));
