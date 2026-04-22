projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'stage1');
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

cases = sweep.designLhsSweep(3000, 42);
tStart = tic;
results = sweep.runSweepBatch(cases, cfg, true);
elapsed = toc(tStart);

matPath = fullfile(outDir, 'stage1_3000_ffd.mat');
csvPath = fullfile(outDir, 'stage1_3000_ffd.csv');
save(matPath, 'results', 'cases', 'cfg', 'elapsed');
writetable(results, csvPath);

valid = ~logical(results.failed);
summaryPath = fullfile(outDir, 'stage1_3000_ffd_summary.md');
fid = fopen(summaryPath, 'w');
assert(fid ~= -1, 'Failed to open %s', summaryPath);
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Stage 1 Full Sweep\n\n');
fprintf(fid, '- cases: %d\n', height(cases));
fprintf(fid, '- elapsed_s: %.6f\n', elapsed);
fprintf(fid, '- elapsed_min: %.6f\n', elapsed / 60.0);
fprintf(fid, '- per_case_ms: %.6f\n', elapsed / height(cases) * 1000.0);
fprintf(fid, '- failed: %d\n', sum(logical(results.failed)));
fprintf(fid, '- valid_los: %d\n', sum(logical(results.is_los) & valid));
fprintf(fid, '- valid_nlos: %d\n', sum(logical(results.is_nlos) & valid));

fprintf('Saved:\n');
fprintf('  %s\n', matPath);
fprintf('  %s\n', csvPath);
fprintf('  %s\n', summaryPath);
fprintf('Elapsed %.2f min, per case %.3f ms\n', elapsed / 60.0, elapsed / height(cases) * 1000.0);
