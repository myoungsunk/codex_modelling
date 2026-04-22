projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'week25');
matPath = fullfile(outDir, 'smoke_sweep_200.mat');
if exist(matPath, 'file') ~= 2
    run(fullfile(projectRoot, 'scripts', 'week25_smoke_sweep.m'));
end

S = load(matPath, 'results');
results = S.results;
results = results(~logical(results.failed), :);
nlos_rows = results(logical(results.is_nlos), :);

sd = std(nlos_rows.kurtosis_total);
lo = min(nlos_rows.kurtosis_total);
hi = max(nlos_rows.kurtosis_total);

reportPath = fullfile(outDir, 'nlos_kurtosis_diag.md');
fid = fopen(reportPath, 'w');
assert(fid ~= -1, 'Failed to open %s', reportPath);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, '# NLoS Kurtosis Diagnostic\n\n');
fprintf(fid, '- NLoS kurtosis std = %.3f\n', sd);
fprintf(fid, '- NLoS kurtosis range = [%.3f, %.3f]\n', lo, hi);
fprintf(fid, '- Verdict: **%s**\n', ternary(sd > 1.5, 'diversity_ok', 'diversity_low'));

fprintf('NLoS kurtosis std = %.3f, range = [%.3f, %.3f]\n', sd, lo, hi);
fprintf('Saved:\n');
fprintf('  %s\n', reportPath);

function out = ternary(cond, a, b)
    if cond
        out = a;
    else
        out = b;
    end
end
