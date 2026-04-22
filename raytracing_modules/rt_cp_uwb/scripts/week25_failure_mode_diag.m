projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'week25');
csvPath = fullfile(outDir, 'smoke_sweep_200.csv');
if exist(csvPath, 'file') ~= 2
    run(fullfile(projectRoot, 'scripts', 'week25_smoke_sweep.m'));
end

results = readtable(csvPath);
failed = results(logical(results.failed), :);

summary = groupsummary(failed, {'error_msg', 'antenna_type', 'los_blockage', 'num_slabs'});
summary = sortrows(summary, 'GroupCount', 'descend');

detailVars = intersect({'case_id', 'error_msg', 'antenna_type', 'los_blockage', 'num_slabs', ...
    'tx_height', 'rx_height', 'incidence_deg', 'tx_rx_dist_m', 'slab_size_m', 'slab_tilt_deg'}, ...
    failed.Properties.VariableNames, 'stable');
details = failed(:, detailVars);

summaryCsv = fullfile(outDir, 'failure_mode_summary.csv');
detailCsv = fullfile(outDir, 'failure_mode_cases.csv');
writetable(summary, summaryCsv);
writetable(details, detailCsv);

reportPath = fullfile(outDir, 'failure_mode_diag.md');
fid = fopen(reportPath, 'w');
assert(fid ~= -1, 'Failed to open %s', reportPath);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, '# Failure Mode Diagnostic\n\n');
fprintf(fid, '- total failed: %d / %d\n\n', height(failed), height(results));
fprintf(fid, '| error | antenna | blockage | num_slabs | count |\n');
fprintf(fid, '| --- | --- | --- | ---: | ---: |\n');
for i = 1:height(summary)
    fprintf(fid, '| %s | %s | %s | %d | %d |\n', ...
        summary.error_msg{i}, summary.antenna_type{i}, ternary(summary.los_blockage(i), 'true', 'false'), ...
        summary.num_slabs(i), summary.GroupCount(i));
end

fprintf('Saved:\n');
fprintf('  %s\n', summaryCsv);
fprintf('  %s\n', detailCsv);
fprintf('  %s\n', reportPath);

function out = ternary(cond, a, b)
    if cond
        out = a;
    else
        out = b;
    end
end
