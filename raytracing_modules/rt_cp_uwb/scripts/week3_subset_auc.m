projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'stage1');
matPath = fullfile(outDir, 'smoke_sweep_200.mat');
if exist(matPath, 'file') ~= 2
    run(fullfile(projectRoot, 'scripts', 'week3_smoke_sweep.m'));
end

S = load(matPath, 'results');
results = S.results;
results = results(~logical(results.failed), :);

featureName = 'gamma_cp_3_fp_only';
subsets = {};

antennaTypes = {'ideal', 'patch'};
for ai = 1:numel(antennaTypes)
    mask = strcmp(results.antenna_type, antennaTypes{ai});
    subsets(end + 1, :) = subsetRow(results(mask, :), sprintf('%s_all', antennaTypes{ai}), featureName); %#ok<AGROW>
end

placements = unique(string(results.slab_placement_cases_tbl));
for ai = 1:numel(antennaTypes)
    for pi = 1:numel(placements)
        mask = strcmp(results.antenna_type, antennaTypes{ai}) & strcmp(string(results.slab_placement_cases_tbl), placements(pi));
        subsets(end + 1, :) = subsetRow(results(mask, :), sprintf('%s_%s', antennaTypes{ai}, char(placements(pi))), featureName); %#ok<AGROW>
    end
end

subsetTbl = cell2table(subsets, 'VariableNames', ...
    {'subset', 'n', 'n_los', 'n_nlos', 'mean_gamma_los', 'mean_gamma_nlos', 'auc_gamma_cp_3'});
csvPath = fullfile(outDir, 'subset_auc.csv');
writetable(subsetTbl, csvPath);

patchRow = strcmp(subsetTbl.subset, 'patch_all');
patchAuc = subsetTbl.auc_gamma_cp_3(patchRow);
verdict = 'additional_tuning_needed';
if isfinite(patchAuc) && patchAuc >= 0.65 && patchAuc <= 0.85
    verdict = 'target_range';
elseif isfinite(patchAuc) && patchAuc > 0.85
    verdict = 'too_easy';
elseif isfinite(patchAuc) && patchAuc < 0.65
    verdict = 'too_weak';
end

mdPath = fullfile(outDir, 'subset_auc_report.md');
fid = fopen(mdPath, 'w');
assert(fid ~= -1, 'Failed to open %s', mdPath);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Stage 1 Subset AUC\n\n');
fprintf(fid, 'Primary feature: `%s`\n\n', featureName);
fprintf(fid, '| subset | n | n_los | n_nlos | mean_gamma_los | mean_gamma_nlos | auc_gamma_cp_3 |\n');
fprintf(fid, '| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n');
for i = 1:height(subsetTbl)
    fprintf(fid, '| %s | %d | %d | %d | %s | %s | %s |\n', ...
        subsetTbl.subset{i}, subsetTbl.n(i), subsetTbl.n_los(i), subsetTbl.n_nlos(i), ...
        fmtNum(subsetTbl.mean_gamma_los(i)), fmtNum(subsetTbl.mean_gamma_nlos(i)), fmtNum(subsetTbl.auc_gamma_cp_3(i)));
end
fprintf(fid, '\n');
fprintf(fid, '- patch overall AUC verdict: **%s**\n', verdict);

disp(subsetTbl);

function row = subsetRow(tbl, subsetName, featureName)
    y = double(logical(tbl.is_nlos));
    x = double(tbl.(featureName));
    valid = isfinite(x) & isfinite(y);
    x = x(valid);
    y = y(valid);
    auc = NaN;
    if numel(x) >= 2 && numel(unique(y)) >= 2
        [~, ~, ~, auc] = perfcurve(y, x, 1);
    end
    row = {subsetName, height(tbl), sum(logical(tbl.is_los)), sum(logical(tbl.is_nlos)), localMean(tbl, featureName, false), localMean(tbl, featureName, true), auc};
end

function value = localMean(tbl, featureName, useNlos)
    if isempty(tbl)
        value = NaN;
        return;
    end
    mask = logical(tbl.is_nlos);
    if ~useNlos
        mask = logical(tbl.is_los);
    end
    x = double(tbl.(featureName));
    x = x(mask & isfinite(x));
    if isempty(x)
        value = NaN;
    else
        value = mean(x);
    end
end

function txt = fmtNum(x)
    if isfinite(x)
        txt = sprintf('%.4f', x);
    else
        txt = 'NaN';
    end
end
