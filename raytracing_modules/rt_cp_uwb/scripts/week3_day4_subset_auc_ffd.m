projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'week3');
matPath = fullfile(outDir, 'smoke_200_ffd.mat');
if exist(matPath, 'file') ~= 2
    run(fullfile(projectRoot, 'scripts', 'week3_day4_smoke_ffd.m'));
end

S = load(matPath, 'results');
results = S.results;
results = results(~logical(results.failed), :);

featureName = 'gamma_cp_3_fp_only';
subsets = {};
subsets(end + 1, :) = subsetRow(results(strcmp(columnText(results, 'antenna_type'), 'ideal'), :), 'ideal_all', featureName); %#ok<AGROW>
subsets(end + 1, :) = subsetRow(results(strcmp(columnText(results, 'antenna_type'), 'patch_ffd'), :), 'patch_ffd_all', featureName); %#ok<AGROW>

slabPlacement = columnText(results, 'slab_placement');
patchResults = results(strcmp(columnText(results, 'antenna_type'), 'patch_ffd'), :);
patchPlacement = slabPlacement(strcmp(columnText(results, 'antenna_type'), 'patch_ffd'));
placements = unique(string(patchPlacement));
for i = 1:numel(placements)
    mask = strcmp(string(patchPlacement), placements(i));
    subsets(end + 1, :) = subsetRow(patchResults(mask, :), sprintf('patch_ffd_%s', char(placements(i))), featureName); %#ok<AGROW>
end

angleValues = columnNumeric(patchResults, 'los_angle_from_anchor_bore_deg');
edges = unique(quantile(angleValues, linspace(0, 1, 5)));
if numel(edges) < 3
    edges = linspace(min(angleValues), max(angleValues) + 1e-6, 5);
end
binIdx = discretize(angleValues, edges);
for b = 1:max(binIdx)
    mask = binIdx == b;
    if ~any(mask)
        continue;
    end
    label = sprintf('patch_ffd_angle_bin%d_[%.1f,%.1f]', b, edges(b), edges(b + 1));
    subsets(end + 1, :) = subsetRow(patchResults(mask, :), label, featureName); %#ok<AGROW>
end

subsetTbl = cell2table(subsets, 'VariableNames', ...
    {'subset', 'n', 'n_los', 'n_nlos', 'mean_gamma_los', 'mean_gamma_nlos', 'auc_gamma_cp_3'});

patchRow = strcmp(subsetTbl.subset, 'patch_ffd_all');
patchAuc = subsetTbl.auc_gamma_cp_3(patchRow);
verdict = 'scene_too_simple_or_ffd_too_clean';
if isfinite(patchAuc) && patchAuc >= 0.55 && patchAuc <= 0.85
    verdict = 'target_range';
elseif isfinite(patchAuc) && patchAuc < 0.55
    verdict = 'negative_finding_patch_ffd_gamma_weak';
end

csvPath = fullfile(outDir, 'subset_auc_ffd.csv');
writetable(subsetTbl, csvPath);

fig = figure('Visible', 'off', 'Position', [100 100 1200 450]);
subplot(1, 2, 1);
placementRows = startsWith(string(subsetTbl.subset), "patch_ffd_") & ~contains(string(subsetTbl.subset), "angle_bin") & ~strcmp(string(subsetTbl.subset), "patch_ffd_all");
bar(categorical(subsetTbl.subset(placementRows)), subsetTbl.auc_gamma_cp_3(placementRows));
yline(0.55, 'r--');
yline(0.85, 'r--');
ylabel('AUC');
title('Patch FFD by Slab Placement');
xtickangle(30);
grid on;

subplot(1, 2, 2);
angleRows = contains(string(subsetTbl.subset), "angle_bin");
bar(categorical(subsetTbl.subset(angleRows)), subsetTbl.auc_gamma_cp_3(angleRows));
yline(0.55, 'r--');
yline(0.85, 'r--');
ylabel('AUC');
title('Patch FFD by LoS Angle Bin');
xtickangle(30);
grid on;

plotPath = fullfile(outDir, 'subset_auc_ffd.png');
saveas(fig, plotPath);
close(fig);

mdPath = fullfile(outDir, 'subset_auc_ffd_report.md');
fid = fopen(mdPath, 'w');
assert(fid ~= -1, 'Failed to open %s', mdPath);
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Week 3 Day 4 Subset AUC (FFD)\n\n');
fprintf(fid, '- primary feature: `%s`\n', featureName);
fprintf(fid, '- patch_ffd_all verdict: **%s**\n\n', verdict);
fprintf(fid, '| subset | n | n_los | n_nlos | mean_gamma_los | mean_gamma_nlos | auc_gamma_cp_3 |\n');
fprintf(fid, '| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n');
for i = 1:height(subsetTbl)
    fprintf(fid, '| %s | %d | %d | %d | %s | %s | %s |\n', ...
        subsetTbl.subset{i}, subsetTbl.n(i), subsetTbl.n_los(i), subsetTbl.n_nlos(i), ...
        fmtNum(subsetTbl.mean_gamma_los(i)), fmtNum(subsetTbl.mean_gamma_nlos(i)), fmtNum(subsetTbl.auc_gamma_cp_3(i)));
end

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

function values = columnText(tbl, baseName)
    names = {baseName, [baseName '_cases_tbl'], [baseName '_results_tbl']};
    for i = 1:numel(names)
        if ismember(names{i}, tbl.Properties.VariableNames)
            values = cellstr(string(tbl.(names{i})));
            return;
        end
    end
    error('Column %s not found', baseName);
end

function values = columnNumeric(tbl, baseName)
    names = {baseName, [baseName '_cases_tbl'], [baseName '_results_tbl']};
    for i = 1:numel(names)
        if ismember(names{i}, tbl.Properties.VariableNames)
            values = double(tbl.(names{i}));
            return;
        end
    end
    error('Column %s not found', baseName);
end

function txt = fmtNum(x)
    if isfinite(x)
        txt = sprintf('%.4f', x);
    else
        txt = 'NaN';
    end
end
