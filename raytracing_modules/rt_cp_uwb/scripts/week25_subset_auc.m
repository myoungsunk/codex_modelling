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

gamma_name = 'gamma_cp_3_fp_only';
subsetRows = {};
mode_label = '';

antenna_vals = {'ideal', 'patch'};
if hasVariable(results, 'los_blockage') && numel(unique(logical(results.los_blockage))) > 1
    blockage_vals = [false, true];
    mode_label = 'los_blockage';
    for ai = 1:numel(antenna_vals)
        for bi = 1:numel(blockage_vals)
            mask = strcmp(results.antenna_type, antenna_vals{ai}) & logical(results.los_blockage) == blockage_vals(bi);
            subsetRows(end + 1, :) = {sprintf('%s_%s', antenna_vals{ai}, ternary(blockage_vals(bi), 'blocked', 'clear')), ... %#ok<AGROW>
                antenna_vals{ai}, logical(blockage_vals(bi)), sum(mask), sum(results.is_los(mask)), sum(results.is_nlos(mask)), ...
                localMean(results(mask, :), gamma_name, false), localMean(results(mask, :), gamma_name, true), ...
                localAuc(results(mask, :), gamma_name)}; %#ok<AGROW>
        end
    end
else
    placement_name = findPlacementColumn(results.Properties.VariableNames);
    placements = unique(string(results.(placement_name)));
    mode_label = sprintf('slab_placement (%s)', placement_name);
    for ai = 1:numel(antenna_vals)
        for pi = 1:numel(placements)
            mask = strcmp(results.antenna_type, antenna_vals{ai}) & strcmp(string(results.(placement_name)), placements(pi));
            subsetRows(end + 1, :) = {sprintf('%s_%s', antenna_vals{ai}, char(placements(pi))), ... %#ok<AGROW>
                antenna_vals{ai}, NaN, sum(mask), sum(results.is_los(mask)), sum(results.is_nlos(mask)), ...
                localMean(results(mask, :), gamma_name, false), localMean(results(mask, :), gamma_name, true), ...
                localAuc(results(mask, :), gamma_name)}; %#ok<AGROW>
        end
    end
end

for ai = 1:numel(antenna_vals)
    mask = strcmp(results.antenna_type, antenna_vals{ai});
    subsetRows(end + 1, :) = {sprintf('%s_all', antenna_vals{ai}), antenna_vals{ai}, NaN, sum(mask), sum(results.is_los(mask)), sum(results.is_nlos(mask)), ...
        localMean(results(mask, :), gamma_name, false), localMean(results(mask, :), gamma_name, true), ...
        localAuc(results(mask, :), gamma_name)}; %#ok<AGROW>
end

subset_tbl = cell2table(subsetRows, 'VariableNames', {'subset', 'antenna_type', 'los_blockage', 'n', 'n_los', 'n_nlos', 'mean_gamma_los', 'mean_gamma_nlos', 'auc_gamma_cp_3'});
csvPath = fullfile(outDir, 'subset_auc.csv');
writetable(subset_tbl, csvPath);

ideal_auc = subset_tbl.auc_gamma_cp_3(strcmp(subset_tbl.subset, 'ideal_all'));
patch_auc = subset_tbl.auc_gamma_cp_3(strcmp(subset_tbl.subset, 'patch_all'));
signal1 = classifySignal1(ideal_auc, patch_auc);

reportPath = fullfile(outDir, 'subset_auc_report.md');
fid = fopen(reportPath, 'w');
assert(fid ~= -1, 'Failed to open %s', reportPath);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Subset AUC Diagnostic\n\n');
fprintf(fid, 'Primary feature: `%s`\n\n', gamma_name);
fprintf(fid, 'Subset mode: `%s`\n\n', mode_label);
fprintf(fid, '| subset | antenna_type | los_blockage | n | n_los | n_nlos | mean_gamma_los | mean_gamma_nlos | auc_gamma_cp_3 |\n');
fprintf(fid, '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n');
for i = 1:height(subset_tbl)
    fprintf(fid, '| %s | %s | %s | %d | %d | %d | %s | %s | %s |\n', ...
        subset_tbl.subset{i}, subset_tbl.antenna_type{i}, fmtBlock(subset_tbl.los_blockage(i)), ...
        subset_tbl.n(i), subset_tbl.n_los(i), subset_tbl.n_nlos(i), ...
        fmtNum(subset_tbl.mean_gamma_los(i)), fmtNum(subset_tbl.mean_gamma_nlos(i)), fmtNum(subset_tbl.auc_gamma_cp_3(i)));
end
fprintf(fid, '\n');
fprintf(fid, '## Signal 1 Decision\n\n');
fprintf(fid, '- ideal gamma_cp_3 AUC: %s\n', fmtNum(ideal_auc));
fprintf(fid, '- patch gamma_cp_3 AUC: %s\n', fmtNum(patch_auc));
fprintf(fid, '- verdict: **%s**\n', signal1.verdict);
fprintf(fid, '- rationale: %s\n', signal1.reason);
patch_row = strcmp(subset_tbl.subset, 'patch_all');
fprintf(fid, '- patch mean gamma_cp_3 (LoS): %s\n', fmtNum(subset_tbl.mean_gamma_los(patch_row)));
fprintf(fid, '- patch mean gamma_cp_3 (NLoS): %s\n', fmtNum(subset_tbl.mean_gamma_nlos(patch_row)));

fprintf('Saved:\n');
fprintf('  %s\n', csvPath);
fprintf('  %s\n', reportPath);

function tf = hasVariable(tbl, name)
    tf = ismember(name, tbl.Properties.VariableNames);
end

function name = findPlacementColumn(varNames)
    candidates = {'slab_placement', 'slab_placement_cases_tbl', 'slab_placement_results_tbl'};
    name = '';
    for i = 1:numel(candidates)
        if ismember(candidates{i}, varNames)
            name = candidates{i};
            return;
        end
    end
    error('week25:subset_auc:PlacementColumn', 'Could not find slab placement column');
end

function auc = localAuc(tbl, feat_name)
    auc = NaN;
    if isempty(tbl) || ~ismember(feat_name, tbl.Properties.VariableNames)
        return;
    end
    y = double(logical(tbl.is_nlos));
    x = double(tbl.(feat_name));
    valid = isfinite(x) & isfinite(y);
    x = x(valid);
    y = y(valid);
    if numel(x) < 2 || numel(unique(y)) < 2
        return;
    end
    [~, ~, ~, auc] = perfcurve(y, x, 1);
end

function value = localMean(tbl, feat_name, use_nlos)
    value = NaN;
    if isempty(tbl) || ~ismember(feat_name, tbl.Properties.VariableNames)
        return;
    end
    if use_nlos
        mask = logical(tbl.is_nlos);
    else
        mask = logical(tbl.is_los);
    end
    x = double(tbl.(feat_name));
    x = x(mask & isfinite(x));
    if isempty(x)
        return;
    end
    value = mean(x);
end

function out = classifySignal1(ideal_auc, patch_auc)
    out = struct();
    if isfinite(ideal_auc) && isfinite(patch_auc) && ideal_auc > 0.95 && patch_auc < 0.70
        out.verdict = 'yes';
        out.reason = 'ideal AUC is near-perfect while patch AUC is weak';
    elseif isfinite(patch_auc) && patch_auc > 0.90
        out.verdict = 'scene_too_simple';
        out.reason = 'patch subset is already almost perfectly separable';
    elseif isfinite(patch_auc) && patch_auc >= 0.65 && patch_auc <= 0.85
        out.verdict = 'no';
        out.reason = 'patch subset AUC is in the acceptable mid range';
    elseif isfinite(patch_auc) && patch_auc < 0.65
        out.verdict = 'partial';
        out.reason = 'patch subset AUC is weak, suggesting limited CP usefulness in this setup';
    else
        out.verdict = 'undetermined';
        out.reason = 'subset AUC is not defined because one class is missing or features are invalid';
    end
end

function txt = fmtNum(x)
    if isfinite(x)
        txt = sprintf('%.4f', x);
    else
        txt = 'NaN';
    end
end

function txt = fmtBlock(x)
    if isnan(x)
        txt = 'all';
    elseif logical(x)
        txt = 'true';
    else
        txt = 'false';
    end
end

function out = ternary(cond, a, b)
    if cond
        out = a;
    else
        out = b;
    end
end
