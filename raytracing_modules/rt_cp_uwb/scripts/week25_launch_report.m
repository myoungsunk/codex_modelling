projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'week25');
ensure(fullfile(projectRoot, 'scripts', 'week25_smoke_sweep.m'), fullfile(outDir, 'smoke_sweep_200.csv'));
ensure(fullfile(projectRoot, 'scripts', 'week25_initial_auc.m'), fullfile(outDir, 'initial_auc.csv'));
ensure(fullfile(projectRoot, 'scripts', 'week25_subset_auc.m'), fullfile(outDir, 'subset_auc.csv'));
ensure(fullfile(projectRoot, 'scripts', 'week25_nlos_kurtosis_diag.m'), fullfile(outDir, 'nlos_kurtosis_diag.md'));

results = readtable(fullfile(outDir, 'smoke_sweep_200.csv'));
aucTable = readtable(fullfile(outDir, 'initial_auc.csv'));
subsetTbl = readtable(fullfile(outDir, 'subset_auc.csv'));

dupKeep = {'fp_to_total_ratio', 'fp_kurtosis', 'rise_time_fp', 'a_fp_6_fp_to_2nd_peak'};
dupDrop = {'a_fp_1_norm_energy', 'a_fp_4_kurt_local', 'a_fp_5_rise_time', 'a_fp_3_peak_to_max'};
nlos_sd = std(results.kurtosis_total(logical(results.is_nlos) & ~logical(results.failed)));
patch_auc = subsetTbl.auc_gamma_cp_3(strcmp(subsetTbl.subset, 'patch_all'));
patch_mean_los = subsetTbl.mean_gamma_los(strcmp(subsetTbl.subset, 'patch_all'));
patch_mean_nlos = subsetTbl.mean_gamma_nlos(strcmp(subsetTbl.subset, 'patch_all'));
failed_ratio = mean(logical(results.failed));
failedRows = results(logical(results.failed), :);
[failureGroups, failureNames] = findgroups(string(failedRows.error_msg));
failureCounts = splitapply(@numel, failedRows.case_id, failureGroups);

launch = 'GO';
reason = 'Subset AUC and kurtosis diversity are acceptable for Week 3.';
if ~isfinite(patch_auc) || patch_auc > 0.90 || patch_auc < 0.65 || nlos_sd <= 1.5 || failed_ratio >= 0.05
    launch = 'ADDITIONAL_CALIBRATION_NEEDED';
    reason = localReason(patch_auc, patch_mean_los, patch_mean_nlos, nlos_sd, failed_ratio);
end

reportPath = fullfile(outDir, 'week25_report.md');
fid = fopen(reportPath, 'w');
assert(fid ~= -1, 'Failed to open %s', reportPath);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, '# Week 2.5 Report\n\n');
fprintf(fid, 'Generated: %s\n\n', datestr(now, 31));
fprintf(fid, 'Week 3 launch decision: **%s**\n\n', launch);
fprintf(fid, '- rationale: %s\n\n', reason);

fprintf(fid, '## Checklist\n\n');
fprintf(fid, '- [x] Subset AUC diagnostic\n');
fprintf(fid, '- [x] Signal 1 decision recorded\n');
fprintf(fid, '- [x] Feature duplicate cleanup to canonical 18\n');
fprintf(fid, '- [x] NLoS kurtosis diversity diagnostic\n');
fprintf(fid, '- [x] 200-case smoke rerun with updated sweep design\n\n');

fprintf(fid, '## Key Numbers\n\n');
fprintf(fid, '- failed ratio: %.4f\n', failed_ratio);
fprintf(fid, '- patch gamma_cp_3 AUC: %s\n', fmtNum(patch_auc));
fprintf(fid, '- patch mean gamma_cp_3 (LoS): %s\n', fmtNum(patch_mean_los));
fprintf(fid, '- patch mean gamma_cp_3 (NLoS): %s\n', fmtNum(patch_mean_nlos));
fprintf(fid, '- NLoS kurtosis std: %.4f\n', nlos_sd);
fprintf(fid, '- top 3 AUC: %s, %s, %s\n\n', aucTable.feature{1}, aucTable.feature{2}, aucTable.feature{3});

fprintf(fid, '## Failure Breakdown\n\n');
if isempty(failureCounts)
    fprintf(fid, '- none\n\n');
else
    for i = 1:numel(failureCounts)
        fprintf(fid, '- `%s`: %d\n', failureNames(i), failureCounts(i));
    end
    fprintf(fid, '\n');
end

fprintf(fid, '## Feature Cleanup\n\n');
fprintf(fid, '- canonical feature count: %d\n', numel(features.canonicalFeatureNames()));
fprintf(fid, '- kept duplicate-side representatives: `%s`\n', strjoin(dupKeep, '`, `'));
fprintf(fid, '- dropped aliases: `%s`\n\n', strjoin(dupDrop, '`, `'));

fprintf(fid, '## Artifacts\n\n');
fprintf(fid, '- `%s`\n', fullfile(outDir, 'smoke_sweep_200.csv'));
fprintf(fid, '- `%s`\n', fullfile(outDir, 'initial_auc.csv'));
fprintf(fid, '- `%s`\n', fullfile(outDir, 'subset_auc_report.md'));
fprintf(fid, '- `%s`\n', fullfile(outDir, 'nlos_kurtosis_diag.md'));

fprintf('Saved:\n');
fprintf('  %s\n', reportPath);

function ensure(scriptPath, artifactPath)
    if exist(artifactPath, 'file') ~= 2
        run(scriptPath);
    end
end

function txt = fmtNum(x)
    if isfinite(x)
        txt = sprintf('%.4f', x);
    else
        txt = 'NaN';
    end
end

function txt = localReason(patch_auc, patch_mean_los, patch_mean_nlos, nlos_sd, failed_ratio)
    notes = strings(0, 1);
    if failed_ratio >= 0.05
        notes(end + 1) = sprintf('failed ratio %.1f%% exceeds the 5%% launch threshold', 100.0 * failed_ratio); %#ok<AGROW>
    end
    if nlos_sd <= 1.5
        notes(end + 1) = sprintf('NLoS kurtosis std %.3f indicates low geometry diversity', nlos_sd); %#ok<AGROW>
    end
    if ~isfinite(patch_auc)
        notes(end + 1) = "patch gamma_cp AUC is undefined"; %#ok<AGROW>
    elseif patch_auc > 0.90
        notes(end + 1) = sprintf('patch gamma_cp AUC %.3f is too high, so the scene is still too simple', patch_auc); %#ok<AGROW>
    elseif patch_auc < 0.65
        if isfinite(patch_mean_los) && isfinite(patch_mean_nlos) && patch_mean_los > patch_mean_nlos
            notes(end + 1) = sprintf('patch gamma_cp AUC %.3f is weak/inverted because mean gamma is higher in LoS than NLoS', patch_auc); %#ok<AGROW>
        else
            notes(end + 1) = sprintf('patch gamma_cp AUC %.3f is below the useful range', patch_auc); %#ok<AGROW>
        end
    end
    txt = strjoin(cellstr(notes), '; ');
end
