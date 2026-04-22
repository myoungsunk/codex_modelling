projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
week2Dir = fullfile(cfg.results_dir, 'week2');
if ~exist(week2Dir, 'dir')
    mkdir(week2Dir);
end

smokeCsv = fullfile(week2Dir, 'smoke_sweep_200.csv');
aucCsv = fullfile(week2Dir, 'initial_auc.csv');
diagPng = fullfile(week2Dir, 'smoke_diagnostics.png');
convLog = fullfile(cfg.results_dir, 'week2_day1_convention_test.log');

ensureArtifact(smokeCsv, fullfile(projectRoot, 'scripts', 'week2_day4_smoke_sweep.m'));
ensureArtifact(aucCsv, fullfile(projectRoot, 'scripts', 'week2_day4_initial_auc.m'));
ensureArtifact(diagPng, fullfile(projectRoot, 'scripts', 'week2_day4_diagnostic_plots.m'));
ensureArtifact(convLog, fullfile(projectRoot, 'scripts', 'test_gamma_cp_convention.m'));

results = readtable(smokeCsv);
aucTable = readtable(aucCsv);

cir_features = {'kurtosis_total', 'peak_to_avg_ratio', 'rms_delay_spread', 'k_factor_estimate'};
cp_features = {'gamma_cp_2_freq_db', 'gamma_cp_3_fp_only', 'gamma_cp_4_total_energy', 'gamma_cp_5_post_fp'};
joint_features = [cir_features, cp_features];

[delta_auc, cond_bins] = analysis.computeConditionalAuc(results, 'incidence_deg', cir_features, joint_features, 5);
disagreement = analysis.computeDisagreementAnalysis(results, cir_features, cp_features);
dup_candidates = findDuplicateCandidates(results);
failed_summary = summarizeFailedCases(results);
checklist = buildChecklist(results, aucTable, convLog, cond_bins, disagreement, failed_summary, cfg);

reportPath = fullfile(week2Dir, 'week2_report.md');
writeReport(reportPath, checklist, results, aucTable, cond_bins, delta_auc, disagreement, dup_candidates, failed_summary, ...
    smokeCsv, aucCsv, diagPng, convLog);

fprintf('Saved:\n');
fprintf('  %s\n', reportPath);

function ensureArtifact(targetPath, scriptPath)
    if exist(targetPath, 'file') == 2
        return;
    end
    run(scriptPath);
end

function checklist = buildChecklist(results, aucTable, convLog, cond_bins, disagreement, failed_summary, cfg)
    results_failed = logical(results.failed);
    failed_ratio = mean(results_failed);
    top3 = aucTable(1:min(3, height(aucTable)), :);
    dup_candidates = findDuplicateCandidates(results);

    checklist = struct('item', {}, 'passed', {}, 'detail', {});
    checklist(end + 1) = makeItem('gamma_CP convention fixed and regression test passed', ...
        conventionPassed(convLog), 'Week 2 Day 1 convention log exists and contains pass marker');

    checklist(end + 1) = makeItem('materialsLibrary 8 materials work', ...
        materialsLibraryPass(), 'All 8 predefined materials instantiate without error');

    [scene_ok, scene_detail] = singleSlabAndCasePass(cfg);
    checklist(end + 1) = makeItem('Single-slab scene + runOneCase end-to-end', scene_ok, scene_detail);

    checklist(end + 1) = makeItem('200-case smoke sweep failed-case ratio < 5%', ...
        failed_ratio < 0.05, sprintf('failed=%d/%d (%.2f%%)', sum(results_failed), height(results), 100 * failed_ratio));

    checklist(end + 1) = makeItem('Per-feature AUC ranking available and top 3 identified', ...
        height(top3) >= 3 && all(isfinite(top3.AUC(1:3))), ...
        sprintf('top3=%s, %s, %s', top3.feature{1}, top3.feature{2}, top3.feature{3}));

    if isempty(dup_candidates)
        dup_detail = 'No feature pairs exceeded the duplicate-candidate threshold';
    else
        dup_detail = sprintf('%s vs %s (|r|=%.4f)', dup_candidates.feature_a{1}, dup_candidates.feature_b{1}, dup_candidates.abs_corr(1));
    end
    checklist(end + 1) = makeItem('Feature correlation checked and duplicate candidates identified', true, dup_detail);

    cond_ok = any(isfinite(cond_bins.delta_auc));
    checklist(end + 1) = makeItem('Conditional AUC utility works on smoke set', ...
        cond_ok, sprintf('finite bins=%d/%d, mean delta AUC=%.4f', sum(isfinite(cond_bins.delta_auc)), height(cond_bins), mean(cond_bins.delta_auc, 'omitnan')));

    disagree_ok = isfinite(disagreement.misc_rate) && isfinite(disagreement.auc_cir_cv);
    checklist(end + 1) = makeItem('Disagreement analysis utility works', ...
        disagree_ok, sprintf('misc_rate=%.4f, auc_cir_cv=%.4f', disagreement.misc_rate, disagreement.auc_cir_cv));

    checklist(end + 1) = makeItem('Failed-case analysis completed', ...
        failed_summary.analyzed, failed_summary.summary);
end

function out = makeItem(item, passed, detail)
    out = struct('item', item, 'passed', logical(passed), 'detail', detail);
end

function ok = conventionPassed(convLog)
    ok = exist(convLog, 'file') == 2;
    if ~ok
        return;
    end
    txt = fileread(convLog);
    ok = contains(txt, 'Week 2 Day 1 convention test passed.');
end

function ok = materialsLibraryPass()
    mats = {'wood', 'glass', 'concrete', 'metal_pec', 'drywall', 'brick', 'ceramic_tile', 'vacuum'};
    ok = true;
    for i = 1:numel(mats)
        try
            m = materials.materialsLibrary(mats{i}); %#ok<NASGU>
        catch
            ok = false;
            return;
        end
    end
end

function [ok, detail] = singleSlabAndCasePass(cfg)
    try
        scene = scenes.makeSingleSlabScene('wood', [2 2], [1; 0; 0], [0; 0; 1]);
        paths = trace.enumeratePaths(scene, [0; 0; 1], [2; 0; 1], 1);
        one = sweep.runOneCase(sweep.designLhsSweep(1, 42), cfg);
        ok = ~isempty(paths) && ~one.failed;
        detail = sprintf('single_slab_paths=%d, runOneCase_num_paths=%d', numel(paths), one.num_paths);
    catch ME
        ok = false;
        detail = ME.message;
    end
end

function pairs = findDuplicateCandidates(results)
    candidate_names = intersect(results.Properties.VariableNames, { ...
        'gamma_cp_1_freq_avg', 'gamma_cp_2_freq_db', 'gamma_cp_3_fp_only', ...
        'gamma_cp_4_total_energy', 'gamma_cp_5_post_fp', 'gamma_cp_6_phase_circvar', ...
        'a_fp_1_norm_energy', 'a_fp_2_peak_to_total', 'a_fp_3_peak_to_max', ...
        'a_fp_4_kurt_local', 'a_fp_5_rise_time', 'a_fp_6_fp_to_2nd_peak', ...
        'rms_delay_spread', 'mean_excess_delay', 'max_excess_delay', ...
        'fp_to_total_ratio', 'rise_time_fp', 'fp_kurtosis', 'kurtosis_total', ...
        'skewness_total', 'energy_concentration_50ns', 'num_significant_peaks', ...
        'peak_to_avg_ratio', 'k_factor_estimate'}, 'stable');

    rows = {};
    for i = 1:numel(candidate_names)
        xi = double(results.(candidate_names{i}));
        for j = (i + 1):numel(candidate_names)
            xj = double(results.(candidate_names{j}));
            valid = isfinite(xi) & isfinite(xj);
            if sum(valid) < 10
                continue;
            end
            r = corr(xi(valid), xj(valid));
            if ~isfinite(r)
                continue;
            end
            exact_dup = max(abs(xi(valid) - xj(valid))) < 1e-12;
            if abs(r) >= 0.95 || exact_dup
                rows(end + 1, :) = {candidate_names{i}, candidate_names{j}, r, abs(r), exact_dup}; %#ok<AGROW>
            end
        end
    end

    if isempty(rows)
        pairs = table();
        return;
    end
    pairs = cell2table(rows, 'VariableNames', {'feature_a', 'feature_b', 'corr', 'abs_corr', 'exact_duplicate'});
    pairs = sortrows(pairs, {'exact_duplicate', 'abs_corr'}, {'descend', 'descend'});
end

function summary = summarizeFailedCases(results)
    failed_mask = logical(results.failed);
    n_failed = sum(failed_mask);
    summary = struct();
    summary.n_failed = n_failed;
    summary.analyzed = true;

    if n_failed == 0
        summary.summary = '0 failed cases in the 200-case smoke sweep; no active geometry/tracer failure mode observed';
        summary.mode = 'none';
        summary.error_counts = table();
        return;
    end

    failed_tbl = results(failed_mask, :);
    error_strings = strings(height(failed_tbl), 1);
    if ismember('error_msg', failed_tbl.Properties.VariableNames)
        for i = 1:height(failed_tbl)
            error_strings(i) = string(failed_tbl.error_msg{i});
        end
    end

    [groups, keys] = findgroups(error_strings);
    counts = splitapply(@numel, error_strings, groups);
    error_counts = table(keys, counts, 'VariableNames', {'error_msg', 'count'});

    if all(strcmpi(strtrim(error_strings), "no_paths"))
        summary.mode = 'geometry_no_paths';
        summary.summary = sprintf('%d failed cases, all due to no_paths (geometry/blockage configuration)', n_failed);
    else
        summary.mode = 'mixed_runtime';
        summary.summary = sprintf('%d failed cases with mixed runtime/geometry errors', n_failed);
    end
    summary.error_counts = error_counts;
end

function writeReport(reportPath, checklist, results, aucTable, cond_bins, delta_auc, disagreement, dup_candidates, failed_summary, ...
    smokeCsv, aucCsv, diagPng, convLog)
    topN = min(5, height(aucTable));
    aucTop = aucTable(1:topN, :);
    is_los = logical(results.is_los);
    is_nlos = logical(results.is_nlos);
    all_passed = all([checklist.passed]);
    status = 'GO';
    if ~all_passed
        status = 'HOLD';
    end

    strongest_cp = pickStrongestCp(disagreement.cp_separability);

    fid = fopen(reportPath, 'w');
    assert(fid ~= -1, 'Failed to open report for writing: %s', reportPath);
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fprintf(fid, '# Week 2 Report\n\n');
    fprintf(fid, 'Generated: %s\n\n', datestr(now, 31));
    fprintf(fid, 'Week 3 launch status: **%s**\n\n', status);

    fprintf(fid, '## Deliverables\n\n');
    fprintf(fid, '- gamma convention fix and regression log: `%s`\n', convLog);
    fprintf(fid, '- smoke sweep artifacts: `%s`, `%s`, `%s`\n', smokeCsv, fullfile(fileparts(smokeCsv), 'smoke_sweep_200.mat'), diagPng);
    fprintf(fid, '- initial per-feature AUC ranking: `%s`\n', aucCsv);
    fprintf(fid, '- analysis utilities: `+analysis/computeConditionalAuc.m`, `+analysis/computeDisagreementAnalysis.m`\n\n');

    fprintf(fid, '## Launch Checklist\n\n');
    fprintf(fid, '| Item | Status | Detail |\n');
    fprintf(fid, '| --- | --- | --- |\n');
    for i = 1:numel(checklist)
        status_str = 'FAIL';
        if checklist(i).passed
            status_str = 'PASS';
        end
        fprintf(fid, '| %s | %s | %s |\n', escapePipe(checklist(i).item), status_str, escapePipe(checklist(i).detail));
    end
    fprintf(fid, '\n');

    fprintf(fid, '## Smoke Sweep Statistics\n\n');
    fprintf(fid, '- Cases: %d\n', height(results));
    fprintf(fid, '- Failed: %d (%.2f%%)\n', sum(logical(results.failed)), 100 * mean(logical(results.failed)));
    fprintf(fid, '- LoS: %d\n', sum(is_los));
    fprintf(fid, '- NLoS: %d\n', sum(is_nlos));
    fprintf(fid, '- Numeric non-finite count: %d\n\n', countNonFiniteNumeric(results));

    fprintf(fid, '## Top AUC Features\n\n');
    fprintf(fid, '| Rank | Feature | AUC | n_valid |\n');
    fprintf(fid, '| --- | --- | ---: | ---: |\n');
    for i = 1:topN
        fprintf(fid, '| %d | %s | %.4f | %d |\n', i, escapePipe(aucTop.feature{i}), aucTop.AUC(i), aucTop.n_valid(i));
    end
    fprintf(fid, '\n');

    fprintf(fid, '## Correlation And Redundancy\n\n');
    if isempty(dup_candidates)
        fprintf(fid, '- No duplicate-candidate pairs exceeded the threshold.\n\n');
    else
        fprintf(fid, '| Feature A | Feature B | corr | exact duplicate |\n');
        fprintf(fid, '| --- | --- | ---: | --- |\n');
        for i = 1:min(6, height(dup_candidates))
            fprintf(fid, '| %s | %s | %.4f | %s |\n', ...
                escapePipe(dup_candidates.feature_a{i}), escapePipe(dup_candidates.feature_b{i}), ...
                dup_candidates.corr(i), tfText(dup_candidates.exact_duplicate(i)));
        end
        fprintf(fid, '\n');
    end

    fprintf(fid, '## Conditional AUC Smoke Test\n\n');
    fprintf(fid, '- Condition variable: `incidence_deg`\n');
    fprintf(fid, '- Mean delta AUC (joint - CIR): %.4f\n', mean(delta_auc, 'omitnan'));
    fprintf(fid, '- Finite bins: %d / %d\n', sum(isfinite(delta_auc)), numel(delta_auc));
    fprintf(fid, '| Bin | n | delta_auc |\n');
    fprintf(fid, '| --- | ---: | ---: |\n');
    for i = 1:height(cond_bins)
        fprintf(fid, '| %d | %d | %s |\n', cond_bins.bin(i), cond_bins.n(i), fmtNum(cond_bins.delta_auc(i)));
    end
    fprintf(fid, '\n');

    fprintf(fid, '## Disagreement Analysis Smoke Test\n\n');
    fprintf(fid, '- CIR-only CV AUC: %.4f\n', disagreement.auc_cir_cv);
    fprintf(fid, '- Misclassification rate: %.4f (%d / %d)\n', disagreement.misc_rate, disagreement.n_misc, disagreement.n_total);
    fprintf(fid, '- Strongest CP rescue signal: `%s` with KS=%.4f, p=%.4g\n\n', ...
        strongest_cp.name, strongest_cp.ks_stat, strongest_cp.p_value);

    fprintf(fid, '## Failed Case Analysis\n\n');
    fprintf(fid, '- %s\n\n', failed_summary.summary);
    if isfield(failed_summary, 'error_counts') && ~isempty(failed_summary.error_counts)
        fprintf(fid, '| Error | Count |\n');
        fprintf(fid, '| --- | ---: |\n');
        for i = 1:height(failed_summary.error_counts)
            fprintf(fid, '| %s | %d |\n', escapePipe(char(string(failed_summary.error_counts.error_msg(i)))), failed_summary.error_counts.count(i));
        end
        fprintf(fid, '\n');
    end

    fprintf(fid, '## Known Issues And Limitations\n\n');
    fprintf(fid, '- Smoke sweep is only 200 cases; Week 3 full run is still pending at n=3000.\n');
    fprintf(fid, '- Stage 1 geometry is a single-slab scene with a synthetic blocker, so the current separation trends are not yet room-scale results.\n');
    fprintf(fid, '- Patch antenna cases still use the synthetic patch surrogate rather than full angular pattern interpolation from measured/simulated FFD data.\n');
    if ~isempty(dup_candidates)
        fprintf(fid, '- Several feature pairs are redundant enough to be pruning candidates before the 3000-case run.\n');
    end
    fprintf(fid, '\n');
end

function value = countNonFiniteNumeric(results)
    value = 0;
    skip_names = {'error_msg'};
    for i = 1:width(results)
        name = results.Properties.VariableNames{i};
        if ismember(name, skip_names)
            continue;
        end
        v = results.(name);
        if isnumeric(v)
            value = value + sum(~isfinite(v));
        end
    end
end

function out = pickStrongestCp(cp_separability)
    names = fieldnames(cp_separability);
    best_idx = 1;
    best_score = -inf;
    for i = 1:numel(names)
        stats = cp_separability.(names{i});
        score = stats.ks_stat;
        if isfinite(score) && score > best_score
            best_score = score;
            best_idx = i;
        end
    end
    stats = cp_separability.(names{best_idx});
    out = struct('name', names{best_idx}, 'ks_stat', stats.ks_stat, 'p_value', stats.p_value);
end

function txt = tfText(tf)
    txt = 'false';
    if tf
        txt = 'true';
    end
end

function txt = fmtNum(x)
    if isfinite(x)
        txt = sprintf('%.4f', x);
    else
        txt = 'NaN';
    end
end

function txt = escapePipe(txt)
    txt = strrep(char(string(txt)), '|', '\|');
end
