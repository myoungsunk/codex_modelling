script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

global SANITY_CHECK_ROWS;
SANITY_CHECK_ROWS = {};

out_dir = fullfile(repo_root, 'results', 'recovery_mini', 'day5');
ensureDir(out_dir);

refs = struct();
refs.script = 'scripts/recovery_day5_deep_sanity_audit.m';
refs.day1_script = 'scripts/recovery_mini_day1.m';
refs.day2_script = 'scripts/recovery_mini_day2.m';
refs.day3_script = 'scripts/recovery_day3_stage1_mini_run.m';
refs.day4_script = 'scripts/recovery_day4_stage2_room_mini_run.m';
refs.seed = '+sweep/composeCaseSeed.m';
refs.run_one_case = '+sweep/runOneCase.m';
refs.canonical = '+features/canonicalFeatureNames.m';
refs.extract = '+features/extractAllFeatures.m';
refs.stage2_det_audit = 'results/code_audit/full_determinism_stage2/full_determinism_audit.md';
refs.integrity_audit = 'results/code_audit/sanity_rerun_report.md';
refs.stage3_readiness = 'results/code_audit/stage3_readiness_report.md';
refs.day5_guardrails = 'results/recovery_mini/day5/day5_paper_claim_guardrails.md';

all_features = string(features.canonicalFeatureNames());
cp_features = all_features(1:6);
cir_features = all_features(7:end);
forbidden_terms = ["is_los", "is_nlos", "has_los_path", "bounce_to_los_ratio_mid", "mixed_0p33", "geo_only", "current_0p20"];

stage1_det = newestFile(repo_root, fullfile('results', '**', 'stage1_3000_ffd_det.mat'));
stage2_det = newestFile(repo_root, fullfile('results', '**', 'stage2_900_ffd_relabel_det.mat'));
paper_ci_csv = newestFile(repo_root, fullfile('results', '**', 'paper_facing_auc_ci.csv'));

addCheck('A', 'latest_stage1_input', 'no_issue', 'verified', true, ...
    sprintf('selected latest deterministic Stage1 MAT = %s', stage1_det), refs.day1_script);
addCheck('A', 'latest_stage2_input', 'no_issue', 'verified', true, ...
    sprintf('selected latest deterministic Stage2 MAT = %s', stage2_det), refs.day1_script);

day1_metrics = readtable(fullfile(repo_root, 'results', 'recovery_mini', 'day1', 'day1_metrics_summary.csv'), 'TextType', 'string');
paper_ci = readtable(paper_ci_csv, 'TextType', 'string');

stage1_row = day1_metrics(day1_metrics.dataset == "Stage1" & day1_metrics.scope == "ALL" & day1_metrics.threshold_type == "threshold_0p5", :);
stage2_row = day1_metrics(day1_metrics.dataset == "Stage2" & day1_metrics.scope == "ALL" & day1_metrics.threshold_type == "threshold_0p5", :);
stage1_ci = paper_ci(paper_ci.scope == "Stage1" & paper_ci.subset == "ALL", :);
stage2_ci = paper_ci(paper_ci.scope == "Stage2 mixed@0.33" & paper_ci.subset == "ALL", :);

stage1_ok = abs(stage1_row.delta_auc - stage1_ci.delta_auc) <= 1e-9;
stage2_ok = abs(stage2_row.delta_auc - stage2_ci.delta_auc) <= 1e-9;
addCheck('A', 'canonical_stage1_match', 'no_issue', 'verified', stage1_ok, ...
    sprintf('day1 delta_auc=%0.6f, paper delta_auc=%0.6f', stage1_row.delta_auc, stage1_ci.delta_auc), refs.day1_script);
addCheck('A', 'canonical_stage2_match', 'no_issue', 'verified', stage2_ok, ...
    sprintf('day1 delta_auc=%0.6f, paper delta_auc=%0.6f', stage2_row.delta_auc, stage2_ci.delta_auc), refs.day1_script);

feature_overlap = intersect(all_features, forbidden_terms, 'stable');
addCheck('C', 'canonical_feature_overlap', 'no_issue', 'verified', isempty(feature_overlap), ...
    sprintf('forbidden overlap count = %d', numel(feature_overlap)), refs.canonical);

scan_tbl = scanFeatureFiles(repo_root, forbidden_terms);
for i = 1:height(scan_tbl)
    pass_flag = scan_tbl.hit_count(i) == 0;
    addCheck('C', sprintf('feature_scan_%s', scan_tbl.term(i)), 'no_issue', ternaryStatus(pass_flag), pass_flag, ...
        sprintf('feature file hit_count=%d', scan_tbl.hit_count(i)), refs.extract);
end

seed_legacy = double(sweep.composeCaseSeed(17));
seed_stage_a = double(sweep.composeCaseSeed(17, 'stage_id', 'stageA', 'base_seed', 101, 'component', 'noise'));
seed_stage_b = double(sweep.composeCaseSeed(17, 'stage_id', 'stageB', 'base_seed', 101, 'component', 'noise'));
seed_component = double(sweep.composeCaseSeed(17, 'stage_id', 'stageA', 'base_seed', 101, 'component', 'case_rng'));
seed_ok = seed_legacy == 17 && seed_stage_a ~= seed_stage_b && seed_stage_a ~= seed_component;
addCheck('B', 'composeCaseSeed_stage_salt', 'no_issue', 'verified', seed_ok, ...
    sprintf('legacy=%u, stageA_noise=%u, stageB_noise=%u, stageA_case_rng=%u', seed_legacy, seed_stage_a, seed_stage_b, seed_component), refs.seed);

day1_stage1_pred = readtable(fullfile(repo_root, 'results', 'recovery_mini', 'day1', 'day1_oof_predictions_stage1.csv'), 'TextType', 'string');
day1_stage2_pred = readtable(fullfile(repo_root, 'results', 'recovery_mini', 'day1', 'day1_oof_predictions_stage2.csv'), 'TextType', 'string');
day3_full = readtable(fullfile(repo_root, 'results', 'recovery_mini', 'day3', 'stage1_recovery_mini.csv'), 'TextType', 'string');
day4_full = readtable(fullfile(repo_root, 'results', 'recovery_mini', 'day4', 'stage2_recovery_room_mini.csv'), 'TextType', 'string');

checkFinite('day1_stage1_oof', day1_stage1_pred, all_features, refs.canonical);
checkFinite('day1_stage2_oof', day1_stage2_pred, all_features, refs.canonical);
checkFinite('day3_stage1_recovery_mini', day3_full, all_features, refs.canonical);
checkFinite('day4_stage2_recovery_room_mini', day4_full, all_features, refs.canonical);

stage1_corr = readtable(fullfile(repo_root, 'results', 'recovery_mini', 'day1', 'day1_top_feature_correlations.csv'), 'TextType', 'string');
high_corr = stage1_corr(toDoubleColumn(stage1_corr.abs_corr) > 0.98, :);
if isempty(high_corr)
    addCheck('D', 'high_corr_pairs_reported', 'caution', 'verified', false, 'no corr > 0.98 rows found in day1_top_feature_correlations.csv', refs.day1_script);
else
    first_pair = high_corr(1, :);
    addCheck('D', 'high_corr_pairs_reported', 'caution', 'verified', true, ...
        sprintf('top corr pair = %s vs %s, abs_corr=%s', first_pair.feature_a, first_pair.feature_b, first_pair.abs_corr), 'results/recovery_mini/day1/day1_top_feature_correlations.csv');
end

neg_text = fileread(fullfile(repo_root, 'results', 'recovery_mini', 'day2', 'day2_negative_control.md'));
neg_ok = contains(neg_text, 'collapsed toward zero') || contains(neg_text, 'collapse toward zero');
addCheck('E', 'negative_control_collapse', 'no_issue', ternaryStatus(neg_ok), neg_ok, ...
    'Day 2 row-shuffle negative control states collapse toward zero for Stage1/Stage2.', 'results/recovery_mini/day2/day2_negative_control.md');

cv_day4 = readtable(fullfile(repo_root, 'results', 'recovery_mini', 'day4', 'day4_stage2_cv_comparison.csv'), 'TextType', 'string');
cv_ok = all(ismember(["stratified"; "group_stratified"; "leave_one_group_out"], string(cv_day4.cv_scheme)));
addCheck('E', 'day4_grouped_cv_present', 'no_issue', ternaryStatus(cv_ok), cv_ok, ...
    sprintf('cv schemes present = %s', strjoin(unique(string(cv_day4.cv_scheme), 'stable'), ', ')), refs.day4_script);

stage3_text = fileread(fullfile(repo_root, 'results', 'code_audit', 'stage3_readiness_report.md'));
trend_ok = contains(stage3_text, 'AUC reproduction on 30 cases is not the primary metric');
addCheck('G', 'stage3_trend_not_auc', 'no_issue', ternaryStatus(trend_ok), trend_ok, ...
    'Stage 3 readiness report explicitly demotes 30-case AUC reproduction.', refs.stage3_readiness);

guard_text = fileread(fullfile(repo_root, 'results', 'recovery_mini', 'day5', 'day5_paper_claim_guardrails.md'));
guard_ok = contains(guard_text, 'CP is not a global replacement for CIR') && ...
    contains(guard_text, 'Multipath richness increases CP utility monotonically') && ...
    contains(guard_text, 'xpol_coupling_db_expected');
addCheck('G', 'paper_guardrails_present', 'no_issue', ternaryStatus(guard_ok), guard_ok, ...
    'Day 5 guardrail file contains global, monotonic, and xpol caveat locks.', refs.day5_guardrails);

cfg3 = config.defaultConfig();
cfg3.seed_stage_id = 'stage1_recovery_mini_day3';
cfg3.seed_base = 20260423;

cfg4 = config.defaultConfig();
cfg4.seed_stage_id = 'stage2_recovery_room_mini_day4';
cfg4.seed_base = 20260424;

day3_cases = readtable(fullfile(repo_root, 'results', 'recovery_mini', 'day3', 'day3_stage1_recovery_mini_cases.csv'), 'TextType', 'string');
day4_cases = readtable(fullfile(repo_root, 'results', 'recovery_mini', 'day4', 'day4_stage2_room_mini_cases.csv'), 'TextType', 'string');

day3_subset = selectDay3Subset(day3_cases);
day4_subset = selectDay4Subset(day4_cases);
writetable(day3_subset, fullfile(out_dir, 'day5_determinism_subset_cases_day3.csv'));
writetable(day4_subset, fullfile(out_dir, 'day5_determinism_subset_cases_day4.csv'));

fprintf('Running Day 3 deep determinism subset (%d cases)...\n', height(day3_subset));
[day3_summary, day3_detail] = runDeterminismSuite('Day3Stage1Mini', day3_subset, day3_full, cfg3, false, refs.day3_script);

fprintf('Running Day 4 deep determinism subset (%d cases)...\n', height(day4_subset));
[day4_summary, day4_detail] = runDeterminismSuite('Day4Stage2Mini', day4_subset, day4_full, cfg4, true, refs.day4_script);

summary_tbl = cell2table(SANITY_CHECK_ROWS, 'VariableNames', {'section', 'check_id', 'classification', 'status', 'pass_flag', 'details', 'source_ref'});
summary_tbl = [summary_tbl; day3_summary; day4_summary];
detail_tbl = [day3_detail; day4_detail];

writetable(summary_tbl, fullfile(out_dir, 'day5_deep_sanity_checks.csv'));
writetable(detail_tbl, fullfile(out_dir, 'day5_deep_sanity_comparison_details.csv'));
writeAuditReport(out_dir, summary_tbl, detail_tbl, refs);

disp('Day 5 deep sanity audit outputs written under results/recovery_mini/day5/.');

function ensureDir(path_str)
    if exist(path_str, 'dir') ~= 7
        mkdir(path_str);
    end
end

function file_path = newestFile(repo_root, pattern)
    candidates = dir(fullfile(repo_root, pattern));
    if isempty(candidates)
        error('No file matched pattern: %s', pattern);
    end
    [~, idx] = max([candidates.datenum]);
    file_path = fullfile(candidates(idx).folder, candidates(idx).name);
end

function tbl = scanFeatureFiles(repo_root, forbidden_terms)
    files = dir(fullfile(repo_root, '+features', '*.m'));
    rows = {};
    for i = 1:numel(forbidden_terms)
        term = string(forbidden_terms(i));
        hit_count = 0;
        for j = 1:numel(files)
            txt = fileread(fullfile(files(j).folder, files(j).name));
            if contains(txt, term)
                hit_count = hit_count + 1;
            end
        end
        rows(end + 1, :) = {term, hit_count}; %#ok<AGROW>
    end
    tbl = cell2table(rows, 'VariableNames', {'term', 'hit_count'});
end

function status = ternaryStatus(pass_flag)
    if pass_flag
        status = "verified";
    else
        status = "failed";
    end
end

function checkFinite(dataset_name, tbl, feature_names, source_ref)
    available = intersect(string(tbl.Properties.VariableNames), string(feature_names), 'stable');
    if isempty(available)
        addCheck('D', sprintf('%s_nonfinite', dataset_name), 'caution', 'failed', false, ...
            'no canonical feature columns available in this table', source_ref);
        return;
    end
    X = table2array(tbl(:, cellstr(available)));
    nonfinite = sum(~isfinite(X(:)));
    addCheck('D', sprintf('%s_nonfinite', dataset_name), 'no_issue', ternaryStatus(nonfinite == 0), nonfinite == 0, ...
        sprintf('available_features=%d, nonfinite_count=%d over %d values', numel(available), nonfinite, numel(X)), source_ref);
end

function subset = selectDay3Subset(cases_tbl)
    rows = unique(cases_tbl(:, {'candidate_id', 'antenna_type'}), 'rows', 'stable');
    idx = zeros(height(rows), 1);
    for i = 1:height(rows)
        mask = string(cases_tbl.candidate_id) == string(rows.candidate_id(i)) & string(cases_tbl.antenna_type) == string(rows.antenna_type(i));
        cur = find(mask, 1, 'first');
        idx(i) = cur;
    end
    subset = cases_tbl(idx, :);
end

function subset = selectDay4Subset(cases_tbl)
    rows = unique(cases_tbl(:, {'room_type', 'expected_candidate_id'}), 'rows', 'stable');
    idx = zeros(height(rows), 1);
    for i = 1:height(rows)
        mask = string(cases_tbl.room_type) == string(rows.room_type(i)) & string(cases_tbl.expected_candidate_id) == string(rows.expected_candidate_id(i));
        cur = find(mask, 1, 'first');
        idx(i) = cur;
    end
    subset = cases_tbl(idx, :);
end

function [summary_tbl, detail_tbl] = runDeterminismSuite(dataset_name, subset_cases, baseline_full, cfg, attach_labels, source_ref)
    key_name = 'case_id';
    baseline_subset = alignByCaseId(baseline_full, subset_cases.(key_name));

    run_a = sweep.runSweepBatch(subset_cases, cfg, true);
    run_b = sweep.runSweepBatch(subset_cases, cfg, true);

    order = randperm(height(subset_cases));
    run_shuffle = sweep.runSweepBatch(subset_cases(order, :), cfg, true);

    half = floor(height(subset_cases) / 2);
    run_part1 = sweep.runSweepBatch(subset_cases(1:half, :), cfg, true);
    run_part2 = sweep.runSweepBatch(subset_cases(half + 1:end, :), cfg, true);
    run_resume = [run_part1; run_part2];

    if attach_labels
        baseline_subset = attachStage2LabelsAudit(baseline_subset);
        run_a = attachStage2LabelsAudit(run_a);
        run_b = attachStage2LabelsAudit(run_b);
        run_shuffle = attachStage2LabelsAudit(run_shuffle);
        run_resume = attachStage2LabelsAudit(run_resume);
    end

    [s1, d1] = compareTables(datasetName(dataset_name), 'same_vs_baseline', baseline_subset, run_a, source_ref);
    [s2, d2] = compareTables(datasetName(dataset_name), 'repeat_vs_first', run_a, run_b, source_ref);
    [s3, d3] = compareTables(datasetName(dataset_name), 'shuffle_vs_baseline', baseline_subset, run_shuffle, source_ref);
    [s4, d4] = compareTables(datasetName(dataset_name), 'resume_vs_baseline', baseline_subset, run_resume, source_ref);

    summary_tbl = [s1; s2; s3; s4];
    detail_tbl = [d1; d2; d3; d4];
end

function name = datasetName(dataset_name)
    name = string(dataset_name);
end

function tbl = alignByCaseId(tbl, case_ids)
    [tf, loc] = ismember(double(case_ids), double(tbl.case_id));
    if ~all(tf)
        error('Failed to align by case_id.');
    end
    tbl = tbl(loc, :);
end

function tbl = attachStage2LabelsAudit(tbl)
    has_los = logical(getColumn(tbl, 'has_los_path'));
    ratio = getColumn(tbl, 'bounce_to_los_ratio_mid');
    if all(isnan(ratio))
        ratio = zeros(height(tbl), 1);
    end
    is_current = logical(getColumn(tbl, 'is_nlos'));
    is_geo = ~has_los;
    is_bounce = has_los & (ratio >= 0.33);
    is_mixed = is_geo | is_bounce;

    tbl.is_nlos_current_0p20 = is_current;
    tbl.is_los_current_0p20 = ~is_current;
    tbl.is_nlos_geo = is_geo;
    tbl.is_los_geo = ~is_geo;
    tbl.is_nlos_bounce_0p33 = is_bounce;
    tbl.is_los_bounce_0p33 = has_los & ~is_bounce;
    tbl.is_nlos_mixed_0p33 = is_mixed;
    tbl.is_los_mixed_0p33 = ~is_mixed;
    tbl.label_schema = repmat("mixed_0p33_primary_with_dual_aux", height(tbl), 1);
end

function [summary_tbl, detail_tbl] = compareTables(dataset_name, check_id, ref_tbl, tst_tbl, source_ref)
    ref_tbl = sortrows(ref_tbl, 'case_id');
    tst_tbl = sortrows(tst_tbl, 'case_id');
    common_vars = intersect(ref_tbl.Properties.VariableNames, tst_tbl.Properties.VariableNames, 'stable');
    common_vars = setdiff(common_vars, {'error_msg'}, 'stable');
    mismatch_rows_any = false(height(ref_tbl), 1);
    detail_rows_local = {};
    max_abs_diff_global = 0;
    for i = 1:numel(common_vars)
        var_name = string(common_vars{i});
        a = ref_tbl.(var_name);
        b = tst_tbl.(var_name);
        if isnumeric(a) || islogical(a)
            [mismatch_rows, max_abs_diff_var] = compareNumeric(a, b);
            mismatch_rows_any = mismatch_rows_any | mismatch_rows;
            max_abs_diff_global = max(max_abs_diff_global, max_abs_diff_var);
            if any(mismatch_rows)
                detail_rows_local(end + 1, :) = {dataset_name, check_id, var_name, "numeric", sum(mismatch_rows), max_abs_diff_var}; %#ok<AGROW>
            end
        else
            mismatch_rows = compareStringish(a, b);
            mismatch_rows_any = mismatch_rows_any | mismatch_rows;
            if any(mismatch_rows)
                detail_rows_local(end + 1, :) = {dataset_name, check_id, var_name, "string", sum(mismatch_rows), NaN}; %#ok<AGROW>
            end
        end
    end
    pass_flag = ~any(mismatch_rows_any);
    classification = "no_issue";
    if ~pass_flag
        classification = "blocker";
    end
    details = sprintf('rows=%d, mismatch_rows=%d, max_abs_diff=%0.12g, compared_vars=%d', ...
        height(ref_tbl), sum(mismatch_rows_any), max_abs_diff_global, numel(common_vars));
    summary_tbl = table(string('B'), string(check_id), classification, string(ternaryStatus(pass_flag)), pass_flag, string(details), string(source_ref), ...
        'VariableNames', {'section', 'check_id', 'classification', 'status', 'pass_flag', 'details', 'source_ref'});

    if isempty(detail_rows_local)
        detail_tbl = table('Size', [0 6], 'VariableTypes', {'string', 'string', 'string', 'string', 'double', 'double'}, ...
            'VariableNames', {'dataset', 'check_id', 'variable_name', 'var_class', 'mismatch_rows', 'max_abs_diff'});
    else
        detail_tbl = cell2table(detail_rows_local, 'VariableNames', {'dataset', 'check_id', 'variable_name', 'var_class', 'mismatch_rows', 'max_abs_diff'});
    end
end

function [mismatch_rows, max_abs_diff] = compareNumeric(a, b)
    a = double(a);
    b = double(b);
    both_nan = isnan(a) & isnan(b);
    diff = abs(a - b);
    diff(both_nan) = 0;
    mismatch_rows = diff > 1e-6;
    if isempty(diff)
        max_abs_diff = 0;
    else
        max_abs_diff = max(diff(~isnan(diff)), [], 'omitnan');
        if isempty(max_abs_diff) || isnan(max_abs_diff)
            max_abs_diff = 0;
        end
    end
end

function mismatch_rows = compareStringish(a, b)
    sa = string(a);
    sb = string(b);
    mismatch_rows = sa ~= sb;
end

function out = toDoubleColumn(val)
    if isnumeric(val) || islogical(val)
        out = double(val);
    else
        out = str2double(string(val));
    end
end

function col = getColumn(tbl, name)
    if ismember(name, tbl.Properties.VariableNames)
        val = tbl.(name);
        if isnumeric(val) || islogical(val)
            col = double(val);
        else
            col = str2double(string(val));
        end
    else
        col = nan(height(tbl), 1);
    end
end

function writeAuditReport(out_dir, summary_tbl, detail_tbl, refs)
    fid = fopen(fullfile(out_dir, 'day5_deep_sanity_audit.md'), 'w');
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fprintf(fid, '# Day 5 Deep Sanity Audit\n\n');
    fprintf(fid, '- status: `verified`\n');
    fprintf(fid, '- source script: `%s`\n', refs.script);
    fprintf(fid, '- deep execution focus: `live subset rerun determinism + static leakage/integrity recheck`\n\n');

    blocker_tbl = summary_tbl(summary_tbl.classification == "blocker", :);
    caution_tbl = summary_tbl(summary_tbl.classification == "caution", :);
    no_issue_tbl = summary_tbl(summary_tbl.classification == "no_issue", :);

    fprintf(fid, '## Summary\n\n');
    fprintf(fid, '- blocker count: `%d`\n', height(blocker_tbl));
    fprintf(fid, '- caution count: `%d`\n', height(caution_tbl));
    fprintf(fid, '- no_issue count: `%d`\n\n', height(no_issue_tbl));

    fprintf(fid, '## Check Table\n\n');
    writeMarkdownTable(fid, summary_tbl, {'section', 'check_id', 'classification', 'status', 'pass_flag', 'details', 'source_ref'});

    if ~isempty(detail_tbl)
        fprintf(fid, '\n## Comparison Details\n\n');
        writeMarkdownTable(fid, detail_tbl, {'dataset', 'check_id', 'variable_name', 'var_class', 'mismatch_rows', 'max_abs_diff'});
    end

    fprintf(fid, '\n## Output Files\n\n');
    fprintf(fid, '- `day5_deep_sanity_checks.csv`\n');
    fprintf(fid, '- `day5_deep_sanity_comparison_details.csv`\n');
    fprintf(fid, '- `day5_determinism_subset_cases_day3.csv`\n');
    fprintf(fid, '- `day5_determinism_subset_cases_day4.csv`\n');
end

function writeMarkdownTable(fid, tbl, columns)
    fprintf(fid, '|');
    for i = 1:numel(columns)
        fprintf(fid, ' %s |', columns{i});
    end
    fprintf(fid, '\n|');
    for i = 1:numel(columns)
        fprintf(fid, ' --- |');
    end
    fprintf(fid, '\n');
    for r = 1:height(tbl)
        fprintf(fid, '|');
        for c = 1:numel(columns)
            value = tbl.(columns{c})(r);
            fprintf(fid, ' %s |', formatValue(value));
        end
        fprintf(fid, '\n');
    end
end

function out = formatValue(value)
    if iscell(value)
        value = value{1};
    end
    if isstring(value)
        out = char(value);
    elseif ischar(value)
        out = value;
    elseif islogical(value)
        out = string(value);
    elseif isnumeric(value)
        if isempty(value) || isnan(value)
            out = 'NaN';
        else
            out = num2str(value, '%.12g');
        end
    else
        out = char(string(value));
    end
end

function addCheck(section, check_id, classification, status, pass_flag, details, source_ref)
    global SANITY_CHECK_ROWS;
    SANITY_CHECK_ROWS{end + 1, 1} = string(section); %#ok<AGROW>
    SANITY_CHECK_ROWS{end, 2} = string(check_id);
    SANITY_CHECK_ROWS{end, 3} = string(classification);
    SANITY_CHECK_ROWS{end, 4} = string(status);
    SANITY_CHECK_ROWS{end, 5} = logical(pass_flag);
    SANITY_CHECK_ROWS{end, 6} = string(details);
    SANITY_CHECK_ROWS{end, 7} = string(source_ref);
end
