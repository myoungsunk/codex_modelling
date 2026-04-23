script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

out_dir = fullfile(repo_root, 'results', 'code_audit', 'full_determinism_stage2');
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

baseline_path = fullfile(repo_root, 'results', 'stage2', 'stage2_900_ffd_det.mat');
resume_dir = fullfile(out_dir, 'resume_chunks');
shuffle_dir = fullfile(out_dir, 'shuffle_chunks');

baseline = loadResults(baseline_path);
resume_results = loadChunkTables(fullfile(resume_dir, '*.mat'));
shuffle_results = loadChunkTables(fullfile(shuffle_dir, '*.mat'));

baseline_second_half = baseline(double(baseline.case_id) >= 451, :);
resume_cmp = compareByCaseId(baseline_second_half, resume_results);
shuffle_cmp = compareByCaseId(baseline, shuffle_results);

seed_legacy = double(sweep.composeCaseSeed(17, 'component', 'case_rng'));
seed_stage1 = double(sweep.composeCaseSeed(17, 'stage_id', 'stage1_full_det', 'base_seed', 20260422, 'component', 'noise_awgn'));
seed_stage2 = double(sweep.composeCaseSeed(17, 'stage_id', 'stage2_full_det', 'base_seed', 20260422, 'component', 'noise_awgn'));
seed_component_case = double(sweep.composeCaseSeed(17, 'stage_id', 'stage2_full_det', 'base_seed', 20260422, 'component', 'case_rng'));
seed_component_noise = double(sweep.composeCaseSeed(17, 'stage_id', 'stage2_full_det', 'base_seed', 20260422, 'component', 'noise_awgn'));

[sample_pass, sample_details] = verifyLegacySampleMatch(repo_root);

audit_rows = { ...
    1, 'sample case match after backward-compatible seed patch', sample_pass, sample_details.max_abs_diff, 'max_abs_diff=0', sample_details.note; ...
    2, 'full 900 resume-second-half matches baseline by case_id', resume_cmp.pass, resume_cmp.max_abs_diff, '0', resume_cmp.note; ...
    3, 'full 900 shuffled-order run matches baseline by case_id', shuffle_cmp.pass, shuffle_cmp.max_abs_diff, '0', shuffle_cmp.note; ...
    4, 'salted seed differs across stage_id and component', (seed_stage1 ~= seed_stage2) && (seed_component_case ~= seed_component_noise), 0, 'stage1!=stage2 and case_rng!=noise_awgn', ...
        sprintf('legacy=%u, stage1_noise=%u, stage2_noise=%u, stage2_case_rng=%u', seed_legacy, seed_stage1, seed_stage2, seed_component_case)};
audit_tbl = cell2table(audit_rows, 'VariableNames', {'check_id', 'description', 'passed', 'metric', 'expected', 'details'});
writetable(audit_tbl, fullfile(out_dir, 'full_determinism_audit.csv'));

fid = fopen(fullfile(out_dir, 'full_determinism_audit.md'), 'w');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Full Stage 2 Determinism Audit\n\n');
fprintf(fid, '## Scope\n\n');
fprintf(fid, '- baseline: `results/stage2/stage2_900_ffd_det.mat`\n');
fprintf(fid, '- resume audit: rerun second half only (`case_id >= 451`) and compare against the baseline fresh run by `case_id`\n');
fprintf(fid, '- shuffle audit: rerun all 900 cases in a deterministic shuffled order and compare against the baseline fresh run by `case_id`\n');
fprintf(fid, '- seed hardening: verify new `composeCaseSeed(case_id, stage_id, base_seed, component)` separates stages/components while preserving backward compatibility when unset\n\n');

fprintf(fid, '## Checks\n\n');
fprintf(fid, '| check_id | description | status | metric | expected | details |\n');
fprintf(fid, '|---:|---|---|---|---|---|\n');
for idx = 1:height(audit_tbl)
    fprintf(fid, '| %d | %s | %s | %s | %s | %s |\n', ...
        audit_tbl.check_id(idx), ...
        escapePipe(audit_tbl.description{idx}), ...
        ternary(logical(audit_tbl.passed(idx)), 'PASS', 'FAIL'), ...
        escapePipe(audit_tbl.metric(idx)), ...
        escapePipe(audit_tbl.expected{idx}), ...
        escapePipe(audit_tbl.details{idx}));
end

fprintf(fid, '\n## Comparison summary\n\n');
fprintf(fid, '- resume rows compared: `%d`\n', resume_cmp.n_rows);
fprintf(fid, '- resume mismatched rows: `%d`\n', resume_cmp.n_row_mismatch);
fprintf(fid, '- resume numeric max abs diff: `%.3g`\n', resume_cmp.max_abs_diff);
fprintf(fid, '- shuffle rows compared: `%d`\n', shuffle_cmp.n_rows);
fprintf(fid, '- shuffle mismatched rows: `%d`\n', shuffle_cmp.n_row_mismatch);
fprintf(fid, '- shuffle numeric max abs diff: `%.3g`\n', shuffle_cmp.max_abs_diff);
fprintf(fid, '- salted stage separation: stage1 noise seed `%u`, stage2 noise seed `%u`\n', seed_stage1, seed_stage2);
fprintf(fid, '- salted component separation: stage2 case-rng seed `%u`, stage2 noise seed `%u`\n', seed_component_case, seed_component_noise);

function tbl = loadResults(mat_path)
    S = load(mat_path, 'results');
    tbl = S.results;
    tbl = sortrows(tbl, 'case_id');
end

function tbl = loadChunkTables(pattern)
    files = dir(pattern);
    assert(~isempty(files), 'No chunk files found for %s', pattern);
    chunks = cell(numel(files), 1);
    for idx = 1:numel(files)
        S = load(fullfile(files(idx).folder, files(idx).name), 'results');
        chunks{idx} = S.results;
    end
    tbl = vertcat(chunks{:});
    tbl = sortrows(tbl, 'case_id');
end

function cmp = compareByCaseId(reference, candidate)
    ref = sortrows(reference, 'case_id');
    cand = sortrows(candidate, 'case_id');
    common = intersect(ref.Properties.VariableNames, cand.Properties.VariableNames, 'stable');
    ref = ref(:, common);
    cand = cand(:, common);
    assert(isequal(double(ref.case_id), double(cand.case_id)), 'case_id mismatch between compared tables');

    n_rows = height(ref);
    n_row_mismatch = 0;
    max_abs_diff = 0.0;
    mismatch_examples = strings(0, 1);
    for row_idx = 1:n_rows
        row_ok = true;
        for vidx = 1:numel(common)
            name = common{vidx};
            a = ref.(name)(row_idx, :);
            b = cand.(name)(row_idx, :);
            if isnumeric(a) || islogical(a)
                da = double(a);
                db = double(b);
                if any(~isequaln(da, db))
                    row_ok = false;
                    diff_val = max(abs(da(:) - db(:)), [], 'omitnan');
                    if isempty(diff_val) || ~isfinite(diff_val)
                        diff_val = 0.0;
                    end
                    max_abs_diff = max(max_abs_diff, diff_val);
                end
            else
                if ~isequaln(string(a), string(b))
                    row_ok = false;
                end
            end
        end
        if ~row_ok
            n_row_mismatch = n_row_mismatch + 1;
            if numel(mismatch_examples) < 5
                mismatch_examples(end + 1, 1) = sprintf('case_id=%d', double(ref.case_id(row_idx))); %#ok<AGROW>
            end
        end
    end

    cmp = struct();
    cmp.pass = (n_row_mismatch == 0);
    cmp.n_rows = n_rows;
    cmp.n_row_mismatch = n_row_mismatch;
    cmp.max_abs_diff = max_abs_diff;
    cmp.note = sprintf('rows=%d, mismatched=%d, examples=%s', n_rows, n_row_mismatch, strjoin(mismatch_examples, ', '));
end

function [pass, details] = verifyLegacySampleMatch(repo_root)
    S = load(fullfile(repo_root, 'results', 'stage2', 'stage2_900_ffd_det.mat'), 'results', 'cases', 'cfg');
    stored = S.results;
    cases = S.cases;
    cfg = S.cfg;
    sample_ids = [1, 123, 450, 777, 900];
    max_abs_diff = 0.0;
    pass = true;
    for idx = 1:numel(sample_ids)
        case_id = sample_ids(idx);
        row = cases(double(cases.case_id) == case_id, :);
        rerun = sweep.runOneCase(row, cfg);
        stored_row = stored(double(stored.case_id) == case_id, :);
        [row_pass, row_diff] = compareStructToRow(rerun, stored_row);
        pass = pass && row_pass;
        max_abs_diff = max(max_abs_diff, row_diff);
    end
    details = struct('max_abs_diff', max_abs_diff, 'note', sprintf('sample_ids=%s', mat2str(sample_ids)));
end

function [pass, max_abs_diff] = compareStructToRow(S, row_tbl)
    pass = true;
    max_abs_diff = 0.0;
    fns = fieldnames(S);
    for idx = 1:numel(fns)
        name = fns{idx};
        if ~ismember(name, row_tbl.Properties.VariableNames)
            continue;
        end
        a = S.(name);
        b = row_tbl.(name)(1, :);
        if isnumeric(a) || islogical(a)
            da = double(a);
            db = double(b);
            same = isequaln(da, db);
            if ~same
                pass = false;
                diff_val = max(abs(da(:) - db(:)), [], 'omitnan');
                if isempty(diff_val) || ~isfinite(diff_val)
                    diff_val = 0.0;
                end
                max_abs_diff = max(max_abs_diff, diff_val);
            end
        else
            if ~isequaln(string(a), string(b))
                pass = false;
            end
        end
    end
end

function out = escapePipe(x)
    sx = string(x);
    if ismissing(sx)
        sx = "";
    end
    out = char(sx);
    out = strrep(out, '|', '\|');
end

function out = ternary(tf, a, b)
    if tf
        out = a;
    else
        out = b;
    end
end
