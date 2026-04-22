script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

cfg = config.defaultConfig();
out_dir = fullfile(repo_root, 'results', 'stage1');
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

cases = sweep.designLhsSweep(3000, 42);
checkpoint_path = fullfile(out_dir, 'stage1_3000_ffd_det_checkpoint.mat');
mat_path = fullfile(out_dir, 'stage1_3000_ffd_det.mat');
csv_path = fullfile(out_dir, 'stage1_3000_ffd_det.csv');
summary_path = fullfile(out_dir, 'stage1_3000_ffd_det_summary.md');

n = height(cases);
checkpoint_every = 250;
start_idx = 1;
all_feats = cell(n, 1);
t_start = tic;
elapsed_before = 0.0;

if exist(checkpoint_path, 'file') == 2
    C = load(checkpoint_path, 'cases_saved', 'all_feats', 'last_completed', 'elapsed_before', 'cfg_saved');
    if isfield(C, 'cases_saved') && height(C.cases_saved) == n && isequal(C.cases_saved.case_id, cases.case_id)
        all_feats = C.all_feats;
        start_idx = double(C.last_completed) + 1;
        if isfield(C, 'elapsed_before')
            elapsed_before = double(C.elapsed_before);
        end
        if isfield(C, 'cfg_saved')
            cfg = C.cfg_saved;
        end
        fprintf('Resuming deterministic Stage 1 full sweep from case %d of %d\n', start_idx, n);
    else
        fprintf('Deterministic Stage 1 checkpoint exists but case layout differs. Starting fresh.\n');
    end
end

for i = start_idx:n
    try
        all_feats{i} = sweep.runOneCase(cases(i, :), cfg);
    catch ME
        all_feats{i} = struct( ...
            'case_id', cases.case_id(i), ...
            'failed', true, ...
            'error_msg', ME.message);
    end

    if mod(i, 50) == 0 || i == n
        elapsed = elapsed_before + toc(t_start);
        eta = elapsed / max(i, 1) * (n - i);
        fprintf('[%d/%d] elapsed %.1fs, ETA %.1fs\n', i, n, elapsed, eta);
    end

    if mod(i, checkpoint_every) == 0 || i == n
        elapsed_running = elapsed_before + toc(t_start);
        cases_saved = cases; %#ok<NASGU>
        last_completed = i; %#ok<NASGU>
        cfg_saved = cfg; %#ok<NASGU>
        elapsed_before = elapsed_running; %#ok<NASGU>
        save(checkpoint_path, 'cases_saved', 'all_feats', 'last_completed', 'elapsed_before', 'cfg_saved');
        t_start = tic;
    end
end

results = sweep.structArrayToTable(all_feats);
results = outerjoin(cases, results, 'Keys', 'case_id', 'MergeKeys', true, 'Type', 'left');

elapsed = 0.0;
if exist(checkpoint_path, 'file') == 2
    C = load(checkpoint_path, 'elapsed_before');
    if isfield(C, 'elapsed_before')
        elapsed = double(C.elapsed_before);
    end
end
if elapsed <= 0.0
    elapsed = toc(t_start);
end

save(mat_path, 'results', 'cases', 'cfg', 'elapsed');
writetable(results, csv_path);

valid = ~logical(results.failed);

fid = fopen(summary_path, 'w');
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Stage 1 Full Sweep (Deterministic)\n\n');
fprintf(fid, '- cases: %d\n', height(cases));
fprintf(fid, '- elapsed_s: %.6f\n', elapsed);
fprintf(fid, '- elapsed_min: %.6f\n', elapsed / 60.0);
fprintf(fid, '- per_case_ms: %.6f\n', elapsed / height(cases) * 1000.0);
fprintf(fid, '- failed: %d\n', sum(logical(results.failed)));
fprintf(fid, '- valid_los: %d\n', sum(logical(results.is_los) & valid));
fprintf(fid, '- valid_nlos: %d\n', sum(logical(results.is_nlos) & valid));
fprintf(fid, '- checkpoint_every: %d\n', checkpoint_every);
fprintf(fid, '- deterministic_seed_rule: case_id -> runOneCase local rng + injectSnr global randn\n');

fprintf('Stage 1 deterministic full: %.2f min, %.3f ms/case\n', elapsed / 60.0, elapsed / height(cases) * 1000.0);
fprintf('Saved:\n  %s\n  %s\n  %s\n', mat_path, csv_path, summary_path);
