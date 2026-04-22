script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

cfg = config.defaultConfig();
out_dir = fullfile(repo_root, 'results', 'stage2');
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

cases = sweep.designStage2Cases(300, 42);
checkpoint_path = fullfile(out_dir, 'stage2_900_ffd_checkpoint.mat');
mat_path = fullfile(out_dir, 'stage2_900_ffd.mat');
csv_path = fullfile(out_dir, 'stage2_900_ffd.csv');
summary_path = fullfile(out_dir, 'stage2_900_ffd_summary.md');

n = height(cases);
checkpoint_every = 100;

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
        fprintf('Resuming Stage 2 full sweep from case %d of %d\n', start_idx, n);
    else
        fprintf('Checkpoint exists but case layout differs. Starting fresh.\n');
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
room_col = columnText(results, 'room_type');
rooms = {'A', 'B', 'C'};

fid = fopen(summary_path, 'w');
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Stage 2 Full Sweep\n\n');
fprintf(fid, '- cases: %d\n', height(cases));
fprintf(fid, '- elapsed_s: %.6f\n', elapsed);
fprintf(fid, '- elapsed_min: %.6f\n', elapsed / 60.0);
fprintf(fid, '- per_case_ms: %.6f\n', elapsed / height(cases) * 1000.0);
fprintf(fid, '- failed: %d\n', sum(logical(results.failed)));
fprintf(fid, '- checkpoint_every: %d\n\n', checkpoint_every);
fprintf(fid, '| room | n_total | n_valid | n_failed | n_los | n_nlos |\n');
fprintf(fid, '|---|---:|---:|---:|---:|---:|\n');
for idx = 1:numel(rooms)
    room_mask = strcmp(room_col, rooms{idx});
    fprintf(fid, '| %s | %d | %d | %d | %d | %d |\n', ...
        rooms{idx}, ...
        sum(room_mask), ...
        sum(room_mask & valid), ...
        sum(room_mask & ~valid), ...
        sum(room_mask & valid & logical(results.is_los)), ...
        sum(room_mask & valid & logical(results.is_nlos)));
end

fprintf('Stage 2 full: %.1f min (%.1f ms/case)\n', elapsed / 60.0, elapsed / height(cases) * 1000.0);
fprintf('Saved:\n  %s\n  %s\n  %s\n', mat_path, csv_path, summary_path);

function values = columnText(tbl, base_name)
    names = resolveColumnNames(tbl, base_name);
    values = cellstr(string(tbl.(names{1})));
end

function names = resolveColumnNames(tbl, base_name)
    direct = tbl.Properties.VariableNames(strcmp(tbl.Properties.VariableNames, base_name));
    if ~isempty(direct)
        names = direct;
        return;
    end
    prefixed = tbl.Properties.VariableNames(startsWith(tbl.Properties.VariableNames, [base_name '_']));
    if ~isempty(prefixed)
        names = prefixed;
        return;
    end
    error('Column %s not found', base_name);
end
