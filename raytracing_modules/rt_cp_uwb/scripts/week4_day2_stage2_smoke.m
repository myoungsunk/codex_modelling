script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

cfg = config.defaultConfig();
results_dir = fullfile(repo_root, 'results', 'week4');
if exist(results_dir, 'dir') ~= 7
    mkdir(results_dir);
end

cases = sweep.designStage2Cases(75, 42);
t_start = tic;
results = sweep.runSweepBatch(cases, cfg, true);
elapsed = toc(t_start);

csv_path = fullfile(results_dir, 'smoke_stage2_225.csv');
mat_path = fullfile(results_dir, 'smoke_stage2_225.mat');
summary_path = fullfile(results_dir, 'smoke_stage2_225_summary.md');
writetable(results, csv_path);
save(mat_path, 'results', 'cases', 'cfg', 'elapsed');

failed = logical(results.failed);
valid = ~failed;
room_col = columnText(results, 'room_type');
rooms = {'A', 'B', 'C'};
summary_lines = {
    '# Week 4 Day 2 Stage 2 Smoke'
    ''
    sprintf('- elapsed_s: %.3f', elapsed)
    sprintf('- elapsed_ms_per_case: %.3f', elapsed / height(results) * 1000.0)
    sprintf('- cases_total: %d', height(results))
    sprintf('- failed_cases: %d', sum(failed))
    sprintf('- failed_rate: %.4f', mean(failed))
    ''
    '| room | n_total | n_valid | n_failed | n_los | n_nlos |'
    '|---|---:|---:|---:|---:|---:|'};

for idx = 1:numel(rooms)
    room_mask = strcmp(room_col, rooms{idx});
    room_valid = room_mask & valid;
    summary_lines{end + 1} = sprintf('| %s | %d | %d | %d | %d | %d |', ... %#ok<SAGROW>
        rooms{idx}, ...
        sum(room_mask), ...
        sum(room_valid), ...
        sum(room_mask & failed), ...
        sum(room_valid & logical(results.is_los)), ...
        sum(room_valid & logical(results.is_nlos)));
end

numeric_names = results.Properties.VariableNames(varfun(@isnumeric, results, 'OutputFormat', 'uniform'));
warn_lines = {};
for idx = 1:numel(numeric_names)
    x = results.(numeric_names{idx});
    if isnumeric(x)
        n_bad = sum(~isfinite(x(valid)));
        if n_bad > 0
            warn_lines{end + 1} = sprintf('- %s: %d non-finite values', numeric_names{idx}, n_bad); %#ok<SAGROW>
        end
    end
end
if isempty(warn_lines)
    warn_lines = {'- none'};
end

summary_lines{end + 1} = '';
summary_lines{end + 1} = '## Non-finite Feature Check';
summary_lines{end + 1} = '';
summary_lines = [summary_lines; warn_lines(:)]; %#ok<AGROW>

fid = fopen(summary_path, 'w');
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', summary_lines{:});

fprintf('Stage 2 smoke: %.1fs (%.1f ms/case)\n', elapsed, elapsed / height(results) * 1000.0);

function values = columnText(tbl, base_name)
    names = {base_name, [base_name '_cases_tbl'], [base_name '_results_tbl']};
    for i = 1:numel(names)
        if ismember(names{i}, tbl.Properties.VariableNames)
            values = cellstr(string(tbl.(names{i})));
            return;
        end
    end
    error('Column %s not found', base_name);
end
