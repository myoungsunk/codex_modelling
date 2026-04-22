script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

results_dir = fullfile(repo_root, 'results', 'week4');
mat_path = fullfile(results_dir, 'smoke_stage2_225.mat');
stage1_path = fullfile(repo_root, 'results', 'stage1', 'stage1_3000_ffd.mat');
if exist(mat_path, 'file') ~= 2
    run(fullfile(repo_root, 'scripts', 'week4_day2_stage2_smoke.m'));
end

S2 = load(mat_path, 'results');
stage2 = S2.results;
stage2 = stage2(~logical(stage2.failed), :);

S1 = load(stage1_path, 'results');
stage1 = S1.results;
stage1 = stage1(~logical(stage1.failed), :);
stage1_room = stage1(strcmp(columnText(stage1, 'antenna_type'), 'patch_ffd'), :);

room_col = columnText(stage2, 'room_type');
rooms = {'A', 'B', 'C'};

diversity_rows = {
    'Stage1_patch', height(stage1_room), mean(stage1_room.num_paths), median(stage1_room.num_paths), mean(stage1_room.rms_delay_spread), median(stage1_room.rms_delay_spread)
    };
for idx = 1:numel(rooms)
    room_tbl = stage2(strcmp(room_col, rooms{idx}), :);
    diversity_rows(end + 1, :) = {rooms{idx}, height(room_tbl), mean(room_tbl.num_paths), median(room_tbl.num_paths), mean(room_tbl.rms_delay_spread), median(room_tbl.rms_delay_spread)}; %#ok<AGROW>
end
diversity_tbl = cell2table(diversity_rows, 'VariableNames', ...
    {'group', 'n', 'mean_num_paths', 'median_num_paths', 'mean_rms_delay_spread', 'median_rms_delay_spread'});

csv_path = fullfile(results_dir, 'scene_diversity_summary.csv');
md_path = fullfile(results_dir, 'scene_diversity_report.md');
plot_path = fullfile(results_dir, 'scene_diversity_distributions.png');
writetable(diversity_tbl, csv_path);

fig = figure('Visible', 'off', 'Position', [100 100 1100 450]);
subplot(1, 2, 1);
boxchart(categorical([repmat({'Stage1_patch'}, height(stage1_room), 1); room_col]), ...
    [double(stage1_room.num_paths); double(stage2.num_paths)]);
ylabel('Number of paths');
title('Path count distribution');
grid on;

subplot(1, 2, 2);
boxchart(categorical([repmat({'Stage1_patch'}, height(stage1_room), 1); room_col]), ...
    [double(stage1_room.rms_delay_spread); double(stage2.rms_delay_spread)]);
ylabel('RMS delay spread (s)');
title('RMS delay spread distribution');
grid on;
saveas(fig, plot_path);
close(fig);

fid = fopen(md_path, 'w');
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Week 4 Day 2 Scene Diversity\n\n');
fprintf(fid, '| group | n | mean_num_paths | median_num_paths | mean_rms_delay_spread | median_rms_delay_spread |\n');
fprintf(fid, '|---|---:|---:|---:|---:|---:|\n');
for idx = 1:height(diversity_tbl)
    fprintf(fid, '| %s | %d | %.3f | %.3f | %.4e | %.4e |\n', ...
        diversity_tbl.group{idx}, diversity_tbl.n(idx), diversity_tbl.mean_num_paths(idx), diversity_tbl.median_num_paths(idx), ...
        diversity_tbl.mean_rms_delay_spread(idx), diversity_tbl.median_rms_delay_spread(idx));
end

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
