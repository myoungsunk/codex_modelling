script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));
cfg = config.defaultConfig();
results_dir = fullfile(repo_root, 'results', 'week4');
if exist(results_dir, 'dir') ~= 7
    mkdir(results_dir);
end

room_types = {'A', 'B', 'C'};
room_size = [5, 4, 3];
selected_rows = table();

for room_idx = 1:numel(room_types)
    [grid_tbl, ~] = scenes.generateRoomTxRxGrid(room_types{room_idx}, room_size, 6, 100 + room_idx);
    grid_tbl = sortrows(grid_tbl, 'grid_layer');
    coarse = grid_tbl(find(grid_tbl.grid_layer == 1, 1, 'first'), :); %#ok<FNDSB>
    targeted = grid_tbl(find(grid_tbl.grid_layer == 2, 1, 'first'), :); %#ok<FNDSB>

    room_cases = table();
    room_cases.room_type = repmat(room_types(room_idx), 2, 1);
    room_cases.anchor_x = [coarse.anchor_x; targeted.anchor_x];
    room_cases.anchor_y = [coarse.anchor_y; targeted.anchor_y];
    room_cases.anchor_z = [coarse.anchor_z; targeted.anchor_z];
    room_cases.tag_x = [coarse.tag_x; targeted.tag_x];
    room_cases.tag_y = [coarse.tag_y; targeted.tag_y];
    room_cases.tag_z = [coarse.tag_z; targeted.tag_z];
    room_cases.grid_layer = [coarse.grid_layer; targeted.grid_layer];
    room_cases.los_distance_m = [coarse.los_distance_m; targeted.los_distance_m];
    room_cases.los_angle_from_anchor_bore_deg = [coarse.los_angle_from_anchor_bore_deg; targeted.los_angle_from_anchor_bore_deg];
    room_cases.los_angle_from_tag_bore_deg = [coarse.los_angle_from_tag_bore_deg; targeted.los_angle_from_tag_bore_deg];
    room_cases.snr_db = [18; 28];
    room_cases.xpol_coupling_db = [26; 34];
    room_cases.eps_r_multiplier = [0.95; 1.05];
    room_cases.dominant_wall_material = {'drywall'; 'brick'};
    room_cases.antenna_type = {'patch_ffd'; 'patch_ffd'};
    room_cases.material_name = room_cases.dominant_wall_material;
    selected_rows = [selected_rows; room_cases]; %#ok<AGROW>
end

selected_rows.case_id = (1:height(selected_rows)).';
selected_rows = movevars(selected_rows, 'case_id', 'Before', 1);

outputs = cell(height(selected_rows), 1);
t_start = tic;
for idx = 1:height(selected_rows)
    outputs{idx} = sweep.runOneCase(selected_rows(idx, :), cfg);
end
elapsed_s = toc(t_start);

results_tbl = sweep.structArrayToTable(outputs);
if ismember('case_id', results_tbl.Properties.VariableNames)
    results_tbl = sortrows(results_tbl, 'case_id');
end
selected_rows = sortrows(selected_rows, 'case_id');
result_only_names = setdiff(results_tbl.Properties.VariableNames, [{'case_id'}, selected_rows.Properties.VariableNames]);
joined = [selected_rows, results_tbl(:, result_only_names)];
summary_lines = {
    '# Week 4 Day 1 Smoke Test'
    ''
    sprintf('- cases_run: %d', height(results_tbl))
    sprintf('- elapsed_s: %.3f', elapsed_s)
    sprintf('- failed_cases: %d', sum(results_tbl.failed))
    sprintf('- los_cases: %d', sum(results_tbl.is_los))
    sprintf('- nlos_cases: %d', sum(results_tbl.is_nlos))
    ''
    '| case_id | room | layer | failed | num_paths | is_los | is_nlos | gamma_cp_3_fp_only |'
    '|---:|---|---:|---|---:|---:|---:|---:|'};

for idx = 1:height(joined)
    summary_lines{end + 1} = sprintf('| %d | %s | %d | %s | %d | %d | %d | %.4f |', ... %#ok<SAGROW>
        joined.case_id(idx), ...
        joined.room_type{idx}, ...
        joined.grid_layer(idx), ...
        tfText(joined.failed(idx)), ...
        joined.num_paths(idx), ...
        joined.is_los(idx), ...
        joined.is_nlos(idx), ...
        joined.gamma_cp_3_fp_only(idx));
end

writetable(joined, fullfile(results_dir, 'week4_day1_smoke_cases.csv'));
fid = fopen(fullfile(results_dir, 'week4_day1_smoke_summary.md'), 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', summary_lines{:});

function out = tfText(value)
    if value
        out = 'true';
    else
        out = 'false';
    end
end
