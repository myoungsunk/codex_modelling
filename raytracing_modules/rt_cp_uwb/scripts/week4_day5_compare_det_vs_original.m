script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

stage2_dir = fullfile(repo_root, 'results', 'stage2');
stage3_dir = fullfile(repo_root, 'results', 'stage3');

orig_auc = readtable(fullfile(stage2_dir, 'label_reanalysis_auc.csv'), 'TextType', 'string');
det_auc = readtable(fullfile(stage2_dir, 'label_reanalysis_auc_det.csv'), 'TextType', 'string');
orig_room = readtable(fullfile(stage2_dir, 'label_reanalysis_room_auc.csv'), 'TextType', 'string');
det_room = readtable(fullfile(stage2_dir, 'label_reanalysis_room_auc_det.csv'), 'TextType', 'string');
orig_cells = readtable(fullfile(stage3_dir, 'stage3_reselection_cells.csv'), 'TextType', 'string');
det_cells = readtable(fullfile(stage3_dir, 'stage3_reselection_cells_det.csv'), 'TextType', 'string');
orig_cases = readtable(fullfile(stage3_dir, 'hfss_case_list.csv'), 'TextType', 'string');
det_cases = readtable(fullfile(stage3_dir, 'hfss_case_list_det.csv'), 'TextType', 'string');

delta_rows = {};

labels = intersect(string(orig_auc.label_name), string(det_auc.label_name), 'stable');
for idx = 1:numel(labels)
    name = labels(idx);
    row_orig = orig_auc(string(orig_auc.label_name) == name, :);
    row_det = det_auc(string(det_auc.label_name) == name, :);
    delta_rows(end + 1, :) = {char(name), "ALL", ... %#ok<AGROW>
        row_orig.delta_auc, row_det.delta_auc, row_det.delta_auc - row_orig.delta_auc, ...
        row_orig.auc_cir, row_det.auc_cir, row_orig.auc_joint, row_det.auc_joint};
end

mixed_name = "mixed_0p33";
rooms = intersect(string(orig_room.room_type), string(det_room.room_type), 'stable');
for idx = 1:numel(rooms)
    room = rooms(idx);
    row_orig = orig_room(string(orig_room.label_name) == mixed_name & string(orig_room.room_type) == room, :);
    row_det = det_room(string(det_room.label_name) == mixed_name & string(det_room.room_type) == room, :);
    if isempty(row_orig) || isempty(row_det)
        continue;
    end
    delta_rows(end + 1, :) = {char(mixed_name), char(room), ...
        row_orig.delta_auc, row_det.delta_auc, row_det.delta_auc - row_orig.delta_auc, ...
        row_orig.auc_cir, row_det.auc_cir, row_orig.auc_joint, row_det.auc_joint}; %#ok<AGROW>
end

delta_tbl = cell2table(delta_rows, 'VariableNames', ...
    {'label_name', 'room_type', 'delta_auc_orig', 'delta_auc_det', 'delta_auc_shift', ...
     'auc_cir_orig', 'auc_cir_det', 'auc_joint_orig', 'auc_joint_det'});
writetable(delta_tbl, fullfile(stage2_dir, 'deterministic_delta_compare.csv'));

mixed_all = delta_tbl(strcmp(delta_tbl.label_name, mixed_name) & strcmp(delta_tbl.room_type, 'ALL'), :);
delta_ok = abs(mixed_all.delta_auc_shift) < 0.01;

orig_regimes = strcat(string(orig_cells.group), " | ", string(orig_cells.regime_desc));
det_regimes = strcat(string(det_cells.group), " | ", string(det_cells.regime_desc));
regime_overlap = intersect(orig_regimes, det_regimes, 'stable');
regime_overlap_count = numel(regime_overlap);
orig_case_ids = double(orig_cases.case_id);
det_case_ids = double(det_cases.case_id);
case_overlap = intersect(orig_case_ids, det_case_ids, 'stable');
case_overlap_count = numel(case_overlap);

fid = fopen(fullfile(stage2_dir, 'deterministic_delta_compare.md'), 'w');
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Deterministic Rerun vs Original Comparison\n\n');
fprintf(fid, '- mixed@0.33 global delta shift: %.6f\n', mixed_all.delta_auc_shift);
fprintf(fid, '- mixed@0.33 stability gate (|shift| < 0.01): %s\n', passFail(delta_ok));
fprintf(fid, '- regime overlap count: %d / %d\n', regime_overlap_count, numel(orig_regimes));
fprintf(fid, '- HFSS candidate overlap count: %d / %d\n\n', case_overlap_count, numel(orig_case_ids));

fprintf(fid, '## Delta AUC Comparison\n\n');
fprintf(fid, '| label | room | delta orig | delta det | shift | cir orig | cir det | joint orig | joint det |\n');
fprintf(fid, '|---|---|---:|---:|---:|---:|---:|---:|---:|\n');
for idx = 1:height(delta_tbl)
    fprintf(fid, '| %s | %s | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f |\n', ...
        delta_tbl.label_name{idx}, delta_tbl.room_type{idx}, ...
        delta_tbl.delta_auc_orig(idx), delta_tbl.delta_auc_det(idx), delta_tbl.delta_auc_shift(idx), ...
        delta_tbl.auc_cir_orig(idx), delta_tbl.auc_cir_det(idx), ...
        delta_tbl.auc_joint_orig(idx), delta_tbl.auc_joint_det(idx));
end

fprintf(fid, '\n## Regime Overlap\n\n');
for idx = 1:numel(regime_overlap)
    fprintf(fid, '- %s\n', regime_overlap(idx));
end

fprintf(fid, '\n## Candidate Overlap\n\n');
fprintf(fid, '- case_ids: %s\n', mat2str(case_overlap'));

function txt = passFail(tf)
    if tf
        txt = 'PASS';
    else
        txt = 'FAIL';
    end
end
