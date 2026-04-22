script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

stage1_dir = fullfile(repo_root, 'results', 'stage1');

orig_auc = readtable(fullfile(stage1_dir, 'global_auc_summary.csv'), 'TextType', 'string');
det_auc = readtable(fullfile(stage1_dir, 'global_auc_summary_det.csv'), 'TextType', 'string');
orig_cells = readtable(fullfile(stage1_dir, 'conditional_auc_cells.csv'), 'TextType', 'string');
det_cells = readtable(fullfile(stage1_dir, 'conditional_auc_cells_det.csv'), 'TextType', 'string');

[delta_orig, cir_orig, cp_orig, joint_orig] = extractGlobal(orig_auc);
[delta_det, cir_det, cp_det, joint_det] = extractGlobal(det_auc);

top3_orig = topCells(orig_cells, 3);
top3_det = topCells(det_cells, 3);
top10_orig = topCells(orig_cells, 10);
top10_det = topCells(det_cells, 10);

id_top3_orig = cellIds(top3_orig);
id_top3_det = cellIds(top3_det);
id_top10_orig = cellIds(top10_orig);
id_top10_det = cellIds(top10_det);

top3_overlap = intersect(id_top3_orig, id_top3_det, 'stable');
top10_overlap = intersect(id_top10_orig, id_top10_det, 'stable');

summary_tbl = table();
summary_tbl.delta_auc_orig = delta_orig;
summary_tbl.delta_auc_det = delta_det;
summary_tbl.delta_auc_shift = delta_det - delta_orig;
summary_tbl.auc_cir_orig = cir_orig;
summary_tbl.auc_cir_det = cir_det;
summary_tbl.auc_cir_shift = cir_det - cir_orig;
summary_tbl.auc_cp_orig = cp_orig;
summary_tbl.auc_cp_det = cp_det;
summary_tbl.auc_cp_shift = cp_det - cp_orig;
summary_tbl.auc_joint_orig = joint_orig;
summary_tbl.auc_joint_det = joint_det;
summary_tbl.auc_joint_shift = joint_det - joint_orig;
summary_tbl.top3_overlap_count = numel(top3_overlap);
summary_tbl.top10_overlap_count = numel(top10_overlap);
summary_tbl.delta_stability_pass = abs(summary_tbl.delta_auc_shift) < 0.01;
writetable(summary_tbl, fullfile(stage1_dir, 'stage1_det_compare_summary.csv'));

overlap_tbl = table();
overlap_tbl.rank = (1:10).';
overlap_tbl.orig_cell = padIds(id_top10_orig, 10);
overlap_tbl.det_cell = padIds(id_top10_det, 10);
writetable(overlap_tbl, fullfile(stage1_dir, 'stage1_det_compare_topcells.csv'));

fid = fopen(fullfile(stage1_dir, 'stage1_det_compare.md'), 'w');
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Stage 1 Deterministic Comparison\n\n');
fprintf(fid, '- delta_auc_orig: %.6f\n', delta_orig);
fprintf(fid, '- delta_auc_det: %.6f\n', delta_det);
fprintf(fid, '- delta_auc_shift: %.6f\n', delta_det - delta_orig);
fprintf(fid, '- delta_stability_gate |shift| < 0.01: %s\n', passFail(abs(delta_det - delta_orig) < 0.01));
fprintf(fid, '- auc_cir_shift: %.6f\n', cir_det - cir_orig);
fprintf(fid, '- auc_cp_shift: %.6f\n', cp_det - cp_orig);
fprintf(fid, '- auc_joint_shift: %.6f\n', joint_det - joint_orig);
fprintf(fid, '- top3_overlap: %d / 3\n', numel(top3_overlap));
fprintf(fid, '- top10_overlap: %d / 10\n\n', numel(top10_overlap));

fprintf(fid, '## Top 3 Original\n\n');
writeTopTable(fid, top3_orig);
fprintf(fid, '\n## Top 3 Deterministic\n\n');
writeTopTable(fid, top3_det);
fprintf(fid, '\n## Top 10 Overlap IDs\n\n');
for i = 1:numel(top10_overlap)
    fprintf(fid, '- %s\n', top10_overlap(i));
end

function [delta_auc, auc_cir, auc_cp, auc_joint] = extractGlobal(tbl)
    auc_cir = tbl.auc(strcmp(tbl.model, 'CIR-only'));
    auc_cp = tbl.auc(strcmp(tbl.model, 'CP-only'));
    auc_joint = tbl.auc(strcmp(tbl.model, 'Joint'));
    delta_auc = auc_joint - auc_cir;
end

function tbl = topCells(cell_tbl, n_top)
    mask = isfinite(cell_tbl.delta_auc);
    tbl = cell_tbl(mask, :);
    tbl = sortrows(tbl, {'delta_auc', 'n'}, {'descend', 'descend'});
    tbl = tbl(1:min(n_top, height(tbl)), :);
end

function ids = cellIds(tbl)
    ids = strcat(string(tbl.x_var), ' | ', string(tbl.x_label), ' | ', string(tbl.y_var), ' | ', string(tbl.y_label));
end

function writeTopTable(fid, tbl)
    fprintf(fid, '| rank | x_var | x_label | y_var | y_label | n | delta_auc |\n');
    fprintf(fid, '|---:|---|---|---|---|---:|---:|\n');
    for i = 1:height(tbl)
        fprintf(fid, '| %d | %s | %s | %s | %s | %d | %.6f |\n', ...
            i, tbl.x_var{i}, char(string(tbl.x_label{i})), tbl.y_var{i}, char(string(tbl.y_label{i})), tbl.n(i), tbl.delta_auc(i));
    end
end

function values = padIds(ids, target_len)
    values = strings(target_len, 1);
    values(1:numel(ids)) = ids;
end

function txt = passFail(tf)
    if tf
        txt = 'PASS';
    else
        txt = 'FAIL';
    end
end
