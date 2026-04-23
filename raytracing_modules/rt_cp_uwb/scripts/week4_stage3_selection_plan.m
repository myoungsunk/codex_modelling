projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

stage1Dir = fullfile(projectRoot, 'results', 'stage1');
stage3Dir = fullfile(projectRoot, 'results', 'stage3');
criteriaPath = fullfile(stage1Dir, 'stage3_selection_criteria.md');
cellsPath = fullfile(stage3Dir, 'stage3_reselection_cells_det.csv');
casesPath = fullfile(stage3Dir, 'hfss_case_list_det.csv');

cells = readtable(cellsPath, 'TextType', 'string');
cases = readtable(casesPath, 'TextType', 'string');

groupCounts = groupsummary(cases, 'group');
roomCounts = groupsummary(cases, {'group', 'room_type'});

fid = fopen(criteriaPath, 'w');
assert(fid ~= -1, 'Failed to open %s', criteriaPath);
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, '# Stage 3 HFSS SBR+ Selection Criteria\n\n');
fprintf(fid, '## Canonical Principle\n\n');
fprintf(fid, '- Canonical Stage 3 target: 30 cases\n');
for idx = 1:height(groupCounts)
    fprintf(fid, '- %s group: %d cases\n', groupCounts.group{idx}, groupCounts.GroupCount(idx));
end
fprintf(fid, '- Source dataset: deterministic Stage 2 relabeled export (`mixed_0p33_primary_with_dual_aux`)\n');
fprintf(fid, '- Legacy 45-case Stage 1-centric plan is archived and is **not** the canonical HFSS launch basis.\n\n');

fprintf(fid, '## Current Selected Regimes\n\n');
fprintf(fid, '| group | regime | n | n_pos | n_neg | delta_auc |\n');
fprintf(fid, '|---|---|---:|---:|---:|---:|\n');
for idx = 1:height(cells)
    fprintf(fid, '| %s | %s | %d | %d | %d | %.6f |\n', ...
        cells.group{idx}, cells.regime_desc{idx}, cells.n(idx), cells.n_pos(idx), cells.n_neg(idx), cells.delta_auc(idx));
end
fprintf(fid, '\n');

fprintf(fid, '## Current Candidate Distribution\n\n');
fprintf(fid, '| group | room | n |\n');
fprintf(fid, '|---|---|---:|\n');
for idx = 1:height(roomCounts)
    fprintf(fid, '| %s | %s | %d |\n', roomCounts.group{idx}, roomCounts.room_type{idx}, roomCounts.GroupCount(idx));
end
fprintf(fid, '\n');

fprintf(fid, 'Interpretation:\n');
fprintf(fid, '- GEO validation is Room C conditional in the current deterministic list.\n');
fprintf(fid, '- The current selector is Stage 2-centric and two-group (`GEO` / `BOUNCE`), not the legacy Stage 1 `G1/G2/G3` plan.\n');
fprintf(fid, '- `xpol_coupling_db_expected` should be treated as a MATLAB proxy / expected depolarization range, not a direct HFSS input.\n\n');

fprintf(fid, '## Canonical Outputs\n\n');
fprintf(fid, '- Regime CSV: `%s`\n', cellsPath);
fprintf(fid, '- Case CSV: `%s`\n', casesPath);

fprintf('Saved:\n  %s\n', criteriaPath);
