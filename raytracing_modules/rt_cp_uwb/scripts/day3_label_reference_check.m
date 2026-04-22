projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'day3');
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

cases = sweep.designLhsSweep(40, 7);
rows = cell(height(cases), 6);
all_same = true;
for idx = 1:height(cases)
    ideal_case = cases(idx, :);
    patch_case = cases(idx, :);
    ideal_case.antenna_type = {'ideal'};
    patch_case.antenna_type = {'patch'};
    out_ideal = sweep.runOneCase(ideal_case, cfg);
    out_patch = sweep.runOneCase(patch_case, cfg);
    same_label = isequal(logical(out_ideal.is_nlos), logical(out_patch.is_nlos)) && ...
        isequal(logical(out_ideal.is_los), logical(out_patch.is_los));
    all_same = all_same && same_label;
    rows(idx, :) = {cases.case_id(idx), out_ideal.is_los, out_patch.is_los, out_ideal.is_nlos, out_patch.is_nlos, same_label};
end

tbl = cell2table(rows, 'VariableNames', {'case_id', 'ideal_is_los', 'patch_is_los', 'ideal_is_nlos', 'patch_is_nlos', 'same_label'});
writetable(tbl, fullfile(outDir, 'day3_label_reference_check.csv'));

fid = fopen(fullfile(outDir, 'day3_label_reference_check.md'), 'w');
assert(fid ~= -1, 'Failed to open label log');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Day 3 Label Reference Check\n\n');
fprintf(fid, '- all labels identical between ideal and patch: %d\n', all_same);
fprintf(fid, '- mismatched cases: %d\n', sum(~tbl.same_label));

fprintf('All labels identical: %d\n', all_same);
disp(tbl(1:min(10, height(tbl)), :));
