script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

stage2_dir = fullfile(repo_root, 'results', 'stage2');
src_mat = fullfile(stage2_dir, 'stage2_900_ffd_det.mat');
if exist(src_mat, 'file') ~= 2
    run(fullfile(repo_root, 'scripts', 'week4_day5_stage2_full_det.m'));
end

S = load(src_mat, 'results', 'cases', 'cfg', 'elapsed');
results = S.results;
cases = S.cases;
cfg = S.cfg;
elapsed = S.elapsed;

valid = ~logical(results.failed);
has_los = logical(results.has_los_path);
ratio = double(results.bounce_to_los_ratio_mid);

is_nlos_current = logical(results.is_nlos);
is_nlos_geo = ~has_los;
is_nlos_bounce_033 = has_los & (ratio >= 0.33);
is_nlos_mixed_033 = is_nlos_geo | is_nlos_bounce_033;

results_relabel = results;
results_relabel.is_nlos_current_0p20 = is_nlos_current;
results_relabel.is_los_current_0p20 = logical(results.is_los);
results_relabel.is_nlos_geo = is_nlos_geo;
results_relabel.is_los_geo = ~is_nlos_geo;
results_relabel.is_nlos_bounce_0p33 = is_nlos_bounce_033;
results_relabel.is_los_bounce_0p33 = has_los & ~is_nlos_bounce_033;
results_relabel.is_nlos_mixed_0p33 = is_nlos_mixed_033;
results_relabel.is_los_mixed_0p33 = ~is_nlos_mixed_033;
results_relabel.is_nlos = is_nlos_mixed_033;
results_relabel.is_los = ~is_nlos_mixed_033;
results_relabel.label_schema = repmat({'mixed_0p33_primary_with_dual_aux'}, height(results_relabel), 1);

save(fullfile(stage2_dir, 'stage2_900_ffd_relabel_det.mat'), 'results_relabel', 'cases', 'cfg', 'elapsed');
writetable(results_relabel, fullfile(stage2_dir, 'stage2_900_ffd_relabel_det.csv'));
