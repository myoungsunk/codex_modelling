%% Week 1 Final Review
clear; clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

report = sanity.runAllChecks();

fig = figure('Visible', 'off', 'Position', [100 100 1400 900]);
plots = dir(fullfile(projectRoot, 'results', 'sanity', 'plot_*.png'));
for idx = 1:min(12, numel(plots))
    subplot(3, 4, idx);
    img = imread(fullfile(plots(idx).folder, plots(idx).name));
    imshow(img);
    title(plots(idx).name, 'Interpreter', 'none');
end
saveas(fig, fullfile(projectRoot, 'results', 'sanity', 'summary_grid.png'));
close(fig);

names = fieldnames(report);
Check = cell(numel(names), 1);
Passed = false(numel(names), 1);
for idx = 1:numel(names)
    Check{idx} = names{idx};
    current = report.(names{idx});
    Passed(idx) = isfield(current, 'passed') && logical(current.passed);
end
T = table(Check, Passed);
disp(T);
writetable(T, fullfile(projectRoot, 'results', 'sanity', 'summary.csv'));
