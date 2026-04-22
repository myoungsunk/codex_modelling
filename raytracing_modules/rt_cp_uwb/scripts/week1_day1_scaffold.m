projectRoot = 'D:\codex\raytracing_modules\rt_cp_uwb';
folders = {
    '+core', '+trace', '+channel', '+features', ...
    '+antennas', '+sanity', '+config', 'scripts', ...
    'data/patch_patterns', 'results/sanity', 'results/stage1'
};

for k = 1:numel(folders)
    p = fullfile(projectRoot, folders{k});
    if ~exist(p, 'dir')
        mkdir(p);
    end
end

addpath(genpath(projectRoot));
fprintf('Scaffold ready: %s\n', projectRoot);
