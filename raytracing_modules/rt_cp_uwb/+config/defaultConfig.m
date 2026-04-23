function cfg = defaultConfig()
% defaultConfig - baseline shared configuration for sanity checks.

    project_root = fileparts(fileparts(mfilename('fullpath')));
    cfg = struct();
    cfg.project_root = project_root;
    cfg.c0 = 299792458.0;
    cfg.f_center = 6.5e9;
    cfg.bw = 500e6;
    cfg.n_freq = 257;
    cfg.freqs = linspace(cfg.f_center - cfg.bw / 2.0, cfg.f_center + cfg.bw / 2.0, cfg.n_freq).';
    cfg.window_type = 'hann';
    cfg.circular_order = 'RL';
    cfg.convention = 'IEEE-RHCP';
    cfg.results_dir = fullfile(project_root, 'results');
    cfg.sanity_dir = fullfile(cfg.results_dir, 'sanity');
    cfg.sweep_dir = fullfile(cfg.results_dir, 'sweep');
    cfg.seed_base = 0;
    cfg.seed_stage_id = '';
    cfg.seed_replicate_id = 0;
end
