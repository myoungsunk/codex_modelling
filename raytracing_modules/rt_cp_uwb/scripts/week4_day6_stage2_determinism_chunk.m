function week4_day6_stage2_determinism_chunk(mode, start_idx, end_idx, out_path, varargin)
% week4_day6_stage2_determinism_chunk - run a deterministic Stage 2 subset.

    script_dir = fileparts(mfilename('fullpath'));
    repo_root = fileparts(script_dir);
    addpath(repo_root);
    addpath(genpath(repo_root));

    p = inputParser;
    p.addParameter('preserve_legacy_seed_rule', true, @(x) islogical(x) || isnumeric(x));
    p.addParameter('seed_stage_id', 'stage2_full_det', @(x) ischar(x) || isstring(x));
    p.addParameter('seed_base', 20260422, @(x) isnumeric(x));
    p.addParameter('shuffle_seed', 20260422, @(x) isnumeric(x));
    p.parse(varargin{:});
    opts = p.Results;

    cfg = config.defaultConfig();
    if ~logical(opts.preserve_legacy_seed_rule)
        cfg.seed_stage_id = char(string(opts.seed_stage_id));
        cfg.seed_base = double(opts.seed_base);
    end

    cases = sweep.designStage2Cases(300, 42);
    mode = lower(char(string(mode)));
    switch mode
        case 'ordered'
            ordered_cases = cases;
        case 'shuffled'
            rng(double(opts.shuffle_seed), 'twister');
            perm = randperm(height(cases));
            ordered_cases = cases(perm, :);
        otherwise
            error('Unsupported mode: %s', mode);
    end

    start_idx = max(1, round(double(start_idx)));
    end_idx = min(height(ordered_cases), round(double(end_idx)));
    assert(end_idx >= start_idx, 'Invalid start/end indices');

    subset = ordered_cases(start_idx:end_idx, :);
    t_start = tic;
    results = sweep.runSweepBatch(subset, cfg, false);
    elapsed = toc(t_start);

    out_dir = fileparts(out_path);
    if exist(out_dir, 'dir') ~= 7
        mkdir(out_dir);
    end
    save(out_path, 'results', 'subset', 'cfg', 'mode', 'start_idx', 'end_idx', 'elapsed');
    fprintf('Saved %s (%s %d:%d, %.2fs)\n', out_path, mode, start_idx, end_idx, elapsed);
end
