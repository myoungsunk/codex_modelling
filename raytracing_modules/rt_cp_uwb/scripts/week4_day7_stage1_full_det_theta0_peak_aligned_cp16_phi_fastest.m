script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

cfg = config.defaultConfig();
cfg.seed_base = 20260422;
cfg.seed_stage_id = 'stage1_full_det';
cfg.stage1_ffd_peak_align_local_posz = true;
cfg.stage1_ffd_peak_align_note = 'Map local FFD theta=0,phi=90 (+z) to Tx -z and Rx +z for Stage 1 patch_ffd cases.';
cfg.stage1_ffd_peak_local_theta_deg = 0.0;
cfg.stage1_ffd_peak_local_phi_deg = 90.0;
cfg.stage1_ffd_tx_peak_world_target = [0; 0; -1];
cfg.stage1_ffd_rx_peak_world_target = [0; 0; 1];
cfg.stage1_ffd_sample_order_policy = 'loadFfdPattern(sample_order=auto), expected resolved_sample_order=phi_fastest';
cfg.stage1_orientation_variant = 'theta0_phi90_local_posz_to_tx_minus_z_rx_plus_z';
cfg.enable_cp16_features = true;

out_dir = fullfile(repo_root, 'results', 'stage1_theta0_peak_aligned_cp16_phi_fastest');
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

cases = sweep.designLhsSweep(3000, 42);
stem = 'stage1_3000_theta0_peak_aligned_cp16_phi_fastest_det';
max_cases_env = string(getenv('STAGE1_THETA0_MAX_CASES'));
if strlength(max_cases_env) > 0
    max_cases = str2double(max_cases_env);
    if isfinite(max_cases) && max_cases > 0 && max_cases < height(cases)
        cases = cases(1:max_cases, :);
        stem = sprintf('stage1_smoke_%d_theta0_peak_aligned_cp16_phi_fastest_det', height(cases));
    end
end

runSweep(cases, cfg, out_dir, stem, 250, 'stage1');

function runSweep(cases, cfg, out_dir, stem, checkpoint_every, stage_name)
    checkpoint_path = fullfile(out_dir, [stem '_checkpoint.mat']);
    mat_path = fullfile(out_dir, [stem '.mat']);
    csv_path = fullfile(out_dir, [stem '.csv']);
    failed_path = fullfile(out_dir, [stem '_failed_cases.csv']);
    summary_path = fullfile(out_dir, [stem '_summary.md']);
    n = height(cases);
    start_idx = 1;
    all_feats = cell(n, 1);
    t_start = tic;
    elapsed_before = 0.0;

    requested_cfg = cfg;
    if exist(checkpoint_path, 'file') == 2
        C = load(checkpoint_path, 'cases_saved', 'all_feats', 'last_completed', 'elapsed_before', 'cfg_saved');
        if isfield(C, 'cases_saved') && height(C.cases_saved) == n && isequal(C.cases_saved.case_id, cases.case_id) && ...
                isfield(C, 'cfg_saved') && checkpointCfgMatches(requested_cfg, C.cfg_saved)
            all_feats = C.all_feats;
            start_idx = double(C.last_completed) + 1;
            elapsed_before = double(C.elapsed_before);
            cfg = C.cfg_saved;
            fprintf('Resuming %s theta0 sweep from case %d of %d\n', stage_name, start_idx, n);
        elseif isfield(C, 'cases_saved') && height(C.cases_saved) == n && isequal(C.cases_saved.case_id, cases.case_id)
            fprintf('Ignoring %s checkpoint because saved cfg signature differs from requested cfg.\n', stage_name);
        end
    end

    for i = start_idx:n
        try
            all_feats{i} = sweep.runOneCase(cases(i, :), cfg);
        catch ME
            all_feats{i} = struct('case_id', cases.case_id(i), 'failed', true, 'error_msg', ME.message);
        end
        if mod(i, 25) == 0 || i == n
            elapsed = elapsed_before + toc(t_start);
            fprintf('[%d/%d] elapsed %.1fs, ETA %.1fs\n', i, n, elapsed, elapsed / max(i, 1) * (n - i));
        end
        if mod(i, checkpoint_every) == 0 || i == n
            elapsed_running = elapsed_before + toc(t_start);
            cases_saved = cases; %#ok<NASGU>
            last_completed = i; %#ok<NASGU>
            cfg_saved = cfg; %#ok<NASGU>
            elapsed_before = elapsed_running; %#ok<NASGU>
            save(checkpoint_path, 'cases_saved', 'all_feats', 'last_completed', 'elapsed_before', 'cfg_saved');
            t_start = tic;
        end
    end

    results = sweep.structArrayToTable(all_feats);
    results = outerjoin(cases, results, 'Keys', 'case_id', 'MergeKeys', true, 'Type', 'left');
    elapsed = elapsedFromCheckpoint(checkpoint_path, toc(t_start));
    [ffd_sample_order_rhcp, ffd_sample_order_lhcp] = probeFfdSampleOrder(fileparts(fileparts(out_dir)));
    save(mat_path, 'results', 'cases', 'cfg', 'elapsed', 'ffd_sample_order_rhcp', 'ffd_sample_order_lhcp');
    writetable(results, csv_path);
    failed = results(logical(results.failed), :);
    writetable(failed, failed_path);
    writeSummary(summary_path, out_dir, results, elapsed, cfg, ffd_sample_order_rhcp, ffd_sample_order_lhcp, checkpoint_every, failed_path);
    fprintf('Saved:\n  %s\n  %s\n  %s\n', mat_path, csv_path, summary_path);
end

function tf = checkpointCfgMatches(requested_cfg, saved_cfg)
    fields = { ...
        'seed_base', ...
        'seed_stage_id', ...
        'stage1_ffd_peak_align_local_posz', ...
        'stage1_ffd_peak_local_theta_deg', ...
        'stage1_ffd_peak_local_phi_deg', ...
        'stage1_orientation_variant', ...
        'enable_cp16_features'};
    tf = cfgFieldsEqual(requested_cfg, saved_cfg, fields);
end

function tf = cfgFieldsEqual(a, b, fields)
    tf = true;
    for i = 1:numel(fields)
        name = fields{i};
        if ~isfield(a, name) || ~isfield(b, name)
            tf = false;
            return;
        end
        if ~isequaln(a.(name), b.(name))
            tf = false;
            return;
        end
    end
end

function elapsed = elapsedFromCheckpoint(checkpoint_path, fallback)
    elapsed = fallback;
    if exist(checkpoint_path, 'file') == 2
        C = load(checkpoint_path, 'elapsed_before');
        if isfield(C, 'elapsed_before') && double(C.elapsed_before) > 0
            elapsed = double(C.elapsed_before);
        end
    end
end

function [rhcp_order, lhcp_order] = probeFfdSampleOrder(repo_root)
    rhcp_order = "unresolved";
    lhcp_order = "unresolved";
    rhcp_path = fullfile(repo_root, 'RHCP_new_6G7G_11pts.ffd');
    lhcp_path = fullfile(repo_root, 'LHCP_new_6G7G_11pts.ffd');
    if exist(rhcp_path, 'file') == 2
        ffd = antennas.loadFfdPattern(rhcp_path, 'notes', 'stage1_theta0_summary_probe');
        rhcp_order = string(ffd.metadata.sample_order);
    end
    if exist(lhcp_path, 'file') == 2
        ffd = antennas.loadFfdPattern(lhcp_path, 'notes', 'stage1_theta0_summary_probe');
        lhcp_order = string(ffd.metadata.sample_order);
    end
end

function writeSummary(path, out_dir, results, elapsed, cfg, rhcp_order, lhcp_order, checkpoint_every, failed_path)
    valid = ~logical(results.failed);
    cp16_names = features.rhLhCp16FeatureNames();
    cp16_missing = setdiff(cp16_names, results.Properties.VariableNames);
    nan_count = 0;
    inf_count = 0;
    for i = 1:numel(cp16_names)
        if ismember(cp16_names{i}, results.Properties.VariableNames)
            x = double(results.(cp16_names{i}));
            nan_count = nan_count + sum(isnan(x));
            inf_count = inf_count + sum(isinf(x));
        end
    end
    fid = fopen(path, 'w');
    cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# Stage 1 Full Sweep (Theta0 Peak-Aligned FFD + CP16)\n\n');
    fprintf(fid, '- cases: %d\n', height(results));
    fprintf(fid, '- output_dir: %s\n', out_dir);
    fprintf(fid, '- elapsed_s: %.6f\n', elapsed);
    fprintf(fid, '- failed: %d\n', sum(~valid));
    fprintf(fid, '- failed_cases_csv: %s\n', failed_path);
    fprintf(fid, '- checkpoint_every: %d\n', checkpoint_every);
    fprintf(fid, '- seed_stage_id: %s\n', cfg.seed_stage_id);
    fprintf(fid, '- seed_base: %.0f\n', cfg.seed_base);
    fprintf(fid, '- cp16_enabled: true\n');
    fprintf(fid, '- cp16_missing_columns: %d\n', numel(cp16_missing));
    fprintf(fid, '- cp16_nan_count: %d\n', nan_count);
    fprintf(fid, '- cp16_inf_count: %d\n', inf_count);
    fprintf(fid, '- resolved_rhcp_sample_order: %s\n', rhcp_order);
    fprintf(fid, '- resolved_lhcp_sample_order: %s\n', lhcp_order);
    fprintf(fid, '- orientation_variant: %s\n', cfg.stage1_orientation_variant);
    fprintf(fid, '- local_peak_direction: theta=0 deg, phi=90 deg, vector=[0;0;1]\n');
    fprintf(fid, '- tx_peak_world_target: vector=[0;0;-1]\n');
    fprintf(fid, '- rx_peak_world_target: vector=[0;0;1]\n');
end
