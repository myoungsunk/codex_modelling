function result = runWeek1PipelineCheck()
% runWeek1PipelineCheck - minimal end-to-end sanity for the Week 1 baseline.

    project_root = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(project_root));

    freqs = linspace(6.25e9, 6.75e9, 257).';
    tx_pos = [0; 0; 1];
    rx_pos = [2; 0; 1];

    wall = makeSurface(1, 'wall_y1', [1; 1; 1], [0; -1; 0], [1; 0; 0], [0; 0; 1], 5.0, 5.0, core.Material('kind', 'PEC', 'name', 'pec_wall'));
    scene = core.Scene({wall});
    paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 1);

    assert(~isempty(paths), 'expected at least one path');
    assert(any([paths.bounce_count] == 0), 'missing LOS path');
    assert(any([paths.bounce_count] == 1), 'missing 1-bounce path');

    tx_ant = antennas.makeIdealCpAntenna('right', tx_pos, [1; 0; 0], [0; 1; 0], [0; 0; 1]);
    rx_ant = antennas.makeIdealCpAntenna('right', rx_pos, [-1; 0; 0], [0; 1; 0], [0; 0; 1]);
    H = channel.buildChannel(paths, tx_ant, rx_ant, freqs);
    feats = features.extractAllFeatures(H, freqs, 'tx_ant', tx_ant, 'rx_ant', rx_ant);

    assert(isfield(feats, 'gamma_cp_1_freq_avg'), 'missing CP feature output');
    assert(isfield(feats, 'a_fp_1_norm_energy'), 'missing first-path feature output');
    assert(isfield(feats, 'rms_delay_spread'), 'missing CIR baseline output');

    wall_y = makeSurface(1, 'wall_y3', [0; 3; 1], [0; -1; 0], [1; 0; 0], [0; 0; 1], 5.0, 5.0, core.Material('kind', 'PEC', 'name', 'pec_y'));
    wall_x = makeSurface(2, 'wall_x3', [3; 0; 1], [-1; 0; 0], [0; 1; 0], [0; 0; 1], 5.0, 5.0, core.Material('kind', 'PEC', 'name', 'pec_x'));
    corner_scene = core.Scene({wall_y, wall_x});
    corner_paths = trace.enumeratePaths(corner_scene, [0; 0; 1], [3; 4; 1], 2);
    assert(any([corner_paths.bounce_count] == 2), 'missing 2-bounce corner path');

    sample_ffd = 'D:\codex\plot_data_save\reflector_3d_mode2\RHCP_6G7G.ffd';
    patch_summary = struct('loaded', false, 'xpd_db_boresight', NaN, 'ar_db_boresight', NaN);
    if exist(sample_ffd, 'file')
        patch = antennas.loadPatchPattern(sample_ffd);
        antennas.makeRealisticPatchAntenna(sample_ffd, tx_pos, [1; 0; 0], [0; 1; 0], [0; 0; 1]); %#ok<NASGU>
        patch_summary.loaded = true;
        patch_summary.xpd_db_boresight = patch.xpd_db_boresight;
        patch_summary.ar_db_boresight = patch.ar_db_boresight;
    end

    result = struct();
    result.path_count = numel(paths);
    result.corner_path_count = numel(corner_paths);
    result.feature_count = numel(fieldnames(feats));
    result.first_path_time_s = feats.t_fp_s;
    result.patch = patch_summary;

    fprintf('Week 1 pipeline sanity passed.\n');
    fprintf('  paths(one-wall) = %d\n', result.path_count);
    fprintf('  paths(corner)   = %d\n', result.corner_path_count);
    fprintf('  first path time = %.3f ns\n', result.first_path_time_s * 1e9);
    fprintf('  features        = %d\n', result.feature_count);
    if result.patch.loaded
        fprintf('  patch summary   = XPD %.2f dB, AR %.2f dB\n', result.patch.xpd_db_boresight, result.patch.ar_db_boresight);
    else
        fprintf('  patch summary   = skipped (sample FFD not found)\n');
    end
end

function surface = makeSurface(surface_id, name, point, normal, u_axis, v_axis, half_u, half_v, material)
    surface = struct( ...
        'surface_id', surface_id, ...
        'name', name, ...
        'point', point(:), ...
        'normal', normal(:), ...
        'u_axis', u_axis(:), ...
        'v_axis', v_axis(:), ...
        'half_u', half_u, ...
        'half_v', half_v, ...
        'material', material);
end
