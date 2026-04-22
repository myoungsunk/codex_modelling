projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
if ~exist(cfg.results_dir, 'dir')
    mkdir(cfg.results_dir);
end
logPath = fullfile(cfg.results_dir, 'week2_day1_convention_test.log');
if exist(logPath, 'file')
    delete(logPath);
end

diary(logPath);
cleanup = onCleanup(@() diary('off')); %#ok<NASGU>

fprintf('=== Week 2 Day 1: gamma_CP convention test ===\n');
fprintf('Timestamp: %s\n\n', datestr(now, 31));
fprintf('Convention: gamma_CP = |cross-pol| / |co-pol| = reversed-hand / same-hand\n');
fprintf('Expectation: LoS -> low gamma_CP, odd-bounce NLoS -> high gamma_CP\n\n');

freqs = linspace(cfg.f_center - cfg.bw/2, cfg.f_center + cfg.bw/2, cfg.n_freq).';
tx_pos = [0; 0; 1];

% Case 1: pure LoS, RHCP->RHCP should dominate.
scene_los = core.Scene({});
rx_pos_los = [3; 0; 1];
tx_los = antennas.makeIdealCpAntenna('right', tx_pos, [1; 0; 0], [0; 1; 0], [0; 0; 1]);
rx_los = antennas.makeIdealCpAntenna('right', rx_pos_los, [-1; 0; 0], [0; 1; 0], [0; 0; 1]);
paths_los = trace.enumeratePaths(scene_los, tx_pos, rx_pos_los, 0);
H_los = channel.buildChannel(paths_los, tx_los, rx_los, freqs);
feats_los = features.extractAllFeatures(H_los, freqs, 'tx_ant', tx_los, 'rx_ant', rx_los, 'fp_method', 'max_peak');
fprintf('[LoS] gamma_cp_1_freq_avg = %.6g\n', feats_los.gamma_cp_1_freq_avg);
fprintf('[LoS] gamma_cp_2_freq_db  = %.3f dB\n', feats_los.gamma_cp_2_freq_db);
assert(feats_los.gamma_cp_2_freq_db < -30.0, ...
    'LoS gamma_CP should be very low; got %.3f dB', feats_los.gamma_cp_2_freq_db);

% Case 2: single PEC bounce only, reversed-hand should dominate.
pec_mat = core.Material('kind', 'PEC', 'pec_tm_sign', -1.0, 'name', 'pec_floor');
slab = core.Surface('surface_id', 1, 'name', 'floor', 'point', [0; 0; 0], 'normal', [0; 0; 1], ...
    'u_axis', [1; 0; 0], 'v_axis', [0; 1; 0], 'half_u', 100.0, 'half_v', 100.0, 'material', pec_mat);
scene_bounce = core.Scene({slab});
rx_pos_bounce = [2; 0; 1];
tx_bounce = antennas.makeIdealCpAntenna('right', tx_pos, [0; 0; -1], [1; 0; 0], [0; 1; 0]);
rx_bounce = antennas.makeIdealCpAntenna('right', rx_pos_bounce, [0; 0; -1], [1; 0; 0], [0; 1; 0]);
paths_bounce = trace.enumeratePaths(scene_bounce, tx_pos, rx_pos_bounce, 1);
one_bounce = paths_bounce([paths_bounce.bounce_count] == 1);
assert(numel(one_bounce) == 1, 'Expected exactly one 1-bounce path, got %d', numel(one_bounce));
H_bounce = channel.buildChannel(one_bounce, tx_bounce, rx_bounce, freqs);
feats_bounce = features.extractAllFeatures(H_bounce, freqs, 'tx_ant', tx_bounce, 'rx_ant', rx_bounce, 'fp_method', 'max_peak');
fprintf('[1-bounce PEC] gamma_cp_1_freq_avg = %.6g\n', feats_bounce.gamma_cp_1_freq_avg);
fprintf('[1-bounce PEC] gamma_cp_2_freq_db  = %.3f dB\n', feats_bounce.gamma_cp_2_freq_db);
assert(feats_bounce.gamma_cp_2_freq_db > 30.0, ...
    'Odd-bounce gamma_CP should be high; got %.3f dB', feats_bounce.gamma_cp_2_freq_db);

% Re-run the relevant sanity checks so the plots are regenerated.
b1 = sanity.checkB1_cpHandednessReversal();
b2 = sanity.checkB2_evenBounceHandedness();
fprintf('\n[B1] min gamma_cp over angle sweep = %.3f dB\n', b1.metric);
fprintf('[B2] same/reversed preservation ratio = %.3f dB\n', b2.metric);
fprintf('[B2] gamma_cp = reversed/same = %.3f dB\n', b2.details.gamma_cp_db);

fprintf('\nArtifacts:\n');
fprintf('  %s\n', fullfile(cfg.sanity_dir, 'plot_b1_handedness_reversal.png'));
fprintf('  %s\n', fullfile(cfg.sanity_dir, 'plot_b2_even_bounce_handedness.png'));
fprintf('  %s\n', logPath);
fprintf('\nWeek 2 Day 1 convention test passed.\n');
