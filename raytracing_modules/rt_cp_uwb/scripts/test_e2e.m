projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

fprintf('Running end-to-end RT CP-UWB check...\n');

% Frequency sweep for the Week 1 baseband CIR pipeline.
freqs_hz = linspace(6.25e9, 6.75e9, 257).';

% Minimal one-slab scene with both LOS and one-bounce paths.
tx_pos = [0; 0; 1];
rx_pos = [2; 0; 1];
slab = struct( ...
    'surface_id', 1, ...
    'name', 'wall_y1', ...
    'point', [1; 1; 1], ...
    'normal', [0; -1; 0], ...
    'u_axis', [1; 0; 0], ...
    'v_axis', [0; 0; 1], ...
    'half_u', 5.0, ...
    'half_v', 5.0, ...
    'material', core.Material('kind', 'PEC', 'name', 'pec_slab'));
scene = core.Scene({slab});

% Ideal CP antennas so the path-level Jones response is exposed cleanly.
tx_ant = antennas.makeIdealCpAntenna('right', tx_pos, [1; 0; 0], [0; 1; 0], [0; 0; 1]);
rx_ant = antennas.makeIdealCpAntenna('right', rx_pos, [-1; 0; 0], [0; 1; 0], [0; 0; 1]);

% 1. Enumerate paths.
paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 1);
assert(~isempty(paths), 'E2E test failed: no paths were found');

% 2. Build H(f) in the antenna port basis (circular here).
H_f = channel.buildChannel(paths, tx_ant, rx_ant, freqs_hz);
assert(ndims(H_f) == 3, 'E2E test failed: H(f) must be Nr x Nt x Nf');

% 3. Convert the primary RHCP->RHCP channel to CIR.
H_primary = squeeze(H_f(1, 1, :));
[h_t, t_axis] = channel.ifftToCir(H_primary, freqs_hz, 'hann');
[~, idx_peak] = max(abs(h_t));
cir_peak_time_ns = t_axis(idx_peak) * 1e9;

% 4. Extract CP + CIR features from the full channel tensor.
feature_struct = features.extractAllFeatures(H_f, freqs_hz, ...
    'tx_ant', tx_ant, ...
    'rx_ant', rx_ant, ...
    'fp_method', 'max_peak');

% 5. Summarize outputs in a struct for manual inspection.
result = struct();
result.scene_name = 'single_slab_los_plus_reflection';
result.path_count = numel(paths);
result.bounce_counts = [paths.bounce_count];
result.path_delays_ns = [paths.delay_s] * 1e9;
result.channel_size = size(H_f);
result.cir_length = numel(h_t);
result.cir_peak_time_ns = cir_peak_time_ns;
result.first_path_time_ns = feature_struct.t_fp_s * 1e9;
result.primary_peak_mag = max(abs(h_t));
result.features = feature_struct;
result.paths = paths;

disp(result);
fprintf('E2E check completed successfully.\n');
