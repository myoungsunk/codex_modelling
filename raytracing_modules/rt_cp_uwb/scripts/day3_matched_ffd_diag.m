projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'day3');
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

ffdFiles = { ...
    'E:\0. CP Antenna\0. TRACK2_3_SIM\ANTENNA_SOURCE\RHCP_new_6G7G_11pts.ffd', ...
    'E:\0. CP Antenna\0. TRACK2_3_SIM\ANTENNA_SOURCE\LHCP_new_6G7G_11pts.ffd'};
assert(all(cellfun(@(p) exist(p, 'file') == 2, ffdFiles)), 'FFD files not found');

tx_pos = [0; 0; 2.0];
rx_z = 0.2;
angles_deg = [0, 30, 45, 60];
los_gamma = zeros(size(angles_deg));
nlos_gamma = zeros(size(angles_deg));

for idx = 1:numel(angles_deg)
    theta = deg2rad(angles_deg(idx));
    radius = (tx_pos(3) - rx_z) * tan(theta);
    rx_pos = [radius; 0; rx_z];

    tx = antennas.makeRealisticPatchAntennaFFD(ffdFiles{1}, ffdFiles{2}, tx_pos, [0; 0; -1], [1; 0; 0], [0; 1; 0]);
    rx = antennas.makeRealisticPatchAntennaFFD(ffdFiles{1}, ffdFiles{2}, rx_pos, [0; 0; 1], [1; 0; 0], [0; 1; 0]);

    sceneLos = core.Scene();
    los_paths = trace.enumeratePaths(sceneLos, tx_pos, rx_pos, 0);
    H_los = channel.buildChannel(los_paths, tx, rx, cfg.freqs);
    feats_los = features.extractAllFeatures(H_los, cfg.freqs, ...
        'window_type', cfg.window_type, ...
        'feature_schema', 'canonical18', ...
        'tx_ant', tx, ...
        'rx_ant', rx, ...
        'tx_handedness', 'R');
    los_gamma(idx) = feats_los.gamma_cp_3_fp_only;

    floor_mat = core.Material('kind', 'PEC', 'pec_tm_sign', -1.0, 'name', 'pec_floor');
    floor_surface = core.Surface('surface_id', 1, 'name', 'floor', 'point', [0; 0; 0], 'normal', [0; 0; 1], ...
        'u_axis', [1; 0; 0], 'v_axis', [0; 1; 0], 'half_u', 100.0, 'half_v', 100.0, 'material', floor_mat);
    sceneNlos = core.Scene({floor_surface});
    paths = trace.enumeratePaths(sceneNlos, tx_pos, rx_pos, 1);
    one_bounce = paths([paths.bounce_count] == 1);
    if isempty(one_bounce)
        nlos_gamma(idx) = NaN;
    else
        H_nlos = channel.buildChannel(one_bounce, tx, rx, cfg.freqs);
        feats_nlos = features.extractAllFeatures(H_nlos, cfg.freqs, ...
            'window_type', cfg.window_type, ...
            'feature_schema', 'canonical18', ...
            'tx_ant', tx, ...
            'rx_ant', rx, ...
            'tx_handedness', 'R');
        nlos_gamma(idx) = feats_nlos.gamma_cp_3_fp_only;
    end
end

tbl = table(angles_deg(:), los_gamma(:), nlos_gamma(:), 'VariableNames', {'angle_deg', 'gamma_los', 'gamma_nlos'});
writetable(tbl, fullfile(outDir, 'day3_matched_ffd_diag.csv'));

fig = figure('Visible', 'off');
semilogy(tbl.angle_deg, max(tbl.gamma_los, 1e-6), 'bo-', 'LineWidth', 1.5); hold on;
semilogy(tbl.angle_deg, max(tbl.gamma_nlos, 1e-6), 'rs--', 'LineWidth', 1.5);
xlabel('Off-axis angle (deg)');
ylabel('\gamma_{CP,3}');
title('Matched FFD patch-patch link: LoS vs 1-bounce NLoS');
legend('LoS', '1-bounce NLoS', 'Location', 'best');
grid on;
saveas(fig, fullfile(outDir, 'day3_matched_ffd_diag.png'));
close(fig);

fid = fopen(fullfile(outDir, 'day3_matched_ffd_diag.md'), 'w');
assert(fid ~= -1, 'Failed to open matched diag log');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Day 3 Matched FFD Diagnostic\n\n');
fprintf(fid, '| angle_deg | gamma_los | gamma_nlos |\n');
fprintf(fid, '| ---: | ---: | ---: |\n');
for idx = 1:height(tbl)
    fprintf(fid, '| %.1f | %.6g | %.6g |\n', tbl.angle_deg(idx), tbl.gamma_los(idx), tbl.gamma_nlos(idx));
end
fprintf(fid, '\n');
fprintf(fid, '- Scenario 1 check (boresight LoS): gamma = %.6g\n', tbl.gamma_los(1));
fprintf(fid, '- LoS monotonic nondecreasing: %d\n', all(diff(tbl.gamma_los) >= -1e-9));
fprintf(fid, '- NLoS dominates LoS at 45 deg: %d\n', tbl.gamma_nlos(tbl.angle_deg == 45) > tbl.gamma_los(tbl.angle_deg == 45));

disp(tbl);
