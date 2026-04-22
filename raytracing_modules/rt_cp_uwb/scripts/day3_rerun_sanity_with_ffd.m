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

freqs = cfg.freqs;
pec_mat = core.Material('kind', 'PEC', 'pec_tm_sign', -1.0, 'name', 'pec_floor');
slab = core.Surface('surface_id', 1, 'name', 'floor', 'point', [0; 0; 0], 'normal', [0; 0; 1], ...
    'u_axis', [1; 0; 0], 'v_axis', [0; 1; 0], 'half_u', 100.0, 'half_v', 100.0, 'material', pec_mat);
scene = core.Scene({slab});

tx_pos = [0; 0; 1];
tx = antennas.makeRealisticPatchAntennaFFD(ffdFiles{1}, ffdFiles{2}, tx_pos, [0; 0; -1], [1; 0; 0], [0; 1; 0]);
incidence_angles = 0:10:70;
ratio_db = zeros(size(incidence_angles));
mid = ceil(numel(freqs) / 2);

for idx = 1:numel(incidence_angles)
    th = deg2rad(incidence_angles(idx));
    rx_pos = [2 * tan(th); 0; 1];
    rx = antennas.makeRealisticPatchAntennaFFD(ffdFiles{1}, ffdFiles{2}, rx_pos, [0; 0; -1], [1; 0; 0], [0; 1; 0]);
    paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 1);
    one_bounce = paths([paths.bounce_count] == 1);
    H = channel.buildChannel(one_bounce, tx, rx, freqs);
    P_same = abs(H(1, 1, mid))^2;
    P_rev = abs(H(2, 1, mid))^2;
    ratio_db(idx) = 10 * log10(max(P_rev, 1e-30) / max(P_same, 1e-30));
end

b3 = sanity.checkB3_brewsterAngle();

fig = figure('Visible', 'off');
plot(incidence_angles, ratio_db, 'bo-', 'LineWidth', 1.5);
xlabel('Incidence angle (deg)');
ylabel('reversed/same (dB)');
title(sprintf('FFD patch B1 rerun, min %.2f dB', min(ratio_db)));
grid on;
saveas(fig, fullfile(outDir, 'day3_b1_ffd_patch.png'));
close(fig);

fid = fopen(fullfile(outDir, 'day3_rerun_sanity_with_ffd.md'), 'w');
assert(fid ~= -1, 'Failed to open sanity rerun log');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Day 3 Sanity Rerun With FFD\n\n');
fprintf(fid, '- B1 (FFD patch) min reversed/same dB: %.6f\n', min(ratio_db));
fprintf(fid, '- B1 (FFD patch) max reversed/same dB: %.6f\n', max(ratio_db));
fprintf(fid, '- B3 metric: %s\n', mat2str(b3.metric, 6));
fprintf(fid, '- B3 passed: %d\n', b3.passed);
fprintf(fid, '\n| angle_deg | gamma_db |\n| ---: | ---: |\n');
for idx = 1:numel(incidence_angles)
    fprintf(fid, '| %.1f | %.6f |\n', incidence_angles(idx), ratio_db(idx));
end

disp(table(incidence_angles(:), ratio_db(:), 'VariableNames', {'angle_deg', 'gamma_db'}));
