projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'stage1');
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

tx_pos = [0; 0; 2.5];
tx_bore = [0; 0; -1];
tx_h = [1; 0; 0];
tx_v = [0; 1; 0];
rx_bore = [0; 0; 1];
rx_h = [1; 0; 0];
rx_v = [0; 1; 0];
patch = antennas.loadPatchPatternSynthetic();

cases = { ...
    'below',  [0.0; 0.0; 0.5]; ...
    'offaxis_45', [2.0; 0.0; 0.5]; ...
    'corner_55', [2.0; 2.0; 0.5]; ...
    'far_63', [4.0; 0.0; 0.5]};

rows = cell(size(cases, 1), 5);
scene = core.Scene();
for i = 1:size(cases, 1)
    name = cases{i, 1};
    rx_pos = cases{i, 2};

    tx = antennas.makeRealisticPatchAntenna(patch, tx_pos, tx_bore, tx_h, tx_v);
    rx = antennas.makeRealisticPatchAntenna(patch, rx_pos, rx_bore, rx_h, rx_v);

    paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 0);
    H = channel.buildChannel(paths, tx, rx, cfg.freqs);
    feats = features.extractAllFeatures(H, cfg.freqs, ...
        'window_type', cfg.window_type, ...
        'feature_schema', 'canonical18', ...
        'tx_ant', tx, ...
        'rx_ant', rx, ...
        'tx_handedness', 'R');

    los_vec = rx_pos - tx_pos;
    angle_deg = acosd(max(min(dot(los_vec / norm(los_vec), tx_bore), 1.0), -1.0));
    rows(i, :) = {name, angle_deg, feats.gamma_cp_3_fp_only, feats.gamma_cp_2_freq_db, norm(H(:, :, ceil(numel(cfg.freqs) / 2)), 'fro')};
end

tbl = cell2table(rows, 'VariableNames', {'case_name', 'los_angle_deg', 'gamma_cp_3_fp_only', 'gamma_cp_2_freq_db', 'midband_norm'});
csvPath = fullfile(outDir, 'diag_symmetric_boresight_position_diversity.csv');
writetable(tbl, csvPath);

fig = figure('Visible', 'off');
yyaxis left;
plot(tbl.los_angle_deg, tbl.gamma_cp_3_fp_only, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 7);
ylabel('\gamma_{CP,3} (LoS only)');
yyaxis right;
plot(tbl.los_angle_deg, tbl.gamma_cp_2_freq_db, 'rs--', 'LineWidth', 1.5, 'MarkerSize', 7);
ylabel('\gamma_{CP,2} (dB)');
xlabel('LoS angle from TX boresight (deg)');
title('Symmetric Boresight Position Diversity Diagnostic');
grid on;
saveas(fig, fullfile(outDir, 'diag_symmetric_boresight_position_diversity.png'));
close(fig);

mdPath = fullfile(outDir, 'diag_symmetric_boresight_position_diversity.md');
fid = fopen(mdPath, 'w');
assert(fid ~= -1, 'Failed to open %s', mdPath);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Symmetric Boresight Position Diversity Diagnostic\n\n');
fprintf(fid, '| case | LoS angle (deg) | gamma_cp_3_fp_only | gamma_cp_2_freq_db | midband_norm |\n');
fprintf(fid, '| --- | ---: | ---: | ---: | ---: |\n');
for i = 1:height(tbl)
    fprintf(fid, '| %s | %.3f | %.6g | %.6f | %.6g |\n', ...
        tbl.case_name{i}, tbl.los_angle_deg(i), tbl.gamma_cp_3_fp_only(i), tbl.gamma_cp_2_freq_db(i), tbl.midband_norm(i));
end
fprintf(fid, '\n');
fprintf(fid, 'Observation: if `gamma_cp_3_fp_only` stays near zero across angle, symmetric boresight + psi-only coupling still cancels for LoS.\n');

disp(tbl);
