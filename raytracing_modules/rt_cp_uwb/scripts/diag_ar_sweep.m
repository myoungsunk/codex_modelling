projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'week25');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

ar_values = [0; 1; 3; 6; 10];
xpd_fixed = 35;
tx_pos = [0; 0; 1];
rx_pos = [2; 0; 1];
boresight_tx = [1; 0; 0];
boresight_rx = [-1; 0; 0];

scene = core.Scene();
paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 0);
assert(numel(paths) == 1 && paths.bounce_count == 0, 'Expected exactly one LoS path');

tx_ideal = antennas.makeIdealCpAntenna('right', tx_pos, boresight_tx, [0; 1; 0], [0; 0; 1]);
rx_ideal = antennas.makeIdealCpAntenna('right', rx_pos, boresight_rx, [0; 1; 0], [0; 0; 1]);

rows = cell(numel(ar_values), 7);
for i = 1:numel(ar_values)
    ar_db = ar_values(i);
    pattern = antennas.loadPatchPatternSynthetic( ...
        'ar_db_boresight', ar_db, ...
        'xpd_db_boresight', xpd_fixed, ...
        'peak_gain_dbi', 7.0, ...
        'fitted_cos_exp', 2.0);

    tx_patch = antennas.makeRealisticPatchAntenna(pattern, tx_pos, boresight_tx, [0; 1; 0], [0; 0; 1]);
    rx_patch = antennas.makeRealisticPatchAntenna(pattern, rx_pos, boresight_rx, [0; 1; 0], [0; 0; 1]);

    H_patch_patch = channel.buildChannel(paths, tx_patch, rx_patch, cfg.freqs);
    H_ideal_patch = channel.buildChannel(paths, tx_ideal, rx_patch, cfg.freqs);
    H_patch_ideal = channel.buildChannel(paths, tx_patch, rx_ideal, cfg.freqs);
    mid = ceil(numel(cfg.freqs) / 2);

    C = tx_patch.couplingMatrix(cfg.freqs(mid));
    rows(i, :) = {ar_db, ...
        safeRatio(abs(H_patch_patch(2, 1, mid)), abs(H_patch_patch(1, 1, mid))), ...
        safeRatio(abs(H_ideal_patch(2, 1, mid)), abs(H_ideal_patch(1, 1, mid))), ...
        safeRatio(abs(H_patch_ideal(2, 1, mid)), abs(H_patch_ideal(1, 1, mid))), ...
        gammaFp(H_patch_patch, cfg, tx_patch, rx_patch), ...
        gammaFp(H_ideal_patch, cfg, tx_ideal, rx_patch), ...
        safeRatio(abs(C(2, 1, 1)), abs(C(1, 1, 1)))};
end

resultTbl = cell2table(rows, 'VariableNames', { ...
    'ar_db', 'gamma_mid_patch_patch', 'gamma_mid_ideal_patch', 'gamma_mid_patch_ideal', ...
    'gamma_fp_patch_patch', 'gamma_fp_ideal_patch', 'coupling_ratio'});

csvPath = fullfile(outDir, 'diag_ar_sweep.csv');
writetable(resultTbl, csvPath);

fig = figure('Visible', 'off', 'Position', [100 100 960 420]);
subplot(1, 2, 1);
plot(resultTbl.ar_db, resultTbl.gamma_mid_patch_patch, 'o-', 'LineWidth', 1.5); hold on;
plot(resultTbl.ar_db, resultTbl.gamma_mid_ideal_patch, 's-', 'LineWidth', 1.5);
plot(resultTbl.ar_db, resultTbl.gamma_mid_patch_ideal, 'd-', 'LineWidth', 1.5);
xlabel('AR (dB)');
ylabel('\gamma mid');
legend('patch \rightarrow patch', 'ideal \rightarrow patch', 'patch \rightarrow ideal', 'Location', 'northwest');
title('AR Sweep: LoS \gamma at Mid-band');
grid on;

subplot(1, 2, 2);
plot(resultTbl.ar_db, resultTbl.coupling_ratio, 'o-', 'LineWidth', 1.5); hold on;
plot(resultTbl.ar_db, resultTbl.gamma_fp_ideal_patch, 's--', 'LineWidth', 1.5);
xlabel('AR (dB)');
ylabel('ratio');
legend('coupling ratio', '\gamma_{FP} ideal \rightarrow patch', 'Location', 'northwest');
title('AR Sweep: Coupling Response');
grid on;

pngPath = fullfile(outDir, 'diag_ar_sweep.png');
saveas(fig, pngPath);
close(fig);

reportPath = fullfile(outDir, 'diag_ar_sweep.md');
fid = fopen(reportPath, 'w');
assert(fid ~= -1, 'Failed to open %s', reportPath);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, '# AR Sweep Diagnostic\n\n');
fprintf(fid, '- XPD fixed at `%d dB`.\n', xpd_fixed);
fprintf(fid, '- Geometry: pure LoS, boresight aligned.\n');
fprintf(fid, '- If matched `patch -> patch` stays near zero while mixed links increase with AR, coupling is active but cancels in the matched link.\n\n');
fprintf(fid, '| AR (dB) | gamma mid patch-patch | gamma mid ideal-patch | gamma mid patch-ideal | gamma FP patch-patch | gamma FP ideal-patch | coupling ratio |\n');
fprintf(fid, '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n');
for i = 1:height(resultTbl)
    fprintf(fid, '| %.0f | %.6g | %.6g | %.6g | %.6g | %.6g | %.6g |\n', ...
        resultTbl.ar_db(i), resultTbl.gamma_mid_patch_patch(i), resultTbl.gamma_mid_ideal_patch(i), ...
        resultTbl.gamma_mid_patch_ideal(i), resultTbl.gamma_fp_patch_patch(i), ...
        resultTbl.gamma_fp_ideal_patch(i), resultTbl.coupling_ratio(i));
end

fprintf('Saved:\n');
fprintf('  %s\n', csvPath);
fprintf('  %s\n', pngPath);
fprintf('  %s\n', reportPath);

function gamma = gammaFp(H, cfg, tx_ant, rx_ant)
    feats = features.extractAllFeatures(H, cfg.freqs, ...
        'window_type', cfg.window_type, ...
        'feature_schema', 'full', ...
        'tx_ant', tx_ant, ...
        'rx_ant', rx_ant, ...
        'tx_handedness', 'R');
    gamma = feats.gamma_cp_3_fp_only;
end

function value = safeRatio(num, den)
    if den <= 0
        value = NaN;
    else
        value = num / den;
    end
end
