projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'stage1');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

tx_pos = [0; 0; 1];
rx_pos = [2; 0; 0.2];
tx_bore = [1; 0; 0];
tx_h = [0; 1; 0];
tx_v = [0; 0; 1];
rx_bore = [0; 0; 1];
rx_h = [1; 0; 0];
rx_v = [0; 1; 0];

tx_ideal = antennas.makeIdealCpAntenna('right', tx_pos, tx_bore, tx_h, tx_v);
rx_ideal = antennas.makeIdealCpAntenna('right', rx_pos, rx_bore, rx_h, rx_v);
tx_patch = antennas.makeRealisticPatchAntenna('synthetic', tx_pos, tx_bore, tx_h, tx_v);
rx_patch = antennas.makeRealisticPatchAntenna('synthetic', rx_pos, rx_bore, rx_h, rx_v);

scene_los = core.Scene();
scene_pec = scenes.makeSingleSlabScene('metal_pec', [4, 4], [1; 0; 0], [0; 0; 1]);

paths_los = trace.enumeratePaths(scene_los, tx_pos, rx_pos, 0);
paths_pec = trace.enumeratePaths(scene_pec, tx_pos, rx_pos, 1);
paths_1b = paths_pec([paths_pec.bounce_count] == 1);
assert(numel(paths_los) == 1, 'Expected one LoS path');
assert(numel(paths_1b) == 1, 'Expected one PEC 1-bounce path');

rows = [ ...
    summarizeCase('ideal_los', tx_ideal, rx_ideal, paths_los, cfg); ...
    summarizeCase('patch_los', tx_patch, rx_patch, paths_los, cfg); ...
    summarizeCase('ideal_pec_1bounce', tx_ideal, rx_ideal, paths_1b, cfg); ...
    summarizeCase('patch_pec_1bounce', tx_patch, rx_patch, paths_1b, cfg)];

resultTbl = cell2table(rows, 'VariableNames', { ...
    'case_name', 'path_type', 'abs_H11_mid', 'abs_H21_mid', 'gamma_mid', ...
    'gamma_fp', 'tx_ar_db', 'tx_xpd_db', 'rx_ar_db', 'rx_xpd_db'});

csvPath = fullfile(outDir, 'diag_asymmetric_boresight.csv');
writetable(resultTbl, csvPath);

fig = figure('Visible', 'off', 'Position', [100 100 900 420]);
subplot(1, 2, 1);
bar(categorical(resultTbl.case_name), resultTbl.gamma_mid);
ylabel('\gamma mid');
title('Asymmetric Boresight: Mid-band \gamma');
grid on;

subplot(1, 2, 2);
bar(categorical(resultTbl.case_name), resultTbl.gamma_fp);
ylabel('\gamma_{FP}');
title('Asymmetric Boresight: First-path \gamma');
grid on;

pngPath = fullfile(outDir, 'diag_asymmetric_boresight.png');
saveas(fig, pngPath);
close(fig);

patch_los_gamma = resultTbl.gamma_fp(strcmp(resultTbl.case_name, 'patch_los'));
reportPath = fullfile(outDir, 'diag_asymmetric_boresight.md');
fid = fopen(reportPath, 'w');
assert(fid ~= -1, 'Failed to open %s', reportPath);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, '# Asymmetric Boresight Diagnostic\n\n');
fprintf(fid, '- TX anchor boresight: `[1; 0; 0]`\n');
fprintf(fid, '- RX tag boresight: `[0; 0; 1]`\n');
fprintf(fid, '- Geometry note: RX is placed near the floor (`z = 0.2 m`) to avoid the exact `90 deg` side-null of the synthetic `cos^n` patch pattern.\n');
fprintf(fid, '- Expectation: `patch_los` should show non-zero handedness leakage, while `ideal_los` remains near zero.\n\n');
fprintf(fid, '| case | path | |H(1,1)| mid | |H(2,1)| mid | gamma mid | gamma FP | TX AR/XPD (dB) | RX AR/XPD (dB) |\n');
fprintf(fid, '| --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n');
for i = 1:height(resultTbl)
    fprintf(fid, '| %s | %s | %.6g | %.6g | %.6g | %.6g | %.2f / %.2f | %.2f / %.2f |\n', ...
        resultTbl.case_name{i}, resultTbl.path_type{i}, resultTbl.abs_H11_mid(i), ...
        resultTbl.abs_H21_mid(i), resultTbl.gamma_mid(i), resultTbl.gamma_fp(i), ...
        resultTbl.tx_ar_db(i), resultTbl.tx_xpd_db(i), resultTbl.rx_ar_db(i), resultTbl.rx_xpd_db(i));
end
fprintf(fid, '\n');
fprintf(fid, '## Day 1 Checkpoint\n\n');
fprintf(fid, '- patch-patch gamma_LoS (FP) = %.6g\n', patch_los_gamma);
fprintf(fid, '- verdict: **%s**\n', ternary(patch_los_gamma > 0.05, 'GO', 'REVIEW_NEEDED'));

fprintf('Saved:\n');
fprintf('  %s\n', csvPath);
fprintf('  %s\n', pngPath);
fprintf('  %s\n', reportPath);

function row = summarizeCase(label, tx_ant, rx_ant, paths, cfg)
    H = channel.buildChannel(paths, tx_ant, rx_ant, cfg.freqs);
    feats = features.extractAllFeatures(H, cfg.freqs, ...
        'window_type', cfg.window_type, ...
        'feature_schema', 'full', ...
        'tx_ant', tx_ant, ...
        'rx_ant', rx_ant, ...
        'tx_handedness', 'R');
    mid = ceil(numel(cfg.freqs) / 2);
    tx_dir = paths(1).launch_dir;
    rx_dir = -paths(1).arrival_dir;
    [tx_ar, tx_xpd] = antennas.couplingFromAngle(tx_ant, tx_dir);
    [rx_ar, rx_xpd] = antennas.couplingFromAngle(rx_ant, rx_dir);
    row = {label, ternary(paths(1).bounce_count == 0, 'LoS', 'PEC 1-bounce'), ...
        abs(H(1, 1, mid)), abs(H(2, 1, mid)), safeRatio(abs(H(2, 1, mid)), abs(H(1, 1, mid))), ...
        feats.gamma_cp_3_fp_only, tx_ar, tx_xpd, rx_ar, rx_xpd};
end

function value = safeRatio(num, den)
    if den <= 0
        value = NaN;
    else
        value = num / den;
    end
end

function out = ternary(cond, a, b)
    if cond
        out = a;
    else
        out = b;
    end
end
