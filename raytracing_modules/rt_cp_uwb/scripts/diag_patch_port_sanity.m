projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'week25');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

tx_pos = [0; 0; 1];
rx_pos = [2; 0; 1];
scene = core.Scene();
paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 0);
assert(numel(paths) == 1 && paths.bounce_count == 0, 'Expected exactly one LoS path');

tx_ideal = antennas.makeIdealCpAntenna('right', tx_pos, [1; 0; 0], [0; 1; 0], [0; 0; 1]);
rx_ideal = antennas.makeIdealCpAntenna('right', rx_pos, [-1; 0; 0], [0; 1; 0], [0; 0; 1]);
tx_patch = antennas.makeRealisticPatchAntenna('synthetic', tx_pos, [1; 0; 0], [0; 1; 0], [0; 0; 1]);
rx_patch = antennas.makeRealisticPatchAntenna('synthetic', rx_pos, [-1; 0; 0], [0; 1; 0], [0; 0; 1]);

links = { ...
    'ideal_to_ideal', tx_ideal, rx_ideal; ...
    'ideal_to_patch', tx_ideal, rx_patch; ...
    'patch_to_ideal', tx_patch, rx_ideal; ...
    'patch_to_patch', tx_patch, rx_patch};

mid = ceil(numel(cfg.freqs) / 2);
rows = cell(size(links, 1), 8);
for i = 1:size(links, 1)
    label = links{i, 1};
    tx_ant = links{i, 2};
    rx_ant = links{i, 3};
    H = channel.buildChannel(paths, tx_ant, rx_ant, cfg.freqs);
    feats = features.extractAllFeatures(H, cfg.freqs, ...
        'window_type', cfg.window_type, ...
        'feature_schema', 'full', ...
        'tx_ant', tx_ant, ...
        'rx_ant', rx_ant, ...
        'tx_handedness', 'R');

    h11 = abs(H(1, 1, mid));
    h21 = abs(H(2, 1, mid));
    gamma_mid = safeRatio(h21, h11);
    rows(i, :) = {label, h11, h21, gamma_mid, feats.gamma_cp_3_fp_only, ...
        theoreticalCouplingRatio(tx_ant, cfg.freqs(mid)), theoreticalCouplingRatio(rx_ant, cfg.freqs(mid)), ...
        logical(h11 > h21)};
end

resultTbl = cell2table(rows, 'VariableNames', { ...
    'link', 'abs_H11_mid', 'abs_H21_mid', 'gamma_mid', 'gamma_fp', ...
    'tx_coupling_ratio', 'rx_coupling_ratio', 'h11_dominant'});

csvPath = fullfile(outDir, 'diag_patch_port_sanity.csv');
writetable(resultTbl, csvPath);

reportPath = fullfile(outDir, 'diag_patch_port_sanity.md');
fid = fopen(reportPath, 'w');
assert(fid ~= -1, 'Failed to open %s', reportPath);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, '# Patch Port Sanity\n\n');
fprintf(fid, '| link | |H(1,1)| mid | |H(2,1)| mid | gamma_mid | gamma_fp | tx coupling ratio | rx coupling ratio | H11 dominant |\n');
fprintf(fid, '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n');
for i = 1:height(resultTbl)
    fprintf(fid, '| %s | %.6g | %.6g | %.6g | %.6g | %.6g | %.6g | %s |\n', ...
        resultTbl.link{i}, resultTbl.abs_H11_mid(i), resultTbl.abs_H21_mid(i), ...
        resultTbl.gamma_mid(i), resultTbl.gamma_fp(i), ...
        resultTbl.tx_coupling_ratio(i), resultTbl.rx_coupling_ratio(i), ...
        ternary(resultTbl.h11_dominant(i), 'yes', 'no'));
end
fprintf(fid, '\n');
fprintf(fid, '- Interpretation: LoS should remain same-hand dominant, so `|H(1,1)| > |H(2,1)|` for every link.\n');
fprintf(fid, '- The synthetic patch baseline is set by the coupling matrix, so `gamma_mid` should stay below `1` and be on the same order as the coupling ratios.\n');

fprintf('Saved:\n');
fprintf('  %s\n', csvPath);
fprintf('  %s\n', reportPath);

function ratio = theoreticalCouplingRatio(ant, freq_hz)
    C = ant.couplingMatrix(freq_hz);
    ratio = safeRatio(abs(C(2, 1, 1)), abs(C(1, 1, 1)));
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
