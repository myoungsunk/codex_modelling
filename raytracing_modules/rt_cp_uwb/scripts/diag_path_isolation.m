projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'week25');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

tx_pos = [0; 0; 1];
rx_pos = [2; 0; 1];
spec_point = (tx_pos + rx_pos) / 2.0;
spec_point(3) = 0.0;

scene_los = core.Scene();
scene_bounce = scenes.makeSingleSlabScene('metal_pec', [4, 4], [1; 0; 0], [0; 0; 1]);
scene_blocked = scenes.makeSingleSlabScene('metal_pec', [4, 4], [1; 0; 0], [0; 0; 1]);
scene_blocked.addSurface(makeLosBlocker(tx_pos, rx_pos));

paths_los = trace.enumeratePaths(scene_los, tx_pos, rx_pos, 0);
paths_all = trace.enumeratePaths(scene_bounce, tx_pos, rx_pos, 1);
paths_1b = filterByBounce(paths_all, 1);
paths_blocked = trace.enumeratePaths(scene_blocked, tx_pos, rx_pos, 1);

assert(numel(paths_los) == 1, 'Expected a single LoS path');
assert(numel(paths_1b) == 1, 'Expected a single 1-bounce path');

ant_rows = cell(4, 8);
ant_rows(1, :) = summarizeCase('ideal_los_only', ...
    antennas.makeIdealCpAntenna('right', tx_pos, normalizeDir(rx_pos - tx_pos), [0; 1; 0], [0; 0; 1]), ...
    antennas.makeIdealCpAntenna('right', rx_pos, normalizeDir(tx_pos - rx_pos), [0; 1; 0], [0; 0; 1]), ...
    paths_los, cfg.freqs, cfg.window_type);
ant_rows(2, :) = summarizeCase('patch_los_only', ...
    antennas.makeRealisticPatchAntenna('synthetic', tx_pos, normalizeDir(rx_pos - tx_pos), [0; 1; 0], [0; 0; 1]), ...
    antennas.makeRealisticPatchAntenna('synthetic', rx_pos, normalizeDir(tx_pos - rx_pos), [0; 1; 0], [0; 0; 1]), ...
    paths_los, cfg.freqs, cfg.window_type);
ant_rows(3, :) = summarizeCase('ideal_1bounce_only', ...
    antennas.makeIdealCpAntenna('right', tx_pos, normalizeDir(spec_point - tx_pos), [1; 0; 0], [0; 1; 0]), ...
    antennas.makeIdealCpAntenna('right', rx_pos, normalizeDir(spec_point - rx_pos), [1; 0; 0], [0; 1; 0]), ...
    paths_1b, cfg.freqs, cfg.window_type);
ant_rows(4, :) = summarizeCase('patch_1bounce_only', ...
    antennas.makeRealisticPatchAntenna('synthetic', tx_pos, normalizeDir(spec_point - tx_pos), [1; 0; 0], [0; 1; 0]), ...
    antennas.makeRealisticPatchAntenna('synthetic', rx_pos, normalizeDir(spec_point - rx_pos), [1; 0; 0], [0; 1; 0]), ...
    paths_1b, cfg.freqs, cfg.window_type);

caseTbl = cell2table(ant_rows, 'VariableNames', { ...
    'case_name', 'bounce_count', 'abs_H11_mid', 'abs_H21_mid', 'gamma_mid', ...
    'gamma_fp', 'path_length_m', 'delay_ns'});
csvPath = fullfile(outDir, 'diag_path_isolation.csv');
writetable(caseTbl, csvPath);

blockedCounts = [paths_blocked.bounce_count];
blockedCountText = mat2str(blockedCounts);

reportPath = fullfile(outDir, 'diag_path_isolation.md');
fid = fopen(reportPath, 'w');
assert(fid ~= -1, 'Failed to open %s', reportPath);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, '# Path Isolation Diagnostic\n\n');
fprintf(fid, '| case | bounce_count | |H(1,1)| mid | |H(2,1)| mid | gamma_mid | gamma_fp | path_length_m | delay_ns |\n');
fprintf(fid, '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n');
for i = 1:height(caseTbl)
    fprintf(fid, '| %s | %d | %.6g | %.6g | %.6g | %.6g | %.6g | %.6g |\n', ...
        caseTbl.case_name{i}, caseTbl.bounce_count(i), caseTbl.abs_H11_mid(i), ...
        caseTbl.abs_H21_mid(i), caseTbl.gamma_mid(i), caseTbl.gamma_fp(i), ...
        caseTbl.path_length_m(i), caseTbl.delay_ns(i));
end
fprintf(fid, '\n');
fprintf(fid, '## Blocked Scene Check\n\n');
fprintf(fid, '- blocked scene bounce counts: `%s`\n', blockedCountText);
fprintf(fid, '- direct path present: **%s**\n', ternary(any(blockedCounts == 0), 'yes', 'no'));

fprintf('Saved:\n');
fprintf('  %s\n', csvPath);
fprintf('  %s\n', reportPath);

function row = summarizeCase(label, tx_ant, rx_ant, paths, freqs, window_type)
    H = channel.buildChannel(paths, tx_ant, rx_ant, freqs);
    feats = features.extractAllFeatures(H, freqs, ...
        'window_type', window_type, ...
        'feature_schema', 'full', ...
        'tx_ant', tx_ant, ...
        'rx_ant', rx_ant, ...
        'tx_handedness', 'R');
    mid = ceil(numel(freqs) / 2);
    row = {label, paths(1).bounce_count, abs(H(1, 1, mid)), abs(H(2, 1, mid)), ...
        safeRatio(abs(H(2, 1, mid)), abs(H(1, 1, mid))), feats.gamma_cp_3_fp_only, ...
        paths(1).path_length_m, 1e9 * paths(1).delay_s};
end

function blocker = makeLosBlocker(tx_pos, rx_pos)
    blocker_mat = materials.materialsLibrary('concrete');
    center = (tx_pos + rx_pos) / 2.0;
    normal = normalizeDir(rx_pos - tx_pos);
    normal(3) = 0.0;
    normal = normalizeDir(normal);
    u_axis = [-normal(2); normal(1); 0];
    if norm(u_axis) < 1e-12
        u_axis = [0; 1; 0];
    end
    blocker = core.Surface( ...
        'surface_id', 2, ...
        'name', 'diag_blocker', ...
        'point', center, ...
        'normal', normal, ...
        'u_axis', u_axis, ...
        'v_axis', [0; 0; 1], ...
        'half_u', 0.5, ...
        'half_v', 0.6, ...
        'material', blocker_mat);
end

function paths_out = filterByBounce(paths_in, bounce_count)
    mask = [paths_in.bounce_count] == bounce_count;
    paths_out = paths_in(mask);
end

function v = normalizeDir(x)
    v = x(:);
    n = norm(v);
    if n <= 1e-12
        v = [1; 0; 0];
    else
        v = v / n;
    end
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
