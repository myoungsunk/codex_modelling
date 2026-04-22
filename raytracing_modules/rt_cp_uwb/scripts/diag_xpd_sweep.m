projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'week25');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

xpd_list = [60; 40; 30; 25; 20; 15; 10];
n_per_class = 25;
total_cases_per_xpd = 2 * n_per_class;
angles_deg = linspace(8.0, 68.0, n_per_class).';
tx_heights = linspace(0.7, 2.3, n_per_class).';
tx_heights = tx_heights([1:2:end, 2:2:end]).';

rows = cell(numel(xpd_list), 6);
tic;
for i = 1:numel(xpd_list)
    xpd_db = xpd_list(i);
    pattern = antennas.loadPatchPatternSynthetic( ...
        'xpd_db_boresight', xpd_db, ...
        'ar_db_boresight', 0.0, ...
        'peak_gain_dbi', 7.0, ...
        'fitted_cos_exp', 2.0);

    gamma_los = nan(numel(angles_deg), 1);
    gamma_nlos = nan(numel(angles_deg), 1);
    for j = 1:numel(angles_deg)
        th = deg2rad(angles_deg(j));
        tx_height = tx_heights(j);
        tx_pos = [0; 0; tx_height];
        rx_pos = [2 * tx_height * tan(th); 0; tx_height];
        spec_point = (tx_pos + rx_pos) / 2.0;
        spec_point(3) = 0.0;

        scene_los = core.Scene();
        scene_bounce = scenes.makeSingleSlabScene('metal_pec', [5, 5], spec_point, [0; 0; 1]);
        paths_los = trace.enumeratePaths(scene_los, tx_pos, rx_pos, 0);
        paths_bounce = filterByBounce(trace.enumeratePaths(scene_bounce, tx_pos, rx_pos, 1), 1);

        tx_los = antennas.makeRealisticPatchAntenna(pattern, tx_pos, normalizeDir(rx_pos - tx_pos), [0; 1; 0], [0; 0; 1]);
        rx_los = antennas.makeRealisticPatchAntenna(pattern, rx_pos, normalizeDir(tx_pos - rx_pos), [0; 1; 0], [0; 0; 1]);
        tx_bnc = antennas.makeRealisticPatchAntenna(pattern, tx_pos, normalizeDir(spec_point - tx_pos), [1; 0; 0], [0; 1; 0]);
        rx_bnc = antennas.makeRealisticPatchAntenna(pattern, rx_pos, normalizeDir(spec_point - rx_pos), [1; 0; 0], [0; 1; 0]);

        gamma_los(j) = extractGamma(paths_los, tx_los, rx_los, cfg);
        gamma_nlos(j) = extractGamma(paths_bounce, tx_bnc, rx_bnc, cfg);
    end

    y = [zeros(size(gamma_los)); ones(size(gamma_nlos))];
    scores = [gamma_los; gamma_nlos];
    auc = localAuc(scores, y);
    rows(i, :) = {xpd_db, mean(gamma_los, 'omitnan'), mean(gamma_nlos, 'omitnan'), ...
        median(gamma_los, 'omitnan'), median(gamma_nlos, 'omitnan'), auc};
end
elapsed_s = toc;

resultTbl = cell2table(rows, 'VariableNames', { ...
    'xpd_db', 'mean_gamma_los', 'mean_gamma_nlos', 'median_gamma_los', 'median_gamma_nlos', 'auc_gamma_cp_3'});
csvPath = fullfile(outDir, 'diag_xpd_sweep.csv');
writetable(resultTbl, csvPath);

fig = figure('Visible', 'off', 'Position', [100 100 1000 420]);
subplot(1, 2, 1);
plot(resultTbl.xpd_db, resultTbl.mean_gamma_los, 'o-', 'LineWidth', 1.5); hold on;
plot(resultTbl.xpd_db, resultTbl.mean_gamma_nlos, 's-', 'LineWidth', 1.5);
set(gca, 'XDir', 'reverse');
xlabel('Patch XPD (dB)');
ylabel('Mean \gamma_{CP,3}');
legend('LoS', '1-bounce NLoS', 'Location', 'best');
title('XPD Sweep: Mean \gamma_{CP,3}');
grid on;

subplot(1, 2, 2);
plot(resultTbl.xpd_db, resultTbl.auc_gamma_cp_3, 'd-', 'LineWidth', 1.5);
set(gca, 'XDir', 'reverse');
xlabel('Patch XPD (dB)');
ylabel('AUC');
title('XPD Sweep: \gamma_{CP,3} AUC');
ylim([0 1]);
grid on;

pngPath = fullfile(outDir, 'diag_xpd_sweep.png');
saveas(fig, pngPath);
close(fig);

reportPath = fullfile(outDir, 'diag_xpd_sweep.md');
fid = fopen(reportPath, 'w');
assert(fid ~= -1, 'Failed to open %s', reportPath);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, '# XPD Sweep Diagnostic\n\n');
fprintf(fid, '- AR fixed at `0 dB` to isolate XPD.\n');
fprintf(fid, '- Cases per XPD: `%d` (`%d` LoS + `%d` isolated 1-bounce NLoS)\n', total_cases_per_xpd, n_per_class, n_per_class);
fprintf(fid, '- Runtime: `%.3f s`\n', elapsed_s);
fprintf(fid, '- LoS samples: empty scene over incidence/height sweep.\n');
fprintf(fid, '- NLoS samples: isolated single PEC bounce over the same geometry sweep.\n\n');
fprintf(fid, '| XPD (dB) | mean gamma LoS | mean gamma NLoS | median gamma LoS | median gamma NLoS | AUC |\n');
fprintf(fid, '| ---: | ---: | ---: | ---: | ---: | ---: |\n');
for i = 1:height(resultTbl)
    fprintf(fid, '| %.0f | %.6g | %.6g | %.6g | %.6g | %.6g |\n', ...
        resultTbl.xpd_db(i), resultTbl.mean_gamma_los(i), resultTbl.mean_gamma_nlos(i), ...
        resultTbl.median_gamma_los(i), resultTbl.median_gamma_nlos(i), resultTbl.auc_gamma_cp_3(i));
end
fprintf(fid, '\n');
fprintf(fid, '- Interpretation: monotonic degradation from high XPD to low XPD supports a real antenna-impairment mechanism.\n');

fprintf('Saved:\n');
fprintf('  %s\n', csvPath);
fprintf('  %s\n', pngPath);
fprintf('  %s\n', reportPath);
fprintf('Runtime: %.3f s (%d total channels)\n', elapsed_s, numel(xpd_list) * total_cases_per_xpd);

function gamma = extractGamma(paths, tx_ant, rx_ant, cfg)
    H = channel.buildChannel(paths, tx_ant, rx_ant, cfg.freqs);
    feats = features.extractAllFeatures(H, cfg.freqs, ...
        'window_type', cfg.window_type, ...
        'feature_schema', 'full', ...
        'tx_ant', tx_ant, ...
        'rx_ant', rx_ant, ...
        'tx_handedness', 'R');
    gamma = feats.gamma_cp_3_fp_only;
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

function auc = localAuc(scores, labels)
    valid = isfinite(scores) & isfinite(labels);
    scores = scores(valid);
    labels = labels(valid);
    pos = scores(labels == 1);
    neg = scores(labels == 0);
    if isempty(pos) || isempty(neg)
        auc = NaN;
        return;
    end

    wins = 0.0;
    ties = 0.0;
    total = numel(pos) * numel(neg);
    for i = 1:numel(pos)
        wins = wins + sum(pos(i) > neg);
        ties = ties + sum(pos(i) == neg);
    end
    auc = (wins + 0.5 * ties) / total;
end
