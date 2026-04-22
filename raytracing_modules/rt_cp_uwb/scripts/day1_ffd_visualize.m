projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'day1');
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

ffdPath = 'E:\0. CP Antenna\0. TRACK2_3_SIM\ANTENNA_SOURCE\RHCP_new_6G7G_11pts.ffd';
ffd = antennas.loadFfdPattern(ffdPath);
metrics = antennas.computeFfdMetrics(ffd, 'handedness', 'RHCP');
metricsOpp = antennas.computeFfdMetrics(ffd, 'handedness', 'LHCP');

[~, stem] = fileparts(ffdPath);
matPath = fullfile(outDir, [stem, '_parsed.mat']);
antennas.saveFfdPatternMat(ffd, matPath);

thetaDeg = rad2deg(ffd.theta_rad(:));
phiDeg = rad2deg(ffd.phi_rad(:));
freqGHz = ffd.freqs_hz(:) / 1e9;
phi0Idx = metrics.phi0_idx;
phi90Idx = metrics.phi90_idx;
fMid = ceil(numel(ffd.freqs_hz) / 2);

boresightFig = figure('Visible', 'off');
tiledlayout(2, 1);
nexttile;
plot(freqGHz, abs(metrics.boresight_E_theta), 'b-o', 'LineWidth', 1.3); hold on;
plot(freqGHz, abs(metrics.boresight_E_phi), 'r-s', 'LineWidth', 1.3);
xlabel('Frequency (GHz)');
ylabel('|E|');
legend('|E_\theta|', '|E_\phi|', 'Location', 'best');
title(sprintf('%s boresight field components', stem), 'Interpreter', 'none');
grid on;
nexttile;
plot(freqGHz, metrics.boresight_ar_db, 'k-o', 'LineWidth', 1.3); hold on;
plot(freqGHz, metrics.boresight_xpd_db, 'm-s', 'LineWidth', 1.3);
xlabel('Frequency (GHz)');
ylabel('dB');
legend('AR', 'XPD', 'Location', 'best');
title('Boresight AR / XPD');
grid on;
saveas(boresightFig, fullfile(outDir, 'plot_ffd_boresight_components_vs_freq.png'));
close(boresightFig);

coPolFig = figure('Visible', 'off');
plot(thetaDeg, 10 * log10(max(metrics.co_pol_power(:, phi0Idx, fMid), 1e-30)), 'b-', 'LineWidth', 1.4); hold on;
plot(thetaDeg, 10 * log10(max(metrics.co_pol_power(:, phi90Idx, fMid), 1e-30)), 'r--', 'LineWidth', 1.4);
xlabel('\theta (deg)');
ylabel('Co-pol power (dB, rel.)');
legend(sprintf('\\phi=%.0f^\\circ', phiDeg(phi0Idx)), sprintf('\\phi=%.0f^\\circ', phiDeg(phi90Idx)), 'Location', 'best');
title(sprintf('Co-pol principal cuts at %.2f GHz', freqGHz(fMid)));
grid on;
saveas(coPolFig, fullfile(outDir, 'plot_ffd_copol_principal_cuts.png'));
close(coPolFig);

crossPolFig = figure('Visible', 'off');
plot(thetaDeg, 10 * log10(max(metrics.cross_pol_power(:, phi0Idx, fMid), 1e-30)), 'b-', 'LineWidth', 1.4); hold on;
plot(thetaDeg, 10 * log10(max(metrics.cross_pol_power(:, phi90Idx, fMid), 1e-30)), 'r--', 'LineWidth', 1.4);
xlabel('\theta (deg)');
ylabel('Cross-pol power (dB, rel.)');
legend(sprintf('\\phi=%.0f^\\circ', phiDeg(phi0Idx)), sprintf('\\phi=%.0f^\\circ', phiDeg(phi90Idx)), 'Location', 'best');
title(sprintf('Cross-pol principal cuts at %.2f GHz', freqGHz(fMid)));
grid on;
saveas(crossPolFig, fullfile(outDir, 'plot_ffd_crosspol_principal_cuts.png'));
close(crossPolFig);

arFig = figure('Visible', 'off');
plot(thetaDeg, metrics.ar_db(:, phi0Idx, fMid), 'b-', 'LineWidth', 1.4); hold on;
plot(thetaDeg, metrics.ar_db(:, phi90Idx, fMid), 'r--', 'LineWidth', 1.4);
yline(3.0, 'k:', '3 dB');
yline(10.0, 'k--', '10 dB');
xlabel('\theta (deg)');
ylabel('AR (dB)');
legend(sprintf('\\phi=%.0f^\\circ', phiDeg(phi0Idx)), sprintf('\\phi=%.0f^\\circ', phiDeg(phi90Idx)), 'Location', 'best');
title(sprintf('AR principal cuts at %.2f GHz', freqGHz(fMid)));
grid on;
saveas(arFig, fullfile(outDir, 'plot_ffd_ar_principal_cuts.png'));
close(arFig);

xpdFig = figure('Visible', 'off');
plot(thetaDeg, metrics.xpd_db(:, phi0Idx, fMid), 'b-', 'LineWidth', 1.4); hold on;
plot(thetaDeg, metrics.xpd_db(:, phi90Idx, fMid), 'r--', 'LineWidth', 1.4);
xlabel('\theta (deg)');
ylabel('XPD (dB)');
legend(sprintf('\\phi=%.0f^\\circ', phiDeg(phi0Idx)), sprintf('\\phi=%.0f^\\circ', phiDeg(phi90Idx)), 'Location', 'best');
title(sprintf('XPD principal cuts at %.2f GHz', freqGHz(fMid)));
grid on;
saveas(xpdFig, fullfile(outDir, 'plot_ffd_xpd_principal_cuts.png'));
close(xpdFig);

powerFig = figure('Visible', 'off');
plot(freqGHz, metrics.total_radiated_power_numeric, 'b-o', 'LineWidth', 1.4);
xlabel('Frequency (GHz)');
ylabel('Integrated power (arb.)');
title('Total radiated power vs frequency');
grid on;
saveas(powerFig, fullfile(outDir, 'plot_ffd_total_power_vs_freq.png'));
close(powerFig);

summaryPath = fullfile(outDir, 'ffd_visual_summary.md');
fid = fopen(summaryPath, 'w');
assert(fid ~= -1, 'Failed to open %s', summaryPath);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# FFD Visualization Summary\n\n');
fprintf(fid, '- source: `%s`\n', ffdPath);
fprintf(fid, '- port label: `%s`\n', ffd.port_label);
fprintf(fid, '- theta grid: `%d`\n', numel(ffd.theta_rad));
fprintf(fid, '- phi grid: `%d`\n', numel(ffd.phi_rad));
fprintf(fid, '- frequency points: `%d`\n', numel(ffd.freqs_hz));
fprintf(fid, '- boresight AR range: `%.3f dB` to `%.3f dB`\n', min(metrics.boresight_ar_db), max(metrics.boresight_ar_db));
fprintf(fid, '- boresight XPD range: `%.3f dB` to `%.3f dB`\n', min(metrics.boresight_xpd_db), max(metrics.boresight_xpd_db));
fprintf(fid, '- total power range: `%.6g` to `%.6g`\n', min(metrics.total_radiated_power_numeric), max(metrics.total_radiated_power_numeric));
if metricsOpp.boresight_xpd_db(fMid) > metrics.boresight_xpd_db(fMid)
    fprintf(fid, '- note: under the current IEEE-RHCP formula, the opposite hand is stronger at boresight (`RHCP XPD = %.3f dB`, `LHCP XPD = %.3f dB` at %.2f GHz). This suggests an HFSS sign/time-convention flip to resolve before downstream handedness claims.\n', ...
        metrics.boresight_xpd_db(fMid), metricsOpp.boresight_xpd_db(fMid), freqGHz(fMid));
end

interpPath = fullfile(outDir, 'interpolation_strategy.md');
fid2 = fopen(interpPath, 'w');
assert(fid2 ~= -1, 'Failed to open %s', interpPath);
cleanup2 = onCleanup(@() fclose(fid2)); %#ok<NASGU>
fprintf(fid2, '# FFD Interpolation Strategy\n\n');
fprintf(fid2, 'Chosen method: **amplitude / phase separated bilinear interpolation on the (theta, phi) grid, followed by linear interpolation in frequency**.\n\n');
fprintf(fid2, 'Reasoning:\n');
fprintf(fid2, '- Real/imag interpolation is simple but can distort phase near rapid phase transitions.\n');
fprintf(fid2, '- Amplitude and unwrapped phase interpolation preserves polarization behavior better on coarse FFD grids.\n');
fprintf(fid2, '- HFSS FFD data is already on a tensor product grid, so bilinear interpolation is fast and stable.\n\n');
fprintf(fid2, 'Pole handling:\n');
fprintf(fid2, '- Near theta = 0 or theta = pi, use the nearest valid theta ring and avoid phi-sensitive interpolation because the spherical basis becomes singular.\n');
fprintf(fid2, '- Clamp theta queries to the interior by a small epsilon before interpolation.\n\n');
fprintf(fid2, 'Phase strategy:\n');
fprintf(fid2, '- Unwrap phase along theta first, then along phi for each frequency slice.\n');
fprintf(fid2, '- Interpolate phase and amplitude separately, then reconstruct the complex field.\n');
fprintf(fid2, '- If amplitude is below a small floor, fall back to nearest-neighbor phase to avoid unstable unwrap artifacts.\n');

fprintf('Saved parsed MAT: %s\n', matPath);
fprintf('Boresight AR at center freq %.2f GHz: %.3f dB\n', freqGHz(fMid), metrics.boresight_ar_db(fMid));
