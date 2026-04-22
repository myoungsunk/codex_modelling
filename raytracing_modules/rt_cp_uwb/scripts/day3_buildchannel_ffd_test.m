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

scene = core.Scene();
tx_pos = [0; 0; 1];
rx_pos_list = {[2; 0; 1], [4; 0; 1]};
rows = cell(numel(rx_pos_list), 8);

for idx = 1:numel(rx_pos_list)
    rx_pos = rx_pos_list{idx};
    tx = antennas.makeRealisticPatchAntennaFFD(ffdFiles{1}, ffdFiles{2}, tx_pos, [0; 0; -1], [1; 0; 0], [0; 1; 0]);
    rx = antennas.makeRealisticPatchAntennaFFD(ffdFiles{1}, ffdFiles{2}, rx_pos, [0; 0; -1], [1; 0; 0], [0; 1; 0]);
    paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 0);
    H = channel.buildChannel(paths, tx, rx, cfg.freqs);
    mid = ceil(numel(cfg.freqs) / 2);
    h11 = abs(H(1, 1, mid));
    h21 = abs(H(2, 1, mid));
    gamma = h21 / max(h11, 1e-30);
    rows(idx, :) = {norm(rx_pos - tx_pos), h11, h21, gamma, all(isfinite(H(:))), max(abs(H(:))), min(abs(H(:))), numel(paths)};
end

tbl = cell2table(rows, 'VariableNames', {'distance_m', 'H11_mid', 'H21_mid', 'gamma_mid', 'all_finite', 'max_abs_H', 'min_abs_H', 'num_paths'});
writetable(tbl, fullfile(outDir, 'day3_buildchannel_ffd_test.csv'));

fid = fopen(fullfile(outDir, 'day3_buildchannel_ffd_test.md'), 'w');
assert(fid ~= -1, 'Failed to open buildchannel log');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Day 3 BuildChannel FFD Test\n\n');
fprintf(fid, '| distance_m | H11_mid | H21_mid | gamma_mid | finite | max_abs_H | min_abs_H | num_paths |\n');
fprintf(fid, '| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |\n');
for idx = 1:height(tbl)
    fprintf(fid, '| %.3f | %.6g | %.6g | %.6g | %d | %.6g | %.6g | %d |\n', ...
        tbl.distance_m(idx), tbl.H11_mid(idx), tbl.H21_mid(idx), tbl.gamma_mid(idx), tbl.all_finite(idx), tbl.max_abs_H(idx), tbl.min_abs_H(idx), tbl.num_paths(idx));
end

disp(tbl);
