script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

out_dir = fullfile(repo_root, 'results', 'theta0_peak_aligned_cp16_preflight');
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

rhcp_path = fullfile(repo_root, 'RHCP_new_6G7G_11pts.ffd');
lhcp_path = fullfile(repo_root, 'LHCP_new_6G7G_11pts.ffd');
ffd_r = antennas.loadFfdPattern(rhcp_path, 'notes', 'theta0_peak_preflight_rhcp');
ffd_l = antennas.loadFfdPattern(lhcp_path, 'notes', 'theta0_peak_preflight_lhcp');

tx_R = frameLocalPosZToTarget([0; 0; -1]);
rx_R = frameLocalPosZToTarget([0; 0; 1]);
assert(norm(tx_R * [0; 0; 1] - [0; 0; -1]) < 1e-12, 'TX nominal local +z is not world -z');
assert(norm(rx_R * [0; 0; 1] - [0; 0; 1]) < 1e-12, 'RX nominal local +z is not world +z');

rows = [
    peakRow("RHCP", ffd_r, tx_R, rx_R)
    peakRow("LHCP", ffd_l, tx_R, rx_R)
    ];
csv_path = fullfile(out_dir, 'theta0_peak_alignment_preflight.csv');
md_path = fullfile(out_dir, 'theta0_peak_alignment_preflight.md');
writetable(rows, csv_path);

fid = fopen(md_path, 'w');
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Theta0 Peak Alignment Preflight\n\n');
fprintf(fid, '- local_peak_direction: theta=0 deg, phi=90 deg, vector=[0;0;1]\n');
fprintf(fid, '- tx_target_world: [0;0;-1]\n');
fprintf(fid, '- rx_target_world: [0;0;1]\n');
fprintf(fid, '- rhcp_sample_order: %s\n', string(ffd_r.metadata.sample_order));
fprintf(fid, '- lhcp_sample_order: %s\n', string(ffd_l.metadata.sample_order));
fprintf(fid, '- csv: %s\n\n', csv_path);
fprintf(fid, '| port | peak_theta_deg | peak_phi_deg | nominal_gain_delta_db | tx_actual_world_dir | rx_actual_world_dir |\n');
fprintf(fid, '|---|---:|---:|---:|---|---|\n');
for i = 1:height(rows)
    fprintf(fid, '| %s | %.6f | %.6f | %.6f | [%+.6f,%+.6f,%+.6f] | [%+.6f,%+.6f,%+.6f] |\n', ...
        rows.port(i), rows.peak_theta_deg(i), rows.peak_phi_deg(i), rows.nominal_gain_delta_db(i), ...
        rows.tx_actual_world_x(i), rows.tx_actual_world_y(i), rows.tx_actual_world_z(i), ...
        rows.rx_actual_world_x(i), rows.rx_actual_world_y(i), rows.rx_actual_world_z(i));
end

disp(rows);
fprintf('Saved preflight:\n  %s\n  %s\n', csv_path, md_path);

function row = peakRow(port_name, ffd, tx_R, rx_R)
    [~, fidx] = min(abs(ffd.freqs_hz - 6.5e9));
    p = abs(ffd.E_theta(:, :, fidx)).^2 + abs(ffd.E_phi(:, :, fidx)).^2;
    [peak_val, flat_idx] = max(p(:));
    [tidx, pidx] = ind2sub(size(p), flat_idx);
    theta = ffd.theta_rad(tidx);
    phi = ffd.phi_rad(pidx);
    nominal_val = samplePower(ffd, 0.0, pi / 2.0, fidx);
    local_peak = sphDir(theta, phi);
    tx_dir = tx_R * local_peak;
    rx_dir = rx_R * local_peak;
    row = table( ...
        string(port_name), rad2deg(theta), rad2deg(phi), 10 * log10(max(peak_val, 1e-30) / max(nominal_val, 1e-30)), ...
        tx_dir(1), tx_dir(2), tx_dir(3), rx_dir(1), rx_dir(2), rx_dir(3), ...
        string(ffd.metadata.sample_order), ...
        'VariableNames', {'port', 'peak_theta_deg', 'peak_phi_deg', 'nominal_gain_delta_db', ...
        'tx_actual_world_x', 'tx_actual_world_y', 'tx_actual_world_z', ...
        'rx_actual_world_x', 'rx_actual_world_y', 'rx_actual_world_z', 'sample_order'});
end

function value = samplePower(ffd, theta, phi, fidx)
    [~, tidx] = min(abs(ffd.theta_rad - theta));
    phi_grid = ffd.phi_rad(:);
    dphi = abs(angle(exp(1j * (phi_grid - phi))));
    [~, pidx] = min(dphi);
    value = abs(ffd.E_theta(tidx, pidx, fidx)).^2 + abs(ffd.E_phi(tidx, pidx, fidx)).^2;
end

function v = sphDir(theta, phi)
    v = [sin(theta) * cos(phi); sin(theta) * sin(phi); cos(theta)];
    v = v / norm(v);
end

function R = frameLocalPosZToTarget(target_world)
    z_world = normalizeVec(target_world);
    ref = [1; 0; 0];
    if abs(dot(z_world, ref)) > 0.95
        ref = [0; 1; 0];
    end
    x_world = ref - dot(ref, z_world) * z_world;
    x_world = normalizeVec(x_world);
    y_world = normalizeVec(cross(z_world, x_world));
    R = [x_world, y_world, z_world];
end

function v = normalizeVec(x)
    v = double(x(:));
    v = v / norm(v);
end
