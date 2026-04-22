function result = checkD3_ffdBoresight()
% checkD3_ffdBoresight - verify that the FFD-backed patch produces sensible CP behavior.

    cfg = config.defaultConfig();
    ffdFiles = { ...
        'E:\0. CP Antenna\0. TRACK2_3_SIM\ANTENNA_SOURCE\RHCP_new_6G7G_11pts.ffd', ...
        'E:\0. CP Antenna\0. TRACK2_3_SIM\ANTENNA_SOURCE\LHCP_new_6G7G_11pts.ffd'};
    assert(all(cellfun(@(p) exist(p, 'file') == 2, ffdFiles)), 'FFD files not found for D3');

    ant = antennas.makeRealisticPatchAntennaFFD(ffdFiles{1}, ffdFiles{2}, [0; 0; 0], [0; 0; 1], [1; 0; 0], [0; 1; 0]);
    freq_hz = cfg.f_center;

    dirs = [ ...
        0.0, 0.0; ...
        deg2rad(45.0), 0.0; ...
        deg2rad(90.0), 0.0];
    labels = {'boresight', 'off45', 'off90'};
    ar_db = zeros(3, 1);
    amp = zeros(3, 1);
    Etheta = complex(zeros(3, 1));
    Ephi = complex(zeros(3, 1));

    for idx = 1:3
        theta = dirs(idx, 1);
        phi = dirs(idx, 2);
        direction = [sin(theta) * cos(phi); sin(theta) * sin(phi); cos(theta)];
        field_world = ant.portVectorWorld(1, freq_hz, direction);
        [Etheta(idx), Ephi(idx)] = projectToLocalSpherical(ant, direction, field_world(:, 1));
        ar_db(idx) = axialRatioDb(Etheta(idx), Ephi(idx));
        amp(idx) = norm(field_world(:, 1));
    end

    ref_metrics = antennas.computeFfdMetrics(ant.ffd_port_r, 'handedness', 'RHCP');
    [~, fmid] = min(abs(ant.ffd_port_r.freqs_hz - cfg.f_center));
    theta_deg_grid = rad2deg(ant.ffd_port_r.theta_rad(:));
    phi0_idx = 1;
    ref_ar = zeros(3, 1);
    angle_deg = [0; 45; 90];
    for idx = 1:numel(angle_deg)
        [~, tidx] = min(abs(theta_deg_grid - angle_deg(idx)));
        ref_ar(idx) = ref_metrics.ar_db(tidx, phi0_idx, fmid);
    end
    ref_boresight_ar = ref_ar(1);

    ar_err = abs(ar_db - ref_ar);
    passed = max(ar_err) < 1.0;

    fig = figure('Visible', 'off');
    tiledlayout(2, 1);
    nexttile;
    plot([0, 45, 90], ar_db, 'bo-', 'LineWidth', 1.5); hold on;
    plot([0, 45, 90], ref_ar, 'r--s', 'LineWidth', 1.5);
    xlabel('Off-axis angle (deg)');
    ylabel('AR (dB)');
    title('D3: FFD-backed patch AR vs direction');
    legend('portVectorWorld', 'Day1 reference', 'Location', 'best');
    grid on;
    nexttile;
    plot([0, 45, 90], amp, 'ks-', 'LineWidth', 1.5);
    xlabel('Off-axis angle (deg)');
    ylabel('|E|');
    title('Field magnitude from port 1');
    grid on;
    sanity.savePlot(fig, 'plot_d3_ffd_boresight.png');

    details = struct();
    details.direction_labels = {labels{:}};
    details.ar_db = ar_db;
    details.amplitude = amp;
    details.E_theta = Etheta;
    details.E_phi = Ephi;
    details.reference_boresight_ar_db = ref_boresight_ar;
    details.reference_ar_db = ref_ar;
    details.ar_error_db = ar_err;
    result = sanity.makeResult('D3_ffdBoresight', passed, ar_db(:).', 'portVectorWorld AR should match direct FFD reference within 1 dB', 1.0, details, 'plot_d3_ffd_boresight.png');
end

function [Etheta, Ephi] = projectToLocalSpherical(ant, direction_world, field_world)
    R = ant.ffd_local_to_world;
    direction_local = R' * direction_world(:);
    theta = acos(max(min(direction_local(3), 1.0), -1.0));
    phi = mod(atan2(direction_local(2), direction_local(1)), 2.0 * pi);
    theta_hat_local = [cos(theta) * cos(phi); cos(theta) * sin(phi); -sin(theta)];
    phi_hat_local = [-sin(phi); cos(phi); 0.0];
    theta_hat_world = R * theta_hat_local;
    phi_hat_world = R * phi_hat_local;
    Etheta = theta_hat_world' * field_world(:);
    Ephi = phi_hat_world' * field_world(:);
end

function ar_db = axialRatioDb(Etheta, Ephi)
    s0 = abs(Etheta)^2 + abs(Ephi)^2;
    if s0 <= 1e-30
        ar_db = Inf;
        return;
    end
    s3 = -2 * imag(Etheta * conj(Ephi));
    chi = 0.5 * asin(max(min(s3 / s0, 1.0), -1.0));
    ar = max(1.0 / max(abs(tan(chi)), 1e-12), 1.0);
    ar_db = 20 * log10(ar);
end
