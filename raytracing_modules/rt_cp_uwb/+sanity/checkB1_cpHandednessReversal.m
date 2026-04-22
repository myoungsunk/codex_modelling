function result = checkB1_cpHandednessReversal()
% checkB1_cpHandednessReversal - verify odd-bounce CP handedness reversal on a PEC plane.

    cfg = config.defaultConfig();
    freqs = linspace(cfg.f_center - cfg.bw/2, cfg.f_center + cfg.bw/2, cfg.n_freq).';
    pec_mat = core.Material('kind', 'PEC', 'pec_tm_sign', -1.0, 'name', 'pec_floor');
    slab = core.Surface('surface_id', 1, 'name', 'floor', 'point', [0; 0; 0], 'normal', [0; 0; 1], ...
        'u_axis', [1; 0; 0], 'v_axis', [0; 1; 0], 'half_u', 100.0, 'half_v', 100.0, 'material', pec_mat);
    scene = core.Scene({slab});

    tx_pos = [0; 0; 1];
    tx = antennas.makeIdealCpAntenna('right', tx_pos, [0; 0; -1], [1; 0; 0], [0; 1; 0]);
    incidence_angles = 0:10:70;
    ratio_db = zeros(size(incidence_angles));
    mid = ceil(cfg.n_freq / 2);

    for idx = 1:numel(incidence_angles)
        th = deg2rad(incidence_angles(idx));
        rx_pos = [2 * tan(th); 0; 1];
        rx = antennas.makeIdealCpAntenna('right', rx_pos, [0; 0; -1], [1; 0; 0], [0; 1; 0]);
        paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 1);
        one_bounce = paths([paths.bounce_count] == 1);
        assert(numel(one_bounce) == 1, 'Expected exactly one 1-bounce path, got %d', numel(one_bounce));
        H = channel.buildChannel(one_bounce, tx, rx, freqs);
        P_same = abs(H(1, 1, mid))^2;
        P_reversed = abs(H(2, 1, mid))^2;
        ratio_db(idx) = 10 * log10(max(P_reversed, 1e-30) / max(P_same, 1e-30));
    end

    fig = figure('Visible', 'off');
    plot(incidence_angles, ratio_db, 'bo-', 'LineWidth', 1.5); hold on;
    yline(30.0, 'r--', '30 dB threshold');
    xlabel('Incidence angle (deg)');
    ylabel('\gamma_{CP} (dB) = reversed-hand / same-hand');
    title(sprintf('B1: CP handedness reversal, min %.2f dB', min(ratio_db)));
    grid on;
    sanity.savePlot(fig, 'plot_b1_handedness_reversal.png');

    details = struct();
    details.incidence_angles_deg = incidence_angles;
    details.gamma_cp_db = ratio_db;
    details.reversed_to_same_ratio_db = ratio_db;
    result = sanity.makeResult('B1_cpHandednessReversal', all(ratio_db > 30.0), min(ratio_db), '> 30 dB', 30.0, details, 'plot_b1_handedness_reversal.png');
end
