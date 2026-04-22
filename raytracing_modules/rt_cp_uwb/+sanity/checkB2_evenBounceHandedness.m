function result = checkB2_evenBounceHandedness()
% checkB2_evenBounceHandedness - verify 2-bounce corner reflection preserves CP handedness.

    cfg = config.defaultConfig();
    freqs = linspace(cfg.f_center - cfg.bw/2, cfg.f_center + cfg.bw/2, cfg.n_freq).';
    pec = core.Material('kind', 'PEC', 'pec_tm_sign', -1.0, 'name', 'pec_corner');

    wall_y = sanity.makeSurface(1, 'wall_y3', [0; 3; 1], [0; -1; 0], [1; 0; 0], [0; 0; 1], 10.0, 10.0, pec);
    wall_x = sanity.makeSurface(2, 'wall_x3', [3; 0; 1], [-1; 0; 0], [0; 1; 0], [0; 0; 1], 10.0, 10.0, pec);
    scene = core.Scene({wall_y, wall_x});

    tx_pos = [0; 0; 1];
    rx_pos = [3; 4; 1];
    % Use the same global h/v frame on both ends so the circular-port
    % labeling stays comparable across the two-bounce path.
    tx = antennas.makeIdealCpAntenna('right', tx_pos, [1; 0; 0], [0; 1; 0], [0; 0; 1]);
    rx = antennas.makeIdealCpAntenna('right', rx_pos, [-1; 0; 0], [0; 1; 0], [0; 0; 1]);

    paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 2);
    two_bounce = paths([paths.bounce_count] == 2);
    assert(numel(two_bounce) == 1, 'Expected exactly one 2-bounce path, got %d', numel(two_bounce));

    H = channel.buildChannel(two_bounce, tx, rx, freqs);
    mid = ceil(cfg.n_freq / 2);
    P_same = abs(H(1, 1, mid))^2;
    P_reversed = abs(H(2, 1, mid))^2;
    ratio_db = 10 * log10(max(P_same, 1e-30) / max(P_reversed, 1e-30));

    fig = figure('Visible', 'off');
    bar(categorical({'Same-hand', 'Reversed-hand'}), [P_same, P_reversed]);
    ylabel('Power');
    title(sprintf('B2: Even-bounce handedness preservation, ratio %.2f dB', ratio_db));
    grid on;
    sanity.savePlot(fig, 'plot_b2_even_bounce_handedness.png');

    details = struct();
    details.tx_pos = tx_pos;
    details.rx_pos = rx_pos;
    details.surface_ids = two_bounce.surface_ids;
    details.same_hand_power = P_same;
    details.reversed_hand_power = P_reversed;
    details.gamma_cp_db = -ratio_db;
    result = sanity.makeResult('B2_evenBounceHandedness', ratio_db > 20.0, ratio_db, '> 20 dB', 20.0, details, 'plot_b2_even_bounce_handedness.png');
end
