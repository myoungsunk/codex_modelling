function result = checkA3_snellLaw()
% checkA3_snellLaw - verify path incidence angles match geometric reflection angles.

    tx_pos = [0; 0; 1];
    rx_x = linspace(0.5, 8.0, 12);
    slab = sanity.makeSurface(1, 'floor_zm1', [0; 0; -1], [0; 0; 1], [1; 0; 0], [0; 1; 0], 20.0, 20.0, ...
        core.Material('kind', 'PEC', 'name', 'pec_floor'));
    scene = core.Scene({slab});

    actual_angles = zeros(size(rx_x));
    expected_angles = zeros(size(rx_x));

    for idx = 1:numel(rx_x)
        rx_pos = [rx_x(idx); 0; 1];
        paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 1);
        one_bounce = paths([paths.bounce_count] == 1);
        assert(numel(one_bounce) == 1, 'Expected exactly one 1-bounce path, got %d', numel(one_bounce));
        actual_angles(idx) = one_bounce.incidence_angles_rad(1);
        expected_angles(idx) = atan(rx_x(idx) / 4.0);
    end

    err = abs(actual_angles - expected_angles);
    max_err = max(err);

    fig = figure('Visible', 'off');
    plot(rx_x, rad2deg(actual_angles), 'bo-', 'LineWidth', 1.5); hold on;
    plot(rx_x, rad2deg(expected_angles), 'r--', 'LineWidth', 1.5);
    xlabel('RX x-position (m)');
    ylabel('Incidence angle (deg)');
    legend('PathRecord', 'Analytic', 'Location', 'best');
    title(sprintf('A3: Snell-law angle consistency, max err %.3e rad', max_err));
    grid on;
    sanity.savePlot(fig, 'plot_a3_snell_law.png');

    details = struct();
    details.rx_x_m = rx_x;
    details.actual_angles_rad = actual_angles;
    details.expected_angles_rad = expected_angles;
    details.error_rad = err;
    result = sanity.makeResult('A3_snellLaw', max_err < 1e-9, max_err, 0.0, 1e-9, details, 'plot_a3_snell_law.png');
end
