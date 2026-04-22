function result = checkA2_singleBouncePath()
% checkA2_singleBouncePath - verify single-bounce path length against image geometry.

    tx_pos = [0; 0; 1];
    rx_pos = [3; 0; 1];
    slab = sanity.makeSurface(1, 'floor_zm1', [0; 0; -1], [0; 0; 1], [1; 0; 0], [0; 1; 0], 10.0, 10.0, ...
        core.Material('kind', 'PEC', 'name', 'pec_floor'));
    scene = core.Scene({slab});

    paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 1);
    one_bounce = paths([paths.bounce_count] == 1);
    assert(numel(one_bounce) == 1, 'Expected exactly one 1-bounce path, got %d', numel(one_bounce));

    actual_length = one_bounce.path_length_m;
    reflected_tx = [0; 0; -3];
    expected_length = norm(rx_pos - reflected_tx);
    err = abs(actual_length - expected_length);

    bounce_point = one_bounce.points{2};

    fig = figure('Visible', 'off');
    plot([tx_pos(1), bounce_point(1), rx_pos(1)], [tx_pos(3), bounce_point(3), rx_pos(3)], 'bo-', 'LineWidth', 1.5); hold on;
    yline(-1, 'k--', 'Slab z=-1');
    xlabel('x (m)');
    ylabel('z (m)');
    title(sprintf('A2: Single-bounce path length, err %.3e m', err));
    grid on;
    axis equal;
    sanity.savePlot(fig, 'plot_a2_single_bounce_path.png');

    details = struct();
    details.actual_length_m = actual_length;
    details.expected_length_m = expected_length;
    details.error_m = err;
    details.bounce_point = bounce_point;
    result = sanity.makeResult('A2_singleBouncePath', err < 1e-6, actual_length, expected_length, 1e-6, details, 'plot_a2_single_bounce_path.png');
end
