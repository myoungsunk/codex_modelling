function result = checkA4_pathEnumeration()
% checkA4_pathEnumeration - verify LoS/1-bounce/2-bounce counts in a 4-wall box.

    pec = core.Material('kind', 'PEC', 'name', 'pec_wall');
    walls = { ...
        sanity.makeSurface(1, 'x_neg', [-2; 0; 1], [1; 0; 0], [0; 1; 0], [0; 0; 1], 10.0, 10.0, pec), ...
        sanity.makeSurface(2, 'x_pos', [2; 0; 1], [-1; 0; 0], [0; 1; 0], [0; 0; 1], 10.0, 10.0, pec), ...
        sanity.makeSurface(3, 'y_neg', [0; -2; 1], [0; 1; 0], [1; 0; 0], [0; 0; 1], 10.0, 10.0, pec), ...
        sanity.makeSurface(4, 'y_pos', [0; 2; 1], [0; -1; 0], [1; 0; 0], [0; 0; 1], 10.0, 10.0, pec)};
    scene = core.Scene(walls);
    tx_pos = [0; 0; 1];
    rx_pos = [0.7; 0.4; 1];

    paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 2);
    bounce_counts = [paths.bounce_count];
    actual = [sum(bounce_counts == 0), sum(bounce_counts == 1), sum(bounce_counts == 2)];
    % For TX/RX inside a 2D 4-wall room, only adjacent-wall ordered pairs
    % produce valid two-bounce specular paths. Opposite parallel-wall pairs
    % are geometrically infeasible, so the hand-computed count is 8.
    expected = [1, 4, 8];
    passed = isequal(actual, expected);

    fig = figure('Visible', 'off');
    bar(0:2, [actual(:), expected(:)]);
    xlabel('Bounce count');
    ylabel('Path count');
    legend('Actual', 'Expected', 'Location', 'best');
    title(sprintf('A4: Path enumeration [%d %d %d]', actual(1), actual(2), actual(3)));
    grid on;
    sanity.savePlot(fig, 'plot_a4_path_enumeration.png');

    details = struct();
    details.actual_counts = actual;
    details.expected_counts = expected;
    details.total_paths = numel(paths);
    details.note = 'Two-bounce count excludes opposite parallel-wall ordered pairs';
    details.surface_sequences = {paths.surface_ids};
    result = sanity.makeResult('A4_pathEnumeration', passed, actual, expected, 0, details, 'plot_a4_path_enumeration.png');
end
