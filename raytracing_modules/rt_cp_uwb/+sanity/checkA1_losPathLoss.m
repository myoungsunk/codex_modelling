function result = checkA1_losPathLoss()
% checkA1_losPathLoss - verify |H| follows 1/r in free space.

    cfg = config.defaultConfig();
    freqs = linspace(cfg.f_center - cfg.bw/2, cfg.f_center + cfg.bw/2, cfg.n_freq).';
    f0 = cfg.f_center;
    lambda = cfg.c0 / f0;
    distances = logspace(log10(0.5), log10(10.0), 20);
    H_peak = zeros(size(distances));

    scene = core.Scene({});
    tx = antennas.makeIdealCpAntenna('right', [0; 0; 0], [1; 0; 0], [0; 1; 0], [0; 0; 1]);
    mid = ceil(cfg.n_freq / 2);

    for idx = 1:numel(distances)
        rx_pos = [distances(idx); 0; 0];
        rx = antennas.makeIdealCpAntenna('right', rx_pos, [-1; 0; 0], [0; 1; 0], [0; 0; 1]);
        paths = trace.enumeratePaths(scene, tx.position, rx.position, 0);
        assert(numel(paths) == 1, 'Expected exactly 1 LoS path, got %d', numel(paths));
        H = channel.buildChannel(paths, tx, rx, freqs);
        H_peak(idx) = abs(H(1, 1, mid));
    end

    expected_curve = lambda ./ (4 * pi * distances);
    fit_coeff = polyfit(log10(distances), log10(H_peak), 1);
    slope = fit_coeff(1);

    fig = figure('Visible', 'off');
    loglog(distances, H_peak, 'bo-', 'LineWidth', 1.5); hold on;
    loglog(distances, expected_curve, 'r--', 'LineWidth', 1.5);
    xlabel('Distance (m)');
    ylabel('|H|');
    legend('RT result', 'Friis 1/r', 'Location', 'best');
    title(sprintf('A1: LoS path loss, slope %.3f', slope));
    grid on;
    sanity.savePlot(fig, 'plot_a1_los_pathloss.png');

    details = struct();
    details.distances_m = distances;
    details.H_peak = H_peak;
    details.expected_curve = expected_curve;
    details.slope = slope;
    result = sanity.makeResult('A1_losPathLoss', abs(slope + 1.0) < 0.1, slope, -1.0, 0.1, details, 'plot_a1_los_pathloss.png');
end
