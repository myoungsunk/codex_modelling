function result = checkC1_losCirPeak()
% checkC1_losCirPeak - verify the LoS CIR peak lands at d/c within one sample.

    cfg = config.defaultConfig();
    freqs = linspace(cfg.f_center - cfg.bw/2, cfg.f_center + cfg.bw/2, cfg.n_freq).';
    tx_pos = [0; 0; 1];
    rx_pos = [3; 0; 1];
    tx = antennas.makeIdealCpAntenna('right', tx_pos, [1; 0; 0], [0; 1; 0], [0; 0; 1]);
    rx = antennas.makeIdealCpAntenna('right', rx_pos, [-1; 0; 0], [0; 1; 0], [0; 0; 1]);
    scene = core.Scene({});

    paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 0);
    H = channel.buildChannel(paths, tx, rx, freqs);
    [h_t, t_axis] = channel.ifftToCir(squeeze(H(1, 1, :)), freqs, cfg.window_type);
    [idx_fp, t_fp] = features.extractFirstPath(h_t, t_axis, 'max_peak');

    expected_time = 3.0 / cfg.c0;
    dt = t_axis(2) - t_axis(1);
    err = abs(t_fp - expected_time);

    fig = figure('Visible', 'off');
    plot(t_axis * 1e9, abs(h_t), 'b-', 'LineWidth', 1.5); hold on;
    xline(expected_time * 1e9, 'r--', 'Expected');
    xline(t_fp * 1e9, 'k:', 'Measured');
    xlabel('Time (ns)');
    ylabel('|h(t)|');
    title(sprintf('C1: LoS CIR peak, err %.3f ns', err * 1e9));
    grid on;
    xlim([0, max(20, t_axis(min(end, idx_fp + 20)) * 1e9)]);
    sanity.savePlot(fig, 'plot_c1_los_cir_peak.png');

    details = struct();
    details.t_axis_s = t_axis;
    details.h_t = h_t;
    details.expected_time_s = expected_time;
    details.measured_time_s = t_fp;
    details.sample_dt_s = dt;
    result = sanity.makeResult('C1_losCirPeak', err <= dt, err, 0.0, dt, details, 'plot_c1_los_cir_peak.png');
end
