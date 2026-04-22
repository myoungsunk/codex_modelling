function result = checkC2_uwbPulseShape()
% checkC2_uwbPulseShape - verify a flat H(f) yields the expected sinc-like width.

    cfg = config.defaultConfig();
    freqs = linspace(cfg.f_center - cfg.bw/2, cfg.f_center + cfg.bw/2, cfg.n_freq).';
    H_flat = ones(cfg.n_freq, 1);
    [h_t, t_axis] = channel.ifftToCir(H_flat, freqs, 'rect');

    mag = abs(h_t);
    peak_mag = max(mag);
    idx_null = find(mag(2:end) <= 1e-8 * peak_mag, 1, 'first') + 1;
    assert(~isempty(idx_null), 'Failed to locate the first null in the rectangular-band IFFT');
    measured_time = t_axis(idx_null);
    expected_time = 1.0 / cfg.bw;
    dt = t_axis(2) - t_axis(1);
    err = abs(measured_time - expected_time);

    fig = figure('Visible', 'off');
    plot(t_axis * 1e9, mag / peak_mag, 'b-', 'LineWidth', 1.5); hold on;
    xline(expected_time * 1e9, 'r--', 'Expected 1/B');
    xline(measured_time * 1e9, 'k:', 'Measured');
    xlabel('Time (ns)');
    ylabel('Normalized |h(t)|');
    title(sprintf('C2: UWB pulse first null, err %.3f ns', err * 1e9));
    grid on;
    xlim([0, 10]);
    sanity.savePlot(fig, 'plot_c2_uwb_pulse_shape.png');

    details = struct();
    details.t_axis_s = t_axis;
    details.h_t = h_t;
    details.measured_first_null_s = measured_time;
    details.expected_first_null_s = expected_time;
    details.sample_dt_s = dt;
    result = sanity.makeResult('C2_uwbPulseShape', err <= dt, measured_time, expected_time, dt, details, 'plot_c2_uwb_pulse_shape.png');
end
