function result = checkD2_couplingUnitarity()
% checkD2_couplingUnitarity - verify the antenna coupling matrix is unitary.

    cfg = config.defaultConfig();
    freqs = linspace(cfg.f_center - cfg.bw/2, cfg.f_center + cfg.bw/2, cfg.n_freq).';
    ant = core.Antenna( ...
        'position', [0; 0; 0], ...
        'boresight', [1; 0; 0], ...
        'h_axis', [0; 1; 0], ...
        'v_axis', [0; 0; 1], ...
        'basis', 'circular', ...
        'cross_pol_leakage_db', 20.0, ...
        'axial_ratio_db', 3.0, ...
        'enable_coupling', true);

    M = ant.couplingMatrix(freqs);
    I2 = eye(2);
    per_freq_err = zeros(size(freqs));
    for idx = 1:numel(freqs)
        err_mat = M(:, :, idx) * M(:, :, idx)' - I2;
        per_freq_err(idx) = max(abs(err_mat(:)));
    end
    max_err = max(per_freq_err);

    fig = figure('Visible', 'off');
    plot(freqs / 1e9, per_freq_err, 'b-', 'LineWidth', 1.5);
    xlabel('Frequency (GHz)');
    ylabel('max|MM^H - I|');
    title(sprintf('D2: Coupling-matrix unitarity, max err %.3e', max_err));
    grid on;
    sanity.savePlot(fig, 'plot_d2_coupling_unitarity.png');

    details = struct();
    details.freqs_hz = freqs;
    details.per_freq_error = per_freq_err;
    result = sanity.makeResult('D2_couplingUnitarity', max_err < 1e-12, max_err, 0.0, 1e-12, details, 'plot_d2_coupling_unitarity.png');
end
