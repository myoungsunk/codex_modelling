function result = checkB4_normalIncidence()
% checkB4_normalIncidence - verify dielectric normal-incidence reflection magnitudes.

    cfg = config.defaultConfig();
    mat = core.Material('kind', 'dielectric', 'eps_r', 4.0, 'tan_delta', 0.0, 'name', 'eps4');
    [gamma_s, gamma_p] = core.fresnelReflection(mat, 0.0, cfg.f_center);

    mag_s = abs(gamma_s(1));
    mag_p = abs(gamma_p(1));
    expected = 1.0 / 3.0;
    err = max(abs([mag_s, mag_p] - expected));

    fig = figure('Visible', 'off');
    bar([mag_s, mag_p, expected]);
    set(gca, 'XTickLabel', {'|Gamma_s|', '|Gamma_p|', 'Expected'});
    ylabel('Magnitude');
    title(sprintf('B4: Normal-incidence magnitude, max err %.3e', err));
    grid on;
    sanity.savePlot(fig, 'plot_b4_normal_incidence.png');

    details = struct();
    details.gamma_s_mag = mag_s;
    details.gamma_p_mag = mag_p;
    details.error = err;
    result = sanity.makeResult('B4_normalIncidence', err < 1e-12, err, 0.0, 1e-12, details, 'plot_b4_normal_incidence.png');
end
