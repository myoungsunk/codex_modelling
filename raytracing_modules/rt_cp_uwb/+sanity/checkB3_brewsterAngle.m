function result = checkB3_brewsterAngle()
% checkB3_brewsterAngle - verify |Gamma_p| has a minimum near Brewster angle.

    cfg = config.defaultConfig();
    mat = core.Material('kind', 'dielectric', 'eps_r', 4.0, 'tan_delta', 0.0, 'name', 'eps4');
    incidence_deg = 0:1:80;
    gamma_s_mag = zeros(size(incidence_deg));
    gamma_p_mag = zeros(size(incidence_deg));

    for idx = 1:numel(incidence_deg)
        theta = deg2rad(incidence_deg(idx));
        [gamma_s, gamma_p] = core.fresnelReflection(mat, theta, cfg.f_center);
        gamma_s_mag(idx) = abs(gamma_s(1));
        gamma_p_mag(idx) = abs(gamma_p(1));
    end

    [min_mag, min_idx] = min(gamma_p_mag);
    measured_deg = incidence_deg(min_idx);
    expected_deg = rad2deg(atan(sqrt(mat.eps_r)));
    err_deg = abs(measured_deg - expected_deg);

    fig = figure('Visible', 'off');
    plot(incidence_deg, gamma_s_mag, 'b-', 'LineWidth', 1.5); hold on;
    plot(incidence_deg, gamma_p_mag, 'r-', 'LineWidth', 1.5);
    xline(expected_deg, 'k--', 'Brewster');
    xlabel('Incidence angle (deg)');
    ylabel('|Gamma|');
    legend('|Gamma_s|', '|Gamma_p|', 'Location', 'best');
    title(sprintf('B3: Brewster angle %.2f deg (expected %.2f)', measured_deg, expected_deg));
    grid on;
    sanity.savePlot(fig, 'plot_b3_brewster_angle.png');

    details = struct();
    details.incidence_deg = incidence_deg;
    details.gamma_s_mag = gamma_s_mag;
    details.gamma_p_mag = gamma_p_mag;
    details.min_gamma_p_mag = min_mag;
    result = sanity.makeResult('B3_brewsterAngle', err_deg < 2.0, measured_deg, expected_deg, 2.0, details, 'plot_b3_brewster_angle.png');
end
