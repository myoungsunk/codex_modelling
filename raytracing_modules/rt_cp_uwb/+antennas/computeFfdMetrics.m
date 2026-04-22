function metrics = computeFfdMetrics(ffd, varargin)
% computeFfdMetrics - derive co/cross, AR, XPD, and power summaries from a canonical FFD struct.

    p = inputParser;
    p.addParameter('handedness', '', @(x) ischar(x) || isstring(x));
    p.parse(varargin{:});

    handedness = upper(char(string(p.Results.handedness)));
    if isempty(handedness)
        handedness = inferHandedness(ffd.port_label);
    end

    E_theta = ffd.E_theta;
    E_phi = ffd.E_phi;
    Er = (E_theta - 1i * E_phi) / sqrt(2.0);
    El = (E_theta + 1i * E_phi) / sqrt(2.0);
    if startsWith(handedness, 'L')
        co_field = El;
        cross_field = Er;
    else
        co_field = Er;
        cross_field = El;
    end

    co_power = abs(co_field).^2;
    cross_power = abs(cross_field).^2;
    xpd_db = 10 * log10(max(co_power, 1e-30) ./ max(cross_power, 1e-30));

    s0 = abs(E_theta).^2 + abs(E_phi).^2;
    s3 = -2 * imag(E_theta .* conj(E_phi));
    ratio = zeros(size(s0));
    mask = s0 > 1e-30;
    ratio(mask) = max(min(s3(mask) ./ s0(mask), 1.0), -1.0);
    chi = 0.5 * asin(ratio);
    tan_chi = abs(tan(chi));
    ar = ones(size(tan_chi));
    ar(mask) = max(1.0 ./ max(tan_chi(mask), 1e-12), 1.0);
    ar(~mask) = Inf;
    ar_db = 20 * log10(max(ar, 1.0));

    [~, theta0_idx] = min(abs(ffd.theta_rad));
    [~, phi0_idx] = min(abs(ffd.phi_rad));
    [~, phi90_idx] = min(abs(ffd.phi_rad - pi / 2.0));

    metrics = struct();
    metrics.handedness = handedness;
    metrics.co_pol_power = co_power;
    metrics.cross_pol_power = cross_power;
    metrics.co_pol_field = co_field;
    metrics.cross_pol_field = cross_field;
    metrics.ar_db = ar_db;
    metrics.xpd_db = xpd_db;
    metrics.total_radiated_power_numeric = ffd.metadata.total_radiated_power_numeric(:);
    metrics.theta0_idx = theta0_idx;
    metrics.phi0_idx = phi0_idx;
    metrics.phi90_idx = phi90_idx;
    metrics.boresight_co_pol_power = squeeze(co_power(theta0_idx, phi0_idx, :));
    metrics.boresight_cross_pol_power = squeeze(cross_power(theta0_idx, phi0_idx, :));
    metrics.boresight_ar_db = squeeze(ar_db(theta0_idx, phi0_idx, :));
    metrics.boresight_xpd_db = squeeze(xpd_db(theta0_idx, phi0_idx, :));
    metrics.boresight_E_theta = squeeze(E_theta(theta0_idx, phi0_idx, :));
    metrics.boresight_E_phi = squeeze(E_phi(theta0_idx, phi0_idx, :));
end

function handedness = inferHandedness(port_label)
    label = upper(char(string(port_label)));
    if startsWith(label, 'L')
        handedness = 'LHCP';
    else
        handedness = 'RHCP';
    end
end
