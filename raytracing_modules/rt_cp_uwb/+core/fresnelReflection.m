function [Gamma_s, Gamma_p] = fresnelReflection(material, theta_i, f_hz)
    f_hz = f_hz(:);
    Nf = numel(f_hz);

    if strcmpi(material.kind, 'PEC')
        Gamma_s = -ones(Nf, 1);
        if material.pec_tm_sign >= 0
            Gamma_p = ones(Nf, 1);
        else
            Gamma_p = -ones(Nf, 1);
        end
        return;
    end

    sin2 = sin(theta_i)^2;
    cos_i = cos(theta_i);
    if cos_i < 1e-5
        warning('fresnel:grazing', 'Grazing incidence: theta_i near 90 deg');
    end

    if ~isempty(material.complex_eps_r)
        eps_c = material.complex_eps_r * ones(Nf, 1);
    else
        eps_c = material.eps_r * (1 - 1i * material.tan_delta) * ones(Nf, 1);
    end

    root = sqrt(eps_c - sin2);

    % Passive/causal branch enforcement
    flip_re = real(root) < 0;
    root(flip_re) = -root(flip_re);
    flip_im = (real(root) == 0) & (imag(root) < 0);
    root(flip_im) = -root(flip_im);

    Gamma_s = (cos_i - root) ./ (cos_i + root);
    Gamma_p = (eps_c * cos_i - root) ./ (eps_c * cos_i + root);

    % Passive-media clamp
    Gamma_s = clampPassive(Gamma_s);
    Gamma_p = clampPassive(Gamma_p);
end

function g = clampPassive(g)
    mag = abs(g);
    phase = exp(1i * angle(g));
    mag_c = min(mag, 1.0);
    g = mag_c .* phase;
end
