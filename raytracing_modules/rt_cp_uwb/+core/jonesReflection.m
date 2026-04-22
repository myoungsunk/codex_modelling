function R = jonesReflection(material, theta_i, f_hz)
% jonesReflection - local 2x2 reflection tensor in (s,p) basis.
%
% MATLAB convention:
%   R is 2 x 2 x Nf

    [Gamma_s, Gamma_p] = core.fresnelReflection(material, theta_i, f_hz);
    Nf = numel(f_hz);
    R = complex(zeros(2, 2, Nf));
    R(1, 1, :) = reshape(Gamma_s, 1, 1, []);
    R(2, 2, :) = reshape(Gamma_p, 1, 1, []);

    xdb = getFieldOr(material, 'xpol_coupling_db', []);
    xph = getFieldOr(material, 'xpol_coupling_phase_deg', 0.0);
    xdb_hv = getFieldOr(material, 'xpol_coupling_hv_db', []);
    xph_hv = getFieldOr(material, 'xpol_coupling_hv_phase_deg', xph);
    xdb_vh = getFieldOr(material, 'xpol_coupling_vh_db', []);
    xph_vh = getFieldOr(material, 'xpol_coupling_vh_phase_deg', -xph);

    if ~isempty(xdb_hv) && ~isfinite(xdb_hv)
        xdb_hv = [];
    end
    if ~isempty(xdb_vh) && ~isfinite(xdb_vh)
        xdb_vh = [];
    end

    if isempty(xdb_hv) && isempty(xdb_vh)
        if ~isempty(xdb) && isfinite(xdb)
            xdb_hv = xdb;
            xdb_vh = xdb;
            xph_hv = xph;
            xph_vh = -xph;
        end
    else
        if isempty(xdb_hv) && ~isempty(xdb_vh) && isfinite(xdb_vh)
            xdb_hv = xdb_vh;
            xph_hv = -xph_vh;
        end
        if isempty(xdb_vh) && ~isempty(xdb_hv) && isfinite(xdb_hv)
            xdb_vh = xdb_hv;
            xph_vh = -xph_hv;
        end
    end

    if ~isempty(xdb_hv) && ~isempty(xdb_vh)
        diag_scale = sqrt(max(abs(Gamma_s .* Gamma_p), 0.0));
        amp_hv = 10.0^(-xdb_hv / 20.0);
        amp_vh = 10.0^(-xdb_vh / 20.0);
        phase_hv = exp(1i * deg2rad(xph_hv));
        phase_vh = exp(1i * deg2rad(xph_vh));
        R(1, 2, :) = reshape(amp_hv * diag_scale * phase_hv, 1, 1, []);
        R(2, 1, :) = reshape(amp_vh * diag_scale * phase_vh, 1, 1, []);
    end
end

function value = getFieldOr(s, fieldName, defaultValue)
    if isstruct(s) && isfield(s, fieldName)
        value = s.(fieldName);
    else
        value = defaultValue;
    end
end
