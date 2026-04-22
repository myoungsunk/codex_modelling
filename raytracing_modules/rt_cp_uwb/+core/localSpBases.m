function [s_in, p_in, s_out, p_out, theta_i, n_eff] = localSpBases(k_in, k_out, normal)
% localSpBases - local s/p bases for incoming and outgoing directions.
%
% Mirrors the Python local_sp_bases logic, including:
% - deterministic effective normal orientation
% - normal-incidence fallback tangent handling
% - stable in/out basis alignment near grazing

    kin = normalizeVec(k_in);
    kout = normalizeVec(k_out);
    n = normalizeVec(normal);

    % Deterministically orient normal against incoming direction.
    % Tie-break near grazing with outgoing direction to keep reversal stable.
    dot_in = kin' * n;
    dot_out = kout' * n;
    if dot_in > 0.0 || (abs(dot_in) <= 1e-12 && dot_out > 0.0)
        n = -n;
    end

    cos_i = -(kin' * n);
    cos_i = min(max(cos_i, 0.0), 1.0);
    theta_i = acos(cos_i);

    s_in_raw = cross(kin, n);
    s_out_raw = cross(kout, n);
    n_in = norm(s_in_raw);
    n_out = norm(s_out_raw);
    eps_val = 1e-9;

    % Near normal incidence, k x n can vanish for both incident/reflected directions.
    if n_in < eps_val && n_out < eps_val
        s_ref = fallbackTangentFromNormal(n);
        s_in = s_ref;
        s_out = s_ref;
    elseif n_in < eps_val
        s_out = normalizeVec(s_out_raw);
        s_in = s_out;
    elseif n_out < eps_val
        s_in = normalizeVec(s_in_raw);
        s_out = s_in;
    else
        s_in = normalizeVec(s_in_raw);
        s_out = normalizeVec(s_out_raw);
        if (s_in' * s_out) < 0.0
            s_out = -s_out;
        end
    end

    p_in = normalizeVec(cross(kin, s_in));
    p_out = normalizeVec(cross(kout, s_out));
    n_eff = n;
end

function t = fallbackTangentFromNormal(n_eff)
    n = normalizeVec(n_eff);
    if abs(n(3)) < 0.8
        alt = [0; 0; 1];
    else
        alt = [0; 1; 0];
    end
    t = cross(n, alt);
    if norm(t) < 1e-9
        t = cross(n, [1; 0; 0]);
    end
    t = normalizeVec(t);
end

function v = normalizeVec(x)
    v = x(:);
    n = norm(v);
    if n == 0.0
        error('core:localSpBases:ZeroVector', 'zero-length vector');
    end
    v = v / n;
end
