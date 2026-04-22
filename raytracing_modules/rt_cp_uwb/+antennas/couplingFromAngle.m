function [ar_db, xpd_db] = couplingFromAngle(ant, look_direction)
% couplingFromAngle - simple angle-dependent AR/XPD taper from boresight.
%
%   AR(psi)  = AR_bore  + (AR_edge  - AR_bore)  * (1 - cos(psi)^2)
%   XPD(psi) = XPD_bore + (XPD_edge - XPD_bore) * (1 - cos(psi)^2)

    bore = normalizeVec(ant.boresight);
    look = normalizeVec(look_direction);
    cos_psi = max(0.0, dot(bore, look));
    taper = 1.0 - cos_psi ^ 2;

    ar_db = ant.axial_ratio_db + (ant.ar_edge_db - ant.axial_ratio_db) * taper;
    xpd_db = ant.cross_pol_leakage_db + (ant.xpd_edge_db - ant.cross_pol_leakage_db) * taper;
end

function v = normalizeVec(x)
    v = x(:);
    n = norm(v);
    if n <= 1e-12
        error('antennas:couplingFromAngle:ZeroVector', 'look direction must be non-zero');
    end
    v = v / n;
end
