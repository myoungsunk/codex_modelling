function [u, v] = transverseBasis(k, up_hint)
    kk = k(:) / norm(k);
    up = up_hint(:);
    up = up / norm(up);

    u = up - (up' * kk) * kk;
    if norm(u) < 1e-9
        if abs(kk(1)) < 0.8
            alt = [1; 0; 0];
        else
            alt = [0; 1; 0];
        end
        u = alt - (alt' * kk) * kk;
    end
    u = u / norm(u);
    v = cross(kk, u);
    v = v / norm(v);
end
