function [tx_pos, rx_pos, slab_normal, slab_center] = computeSingleBounceGeometry(incidence_deg, tx_height, slab_extent)
% computeSingleBounceGeometry - derive a horizontal-slab single-bounce geometry.
%
% incidence_deg : desired incidence angle from the slab normal
% tx_height     : TX/RX height above the slab
% slab_extent   : optional minimum slab span. If provided, reject
%                 geometries whose TX-RX spacing would exceed this extent.

    th = deg2rad(double(incidence_deg));
    assert(abs(th) < pi / 2, 'incidence_deg must be within (-90, 90) degrees');
    tx_height = double(tx_height);
    assert(tx_height > 0.0, 'tx_height must be positive');
    assert(incidence_deg >= 5.0 && incidence_deg <= 70.0, ...
        'incidence_deg must be within [5, 70] degrees for Stage 1');

    tx_rx_dist_local = 2.0 * tx_height * tan(th);
    assert(tx_rx_dist_local > 0.3, 'TX-RX too close for a stable single-bounce geometry');
    if nargin >= 3 && ~isempty(slab_extent)
        assert(tx_rx_dist_local <= double(slab_extent), ...
            'TX-RX spacing exceeds the available slab extent');
    end
    tx_pos = [0; 0; tx_height];
    rx_pos = [tx_rx_dist_local; 0; tx_height];
    slab_normal = [0; 0; 1];
    slab_center = [tx_rx_dist_local / 2.0; 0; 0];
end
