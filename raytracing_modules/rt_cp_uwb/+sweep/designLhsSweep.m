function cases = designLhsSweep(n_samples, seed)
% designLhsSweep - Latin Hypercube Sampling over Stage 1 conditioning axes.
%
% Continuous variables:
%   anchor_height      2.0 - 3.0 m
%   tag_height         0.1 - 1.5 m
%   tag_x              -2.0 - 2.0 m
%   tag_y              -2.0 - 2.0 m
%   eps_r              2 - 10
%   tan_delta          0.001 - 0.05 (log-uniform scaling after LHS)
%   xpol_coupling_db   20 - 40 dB (material)
%   ar_edge_db         5 - 15 dB (patch edge degradation)
%   snr_db             10 - 40 dB
%   slab_size_m        1.0 - 5.0 m
%
% Categorical variables:
%   material_name      {wood, glass, concrete, drywall, brick, ceramic_tile, metal_pec}
%   antenna_type       {ideal, patch_ffd}
%   slab_placement     {floor, wall_x, wall_y, ceiling}

    if nargin < 1 || isempty(n_samples)
        n_samples = 200;
    end
    if nargin < 2 || isempty(seed)
        seed = 1;
    end

    n_samples = double(n_samples);
    assert(n_samples >= 1 && mod(n_samples, 1) == 0, 'n_samples must be a positive integer');
    rng(double(seed), 'twister');

    n_cont = 10;
    lhs = generateLhs(n_samples, n_cont);

    cases = table();
    cases.anchor_height = 2.0 + 1.0 * lhs(:, 1);
    cases.tag_height = 0.1 + 1.4 * lhs(:, 2);
    cases.tag_x = -2.0 + 4.0 * lhs(:, 3);
    cases.tag_y = -2.0 + 4.0 * lhs(:, 4);
    cases.eps_r = 2.0 + 8.0 * lhs(:, 5);
    cases.tan_delta = 10 .^ (log10(0.001) + (log10(0.05) - log10(0.001)) * lhs(:, 6));
    cases.xpol_coupling_db = 20.0 + 20.0 * lhs(:, 7);
    cases.ar_edge_db = 5.0 + 10.0 * lhs(:, 8);
    cases.snr_db = 10.0 + 30.0 * lhs(:, 9);
    cases.slab_size_m = 1.0 + 4.0 * lhs(:, 10);

    materials_list = {'wood', 'glass', 'concrete', 'drywall', 'brick', 'ceramic_tile', 'metal_pec'};
    antenna_types = {'ideal', 'patch_ffd'};
    slab_placements = {'floor', 'wall_x', 'wall_y', 'ceiling'};

    material_idx = balancedIndexSequence(n_samples, numel(materials_list));
    antenna_idx = balancedIndexSequence(n_samples, numel(antenna_types));
    placement_idx = balancedIndexSequence(n_samples, numel(slab_placements));

    cases.material_name = materials_list(material_idx).';
    cases.antenna_type = antenna_types(antenna_idx).';
    cases.slab_placement = slab_placements(placement_idx).';

    los_dx = cases.tag_x;
    los_dy = cases.tag_y;
    los_dz = cases.tag_height - cases.anchor_height;
    cases.los_distance_m = sqrt(los_dx .^ 2 + los_dy .^ 2 + los_dz .^ 2);
    anchor_dot = max((cases.anchor_height - cases.tag_height) ./ max(cases.los_distance_m, 1e-9), -1.0);
    tag_dot = max((cases.anchor_height - cases.tag_height) ./ max(cases.los_distance_m, 1e-9), -1.0);
    cases.los_angle_from_anchor_bore_deg = acosd(min(anchor_dot, 1.0));
    cases.los_angle_from_tag_bore_deg = acosd(min(tag_dot, 1.0));

    % Legacy compatibility columns retained for downstream utilities.
    cases.tx_height = cases.anchor_height;
    cases.rx_height = cases.tag_height;
    cases.incidence_deg = cases.los_angle_from_anchor_bore_deg;
    cases.los_blockage = false(n_samples, 1);
    cases.num_slabs = ones(n_samples, 1);
    cases.slab_tilt_deg = zeros(n_samples, 1);

    idx = randperm(n_samples);
    cases = cases(idx, :);
    cases.case_id = (1:n_samples).';
    cases = movevars(cases, 'case_id', 'Before', 1);
end

function idx = balancedIndexSequence(n_samples, n_values)
    reps = ceil(n_samples / n_values);
    idx = repmat(1:n_values, 1, reps);
    idx = idx(1:n_samples);
    idx = idx(randperm(n_samples));
end

function lhs = generateLhs(n_samples, n_dims)
    if exist('lhsdesign', 'file') == 2
        lhs = lhsdesign(n_samples, n_dims, 'criterion', 'maximin', 'iterations', 10);
        return;
    end

    lhs = zeros(n_samples, n_dims);
    for dim = 1:n_dims
        edges = ((0:(n_samples - 1)).' + rand(n_samples, 1)) / n_samples;
        lhs(:, dim) = edges(randperm(n_samples));
    end
end
