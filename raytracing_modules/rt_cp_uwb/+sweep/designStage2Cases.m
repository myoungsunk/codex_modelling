function cases = designStage2Cases(n_per_room, seed)
% designStage2Cases - Design Stage 2 room cases for patch-FFD sweeps.
%
% Each room receives:
%   - one fixed ceiling anchor with downward boresight
%   - a 2-layer tag-position grid
%   - LHS over SNR, XPD override, and epsilon scaling
%   - stratified dominant wall materials

    if nargin < 1 || isempty(n_per_room)
        n_per_room = 75;
    end
    if nargin < 2 || isempty(seed)
        seed = 1;
    end

    n_per_room = double(n_per_room);
    assert(n_per_room >= 2 && mod(n_per_room, 1) == 0, 'n_per_room must be an integer >= 2');
    rng(double(seed), 'twister');

    room_types = {'A', 'B', 'C'};
    room_size = [5.0, 4.0, 3.0];
    materials_list = {'wood', 'glass', 'concrete', 'drywall', 'brick'};

    all_cases = table();
    for room_idx = 1:numel(room_types)
        room_type = room_types{room_idx};
        [grid_tbl, ~] = scenes.generateRoomTxRxGrid(room_type, room_size, n_per_room, seed + room_idx - 1);

        n = height(grid_tbl);
        lhs = generateLhs(n, 3);
        material_idx = balancedIndexSequence(n, numel(materials_list));

        room_cases = table();
        room_cases.room_type = repmat({room_type}, n, 1);
        room_cases.anchor_x = grid_tbl.anchor_x;
        room_cases.anchor_y = grid_tbl.anchor_y;
        room_cases.anchor_z = grid_tbl.anchor_z;
        room_cases.tag_x = grid_tbl.tag_x;
        room_cases.tag_y = grid_tbl.tag_y;
        room_cases.tag_z = grid_tbl.tag_z;
        room_cases.grid_layer = grid_tbl.grid_layer;
        room_cases.los_distance_m = grid_tbl.los_distance_m;
        room_cases.los_angle_from_anchor_bore_deg = grid_tbl.los_angle_from_anchor_bore_deg;
        room_cases.los_angle_from_tag_bore_deg = grid_tbl.los_angle_from_tag_bore_deg;

        room_cases.snr_db = 10.0 + 30.0 * lhs(:, 1);
        room_cases.xpol_coupling_db = 20.0 + 20.0 * lhs(:, 2);
        room_cases.eps_r_multiplier = 0.80 + 0.40 * lhs(:, 3);
        room_cases.dominant_wall_material = materials_list(material_idx).';

        room_cases.antenna_type = repmat({'patch_ffd'}, n, 1);
        room_cases.material_name = room_cases.dominant_wall_material;

        all_cases = [all_cases; room_cases]; %#ok<AGROW>
    end

    idx = randperm(height(all_cases));
    cases = all_cases(idx, :);
    cases.case_id = (1:height(cases)).';
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
