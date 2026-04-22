function [tx_rx_list, grid_metadata] = generateRoomTxRxGrid(room_type, room_size, n_positions, seed)
% generateRoomTxRxGrid - Build a 2-layer anchor/tag grid for Stage 2 rooms.
%
% Inputs:
%   room_type   : 'A' | 'B' | 'C'
%   room_size   : [L, W, H] in meters, default [5 4 3]
%   n_positions : total number of anchor/tag pairs, split across 2 layers
%   seed        : RNG seed for targeted random fill
%
% Returns:
%   tx_rx_list  : table with anchor/tag positions and LoS-angle metadata
%   grid_metadata : struct summarizing the generated grid

    if nargin < 1 || isempty(room_type)
        room_type = 'A';
    end
    if nargin < 2 || isempty(room_size)
        room_size = [5.0, 4.0, 3.0];
    end
    if nargin < 3 || isempty(n_positions)
        n_positions = 50;
    end
    if nargin < 4 || isempty(seed)
        seed = 1;
    end

    room_size = double(room_size(:).');
    assert(numel(room_size) == 3, 'room_size must be [L, W, H]');
    n_positions = double(n_positions);
    assert(n_positions >= 2 && mod(n_positions, 1) == 0, 'n_positions must be an integer >= 2');
    rng(double(seed), 'twister');

    L = room_size(1);
    W = room_size(2);
    H = room_size(3);

    anchor_pos = [L / 2.0, W / 2.0, H - 0.30];
    n_layer1 = ceil(n_positions / 2.0);
    n_layer2 = n_positions - n_layer1;

    layer1 = buildLayer1Candidates(anchor_pos, room_size, n_layer1);
    layer2 = buildLayer2Candidates(anchor_pos, room_size, room_type, n_layer2);

    tx_rx_list = [layer1; layer2];
    tx_rx_list.room_type = repmat({char(string(room_type))}, height(tx_rx_list), 1);
    tx_rx_list = movevars(tx_rx_list, 'room_type', 'Before', 1);

    grid_metadata = struct();
    grid_metadata.room_type = char(string(room_type));
    grid_metadata.room_size = room_size;
    grid_metadata.anchor_pos = anchor_pos;
    grid_metadata.n_positions = height(tx_rx_list);
    grid_metadata.n_layer1 = sum(tx_rx_list.grid_layer == 1);
    grid_metadata.n_layer2 = sum(tx_rx_list.grid_layer == 2);
    grid_metadata.los_angle_min_deg = min(tx_rx_list.los_angle_from_anchor_bore_deg);
    grid_metadata.los_angle_max_deg = max(tx_rx_list.los_angle_from_anchor_bore_deg);
end

function tbl = buildLayer1Candidates(anchor_pos, room_size, n_keep)
    L = room_size(1);
    W = room_size(2);
    z_grid = [0.20, 1.20];
    n_xy = ceil(n_keep / numel(z_grid));
    x_count = max(3, ceil(sqrt(n_xy * L / W)));
    y_count = max(3, ceil(n_xy / x_count));
    while x_count * y_count * numel(z_grid) < n_keep
        if x_count <= y_count
            x_count = x_count + 1;
        else
            y_count = y_count + 1;
        end
    end
    x_grid = linspace(0.50, L - 0.50, x_count);
    y_grid = linspace(0.50, W - 0.50, y_count);

    rows = zeros(numel(x_grid) * numel(y_grid) * numel(z_grid), 10);
    idx = 1;
    for z_idx = 1:numel(z_grid)
        for y_idx = 1:numel(y_grid)
            for x_idx = 1:numel(x_grid)
                rows(idx, :) = buildRow(anchor_pos, [x_grid(x_idx), y_grid(y_idx), z_grid(z_idx)], 1);
                idx = idx + 1;
            end
        end
    end

    preferred = rows(rows(:, 9) >= 10.0 & rows(:, 9) <= 70.0, :);
    if size(preferred, 1) >= n_keep
        rows = preferred;
    end

    keep_idx = unique(round(linspace(1, size(rows, 1), n_keep)));
    if numel(keep_idx) < n_keep
        remaining = setdiff(1:size(rows, 1), keep_idx, 'stable');
        keep_idx = [keep_idx, remaining(1:(n_keep - numel(keep_idx)))];
    end
    rows = rows(keep_idx(1:n_keep), :);
    tbl = rowsToTable(rows);
end

function tbl = buildLayer2Candidates(anchor_pos, room_size, room_type, n_keep)
    L = room_size(1);
    W = room_size(2);
    wall_offset = 0.30;
    z_candidates = [0.20, 0.65, 1.20];

    fixed_points = [ ...
        wall_offset, W / 2.0, 0.20; ...
        L - wall_offset, W / 2.0, 1.20; ...
        L / 2.0, wall_offset, 0.20; ...
        L / 2.0, W - wall_offset, 1.20; ...
        wall_offset, wall_offset, 0.20; ...
        wall_offset, W - wall_offset, 1.20; ...
        L - wall_offset, wall_offset, 1.20; ...
        L - wall_offset, W - wall_offset, 0.20; ...
        0.20 * L, 0.50 * W, 0.65; ...
        0.80 * L, 0.50 * W, 0.65; ...
        0.50 * L, 0.20 * W, 0.65; ...
        0.50 * L, 0.80 * W, 0.65];

    if strcmpi(room_type, 'C')
        fixed_points = [fixed_points; ...
            0.30 * L, 0.78 * W, 0.40; ...
            0.62 * L, 0.60 * W, 0.95];
    elseif strcmpi(room_type, 'B')
        fixed_points = [fixed_points; ...
            0.70 * L, 0.62 * W, 0.35];
    end

    rows = zeros(max(n_keep, size(fixed_points, 1)) + 32, 10);
    idx = 1;
    for k = 1:size(fixed_points, 1)
        candidate = buildRow(anchor_pos, fixed_points(k, :), 2);
        if candidate(9) >= 10.0 && candidate(9) <= 70.0
            rows(idx, :) = candidate;
            idx = idx + 1;
        end
    end

    while idx <= n_keep
        side = randi(4);
        switch side
            case 1
                tag = [wall_offset + 0.25 * rand(), 0.25 + (W - 0.50) * rand(), z_candidates(randi(numel(z_candidates)))];
            case 2
                tag = [L - wall_offset - 0.25 * rand(), 0.25 + (W - 0.50) * rand(), z_candidates(randi(numel(z_candidates)))];
            case 3
                tag = [0.25 + (L - 0.50) * rand(), wall_offset + 0.25 * rand(), z_candidates(randi(numel(z_candidates)))];
            otherwise
                tag = [0.25 + (L - 0.50) * rand(), W - wall_offset - 0.25 * rand(), z_candidates(randi(numel(z_candidates)))];
        end
        candidate = buildRow(anchor_pos, tag, 2);
        if candidate(9) >= 10.0 && candidate(9) <= 70.0
            rows(idx, :) = candidate;
            idx = idx + 1;
        end
    end

    rows = rows(1:n_keep, :);
    tbl = rowsToTable(rows);
end

function row = buildRow(anchor_pos, tag_pos, layer_id)
    los_vec = tag_pos(:) - anchor_pos(:);
    los_dist = norm(los_vec);
    los_dir = los_vec / max(los_dist, 1e-12);
    anchor_bore = [0; 0; -1];
    tag_bore = [0; 0; 1];
    row = [ ...
        anchor_pos(:).', ...
        tag_pos(:).', ...
        double(layer_id), ...
        los_dist, ...
        acosd(clamp(dot(los_dir, anchor_bore))), ...
        acosd(clamp(dot(-los_dir, tag_bore)))];
end

function tbl = rowsToTable(rows)
    tbl = array2table(rows, 'VariableNames', { ...
        'anchor_x', 'anchor_y', 'anchor_z', ...
        'tag_x', 'tag_y', 'tag_z', ...
        'grid_layer', 'los_distance_m', ...
        'los_angle_from_anchor_bore_deg', 'los_angle_from_tag_bore_deg'});
end

function y = clamp(x)
    y = min(max(double(x), -1.0), 1.0);
end
