script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));
results_dir = fullfile(repo_root, 'results', 'week4');
if exist(results_dir, 'dir') ~= 7
    mkdir(results_dir);
end

room_types = {'A', 'B', 'C'};
room_size = [5, 4, 3];
room_center = [room_size(1) / 2; room_size(2) / 2; room_size(3) / 2];
expected_counts = containers.Map({'A', 'B', 'C'}, [6, 9, 13]);
summary_lines = {'# Week 4 Day 1 Room Visualization', '', '| Room | Surface Count | Expected | Enclosing Normals Inward |', '|---|---:|---:|---|'};

for room_idx = 1:numel(room_types)
    room_type = room_types{room_idx};
    scene = scenes.makeRoomABCScene(room_type, room_size);

    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 960 720]);
    ax = axes(fig); %#ok<LAXES>
    hold(ax, 'on');
    normals_ok = true;

    for surf_idx = 1:numel(scene.surfaces)
        surf = scene.surfaces{surf_idx};
        [vertices, faces] = surfacePatchGeometry(surf);
        patch(ax, 'Vertices', vertices, 'Faces', faces, ...
            'FaceColor', materialColor(surf.material.name), ...
            'FaceAlpha', 0.60, ...
            'EdgeColor', [0.15 0.15 0.15], ...
            'LineWidth', 0.75);

        if isEnclosingSurface(surf.name)
            normals_ok = normals_ok && dot(surf.normal(:), room_center - surf.point(:)) > 0.0;
        end
    end

    view(ax, 3);
    axis(ax, 'equal');
    axis(ax, [0 room_size(1) 0 room_size(2) 0 room_size(3)]);
    grid(ax, 'on');
    xlabel(ax, 'x (m)');
    ylabel(ax, 'y (m)');
    zlabel(ax, 'z (m)');
    title(ax, sprintf('Room %s: %d surfaces', room_type, numel(scene.surfaces)));
    saveas(fig, fullfile(results_dir, sprintf('room_%s_3d.png', room_type)));
    close(fig);

    summary_lines{end + 1} = sprintf('| %s | %d | %d | %s |', room_type, numel(scene.surfaces), expected_counts(room_type), tfStr(normals_ok)); %#ok<SAGROW>
end

fid = fopen(fullfile(results_dir, 'room_visual_summary.md'), 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', summary_lines{:});

function tf = isEnclosingSurface(name)
    tf = startsWith(name, 'wall_') || strcmp(name, 'floor') || strcmp(name, 'ceiling');
end

function [vertices, faces] = surfacePatchGeometry(surf)
    p = surf.point(:);
    u = surf.half_u * surf.u_axis(:);
    v = surf.half_v * surf.v_axis(:);
    vertices = [ ...
        (p - u - v).'; ...
        (p + u - v).'; ...
        (p + u + v).'; ...
        (p - u + v).'];
    faces = [1 2 3 4];
end

function c = materialColor(name)
    switch lower(char(string(name)))
        case 'concrete'
            c = [0.55 0.55 0.55];
        case 'drywall'
            c = [0.86 0.84 0.80];
        case 'glass'
            c = [0.45 0.72 0.92];
        case 'wood'
            c = [0.67 0.49 0.28];
        case 'brick'
            c = [0.74 0.36 0.29];
        case 'metal_pec'
            c = [0.70 0.73 0.78];
        otherwise
            c = [0.65 0.65 0.65];
    end
end

function out = tfStr(value)
    if value
        out = 'yes';
    else
        out = 'no';
    end
end
