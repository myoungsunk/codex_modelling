script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

out_dir = fullfile(repo_root, 'results', 'week4_visuals');
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

stage2_path = resolvePreferredPath( ...
    fullfile(repo_root, 'results', 'stage2', 'stage2_900_ffd_relabel_det.csv'), ...
    fullfile(repo_root, 'results', 'stage2', 'stage2_900_ffd_relabel.csv'));
stage1_path = resolvePreferredPath( ...
    fullfile(repo_root, 'results', 'stage1', 'stage1_3000_ffd_det.csv'), ...
    fullfile(repo_root, 'results', 'stage1', 'stage1_3000_ffd.csv'));
label_auc_path = resolvePreferredPath( ...
    fullfile(repo_root, 'results', 'stage2', 'label_reanalysis_auc_det.csv'), ...
    fullfile(repo_root, 'results', 'stage2', 'label_reanalysis_auc.csv'));
label_room_auc_path = resolvePreferredPath( ...
    fullfile(repo_root, 'results', 'stage2', 'label_reanalysis_room_auc_det.csv'), ...
    fullfile(repo_root, 'results', 'stage2', 'label_reanalysis_room_auc.csv'));
mechanism_path = resolvePreferredPath( ...
    fullfile(repo_root, 'results', 'stage2', 'mechanism_stage2_det.csv'), ...
    fullfile(repo_root, 'results', 'stage2', 'mechanism_stage2.csv'));
hfss_path = resolvePreferredPath( ...
    fullfile(repo_root, 'results', 'stage3', 'hfss_case_list_det.csv'), ...
    fullfile(repo_root, 'results', 'stage3', 'hfss_case_list.csv'));

stage2 = readtable(stage2_path, 'TextType', 'string');
stage1 = readtable(stage1_path, 'TextType', 'string');
label_auc_tbl = readtable(label_auc_path, 'TextType', 'string');
label_room_auc_tbl = readtable(label_room_auc_path, 'TextType', 'string');
mechanism_tbl = readtable(mechanism_path, 'TextType', 'string');
hfss_tbl = readtable(hfss_path, 'TextType', 'string');

stage2.effective_eps_r = computeEffectiveEps(stage2.dominant_wall_material_cases, stage2.eps_r_multiplier_cases);
stage1.effective_eps_r = stage1.eps_r;
stage2.room_type = string(stage2.room_type_cases);
stage2.grid_layer = stage2.grid_layer_cases;
stage2.los_angle_deg = stage2.los_angle_from_anchor_bore_deg_cases;
stage2.los_distance = stage2.los_distance_m;

label_specs = {
    struct('name', 'current_0p20', 'display', 'Current @0.20', 'field', 'is_nlos_current_0p20');
    struct('name', 'geo_only', 'display', 'Geo', 'field', 'is_nlos_geo');
    struct('name', 'bounce_0p33_visible', 'display', 'Bounce @0.33', 'field', 'is_nlos_bounce_0p33');
    struct('name', 'mixed_0p33', 'display', 'Mixed @0.33', 'field', 'is_nlos_mixed_0p33')};
rooms = {'A', 'B', 'C'};
room_colors = [0.10 0.45 0.85; 0.10 0.65 0.35; 0.85 0.35 0.15];

plotF1a(repo_root, out_dir);
plotF1b(repo_root, out_dir);
plotF1c(repo_root, out_dir);
plotF2a(stage2, rooms, room_colors, out_dir);
plotF2b(stage2, rooms, room_colors, out_dir);
plotF2c(stage2, rooms, label_specs, out_dir);
plotF3a(label_room_auc_tbl, rooms, label_specs, out_dir);
plotF3b(label_room_auc_tbl, rooms, label_specs, out_dir);
plotF3c(stage1, label_room_auc_tbl, out_dir);
plotF4a(repo_root, stage2, mechanism_tbl, rooms, room_colors, out_dir);
plotF4b(stage2, rooms, out_dir);
plotF4c(label_auc_tbl, label_specs, out_dir);
plotF5(stage1, stage2, hfss_tbl, out_dir);
writeSummary(out_dir);

fprintf('Saved visualization checklist outputs under %s\n', out_dir);

function plotF1a(repo_root, out_dir)
    room_types = {'A', 'B', 'C'};
    fig = figure('Visible', 'off', 'Position', [100 100 1500 480]);
    tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    room_size = [5 4 3];
    for idx = 1:numel(room_types)
        nexttile(tl);
        scene = scenes.makeRoomABCScene(room_types{idx}, room_size);
        hold on;
        for sidx = 1:numel(scene.surfaces)
            surf = scene.surfaces{sidx};
            [verts, faces] = surfacePatchGeometry(surf);
            patch('Vertices', verts, 'Faces', faces, ...
                'FaceColor', 'none', 'EdgeColor', [0.15 0.15 0.15], 'LineWidth', 1.0);
            quiver3(surf.point(1), surf.point(2), surf.point(3), ...
                0.25 * surf.normal(1), 0.25 * surf.normal(2), 0.25 * surf.normal(3), ...
                'Color', [0.85 0.15 0.15], 'LineWidth', 1.1, 'MaxHeadSize', 0.9);
        end
        title(sprintf('Room %s Wireframe + Normals', room_types{idx}));
        xlabel('x (m)'); ylabel('y (m)'); zlabel('z (m)');
        axis equal;
        xlim([0 room_size(1)]); ylim([0 room_size(2)]); zlim([0 room_size(3)]);
        view(35, 24); grid on;
    end
    saveas(fig, fullfile(out_dir, 'F1a_room_wireframe_normals.png'));
    close(fig);
end

function plotF1b(repo_root, out_dir)
    room_types = {'A', 'B', 'C'};
    fig = figure('Visible', 'off', 'Position', [100 100 1500 480]);
    tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    room_size = [5 4 3];
    layer_colors = [0.20 0.50 0.95; 0.95 0.50 0.20];
    for idx = 1:numel(room_types)
        nexttile(tl);
        scene = scenes.makeRoomABCScene(room_types{idx}, room_size);
        hold on;
        for sidx = 1:numel(scene.surfaces)
            surf = scene.surfaces{sidx};
            [verts, faces] = surfacePatchGeometry(surf);
            patch('Vertices', verts, 'Faces', faces, ...
                'FaceColor', [0.92 0.92 0.95], 'FaceAlpha', 0.12, 'EdgeColor', [0.65 0.65 0.70]);
        end
        [grid_tbl, meta] = scenes.generateRoomTxRxGrid(room_types{idx}, room_size, 75, 42 + idx);
        scatter3(grid_tbl.tag_x(grid_tbl.grid_layer == 1), grid_tbl.tag_y(grid_tbl.grid_layer == 1), grid_tbl.tag_z(grid_tbl.grid_layer == 1), ...
            26, layer_colors(1, :), 'filled', 'MarkerFaceAlpha', 0.85);
        scatter3(grid_tbl.tag_x(grid_tbl.grid_layer == 2), grid_tbl.tag_y(grid_tbl.grid_layer == 2), grid_tbl.tag_z(grid_tbl.grid_layer == 2), ...
            30, layer_colors(2, :), 'filled', 'MarkerFaceAlpha', 0.85);
        scatter3(meta.anchor_pos(1), meta.anchor_pos(2), meta.anchor_pos(3), 120, 'k', '^', 'filled');
        title(sprintf('Room %s Grid Overlay', room_types{idx}));
        xlabel('x (m)'); ylabel('y (m)'); zlabel('z (m)');
        axis equal;
        xlim([0 room_size(1)]); ylim([0 room_size(2)]); zlim([0 room_size(3)]);
        view(35, 24); grid on;
        legend({'Surface', 'Layer 1', 'Layer 2', 'Anchor'}, 'Location', 'southoutside');
    end
    saveas(fig, fullfile(out_dir, 'F1b_room_grid_overlay.png'));
    close(fig);
end

function plotF1c(repo_root, out_dir)
    room_types = {'A', 'B', 'C'};
    room_size = [5 4 3];
    eps_values = [1 10];
    fig = figure('Visible', 'off', 'Position', [100 100 1500 480]);
    tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    cmap = turbo(256);
    for idx = 1:numel(room_types)
        nexttile(tl);
        scene = scenes.makeRoomABCScene(room_types{idx}, room_size);
        hold on;
        for sidx = 1:numel(scene.surfaces)
            surf = scene.surfaces{sidx};
            [verts, faces] = surfacePatchGeometry(surf);
            if strcmpi(surf.material.kind, 'PEC')
                fc = [0.55 0.55 0.58];
            else
                fc = mapColor(surf.material.eps_r, eps_values, cmap);
            end
            patch('Vertices', verts, 'Faces', faces, ...
                'FaceColor', fc, 'FaceAlpha', 0.72, 'EdgeColor', [0.20 0.20 0.20], 'LineWidth', 0.8);
        end
        title(sprintf('Room %s Material Color (\\epsilon_r)', room_types{idx}));
        xlabel('x (m)'); ylabel('y (m)'); zlabel('z (m)');
        axis equal;
        xlim([0 room_size(1)]); ylim([0 room_size(2)]); zlim([0 room_size(3)]);
        view(35, 24); grid on;
        colormap(gca, cmap);
        caxis(eps_values);
        cb = colorbar;
        cb.Label.String = '\epsilon_r (PEC shown in gray)';
    end
    saveas(fig, fullfile(out_dir, 'F1c_room_material_eps_colorbar.png'));
    close(fig);
end

function plotF2a(stage2, rooms, room_colors, out_dir)
    fig = figure('Visible', 'off', 'Position', [100 100 850 620]);
    hold on;
    for idx = 1:numel(rooms)
        mask = stage2.room_type == rooms{idx};
        scatter(stage2.los_angle_deg(mask), stage2.los_distance(mask), 28, ...
            room_colors(idx, :), 'filled', 'MarkerFaceAlpha', 0.55, 'DisplayName', sprintf('Room %s', rooms{idx}));
    end
    xlabel('LoS angle from anchor bore (deg)');
    ylabel('LoS distance (m)');
    title('F2a. Stage 2 Case Distribution in (LoS angle, LoS distance)');
    grid on;
    legend('Location', 'eastoutside');
    saveas(fig, fullfile(out_dir, 'F2a_los_angle_distance_scatter.png'));
    close(fig);
end

function plotF2b(stage2, rooms, room_colors, out_dir)
    fig = figure('Visible', 'off', 'Position', [100 100 1350 420]);
    tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    for idx = 1:numel(rooms)
        nexttile(tl);
        mask = stage2.room_type == rooms{idx};
        scatter(stage2.effective_eps_r(mask), stage2.xpol_coupling_db(mask), 28, ...
            stage2.grid_layer(mask), 'filled', 'MarkerFaceAlpha', 0.65);
        xlabel('Effective \epsilon_r');
        ylabel('xpol\_coupling\_db');
        title(sprintf('Room %s Parameter Coverage', rooms{idx}));
        grid on;
        colormap(gca, [room_colors(idx, :); min(room_colors(idx, :) + 0.25, 1.0)]);
        cb = colorbar;
        cb.Ticks = [1 2];
        cb.TickLabels = {'Layer 1', 'Layer 2'};
    end
    saveas(fig, fullfile(out_dir, 'F2b_eps_xpol_coverage_facets.png'));
    close(fig);
end

function plotF2c(stage2, rooms, label_specs, out_dir)
    fig = figure('Visible', 'off', 'Position', [100 100 1500 780]);
    tl = tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    for ridx = 1:numel(rooms)
        room_mask = stage2.room_type == rooms{ridx};
        for lidx = 1:numel(label_specs)
            nexttile(tl);
            y = logical(stage2.(label_specs{lidx}.field));
            neg = sum(room_mask & ~y);
            pos = sum(room_mask & y);
            bar(1, [neg pos], 'stacked');
            ylim([0 max(1, max(histcounts(categorical(stage2.room_type))) )]);
            xlim([0.25 1.75]);
            set(gca, 'XTick', 1, 'XTickLabel', {''});
            ylabel('Cases');
            title(sprintf('Room %s | %s', rooms{ridx}, label_specs{lidx}.display));
            legend({'LoS', 'NLoS'}, 'Location', 'northoutside');
            text(1, neg / 2, sprintf('%d', neg), 'HorizontalAlignment', 'center', 'Color', 'w', 'FontWeight', 'bold');
            text(1, neg + pos / 2, sprintf('%d', pos), 'HorizontalAlignment', 'center', 'Color', 'k', 'FontWeight', 'bold');
            grid on;
        end
    end
    saveas(fig, fullfile(out_dir, 'F2c_label_balance_grid.png'));
    close(fig);
end

function plotF3a(label_room_auc_tbl, rooms, label_specs, out_dir)
    fig = figure('Visible', 'off', 'Position', [100 100 1500 780]);
    tl = tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    for ridx = 1:numel(rooms)
        for lidx = 1:numel(label_specs)
            nexttile(tl);
            mask = strcmp(label_room_auc_tbl.room_type, rooms{ridx}) & strcmp(label_room_auc_tbl.label_name, label_specs{lidx}.name);
            row = label_room_auc_tbl(mask, :);
            vals = [row.auc_cir, row.auc_cp, row.auc_joint];
            bar(vals, 'FaceColor', 'flat');
            ylim([0.45 1.05]);
            set(gca, 'XTick', 1:3, 'XTickLabel', {'CIR', 'CP', 'Joint'});
            ylabel('AUC');
            title(sprintf('Room %s | %s', rooms{ridx}, label_specs{lidx}.display));
            grid on;
        end
    end
    saveas(fig, fullfile(out_dir, 'F3a_room_label_auc_grid.png'));
    close(fig);
end

function plotF3b(label_room_auc_tbl, rooms, label_specs, out_dir)
    delta_map = NaN(numel(label_specs), numel(rooms));
    for lidx = 1:numel(label_specs)
        for ridx = 1:numel(rooms)
            mask = strcmp(label_room_auc_tbl.label_name, label_specs{lidx}.name) & strcmp(label_room_auc_tbl.room_type, rooms{ridx});
            if any(mask)
                delta_map(lidx, ridx) = label_room_auc_tbl.delta_auc(find(mask, 1, 'first'));
            end
        end
    end
    fig = figure('Visible', 'off', 'Position', [100 100 680 420]);
    imagesc(delta_map);
    axis tight;
    colormap(turbo(256));
    colorbar;
    set(gca, 'XTick', 1:numel(rooms), 'XTickLabel', rooms);
    set(gca, 'YTick', 1:numel(label_specs), 'YTickLabel', cellfun(@(s) s.display, label_specs, 'UniformOutput', false));
    xlabel('Room');
    ylabel('Label rule');
    title('F3b. \DeltaAUC Heatmap (Joint - CIR)');
    for lidx = 1:numel(label_specs)
        for ridx = 1:numel(rooms)
            if isfinite(delta_map(lidx, ridx))
                text(ridx, lidx, sprintf('%.03f', delta_map(lidx, ridx)), ...
                    'HorizontalAlignment', 'center', 'Color', 'w', 'FontWeight', 'bold');
            end
        end
    end
    saveas(fig, fullfile(out_dir, 'F3b_delta_auc_heatmap.png'));
    close(fig);
end

function plotF3c(stage1, label_room_auc_tbl, out_dir)
    all_features = features.canonicalFeatureNames();
    cp_features = all_features(1:6);
    cir_features = all_features(7:end);
    mixed_stage1 = (~logical(stage1.has_los_path)) | (logical(stage1.has_los_path) & (stage1.bounce_to_los_ratio_mid >= 0.33));
    stage1_cir = fitOnly(stage1, cir_features, double(mixed_stage1));
    stage1_joint = fitOnly(stage1, all_features, double(mixed_stage1));

    mask = strcmp(label_room_auc_tbl.label_name, 'mixed_0p33');
    mixed_tbl = label_room_auc_tbl(mask, :);
    stage2_all_cir = mean(mixed_tbl.auc_cir, 'omitnan');
    stage2_all_joint = mean(mixed_tbl.auc_joint, 'omitnan');

    x_labels = {'Stage1 ALL', 'Stage2 A', 'Stage2 B', 'Stage2 C', 'Stage2 mean'};
    x = 1:numel(x_labels);
    cir_vals = [stage1_cir, mixed_tbl.auc_cir.', stage2_all_cir];
    joint_vals = [stage1_joint, mixed_tbl.auc_joint.', stage2_all_joint];

    fig = figure('Visible', 'off', 'Position', [100 100 800 420]);
    hold on;
    for idx = 1:numel(x)
        plot([x(idx) x(idx)], [cir_vals(idx) joint_vals(idx)], '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5);
    end
    scatter(x, cir_vals, 70, [0.10 0.35 0.85], 'filled', 'DisplayName', 'CIR');
    scatter(x, joint_vals, 70, [0.85 0.35 0.20], 'filled', 'DisplayName', 'Joint');
    xlim([0.5 numel(x_labels) + 0.5]);
    ylim([0.55 0.85]);
    set(gca, 'XTick', x, 'XTickLabel', x_labels);
    ylabel('AUC');
    title('F3c. Stage 1 vs Stage 2 (mixed@0.33)');
    grid on;
    legend('Location', 'northwest');
    saveas(fig, fullfile(out_dir, 'F3c_stage1_vs_stage2_mixed_dotplot.png'));
    close(fig);
end

function plotF4a(repo_root, stage2, mechanism_tbl, rooms, room_colors, out_dir)
    mechanism_tbl.case_id = double(mechanism_tbl.case_id);
    mechanism_tbl.room_type = string(mechanism_tbl.room_type);
    stage2.case_id = double(stage2.case_id);
    joined = innerjoin(stage2(:, {'case_id', 'room_type', 'anchor_x', 'anchor_y', 'anchor_z', 'tag_x', 'tag_y', 'tag_z'}), ...
        mechanism_tbl(:, {'case_id', 'gamma_cp_3_fp_only', 'ar_db_at_los', 'xpd_db_at_los'}), 'Keys', 'case_id');

    gain_db = computeLosGainGateDb(repo_root, joined);
    joined.gain_gate_db = gain_db;

    fig = figure('Visible', 'off', 'Position', [100 100 1350 420]);
    tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    for ridx = 1:numel(rooms)
        nexttile(tl);
        mask = joined.room_type == rooms{ridx};
        room_idx = find(mask);
        room_gain = joined.gain_gate_db(room_idx);
        gate = room_gain >= quantile(room_gain, 0.5);
        low_idx = room_idx(~gate);
        high_idx = room_idx(gate);
        scatter(joined.ar_db_at_los(low_idx), joined.gamma_cp_3_fp_only(low_idx), 16, ...
            [0.75 0.75 0.78], 'filled', 'MarkerFaceAlpha', 0.25); hold on;
        scatter(joined.ar_db_at_los(high_idx), joined.gamma_cp_3_fp_only(high_idx), 28, ...
            room_colors(ridx, :), 'filled', 'MarkerFaceAlpha', 0.75);
        xlabel('AR at LoS (dB)');
        ylabel('\gamma_{CP,3}');
        title(sprintf('Room %s | gain-gated highlight', rooms{ridx}));
        grid on;
    end
    saveas(fig, fullfile(out_dir, 'F4a_gamma_vs_ar_gain_gated.png'));
    close(fig);
end

function plotF4b(stage2, rooms, out_dir)
    all_features = features.canonicalFeatureNames();
    cp_features = all_features(1:6);
    cir_features = all_features(7:end);
    y_all = double(stage2.is_nlos_mixed_0p33);

    fig = figure('Visible', 'off', 'Position', [100 100 1350 430]);
    tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    for ridx = 1:numel(rooms)
        mask = stage2.room_type == rooms{ridx};
        tbl = stage2(mask, :);
        y = y_all(mask);
        [~, pred_cir] = fitAndPredict(tbl, cir_features, y);
        [~, pred_cp] = fitAndPredict(tbl, cp_features, y);
        valid = isfinite(pred_cir) & isfinite(pred_cp);
        cir_fail = (pred_cir(valid) > 0.5) ~= (y(valid) > 0.5);
        cp_fail = (pred_cp(valid) > 0.5) ~= (y(valid) > 0.5);
        only_cir = sum(cir_fail & ~cp_fail);
        only_cp = sum(cp_fail & ~cir_fail);
        both = sum(cir_fail & cp_fail);
        both_ok = sum(~cir_fail & ~cp_fail);

        nexttile(tl);
        hold on;
        axis equal;
        rectangle('Position', [0.1 0.2 1.3 1.3], 'Curvature', [1 1], 'FaceColor', [0.10 0.35 0.85 0.18], 'EdgeColor', [0.10 0.35 0.85], 'LineWidth', 1.5);
        rectangle('Position', [0.8 0.2 1.3 1.3], 'Curvature', [1 1], 'FaceColor', [0.85 0.35 0.20 0.18], 'EdgeColor', [0.85 0.35 0.20], 'LineWidth', 1.5);
        text(0.52, 1.63, 'CIR fail', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
        text(1.68, 1.63, 'CP fail', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
        text(0.50, 0.85, sprintf('CP saves\n%d', only_cir), 'HorizontalAlignment', 'center', 'FontSize', 11);
        text(1.45, 0.85, sprintf('both fail\n%d', both), 'HorizontalAlignment', 'center', 'FontSize', 11);
        text(2.00, 0.85, sprintf('CIR saves\n%d', only_cp), 'HorizontalAlignment', 'center', 'FontSize', 11);
        text(1.10, -0.05, sprintf('both correct = %d', both_ok), 'HorizontalAlignment', 'center', 'FontSize', 10);
        xlim([0 2.4]); ylim([-0.2 1.9]);
        axis off;
        title(sprintf('Room %s disagreement', rooms{ridx}));
    end
    saveas(fig, fullfile(out_dir, 'F4b_disagreement_venn.png'));
    close(fig);
end

function plotF4c(label_auc_tbl, label_specs, out_dir)
    rise_vals = NaN(1, numel(label_specs));
    gamma_vals = NaN(1, numel(label_specs));
    for idx = 1:numel(label_specs)
        mask = strcmp(label_auc_tbl.label_name, label_specs{idx}.name);
        rise_vals(idx) = label_auc_tbl.auc_rise_time_fp(find(mask, 1, 'first'));
        gamma_vals(idx) = label_auc_tbl.auc_gamma_cp_3(find(mask, 1, 'first'));
    end
    fig = figure('Visible', 'off', 'Position', [100 100 760 420]);
    bar(categorical(cellfun(@(s) s.display, label_specs, 'UniformOutput', false)), [rise_vals(:), gamma_vals(:)]);
    ylabel('Univariate AUC');
    title('F4c. rise\_time\_fp vs \gamma_{CP,3} by label rule');
    legend({'rise\_time\_fp', '\gamma_{CP,3}'}, 'Location', 'northwest');
    grid on;
    saveas(fig, fullfile(out_dir, 'F4c_rise_time_auc_by_label.png'));
    close(fig);
end

function plotF5(stage1, stage2, hfss_tbl, out_dir)
    [is_member, loc] = ismember(double(hfss_tbl.case_id), double(stage2.case_id));
    hfss_eps = NaN(height(hfss_tbl), 1);
    hfss_xpol = NaN(height(hfss_tbl), 1);
    hfss_eps(is_member) = stage2.effective_eps_r(loc(is_member));
    hfss_xpol(is_member) = stage2.xpol_coupling_db(loc(is_member));
    outside = ~(hfss_eps >= min(stage2.effective_eps_r) & hfss_eps <= max(stage2.effective_eps_r) & ...
        hfss_xpol >= min(stage2.xpol_coupling_db) & hfss_xpol <= max(stage2.xpol_coupling_db));

    fig = figure('Visible', 'off', 'Position', [100 100 900 620]);
    hold on;
    scatter(stage1.effective_eps_r, stage1.xpol_coupling_db, 8, [0.65 0.65 0.68], 'filled', 'MarkerFaceAlpha', 0.15, 'DisplayName', 'Stage 1 cloud');
    scatter(stage2.effective_eps_r, stage2.xpol_coupling_db, 16, [0.20 0.20 0.25], 'filled', 'MarkerFaceAlpha', 0.18, 'DisplayName', 'Stage 2 cloud');
    geo_mask = strcmp(hfss_tbl.group, 'GEO');
    bounce_mask = strcmp(hfss_tbl.group, 'BOUNCE');
    scatter(hfss_eps(geo_mask), hfss_xpol(geo_mask), 64, [0.15 0.45 0.90], 'o', 'filled', 'DisplayName', 'Stage 3 GEO');
    scatter(hfss_eps(bounce_mask), hfss_xpol(bounce_mask), 72, [0.90 0.45 0.15], '^', 'filled', 'DisplayName', 'Stage 3 BOUNCE');
    xlabel('Effective \epsilon_r');
    ylabel('xpol\_coupling\_db');
    title(sprintf('F5. Stage 3 candidates on Stage 1/2 cloud (outside cloud = %d)', sum(outside)));
    grid on;
    legend('Location', 'eastoutside');
    saveas(fig, fullfile(out_dir, 'F5_stage3_candidate_overlay.png'));
    close(fig);
end

function gain_db = computeLosGainGateDb(repo_root, joined)
    persistent ffd_r freq0
    if isempty(ffd_r)
        ffd_r = antennas.loadFfdPattern(fullfile(repo_root, 'RHCP_new_6G7G_11pts.ffd'));
        freq0 = mean(ffd_r.freqs_hz);
    end

    tx_bore = [0; 0; -1];
    rx_bore = [0; 0; 1];
    h_axis = [1; 0; 0];
    v_axis = [0; 1; 0];
    tx_R = [h_axis, v_axis, tx_bore];
    rx_R = [h_axis, v_axis, rx_bore];

    gain_db = NaN(height(joined), 1);
    for idx = 1:height(joined)
        tx = [joined.anchor_x(idx); joined.anchor_y(idx); joined.anchor_z(idx)];
        rx = [joined.tag_x(idx); joined.tag_y(idx); joined.tag_z(idx)];
        d_tx = normalizeVec(rx - tx);
        d_rx = normalizeVec(tx - rx);

        [theta_tx, phi_tx] = worldDirToLocalAngles(tx_R, d_tx);
        [theta_rx, phi_rx] = worldDirToLocalAngles(rx_R, d_rx);

        [Etx_theta, Etx_phi] = antennas.interpolatePattern(ffd_r, theta_tx, phi_tx, freq0);
        [Erx_theta, Erx_phi] = antennas.interpolatePattern(ffd_r, theta_rx, phi_rx, freq0);
        co_tx = abs((Etx_theta - 1i * Etx_phi) / sqrt(2.0)).^2;
        co_rx = abs((Erx_theta - 1i * Erx_phi) / sqrt(2.0)).^2;
        gain_db(idx) = 10 * log10(max(co_tx * co_rx, 1e-18));
    end
end

function writeSummary(out_dir)
    md = fullfile(out_dir, 'visualization_checklist.md');
    names = { ...
        'F1a_room_wireframe_normals.png'; ...
        'F1b_room_grid_overlay.png'; ...
        'F1c_room_material_eps_colorbar.png'; ...
        'F2a_los_angle_distance_scatter.png'; ...
        'F2b_eps_xpol_coverage_facets.png'; ...
        'F2c_label_balance_grid.png'; ...
        'F3a_room_label_auc_grid.png'; ...
        'F3b_delta_auc_heatmap.png'; ...
        'F3c_stage1_vs_stage2_mixed_dotplot.png'; ...
        'F4a_gamma_vs_ar_gain_gated.png'; ...
        'F4b_disagreement_venn.png'; ...
        'F4c_rise_time_auc_by_label.png'; ...
        'F5_stage3_candidate_overlay.png'};
    fid = fopen(md, 'w');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# Visualization Checklist Outputs\n\n');
    for idx = 1:numel(names)
        fprintf(fid, '- [%s](%s)\n', names{idx}, names{idx});
    end
end

function [verts, faces] = surfacePatchGeometry(surf)
    u = surf.u_axis(:) * surf.half_u;
    v = surf.v_axis(:) * surf.half_v;
    c = surf.point(:);
    verts = [c - u - v, c + u - v, c + u + v, c - u + v].';
    faces = [1 2 3 4];
end

function color = mapColor(value, limits, cmap)
    alpha = (value - limits(1)) / max(limits(2) - limits(1), 1e-12);
    alpha = min(max(alpha, 0.0), 1.0);
    idx = 1 + round(alpha * (size(cmap, 1) - 1));
    color = cmap(idx, :);
end

function eps_r = computeEffectiveEps(material_names, multiplier)
    eps_r = NaN(numel(multiplier), 1);
    for idx = 1:numel(multiplier)
        mat = materials.materialsLibrary(material_names(idx));
        eps_r(idx) = mat.eps_r * multiplier(idx);
    end
end

function [theta_rad, phi_rad] = worldDirToLocalAngles(R, dir_world)
    d_local = R' * normalizeVec(dir_world);
    theta_rad = acos(min(max(d_local(3), -1.0), 1.0));
    phi_rad = atan2(d_local(2), d_local(1));
    if phi_rad < 0
        phi_rad = phi_rad + 2.0 * pi;
    end
end

function v = normalizeVec(x)
    v = double(x(:));
    n = norm(v);
    if n <= 1e-12
        error('zero-length vector');
    end
    v = v / n;
end

function [auc, pred] = fitAndPredict(tbl, feature_names, y)
    X = table2array(tbl(:, feature_names));
    [auc, pred] = analysis.cvLogisticAuc(X, double(y));
end

function auc = fitOnly(tbl, feature_names, y)
    [auc, ~] = fitAndPredict(tbl, feature_names, y);
end

function path_out = resolvePreferredPath(primary_path, fallback_path)
    if exist(primary_path, 'file') == 2
        path_out = primary_path;
    else
        path_out = fallback_path;
    end
end
