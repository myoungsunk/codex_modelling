function result = checkD1_antennaCompare()
% checkD1_antennaCompare - compare ideal CP and realistic/synthetic patch antennas on the same off-boresight LoS link.

    cfg = config.defaultConfig();
    freqs = linspace(cfg.f_center - cfg.bw/2, cfg.f_center + cfg.bw/2, cfg.n_freq).';
    scene = core.Scene({});
    tx_pos = [0; 0; 1];
    rx_pos = [3; 3; 1];

    ffd_files = { ...
        'E:\0. CP Antenna\0. TRACK2_3_SIM\ANTENNA_SOURCE\RHCP_new_6G7G_11pts.ffd', ...
        'E:\0. CP Antenna\0. TRACK2_3_SIM\ANTENNA_SOURCE\LHCP_new_6G7G_11pts.ffd'};

    use_realistic_ffd = all(cellfun(@(p) exist(p, 'file') == 2, ffd_files));
    patch_source = [];
    reference_estimate = struct();
    if use_realistic_ffd
        patch_source = ffd_files;
        reference_estimate = antennas.loadPatchPattern(ffd_files);
    else
        ref_file = 'D:\codex\plot_data_save\reflector_3d_mode2\RHCP_6G7G.ffd';
        if exist(ref_file, 'file')
            reference_estimate = antennas.loadPatchPattern(ref_file);
        end
        patch_source = antennas.loadPatchPatternSynthetic( ...
            'ar_db_boresight', 3.0, ...
            'xpd_db_boresight', 20.0, ...
            'peak_gain_dbi', 7.0, ...
            'fitted_cos_exp', 2.0, ...
            'reference_estimate', reference_estimate);
    end

    tx_ideal = antennas.makeIdealCpAntenna('right', tx_pos, [1; 0; 0], [0; 1; 0], [0; 0; 1]);
    rx_ideal = antennas.makeIdealCpAntenna('right', rx_pos, [-1; 0; 0], [0; 1; 0], [0; 0; 1]);
    tx_patch = antennas.makeRealisticPatchAntenna(patch_source, tx_pos, [1; 0; 0], [0; 1; 0], [0; 0; 1]);
    rx_patch = antennas.makeRealisticPatchAntenna(patch_source, rx_pos, [-1; 0; 0], [0; 1; 0], [0; 0; 1]);

    paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 0);
    H_ideal = channel.buildChannel(paths, tx_ideal, rx_ideal, freqs);
    H_patch = channel.buildChannel(paths, tx_patch, rx_patch, freqs);

    same_ideal = squeeze(abs(H_ideal(1, 1, :)));
    reversed_ideal = squeeze(abs(H_ideal(2, 1, :)));
    same_patch = squeeze(abs(H_patch(1, 1, :)));
    reversed_patch = squeeze(abs(H_patch(2, 1, :)));

    mid = ceil(cfg.n_freq / 2);
    spectral_gap_db = 20 * log10(max(same_patch(mid), 1e-30) / max(same_ideal(mid), 1e-30));

    fig = figure('Visible', 'off');
    tiledlayout(2, 1);
    nexttile;
    plot(freqs / 1e9, same_ideal, 'b-', 'LineWidth', 1.5); hold on;
    plot(freqs / 1e9, same_patch, 'r--', 'LineWidth', 1.5);
    ylabel('|H_{11}|');
    legend('Ideal CP', 'Synthetic patch', 'Location', 'best');
    title(sprintf('D1: Off-boresight LoS same-hand comparison, gap %.2f dB', spectral_gap_db));
    grid on;
    nexttile;
    plot(freqs / 1e9, reversed_ideal, 'b-', 'LineWidth', 1.5); hold on;
    plot(freqs / 1e9, reversed_patch, 'r--', 'LineWidth', 1.5);
    xlabel('Frequency (GHz)');
    ylabel('|H_{21}|');
    legend('Ideal CP', 'Synthetic patch', 'Location', 'best');
    title('Reversed-hand response stays near zero for LoS because the scalar channel is preserved under unitary coupling');
    grid on;
    sanity.savePlot(fig, 'plot_d1_antenna_compare.png');

    details = struct();
    details.freqs_hz = freqs;
    details.same_hand_ideal = same_ideal;
    details.same_hand_patch = same_patch;
    details.reversed_hand_ideal = reversed_ideal;
    details.reversed_hand_patch = reversed_patch;
    details.midband_gap_db = spectral_gap_db;
    details.use_realistic_ffd = use_realistic_ffd;
    details.patch_source = patch_source;
    details.reference_estimate = reference_estimate;
    result = sanity.makeResult('D1_antennaCompare', abs(spectral_gap_db) > 3.0, spectral_gap_db, '|gap| > 3 dB vs ideal', 3.0, details, 'plot_d1_antenna_compare.png');
end
