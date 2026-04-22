function out_dir = ensureOutputDir()
% ensureOutputDir - create the sanity output directory if needed.

    cfg = config.defaultConfig();
    out_dir = cfg.sanity_dir;
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
end
