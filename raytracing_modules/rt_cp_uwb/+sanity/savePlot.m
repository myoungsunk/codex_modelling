function savePlot(fig, filename)
% savePlot - save a hidden sanity figure to results/sanity.

    out_dir = sanity.ensureOutputDir();
    saveas(fig, fullfile(out_dir, filename));
    close(fig);
end
