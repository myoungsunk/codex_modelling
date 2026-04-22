projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'day1');
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

ffdPaths = { ...
    'E:\0. CP Antenna\0. TRACK2_3_SIM\ANTENNA_SOURCE\RHCP_new_6G7G_11pts.ffd', ...
    'E:\0. CP Antenna\0. TRACK2_3_SIM\ANTENNA_SOURCE\LHCP_new_6G7G_11pts.ffd'};

summaryRows = cell(numel(ffdPaths), 6);
for idx = 1:numel(ffdPaths)
    filepath = ffdPaths{idx};
    assert(exist(filepath, 'file') == 2, 'FFD file not found: %s', filepath);

    rawLines = splitlines(string(fileread(filepath)));
    preview = rawLines(1:min(numel(rawLines), 50));
    [~, stem] = fileparts(filepath);
    txtPath = fullfile(outDir, [stem, '_inspect.txt']);
    fid = fopen(txtPath, 'w');
    assert(fid ~= -1, 'Failed to open %s', txtPath);
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, 'Path: %s\n\n', filepath);
    for lineIdx = 1:numel(preview)
        fprintf(fid, 'L%02d: %s\n', lineIdx, char(preview(lineIdx)));
    end
    clear cleanup;

    header1 = sscanf(char(preview(1)), '%f');
    header2 = sscanf(char(preview(2)), '%f');
    fileInfo = dir(filepath);
    summaryRows(idx, :) = {stem, fileInfo.bytes, round(header1(3)), round(header2(3)), round((header1(3) * header2(3))), extractFreqCount(preview)};

    fprintf('=== %s ===\n', stem);
    for lineIdx = 1:numel(preview)
        fprintf('L%02d: %s\n', lineIdx, char(preview(lineIdx)));
    end
end

summaryTbl = cell2table(summaryRows, 'VariableNames', ...
    {'file_stem', 'file_bytes', 'n_theta', 'n_phi', 'samples_per_freq', 'n_freq'});
writetable(summaryTbl, fullfile(outDir, 'ffd_inspect_summary.csv'));
disp(summaryTbl);

function nFreq = extractFreqCount(preview)
    nFreq = NaN;
    for i = 1:numel(preview)
        line = strtrim(preview(i));
        if startsWith(lower(line), 'frequencies')
            vals = sscanf(char(line), '%*s %f');
            nFreq = vals(end);
            return;
        end
    end
end
