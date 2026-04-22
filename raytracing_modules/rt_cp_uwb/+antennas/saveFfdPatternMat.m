function outpath = saveFfdPatternMat(ffd, outpath)
% saveFfdPatternMat - Save canonical FFD struct to a MAT file.

    if nargin < 2 || isempty(outpath)
        if isfield(ffd, 'metadata') && isfield(ffd.metadata, 'source_filename')
            [~, stem] = fileparts(ffd.metadata.source_filename);
            outpath = fullfile(tempdir, [stem, '.mat']);
        else
            outpath = fullfile(tempdir, 'ffd_pattern.mat');
        end
    end

    ffd = orderfields(ffd); %#ok<NASGU>
    save(outpath, 'ffd');
end
