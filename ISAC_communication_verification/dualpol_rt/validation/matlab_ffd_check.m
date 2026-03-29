function report = matlab_ffd_check(portFiles, queryFreqHz)
% Manual oracle for single-file multi-frequency HFSS FFD inputs.
% portFiles: cell array or string array of .ffd files, e.g.
%   {
%     'LP_+45_new_6G7G_11pts.ffd'
%     'LP_-45_new_6G7G_11pts.ffd'
%     'RHCP_new_6G7G_11pts.ffd'
%     'LHCP_new_6G7G_11pts.ffd'
%   }
% queryFreqHz: target frequency, e.g. 6.5e9
%
% The script reads the same text FFD files as the Python parser and reports:
% - theta/phi grids
% - nearest-frequency boresight samples
% - peak direction sanity
% - sample cuts for manual Python/MATLAB comparison

if nargin < 1 || isempty(portFiles)
    error('Provide one or more .ffd files.');
end
if nargin < 2 || isempty(queryFreqHz)
    queryFreqHz = 6.5e9;
end
if ischar(portFiles) || isstring(portFiles)
    portFiles = cellstr(portFiles);
end

fprintf('MATLAB multi-frequency FFD validation\n');
fprintf('query frequency = %.3f GHz\n', queryFreqHz/1e9);

ports = struct([]);
for idx = 1:numel(portFiles)
    parsed = parseSingleFileMultiFreqFfd(portFiles{idx});
    [~, fIdx] = min(abs(parsed.freqGridHz - queryFreqHz));
    thetaIdx0 = 1;
    phiIdx0 = nearestIndex(parsed.phiGridDeg, 0.0);
    phiIdx90 = nearestIndex(parsed.phiGridDeg, 90.0);
    mag2 = abs(parsed.field(fIdx,:,:,1)).^2 + abs(parsed.field(fIdx,:,:,2)).^2;
    [~, peakFlat] = max(mag2(:));
    [peakThetaIdx, peakPhiIdx] = ind2sub(size(mag2), peakFlat);

    ports(idx).file = portFiles{idx}; %#ok<AGROW>
    ports(idx).freqGridHz = parsed.freqGridHz;
    ports(idx).thetaGridDeg = parsed.thetaGridDeg;
    ports(idx).phiGridDeg = parsed.phiGridDeg;
    ports(idx).queryFreqHz = parsed.freqGridHz(fIdx);
    ports(idx).boresightPhi0 = squeeze(parsed.field(fIdx, thetaIdx0, phiIdx0, :));
    ports(idx).boresightPhi90 = squeeze(parsed.field(fIdx, thetaIdx0, phiIdx90, :));
    ports(idx).peakThetaDeg = parsed.thetaGridDeg(peakThetaIdx);
    ports(idx).peakPhiDeg = parsed.phiGridDeg(peakPhiIdx);
    ports(idx).theta90Cut = squeeze(parsed.field(fIdx, nearestIndex(parsed.thetaGridDeg, 90.0), :, :));

    fprintf('\n[%d] %s\n', idx, portFiles{idx});
    fprintf('  nearest stored freq = %.3f GHz\n', parsed.freqGridHz(fIdx)/1e9);
    fprintf('  boresight @ theta=0, phi=0:   Etheta=%+.6e%+.6ej, Ephi=%+.6e%+.6ej\n', ...
        real(ports(idx).boresightPhi0(1)), imag(ports(idx).boresightPhi0(1)), ...
        real(ports(idx).boresightPhi0(2)), imag(ports(idx).boresightPhi0(2)));
    fprintf('  boresight @ theta=0, phi=90:  Etheta=%+.6e%+.6ej, Ephi=%+.6e%+.6ej\n', ...
        real(ports(idx).boresightPhi90(1)), imag(ports(idx).boresightPhi90(1)), ...
        real(ports(idx).boresightPhi90(2)), imag(ports(idx).boresightPhi90(2)));
    fprintf('  peak direction = theta %.1f deg, phi %.1f deg\n', ...
        ports(idx).peakThetaDeg, ports(idx).peakPhiDeg);
end

report = struct();
report.queryFreqHz = queryFreqHz;
report.ports = ports;
report.note = [
    "Use boresightPhi0 / boresightPhi90 to confirm the pole handling matches Python. " ...
    "At theta=0 the physical boresight is the +z pole, so phi labels are diagnostic only."
];
end

function parsed = parseSingleFileMultiFreqFfd(path)
lines = string(splitlines(fileread(path)));
lines = strtrim(lines);
lines = lines(lines ~= "" & ~startsWith(lines, "#") & ~startsWith(lines, "!") & ~startsWith(lines, "//"));

thetaMeta = sscanf(lines(1), '%f');
phiMeta = sscanf(lines(2), '%f');
thetaGridDeg = linspace(thetaMeta(1), thetaMeta(2), thetaMeta(3));
phiGridDeg = linspace(phiMeta(1), phiMeta(2), phiMeta(3));
nTheta = numel(thetaGridDeg);
nPhi = numel(phiGridDeg);
nSamples = nTheta * nPhi;

freqHeader = split(lines(3));
if lower(freqHeader(1)) ~= "frequencies"
    error('Expected "Frequencies N" header.');
end
nFreq = str2double(freqHeader(2));
freqGridHz = zeros(nFreq, 1);
field = complex(zeros(nFreq, nTheta, nPhi, 2));

cursor = 4;
for fIdx = 1:nFreq
    freqLine = split(lines(cursor));
    cursor = cursor + 1;
    if lower(freqLine(1)) ~= "frequency"
        error('Expected "Frequency <Hz>" line.');
    end
    freqGridHz(fIdx) = str2double(freqLine(2));
    for flatIdx = 0:(nSamples - 1)
        vals = sscanf(lines(cursor + flatIdx), '%f');
        tIdx = mod(flatIdx, nTheta) + 1;
        pIdx = floor(flatIdx / nTheta) + 1;
        field(fIdx, tIdx, pIdx, 1) = complex(vals(1), vals(2));
        field(fIdx, tIdx, pIdx, 2) = complex(vals(3), vals(4));
    end
    cursor = cursor + nSamples;
end

parsed = struct();
parsed.thetaGridDeg = thetaGridDeg;
parsed.phiGridDeg = phiGridDeg;
parsed.freqGridHz = freqGridHz;
parsed.field = field;
end

function idx = nearestIndex(values, target)
[~, idx] = min(abs(values - target));
end
