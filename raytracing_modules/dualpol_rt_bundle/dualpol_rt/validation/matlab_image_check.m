function report = matlab_image_check()
% Manual oracle for 0/1/2 reflection geometry and phase validation.
% Run this script inside MATLAB with Antenna Toolbox / Communications Toolbox available.
%
% Expected workflow:
% 1) Mirror the Python shoebox geometry: 10 m x 6 m x 3 m, concrete walls.
% 2) Use raytrace(..., "Method","image", "MaxNumReflections",0|1|2).
% 3) Compare path count, path length, delay, bounce points, and per-path phase against Python output.

fc = 6.5e9;
roomSize = [10.0 6.0 3.0];
txPos = [1.5 3.0 2.2];
rxPos = [8.5 3.0 1.2];

fprintf('MATLAB image-method validation\n');
fprintf('fc = %.3f GHz\n', fc/1e9);
fprintf('room = [%.1f %.1f %.1f] m\n', roomSize);
fprintf('tx = [%.2f %.2f %.2f] m\n', txPos);
fprintf('rx = [%.2f %.2f %.2f] m\n', rxPos);

fprintf('\nSuggested commands:\n');
fprintf('  viewer = siteviewer("SceneModel","your_shoebox.stl");\n');
fprintf('  tx = txsite("cartesian","AntennaPosition",txPos);\n');
fprintf('  rx = rxsite("cartesian","AntennaPosition",rxPos);\n');
fprintf('  rays0 = raytrace(tx,rx,"Method","image","MaxNumReflections",0);\n');
fprintf('  rays1 = raytrace(tx,rx,"Method","image","MaxNumReflections",1);\n');
fprintf('  rays2 = raytrace(tx,rx,"Method","image","MaxNumReflections",2);\n');
fprintf('  % Then compare NumInteractions, PropagationDistance, AngleOfDeparture/Arrival.\n');

report = struct();
report.fc = fc;
report.roomSize = roomSize;
report.txPos = txPos;
report.rxPos = rxPos;
report.note = 'Populate this report manually with MATLAB raytrace(image) results and compare to Python Phase 1 baseline.';
end
