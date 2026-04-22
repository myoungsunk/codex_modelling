function report = matlab_raypl_check()
% Manual oracle for polarization / basis convention checks.
% Validate V/H/LHCP/RHCP and custom Jones vectors against the Python conversion rules.

fc = 6.5e9;
fprintf('MATLAB raypl polarization validation\n');
fprintf('fc = %.3f GHz\n', fc/1e9);
fprintf('\nCheck list:\n');
fprintf('  1) Use raypl with Tx/Rx polarization set to V and H.\n');
fprintf('  2) Repeat with LHCP and RHCP.\n');
fprintf('  3) Repeat with custom normalized Jones vectors in [H;V] order.\n');
fprintf('  4) Verify MATLAB circular component order [L;R] matches Python circular_order="LR".\n');
fprintf('  5) If user-facing order needs [R;L], apply an explicit permutation matrix outside the core solver.\n');

report = struct();
report.fc = fc;
report.circularOrder = 'LR';
report.linearOrder = 'HV';
report.timeConvention = 'exp(-j*w*t)';
report.note = 'Use this script as a checklist while comparing MATLAB raypl outputs against Python basis conversion and handedness conventions.';
end
