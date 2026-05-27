% Export impulse boundary data to CSV files for pgfplots
clear; clc;

scriptDir = fileparts(mfilename('fullpath'));
matFile   = 'C:\Users\mariujf\Style-Preview-thesis\tex_impulse_boundary\impulse_boundary_data_L2N128_1.mat';
data      = load(matFile);

% --- File 1: bar chart data (direction index, recoverable, first failure) ---
N    = numel(data.critical_delta_v);
idx  = (1:N).';
rec  = data.critical_delta_v(:);
fail = data.first_failure_delta_v(:);

T1 = table(idx, rec, fail, 'VariableNames', {'idx','recoverable','first_failure'});
writetable(T1, fullfile(scriptDir, 'impulse_bars.csv'));

% --- File 2: 3D scatter data (direction x,y,z, critical velocity) ---
dirs = data.impulseDirections;   % 3 x N
dx = dirs(1,:).';
dy = dirs(2,:).';
dz = dirs(3,:).';
T2 = table(dx, dy, dz, rec, 'VariableNames', {'dx','dy','dz','critical'});
writetable(T2, fullfile(scriptDir, 'impulse_directions.csv'));

% --- File 3: metadata (search limit, controller, color range) ---
fid = fopen(fullfile(scriptDir, 'meta.csv'), 'w');
fprintf(fid, 'key,value\n');
fprintf(fid, 'search_limit,%.10g\n',  data.maxDeltaVLimit);
fprintf(fid, 'controller,%s\n',       data.controllerType);
fprintf(fid, 'num_directions,%d\n',   N);
fprintf(fid, 'critical_min,%.10g\n',  min(rec));
fprintf(fid, 'critical_max,%.10g\n',  max(rec));
fclose(fid);

% --- Files 4-6: spider chart data (one CSV per elevation ring) ---
% Compute spherical coordinates for all directions
azimuth_deg   = mod(atan2d(dy, dx), 360);        % [0, 360)
elevation_deg = atan2d(dz, sqrt(dx.^2 + dy.^2)); % [-90, 90]

% Separate poles (|elevation| > 80 deg) from azimuthal rings
is_pole           = abs(elevation_deg) > 80;
unique_ring_elevs = unique(round(elevation_deg(~is_pole)));  % rounded to nearest degree

assert(numel(unique_ring_elevs) == 3, ...
    'Expected exactly 3 non-pole elevation rings, found %d. Update main.tex accordingly.', ...
    numel(unique_ring_elevs));

ring_elevs_sorted = sort(unique_ring_elevs);   % ascending: lower, equatorial, upper
ring_files = {'spider_lower.csv', 'spider_eq.csv', 'spider_high.csv'};

for k = 1:3
    mask = round(elevation_deg) == ring_elevs_sorted(k) & ~is_pole;

    az_k  = azimuth_deg(mask);
    rec_k = rec(mask);
    ff_k  = fail(mask);

    % Sort by azimuth so the polygon is drawn in order
    [az_sorted, order] = sort(az_k);
    rec_sorted = rec_k(order);
    ff_sorted  = ff_k(order);

    % Close the polygon: repeat the first direction at azimuth = 360
    az_closed  = [az_sorted;  360];
    rec_closed = [rec_sorted; rec_sorted(1)];
    ff_closed  = [ff_sorted;  ff_sorted(1)];

    Tk = table(az_closed, rec_closed, ff_closed, ...
        'VariableNames', {'azimuth', 'recoverable', 'first_failure'});
    writetable(Tk, fullfile(scriptDir, ring_files{k}));
end

% --- File 7: meridian profile (robustness vs. elevation) ---
% Aggregate recoverable & first-failure over azimuth at each elevation
% (the 3 rings + the two poles), giving the elevation cross-section used
% by the meridian panel in main.tex. The poles become the +-90 endpoints.
% Columns rec_min/rec_max/rec_mean (and ff_*) let the plot pick a
% statistic without re-exporting; main.tex currently plots rec_mean.
elev_rounded = round(elevation_deg);
mer_elev     = sort(unique(elev_rounded));   % ascending, includes poles (+-90)
n_e          = numel(mer_elev);

[rec_min, rec_max, rec_mean] = deal(zeros(n_e,1));
[ff_min,  ff_max,  ff_mean]  = deal(zeros(n_e,1));
for k = 1:n_e
    m          = elev_rounded == mer_elev(k);
    rec_min(k)  = min(rec(m));  rec_max(k)  = max(rec(m));  rec_mean(k)  = mean(rec(m));
    ff_min(k)   = min(fail(m)); ff_max(k)   = max(fail(m)); ff_mean(k)   = mean(fail(m));
end

T7 = table(mer_elev, rec_min, rec_max, rec_mean, ff_min, ff_max, ff_mean, ...
    'VariableNames', {'elevation','rec_min','rec_max','rec_mean','ff_min','ff_max','ff_mean'});
% For multi-controller overlays, give each run its own file, e.g.
%   sprintf('meridian_%s.csv', data.controllerType)
% and list them in \meridianlist (main.tex). Single controller -> meridian.csv.
writetable(T7, fullfile(scriptDir, 'meridian.csv'));

fprintf('Exported %d directions to CSV.\n', N);
fprintf('  critical_delta_v range: [%.4f, %.4f] m/s\n', min(rec), max(rec));
fprintf('  search limit: %.2f m/s\n', data.maxDeltaVLimit);
fprintf('  Spider rings written at elevations: %s deg\n', mat2str(ring_elevs_sorted.'));
