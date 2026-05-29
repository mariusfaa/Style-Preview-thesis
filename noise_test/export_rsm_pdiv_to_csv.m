%% export_rsm_pdiv_to_csv.m
% Export the divergence-probability RSM surface used by
% noiseRSM_reducedOrder.m to CSV files for PGFPlots/TikZ.

clearvars; clc;

ctrl_dir = 'NMPC';
runs_path = fullfile(ctrl_dir, 'rsm_runs.csv');

results = readtable(runs_path);

mdl_div = fitglm(results, ...
    ['y_diverged ~ f1 + f2 + f3 + f4 + ' ...
     'f1:f2 + f1:f3 + f1:f4 + f2:f3 + f2:f4 + f3:f4 + ' ...
     'f1^2 + f2^2 + f3^2 + f4^2'], ...
    'Distribution','binomial','Link','logit');

pair_list = nchoosek(1:4, 2);
gx = linspace(-1, 1, 40);
[GX, GY] = meshgrid(gx, gx);

all_grid = table();
levels = linspace(0, 1, 11)';
level_table = table(levels, 'VariableNames', {'p_diverged'});
writetable(level_table, fullfile(ctrl_dir, 'rsm_pdiv_levels.csv'));

for k = 1:size(pair_list, 1)
    i1 = pair_list(k, 1);
    i2 = pair_list(k, 2);

    fvals = zeros(numel(GX), 4);
    fvals(:, i1) = GX(:);
    fvals(:, i2) = GY(:);
    pred = table(fvals(:,1), fvals(:,2), fvals(:,3), fvals(:,4), ...
        'VariableNames', {'f1','f2','f3','f4'});
    z = predict(mdl_div, pred);
    Z = reshape(z, size(GX));

    grid_table = table(GX(:), GY(:), Z(:), ...
        'VariableNames', {'fa_coded','fb_coded','p_diverged'});
    writetable(grid_table, fullfile(ctrl_dir, sprintf('rsm_pdiv_grid_pair%d.csv', k)));

    tagged_grid = table( ...
        repmat(k, numel(GX), 1), ...
        repmat(i1, numel(GX), 1), ...
        repmat(i2, numel(GX), 1), ...
        GX(:), GY(:), Z(:), ...
        'VariableNames', {'pair_id','fa_idx','fb_idx','fa_coded','fb_coded','p_diverged'});
    all_grid = [all_grid; tagged_grid]; %#ok<AGROW>

    % Extract the 50% failure boundary in the same coordinate system.
    C = contourc(gx, gx, Z, [0.5 0.5]);
    segment_id = [];
    x = [];
    y = [];
    idx = 1;
    sid = 0;
    while idx < size(C, 2)
        n = C(2, idx);
        pts = C(:, idx+1:idx+n);
        sid = sid + 1;
        segment_id = [segment_id; repmat(sid, n, 1); NaN]; %#ok<AGROW>
        x = [x; pts(1,:)'; NaN]; %#ok<AGROW>
        y = [y; pts(2,:)'; NaN]; %#ok<AGROW>
        idx = idx + n + 1;
    end

    boundary_table = table(segment_id, x, y, ...
        'VariableNames', {'segment_id','fa_coded','fb_coded'});
    writetable(boundary_table, fullfile(ctrl_dir, sprintf('rsm_pdiv_boundary_pair%d.csv', k)));
end

writetable(all_grid, fullfile(ctrl_dir, 'rsm_pdiv_grid.csv'));
fprintf('Wrote P(diverged) RSM CSV files into %s\n', ctrl_dir);
