% Export MATLAB contourf response-surface figures for inclusion in LaTeX.
% This intentionally mirrors the MATLAB RSM contour rendering instead of
% approximating it with PGFPlots matrix heatmaps.

controllers = { ...
    'NMPC'; ...
    'RL_1000Hz'; ...
    'RL_100Hz' ...
};

pairFiles = { ...
    'rsm_contour_grid_pair1.csv', 1, 2, '\sigma_{pos} [mm]',     '\sigma_{ori} [mrad]',    {'0.01','0.1'}, {'0.1','1'}; ...
    'rsm_contour_grid_pair2.csv', 1, 3, '\sigma_{pos} [mm]',     '\sigma_{vel} [mm/s]',    {'0.01','0.1'}, {'0.1','1'}; ...
    'rsm_contour_grid_pair3.csv', 1, 4, '\sigma_{pos} [mm]',     '\sigma_{ang} [mrad/s]',  {'0.01','0.1'}, {'1','10'}; ...
    'rsm_contour_grid_pair4.csv', 2, 3, '\sigma_{ori} [mrad]',   '\sigma_{vel} [mm/s]',    {'0.1','1'},    {'0.1','1'}; ...
    'rsm_contour_grid_pair5.csv', 2, 4, '\sigma_{ori} [mrad]',   '\sigma_{ang} [mrad/s]',  {'0.1','1'},    {'1','10'}; ...
    'rsm_contour_grid_pair6.csv', 3, 4, '\sigma_{vel} [mm/s]',   '\sigma_{ang} [mrad/s]',  {'0.1','1'},    {'1','10'} ...
};

outDir = fullfile(pwd, 'figures');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

for c = 1:size(controllers, 1)
    ctrl = controllers{c};
    runs = readtable(fullfile(ctrl, 'rsm_runs.csv'));

    fig = figure('Visible', 'off', ...
        'Color', 'w', ...
        'Units', 'centimeters', ...
        'Position', [2, 2, 18.6, 11.8], ...
        'PaperPositionMode', 'auto');

    tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    for k = 1:size(pairFiles, 1)
        nexttile;

        gridPath = fullfile(ctrl, pairFiles{k, 1});
        T = readtable(gridPath);
        x = unique(T.fa_coded, 'sorted');
        y = unique(T.fb_coded, 'sorted');
        Z = reshape(T.log10_y_rms, numel(y), numel(x));

        contourf(x, y, Z, 15, 'LineColor', 'none');
        hold on;
        colormap(parula);
        cb = colorbar;
        cb.TickLabelInterpreter = 'tex';
        cb.FontSize = 7;
        cb.Color = 'k';
        cb.Label.String = '[mm]';
        cb.Label.Interpreter = 'tex';
        cb.Label.FontSize = 7;
        cb.Label.Color = 'k';
        cb.TickLabels = arrayfun(@(v) sprintf('%.1f', 1000 * 10.^v), ...
            cb.Ticks, 'UniformOutput', false);

        i1 = pairFiles{k, 2};
        i2 = pairFiles{k, 3};
        scatter(runs.(sprintf('f%d', i1)), runs.(sprintf('f%d', i2)), ...
            16, 'w', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.45);

        xlabel(pairFiles{k, 4}, 'Interpreter', 'tex');
        ylabel(pairFiles{k, 5}, 'Interpreter', 'tex');
        xlim([-1, 1]);
        ylim([-1, 1]);
        xticks([-1, 1]);
        xticklabels(pairFiles{k, 6});
        yticks([-1, 1]);
        yticklabels(pairFiles{k, 7});
        axis square;
        box on;
        ax = gca;
        set(ax, 'FontSize', 8, 'Layer', 'top', ...
            'XColor', 'k', 'YColor', 'k', ...
            'GridColor', 'k', 'MinorGridColor', 'k');
        ax.XLabel.Color = 'k';
        ax.YLabel.Color = 'k';
    end

    outPdf = fullfile(outDir, sprintf('rsm_contours_%s.pdf', ctrl));
    exportgraphics(fig, outPdf, 'ContentType', 'vector', 'BackgroundColor', 'white');
    close(fig);
end
