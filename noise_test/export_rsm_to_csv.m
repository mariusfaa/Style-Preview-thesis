%% export_rsm_to_csv.m
% Reload a saved RSM .mat file and regenerate all CSV files consumed by
% the LaTeX/PGFPlots noise figures.
%
% Usage:
%   Run from noise_test/ with the updated .mat file present.
%   Writes to NMPC/ (creates it if absent).
%   After this, run export_rsm_pdiv_to_csv.m to refresh the P(div) grids.
%
% Files produced:
%   NMPC/rsm_runs.csv             -- raw Box-Behnken runs
%   NMPC/ofat_runs.csv            -- raw OFAT runs
%   NMPC/ofat_summary.csv         -- per-(factor,sigma) median/IQR/P(div)
%   NMPC/meta_factors.csv         -- per-factor log10_center + thr
%   NMPC/meta_scalars.csv         -- scalar parameters
%   NMPC/rsm_contour_grid_pair{1..6}.csv  -- fitted RSM surface on 40x40 grid

clearvars; clc;

mat_file = 'rsm_noise_results_UPDATEDNMPC.mat';
out_dir  = 'NMPC';

if ~isfile(mat_file)
    error('Cannot find %s — run from noise_test/ directory.', mat_file);
end
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

fprintf('Loading %s ...\n', mat_file);
d = load(mat_file);

results         = d.results;
ofat            = d.ofat;
ofat_thresholds = d.ofat_thresholds;
factor_names    = d.factor_names;
log_center      = d.log_center;
step_per_unit   = d.step_per_unit;
y_rms_base      = d.y_rms_base;
n_replicates    = d.n_replicates;
sim_steps       = d.sim_steps;

%% --- rsm_runs.csv ---
results.log10_y_rms = log10(max(results.y_rms, 1e-8));
writetable(results, fullfile(out_dir, 'rsm_runs.csv'));
fprintf('Wrote rsm_runs.csv  (%d rows)\n', height(results));

%% --- ofat_runs.csv ---
ofat.factor_name = factor_names(ofat.factor)';
writetable(ofat, fullfile(out_dir, 'ofat_runs.csv'));
fprintf('Wrote ofat_runs.csv  (%d rows)\n', height(ofat));

%% --- ofat_summary.csv ---
rows_out = {};
for f = 1:4
    mask_f = ofat.factor == f;
    lvls   = unique(ofat.sigma(mask_f));
    for li = 1:numel(lvls)
        m   = mask_f & ofat.sigma == lvls(li);
        lid = ofat.level_idx(find(m,1));
        rr  = ofat.y_rms(m & ~isnan(ofat.y_rms));
        if isempty(rr)
            med = NaN; p25 = NaN; p75 = NaN;
        else
            med = median(rr);
            p25 = quantile(rr, 0.25);
            p75 = quantile(rr, 0.75);
        end
        pdiv   = mean(ofat.y_diverged(m));
        nreps  = sum(m);
        fname  = factor_names{f};
        rows_out(end+1,:) = {f, lid, lvls(li), med, p25, p75, pdiv, nreps, fname}; %#ok<AGROW>
    end
end
ofat_summary = cell2table(rows_out, 'VariableNames', ...
    {'factor','level_idx','sigma','y_rms_median','y_rms_p25','y_rms_p75', ...
     'p_diverged','n_reps','factor_name'});
writetable(ofat_summary, fullfile(out_dir, 'ofat_summary.csv'));
fprintf('Wrote ofat_summary.csv  (%d rows)\n', height(ofat_summary));

%% --- meta_factors.csv ---
meta_factors = table((1:4)', factor_names', log_center', ofat_thresholds, ...
    'VariableNames', {'factor_idx','factor_name','log10_center','sigma_thr_p50'});
writetable(meta_factors, fullfile(out_dir, 'meta_factors.csv'));
fprintf('Wrote meta_factors.csv\n');

%% --- meta_scalars.csv ---
meta_scalars = table({'y_rms_base';'step_per_unit';'n_replicates';'sim_steps'}, ...
    [y_rms_base; step_per_unit; n_replicates; sim_steps], ...
    'VariableNames', {'name','value'});
writetable(meta_scalars, fullfile(out_dir, 'meta_scalars.csv'));
fprintf('Wrote meta_scalars.csv\n');

%% --- rsm_contour_grid_pair{k}.csv (fitted RSM surface, 40x40 per pair) ---
fprintf('\nFitting RSM quadratic model on surviving runs ...\n');

results.log_y_rms = results.log10_y_rms;
surv = results.y_diverged == 0;
n_surv = sum(surv);
fprintf('  Surviving runs: %d / %d\n', n_surv, height(results));

mdl = fitlm(results(surv,:), ...
    ['log_y_rms ~ f1 + f2 + f3 + f4 + ' ...
     'f1:f2 + f1:f3 + f1:f4 + f2:f3 + f2:f4 + f3:f4 + ' ...
     'f1^2 + f2^2 + f3^2 + f4^2']);
fprintf('  R^2 = %.4f   adj-R^2 = %.4f\n', mdl.Rsquared.Ordinary, mdl.Rsquared.Adjusted);

pair_list = nchoosek(1:4, 2);
gx        = linspace(-1, 1, 40);
[GX, GY]  = meshgrid(gx, gx);

for k = 1:size(pair_list, 1)
    i1 = pair_list(k, 1);
    i2 = pair_list(k, 2);

    fvals = zeros(numel(GX), 4);
    fvals(:, i1) = GX(:);
    fvals(:, i2) = GY(:);
    pred = table(fvals(:,1), fvals(:,2), fvals(:,3), fvals(:,4), ...
        'VariableNames', {'f1','f2','f3','f4'});
    z = predict(mdl, pred);
    Z = reshape(z, size(GX));

    grid_table = table(GX(:), GY(:), Z(:), ...
        'VariableNames', {'fa_coded','fb_coded','log10_y_rms'});
    out_path = fullfile(out_dir, sprintf('rsm_contour_grid_pair%d.csv', k));
    writetable(grid_table, out_path);
    fprintf('  Wrote rsm_contour_grid_pair%d.csv\n', k);
end

%% --- Print TeX scalar updates ---
fprintf('\n============================================================\n');
fprintf('Update these hardcoded values in noise_plots_NMPC.tex:\n\n');
fprintf('  \\newcommand{\\yRmsBase}{%.15g}\n', y_rms_base * 1e3);
for f = 1:4
    thr = ofat_thresholds(f);
    if ~isnan(thr)
        thr_mm = thr * 1e3;
    else
        thr_mm = NaN;
    end
    switch f
        case 1, name = 'sigmaThrPos';
        case 2, name = 'sigmaThrOri';
        case 3, name = 'sigmaThrVel';
        case 4, name = 'sigmaThrAng';
    end
    if isnan(thr_mm)
        fprintf('  \\newcommand{\\%s}{NaN}   %% threshold not reached\n', name);
    else
        fprintf('  \\newcommand{\\%s}{%.15g}\n', name, thr_mm);
    end
end
fprintf('============================================================\n\n');
fprintf('Done. Now run export_rsm_pdiv_to_csv.m to refresh P(div) grids.\n');
