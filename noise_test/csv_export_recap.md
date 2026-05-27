# Experiment 4 — noise testing CSV export recap

Context: noise-tolerance study of the reduced-order NMPC controller, run via
`noiseRSM_reducedOrder.m`. We agreed (with the supervisor) on two plot families
worth keeping for the NMPC-vs-RL comparison:

1. **OFAT sensitivity** — one-factor-at-a-time sweeps of each noise factor.
2. **RSM y_rms contours** — quadratic response surface of `log10(y_rms)` over
   pairs of noise factors (Box–Behnken design).

`export_rsm_to_csv.m` writes everything needed to redraw both into
`csv_export/`. All other figures (P(div) contours, main-effects bar) are
intentionally not exported.

## Noise factors

| idx | name        | physical meaning                  | units  |
|-----|-------------|-----------------------------------|--------|
| 1   | sigma_pos   | position measurement noise std    | m      |
| 2   | sigma_ori   | orientation noise std             | rad    |
| 3   | sigma_vel   | linear-velocity noise std         | m/s    |
| 4   | sigma_ang   | angular-velocity noise std        | rad/s  |

Coded factor `fi ∈ {-1, 0, +1}` maps to real sigma via
`sigma_i = 10^(log10_center_i + step_per_unit * fi)`, with
`step_per_unit = 0.5` (one coded unit ≈ 3.16× in sigma).
`log10_center` and `step_per_unit` are in `meta_factors.csv` /
`meta_scalars.csv`.

## Files

### `rsm_runs.csv` — raw RSM runs (Box–Behnken)
One row per (design point, replicate). 27 design points × 3 reps = 81 rows.
Columns:
- `design`, `rep`, `seed`
- `f1`..`f4`: coded factor levels (−1, 0, +1)
- `sigma_pos`, `sigma_ori`, `sigma_vel`, `sigma_ang`: real sigmas
- `y_rms`: weighted RMS deviation from equilibrium over the settling window
- `y_maxpos`: max ‖position − equilibrium‖ over the run
- `y_diverged`: 1 if the run hit the divergence sanity box
- `log10_y_rms`: log10 of `y_rms`

Use for scatter overlays on the contour plots (the white dots in
`rsm_contours_yrms.fig`).

### `rsm_contour_grid.csv` — fitted RSM surface, gridded
The quadratic model
`log_y_rms ~ f1 + f2 + f3 + f4 + all pairwise interactions + f1^2 + f2^2 + f3^2 + f4^2`
is refit on surviving runs only, then evaluated on a 40×40 grid for each of
the 6 factor pairs (the two off-axis factors held at coded 0). 6 × 1600 = 9600 rows.
Columns: `pair_id`, `fa_idx`, `fb_idx`, `fa_coded`, `fb_coded`, `log10_y_rms`.

### `rsm_contour_pairs.csv` — pair-id lookup
Maps `pair_id` → (`fa_idx`, `fb_idx`, `fa_name`, `fb_name`). 6 rows.

### `ofat_runs.csv` — raw OFAT runs
One row per (factor, sigma level, replicate). 4 factors × 9 levels × 3 reps = 108 rows.
Columns: `factor`, `factor_name`, `level_idx`, `sigma`, `rep`, `y_rms`, `y_diverged`.
`y_rms` is `NaN` for diverged runs.

### `ofat_summary.csv` — what the OFAT subplots actually plot
Per (factor, sigma level): `y_rms_median`, `y_rms_p25`, `y_rms_p75`,
`p_diverged`, `n_reps`. 4 × 9 = 36 rows.

### `meta_factors.csv`
Per factor: `factor_idx`, `factor_name`, `log10_center`, `sigma_thr_p50`
(50%-divergence threshold from OFAT, may be `NaN` if not crossed).

### `meta_scalars.csv`
Scalars: `y_rms_base` (noise-free RMS floor — use as horizontal reference line),
`step_per_unit`, `n_replicates`, `sim_steps`.

## What to plot in TeX

### Figure: OFAT sensitivity (2×2 panels, one per factor)
For each `factor` in `ofat_summary.csv`:
- **Left axis (log–log):** `sigma` (x) vs `y_rms_median` (y), with the
  `y_rms_p25`..`y_rms_p75` band as a shaded region or dotted lines.
- **Right axis (linear, 0..1):** `sigma` (x, still log) vs `p_diverged` (y).
- **Vertical line:** `sigma_thr_p50` from `meta_factors.csv` (skip if NaN).
- **Optional horizontal line on left axis:** `y_rms_base` from `meta_scalars.csv`.

### Figure: RSM y_rms contours (6 panels, one per factor pair)
For each `pair_id` in `rsm_contour_pairs.csv`:
- Filled contour of `log10_y_rms` over (`fa_coded`, `fb_coded`) from
  `rsm_contour_grid.csv` filtered by that `pair_id`.
- Overlay design points: from `rsm_runs.csv`, scatter
  (`f{fa_idx}`, `f{fb_idx}`) as white-filled markers with black edge.
- Axis labels: `fa_name`, `fb_name` (coded units, range −1..+1).

Note: axes are in **coded** units. If you prefer real sigma on the axes,
convert via `sigma = 10^(log10_center + 0.5 * coded)` using `meta_factors.csv`.
