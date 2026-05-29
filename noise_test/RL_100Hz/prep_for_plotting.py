"""
Convert RL_1000Hz CSV outputs to the schema noise_plots.tex expects
(same as NMPC/). Re-run after the raw RL CSVs are refreshed.

Reads (in this folder):
  ofat_aggregated.csv, ofat_results.csv, ofat_thresholds.csv,
  rsm_results.csv, rsm_contour_f{i}_f{j}.csv (6 pairs)

Writes (in this folder):
  ofat_summary.csv, ofat_runs.csv, meta_factors.csv, meta_scalars.csv,
  rsm_runs.csv, rsm_contour_grid_pair{1..6}.csv
"""

import csv
from pathlib import Path
from shutil import copyfile

HERE = Path(__file__).parent

# --- 1. ofat_aggregated.csv -> ofat_summary.csv -----------------------
# Rename cols + add level_idx (1..9 within each factor) and n_reps=3.
src = HERE / "ofat_aggregated.csv"
dst = HERE / "ofat_summary.csv"
with src.open(newline="") as f_in, dst.open("w", newline="") as f_out:
    reader = csv.DictReader(f_in)
    writer = csv.writer(f_out)
    writer.writerow([
        "factor", "level_idx", "sigma", "y_rms_median",
        "y_rms_p25", "y_rms_p75", "p_diverged", "n_reps", "factor_name",
    ])
    level_counters: dict[str, int] = {}
    for row in reader:
        f = row["factor_idx"]
        level_counters[f] = level_counters.get(f, 0) + 1
        writer.writerow([
            f, level_counters[f], row["sigma"], row["rms_median"],
            row["rms_p25"], row["rms_p75"], row["p_div"], 3, row["factor_name"],
        ])

# --- 2. ofat_results.csv -> ofat_runs.csv -----------------------------
# Schema already compatible; copy under the expected name.
copyfile(HERE / "ofat_results.csv", HERE / "ofat_runs.csv")

# --- 3. ofat_thresholds.csv -> meta_factors.csv -----------------------
# Rename `threshold` -> `sigma_thr_p50` and add log10_center (same RSM
# centres as NMPC: the experimental design is shared).
LOG10_CENTERS = {"1": -4.5, "2": -3.5, "3": -3.5, "4": -2.5}
src = HERE / "ofat_thresholds.csv"
dst = HERE / "meta_factors.csv"
with src.open(newline="") as f_in, dst.open("w", newline="") as f_out:
    reader = csv.DictReader(f_in)
    writer = csv.writer(f_out)
    writer.writerow(["factor_idx", "factor_name", "log10_center", "sigma_thr_p50"])
    for row in reader:
        writer.writerow([
            row["factor_idx"], row["factor_name"],
            LOG10_CENTERS[row["factor_idx"]], row["threshold"],
        ])

# --- 4. rsm_results.csv -> rsm_runs.csv -------------------------------
copyfile(HERE / "rsm_results.csv", HERE / "rsm_runs.csv")

# --- 5. rsm_contour_f{i}_f{j}.csv -> rsm_contour_grid_pair{k}.csv -----
# Pair order matches noise_plots.tex (pair1 = pos-vs-ori, etc.).
PAIR_FILES = [
    ("rsm_contour_f1_f2.csv", "rsm_contour_grid_pair1.csv"),
    ("rsm_contour_f1_f3.csv", "rsm_contour_grid_pair2.csv"),
    ("rsm_contour_f1_f4.csv", "rsm_contour_grid_pair3.csv"),
    ("rsm_contour_f2_f3.csv", "rsm_contour_grid_pair4.csv"),
    ("rsm_contour_f2_f4.csv", "rsm_contour_grid_pair5.csv"),
    ("rsm_contour_f3_f4.csv", "rsm_contour_grid_pair6.csv"),
]
for src_name, dst_name in PAIR_FILES:
    with (HERE / src_name).open(newline="") as f_in, \
         (HERE / dst_name).open("w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(["fa_coded", "fb_coded", "log10_y_rms"])
        # First two columns vary by pair (f1_coded/f2_coded, f1_coded/f3_coded, ...).
        a_col, b_col = reader.fieldnames[0], reader.fieldnames[1]
        for row in reader:
            writer.writerow([row[a_col], row[b_col], row["log_y_rms"]])

# --- 6. meta_scalars.csv ---------------------------------------------
# y_rms_base isn't referenced by the .tex (the macro is defined but unused),
# so a placeholder of NaN is harmless. n_replicates is real (matches data).
with (HERE / "meta_scalars.csv").open("w", newline="") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["name", "value"])
    writer.writerow(["y_rms_base", "NaN"])
    writer.writerow(["n_replicates", 3])

print("Wrote NMPC-schema CSVs into", HERE)
