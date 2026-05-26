"""Reformat NMPC and RL mass-mismatch logs into a shared schema for plotting.

Output columns (one file per controller-case combination):
    time_ms, z_mm, horiz_offset_mm, tilt_mag_deg, ang_rate_mag_deg_per_s

The RL CSVs are wide-format with raw states only, so the derived
centre-of-mass and wobble magnitudes are computed here. NMPC already
ships them in nmpc_mass_mismatch_trajectories.csv; we just rescale units.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAD2DEG = 180.0 / math.pi
M2MM = 1000.0

# z setpoint is the same for both controllers (NMPC: z_m - z_error_m,
# RL: target_z column). Hard-coded here so the reference line in the
# plot matches the data exactly.
Z_REF_MM = 0.030119179 * M2MM

OUT_FIELDS = [
    "time_ms",
    "z_mm",
    "horiz_offset_mm",
    "tilt_mag_deg",
    "ang_rate_mag_deg_per_s",
]


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def process_nmpc() -> dict[str, float]:
    src = HERE / "nmpc_mass_mismatch_trajectories.csv"
    buckets: dict[str, list[dict]] = {"min": [], "nominal": [], "max": []}
    multipliers: dict[str, float] = {}

    with src.open() as fh:
        for row in csv.DictReader(fh):
            case = row["case"]
            multipliers[case] = float(row["mass_multiplier"])
            buckets[case].append({
                "time_ms": float(row["time_s"]) * 1000.0,
                "z_mm": float(row["z_m"]) * M2MM,
                "horiz_offset_mm": float(row["horizontal_offset_m"]) * M2MM,
                "tilt_mag_deg": float(row["tilt_magnitude_rad"]) * RAD2DEG,
                "ang_rate_mag_deg_per_s": float(row["angular_rate_magnitude_radps"]) * RAD2DEG,
            })

    name_map = {"min": "min", "nominal": "nom", "max": "max"}
    for case, rows in buckets.items():
        _write(HERE / f"mismatch_nmpc_{name_map[case]}.csv", rows)

    return {name_map[k]: v for k, v in multipliers.items()}


def process_rl() -> dict[str, float]:
    sources = {
        "min": HERE / "mismatch_data_L2N128_1_min_0.csv",
        "nom": HERE / "mismatch_data_L2N128_1_nom_1.csv",
        "max": HERE / "mismatch_data_L2N128_1_max_1.csv",
    }
    multipliers: dict[str, float] = {}

    for case, src in sources.items():
        rows: list[dict] = []
        with src.open() as fh:
            for row in csv.DictReader(fh):
                x = float(row["x"]); y = float(row["y"])
                a = float(row["alpha"]); b = float(row["beta"])
                va = float(row["valpha"]); vb = float(row["vbeta"])
                rows.append({
                    "time_ms": float(row["Time"]) * 1000.0,
                    "z_mm": float(row["z"]) * M2MM,
                    "horiz_offset_mm": math.hypot(x, y) * M2MM,
                    "tilt_mag_deg": math.hypot(a, b) * RAD2DEG,
                    "ang_rate_mag_deg_per_s": math.hypot(va, vb) * RAD2DEG,
                })
                multipliers[case] = float(row["mass_multiplier"])
        _write(HERE / f"mismatch_rl_{case}.csv", rows)

    return multipliers


def main() -> None:
    nmpc_m = process_nmpc()
    rl_m = process_rl()
    print(f"z reference [mm]: {Z_REF_MM:.4f}")
    print("NMPC mass multipliers:", {k: round(v, 3) for k, v in nmpc_m.items()})
    print("RL   mass multipliers:", {k: round(v, 3) for k, v in rl_m.items()})


if __name__ == "__main__":
    main()
