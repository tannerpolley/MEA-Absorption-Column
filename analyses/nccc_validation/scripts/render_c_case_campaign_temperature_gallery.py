from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ANALYSIS / "results" / "runs" / "nccc_2017_no_intercooler_sweep"
DEFAULT_FIGURE_DIR = ANALYSIS / "results" / "final" / "figures" / "nccc_2017_epcsaft_temperature_overlays"
DEFAULT_METRICS = ANALYSIS / "results" / "final" / "tables" / "nccc_2017_epcsaft_temperature_overlay_metrics.csv"
DEFAULT_PROFILE_INDEX = ANALYSIS / "results" / "final" / "tables" / "nccc_2017_epcsaft_temperature_profile_index.csv"
CASE_INPUTS = ROOT / "src" / "mea_absorption_column" / "data" / "C_cases_campaign_inputs.csv"


THERMO_MODEL = "epcsaft_ionic"
ACCEPTED_2017_CASE_IDS = tuple(f"{idx}C" for idx in range(1, 7))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = Path(args.run_dir)
    figure_dir = Path(args.figure_dir)
    metrics_path = Path(args.metrics_path)
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    case_data = pd.read_csv(CASE_INPUTS).set_index("Case")
    results = pd.read_csv(run_dir / "benchmark_results.csv")
    metrics = []
    plot_paths = []

    for case_id in ACCEPTED_2017_CASE_IDS:
        plot_path, case_metrics = _plot_case(case_id, case_data, results, run_dir, figure_dir)
        if plot_path is not None:
            plot_paths.append(plot_path)
            metrics.extend(case_metrics)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(metrics_path, index=False)
    _write_clean_profile_index(metrics_df)
    contact_sheet = _write_contact_sheet(plot_paths, figure_dir)

    print(f"Wrote {figure_dir}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {contact_sheet}")
    return 0


def _write_clean_profile_index(metrics: pd.DataFrame) -> None:
    index = metrics[["case_id", "thermo_model", "plot_png"]].copy()
    index = index.rename(columns={"plot_png": "profile_png"})
    index["clean_profile"] = True
    index["caveat"] = "accepted 2017 one-bed ePC-SAFT temperature-profile validation overlay"
    index.to_csv(DEFAULT_PROFILE_INDEX, index=False)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render accepted 2017 one-bed NCCC ePC-SAFT temperature-profile overlays."
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS))
    return parser.parse_args(argv)


def _plot_case(
    case_id: str,
    case_data: pd.DataFrame,
    results: pd.DataFrame,
    run_dir: Path,
    figure_dir: Path,
) -> tuple[Path | None, list[dict[str, object]]]:
    case_row = case_data.loc[case_id]
    tap_columns = _tap_columns(case_row)
    tap_positions = np.array([float(column) for column in tap_columns], dtype=float)
    tap_temperatures = case_row[tap_columns].astype(float).to_numpy()
    metrics = []

    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=170)
    profile_path = (
        run_dir
        / "profiles"
        / "NCCC_2017_cases"
        / case_id
        / "scipy-bvp"
        / THERMO_MODEL
        / "T.csv"
    )
    result_rows = results[
        (results["case_id"] == case_id)
        & (results["thermo_model"] == THERMO_MODEL)
        & (results["success"])
    ]
    if not profile_path.exists() or result_rows.empty:
        plt.close(fig)
        return None, []

    profile = pd.read_csv(profile_path).sort_values("Position")
    result_row = result_rows.iloc[0]
    capture_pct = float(result_row["capture_pct"])
    capture_error_pct = float(result_row["capture_error_pct"])
    temperature_rmse = float(result_row.get("temperature_rmse_K", np.nan))
    runtime_s = float(result_row["runtime_s"])

    ax.plot(
        profile["Position"],
        profile["Tl"],
        color="#b05a2a",
        linewidth=2.15,
        label=f"ePC-SAFT liquid ({capture_pct:.1f}%, err {capture_error_pct:+.1f})",
    )

    model_taps = np.interp(tap_positions, profile["Position"], profile["Tl"])
    residual = model_taps - tap_temperatures
    metrics.append(
        {
            "case_id": case_id,
            "thermo_model": THERMO_MODEL,
            "capture_pct": capture_pct,
            "capture_error_pct": capture_error_pct,
            "temperature_rmse_K": temperature_rmse,
            "tap_rmse_K": float(np.sqrt(np.mean(residual**2))),
            "tap_bias_K": float(np.mean(residual)),
            "tap_max_abs_K": float(np.max(np.abs(residual))),
            "runtime_s": runtime_s,
            "plot_png": str(
                (figure_dir / f"{case_id}_temperature_overlay.png")
                .relative_to(ROOT)
                .as_posix()
            ),
        }
    )

    ax.scatter(
        tap_positions,
        tap_temperatures,
        s=44,
        color="black",
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
        label="NCCC liquid taps",
    )
    ax.set_title(f"{case_id} ePC-SAFT temperature profile")
    ax.set_xlabel("Normalized column position")
    ax.set_ylabel("Temperature [K]")
    ax.set_xlim(0, 1)
    ax.set_ylim(min(tap_temperatures.min(), 316) - 3, max(tap_temperatures.max(), 352) + 3)
    ax.grid(alpha=0.24, linewidth=0.7)
    ax.legend(loc="best", fontsize=8, frameon=True, framealpha=0.92)
    fig.tight_layout()

    path = figure_dir / f"{case_id}_temperature_overlay.png"
    fig.savefig(path)
    plt.close(fig)
    return path, metrics


def _write_contact_sheet(plot_paths: list[Path], figure_dir: Path) -> Path:
    images = [Image.open(path).convert("RGB") for path in plot_paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    columns = 2
    rows = int(np.ceil(len(images) / columns))
    pad = 24
    sheet = Image.new("RGB", (columns * width + (columns + 1) * pad, rows * height + (rows + 1) * pad), "white")
    for index, image in enumerate(images):
        row = index // columns
        column = index % columns
        sheet.paste(image, (pad + column * (width + pad), pad + row * (height + pad)))

    contact_sheet = figure_dir / "nccc_2017_epcsaft_temperature_overlay_contact_sheet.png"
    sheet.save(contact_sheet, quality=92)
    return contact_sheet


def _tap_columns(case_row: pd.Series) -> list[str]:
    columns = [str(column) for column in case_row.index if _is_float_like(column)]
    return sorted(columns, key=lambda value: float(value))


def _is_float_like(value) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
