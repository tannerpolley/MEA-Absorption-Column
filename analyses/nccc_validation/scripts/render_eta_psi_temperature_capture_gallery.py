from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ANALYSIS / "results" / "runs" / "eta_psi_1c_6c_mass_10_source_backed_profiles"
DEFAULT_FIGURE_DIR = ANALYSIS / "results" / "final" / "figures" / "eta_psi_1c_6c_temperature_capture_profiles"
DEFAULT_METRICS = ANALYSIS / "results" / "final" / "tables" / "eta_psi_1c_6c_temperature_capture_profile_metrics.csv"
SOURCE_CASES = ROOT / "src" / "mea_absorption_column" / "data" / "NCCC_2017_cases.csv"
MODEL_INPUTS = ROOT / "src" / "mea_absorption_column" / "data" / "NCCC_2017_model_inputs_mass.csv"
ABSORBER_TEMPERATURE_PROFILES = (
    ROOT / "src" / "mea_absorption_column" / "data" / "NCCC_2017_absorber_temperature_profiles.csv"
)

CASE_IDS = ("1C", "2C", "3C", "4C", "5C", "6C")
THERMO_MODELS = ("ideal_henry", "epcsaft_ionic")
MODEL_LABELS = {
    "ideal_henry": "Ideal Henry",
    "epcsaft_ionic": "ePC-SAFT ionic",
}
MODEL_COLORS = {
    "ideal_henry": "#4c78a8",
    "epcsaft_ionic": "#b05a2a",
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = Path(args.run_dir)
    figure_dir = Path(args.figure_dir)
    metrics_path = Path(args.metrics_path)
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    source_cases = pd.read_csv(SOURCE_CASES).set_index("case_no")
    model_inputs = pd.read_csv(MODEL_INPUTS).set_index("case_no")
    temperature_profiles = pd.read_csv(ABSORBER_TEMPERATURE_PROFILES).set_index("case_no")
    results = pd.read_csv(run_dir / "benchmark_results.csv")

    metrics: list[dict[str, object]] = []
    plot_paths: list[Path] = []
    for case_id in CASE_IDS:
        path, rows = _plot_case(
            case_id,
            source_cases,
            model_inputs,
            temperature_profiles,
            results,
            run_dir,
            figure_dir,
        )
        plot_paths.append(path)
        metrics.extend(rows)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(metrics_path, index=False)
    contact_sheet = _write_contact_sheet(plot_paths, figure_dir)

    print(f"Wrote {figure_dir}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {contact_sheet}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render eta_psi=1 NCCC 2017 1C-6C temperature/capture validation PNGs."
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS))
    return parser.parse_args(argv)


def _plot_case(
    case_id: str,
    source_cases: pd.DataFrame,
    model_inputs: pd.DataFrame,
    temperature_profiles: pd.DataFrame,
    results: pd.DataFrame,
    run_dir: Path,
    figure_dir: Path,
) -> tuple[Path, list[dict[str, object]]]:
    source = source_cases.loc[case_id]
    model_input = model_inputs.loc[case_id]
    source_profile = temperature_profiles.loc[case_id]
    reported_capture = float(source["absorber_capture_pct_avg"])
    reported_model_capture = _optional_float(source.get("reported_model_capture_pct"))
    summary_capture = _optional_float(source.get("summary_table_capture_pct"))

    fig, ax = plt.subplots(figsize=(8.4, 5.3), dpi=170)
    metrics: list[dict[str, object]] = []
    annotation_lines = [f"NCCC measured: {reported_capture:.1f}%"]
    if summary_capture is not None and not np.isclose(summary_capture, reported_capture):
        annotation_lines.append(f"NCCC summary: {summary_capture:.1f}%")
    if reported_model_capture is not None:
        annotation_lines.append(f"Morgan model: {reported_model_capture:.1f}%")

    for thermo_model in THERMO_MODELS:
        profile_path = (
            run_dir
            / "profiles"
            / "NCCC_2017_cases"
            / case_id
            / "scipy-bvp"
            / thermo_model
            / "T.csv"
        )
        result_rows = results[
            (results["case_id"] == case_id)
            & (results["thermo_model"] == thermo_model)
            & (results["success"].astype(str).str.lower().isin({"true", "1"}))
        ]
        if not profile_path.exists() or result_rows.empty:
            continue

        profile = pd.read_csv(profile_path).sort_values("Position")
        result = result_rows.iloc[0]
        color = MODEL_COLORS[thermo_model]
        label = MODEL_LABELS[thermo_model]
        capture_pct = float(result["capture_pct"])
        capture_error = float(result["capture_error_pct"])
        annotation_lines.append(f"{label}: {capture_pct:.1f}% ({capture_error:+.1f} pp)")

        ax.plot(profile["Position"], profile["Tl"], color=color, linewidth=2.1, label=f"{label} liquid")
        ax.plot(
            profile["Position"],
            profile["Tv"],
            color=color,
            linewidth=1.9,
            linestyle="--",
            label=f"{label} vapor",
        )
        source_positions, source_temperatures = _source_temperature_profile(source_profile)
        liquid_at_source = np.interp(source_positions, profile["Position"], profile["Tl"])
        metrics.append(
            {
                "case_id": case_id,
                "thermo_model": thermo_model,
                "capture_pct": capture_pct,
                "nccc_capture_pct": reported_capture,
                "capture_error_pct": capture_error,
                "reported_model_capture_pct": reported_model_capture,
                "liquid_outlet_model_K": float(profile.iloc[0]["Tl"]),
                "liquid_inlet_model_K": float(profile.iloc[-1]["Tl"]),
                "vapor_inlet_model_K": float(profile.iloc[0]["Tv"]),
                "vapor_outlet_model_K": float(profile.iloc[-1]["Tv"]),
                "absorber_profile_rmse_K": float(np.sqrt(np.mean((liquid_at_source - source_temperatures) ** 2))),
                "absorber_profile_bias_K": float(np.mean(liquid_at_source - source_temperatures)),
                "profile_png": str((figure_dir / f"{case_id}_temperature_capture_profile.png").relative_to(ROOT)),
            }
        )

    _plot_temperature_anchors(ax, source, model_input)
    _plot_source_temperature_profile(ax, source_profile)
    ax.set_title(f"NCCC 2017 {case_id} eta_psi=1.0 source-backed profiles")
    ax.set_xlabel("Normalized column position, bottom to top")
    ax.set_ylabel("Temperature [K]")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92, ncols=2)
    ax.text(
        0.985,
        0.03,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.4,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#999999", "alpha": 0.95},
    )
    fig.tight_layout()

    path = figure_dir / f"{case_id}_temperature_capture_profile.png"
    fig.savefig(path)
    plt.close(fig)
    return path, metrics


def _plot_temperature_anchors(ax, source: pd.Series, model_input: pd.Series) -> None:
    liquid_top = float(model_input["Tl"])
    liquid_bottom = _celsius_to_kelvin(source.get("absorber_rich_solvent_temp_C"))
    vapor_bottom = _celsius_to_kelvin(source.get("absorber_inlet_gas_temp_C"))
    vapor_top = _celsius_to_kelvin(source.get("absorber_outlet_gas_temp_C"))

    anchors = [
        (0.0, liquid_bottom, "NCCC rich liquid", "o", "#7f2704"),
        (1.0, liquid_top, "NCCC lean liquid", "o", "#d94801"),
        (0.0, vapor_bottom, "NCCC inlet gas", "s", "#08519c"),
        (1.0, vapor_top, "NCCC outlet gas", "s", "#3182bd"),
    ]
    for x_value, y_value, label, marker, color in anchors:
        if y_value is None:
            continue
        ax.scatter(
            [x_value],
            [y_value],
            s=58,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
            label=label,
        )


def _plot_source_temperature_profile(ax, source_profile: pd.Series) -> None:
    positions, temperatures = _source_temperature_profile(source_profile)
    ax.plot(
        positions,
        temperatures,
        color="black",
        linewidth=1.4,
        linestyle=":",
        marker="o",
        markersize=4.2,
        label="NCCC absorber profile",
        zorder=4,
    )


def _source_temperature_profile(source_profile: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    positions = []
    temperatures = []
    for column, value in source_profile.items():
        if not column.startswith("position_") or not column.endswith("_C"):
            continue
        parsed = _optional_float(value)
        if parsed is None:
            continue
        position = float(column.removeprefix("position_").removesuffix("_C"))
        positions.append(position)
        temperatures.append(parsed + 273.15)
    return np.array(positions, dtype=float), np.array(temperatures, dtype=float)


def _celsius_to_kelvin(value) -> float | None:
    value = _optional_float(value)
    if value is None:
        return None
    return value + 273.15


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


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

    contact_sheet = figure_dir / "eta_psi_1c_6c_temperature_capture_contact_sheet.png"
    sheet.save(contact_sheet, quality=92)
    return contact_sheet


if __name__ == "__main__":
    raise SystemExit(main())
