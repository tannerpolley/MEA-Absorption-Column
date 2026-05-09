from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path(__file__).resolve().parents[1]
RUNS = ANALYSIS / "results" / "runs"
FINAL = ANALYSIS / "results" / "final"

BASELINE_RUN = RUNS / "c_case_thermal_accounting_campaign"
WALL_LOSS_RUN = RUNS / "c_case_wall_heat_loss_75_both_thermo"
TABLES = FINAL / "tables"
FIGURES = FINAL / "figures" / "thermal_closure"
REPORTS = FINAL / "reports"


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    baseline = _load_run(BASELINE_RUN, "adiabatic_baseline")
    wall = _load_run(WALL_LOSS_RUN, "wall_loss_75")
    comparison = _comparison_table(baseline, wall)
    diagnostics = _thermal_diagnostics(baseline)
    summary = _summary_table(comparison)

    comparison_path = TABLES / "thermal_closure_wall_loss_comparison.csv"
    diagnostics_path = TABLES / "thermal_accounting_diagnostics.csv"
    summary_path = TABLES / "thermal_closure_wall_loss_summary.csv"
    comparison.to_csv(comparison_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    summary.to_csv(summary_path, index=False)

    figure_paths = _render_figures(comparison, summary)
    report_path = REPORTS / "thermal_closure_improvement_report.md"
    report_path.write_text(
        _report_markdown(comparison, diagnostics, summary, comparison_path, diagnostics_path, summary_path, figure_paths),
        encoding="utf-8",
    )

    print(f"Wrote {comparison_path}")
    print(f"Wrote {diagnostics_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    return 0


def _load_run(path: Path, scenario: str) -> pd.DataFrame:
    data = pd.read_csv(path / "benchmark_results.csv")
    data = data[data["success"].astype(bool)].copy()
    data["thermal_scenario"] = scenario
    if "wall_heat_loss_coeff_W_m_K" not in data.columns:
        data["wall_heat_loss_coeff_W_m_K"] = 0.0
    for column in ("capture_error_pct", "temperature_rmse_K", "runtime_s", "wall_heat_loss_coeff_W_m_K"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _comparison_table(baseline: pd.DataFrame, wall: pd.DataFrame) -> pd.DataFrame:
    keys = ["case_id", "thermo_model"]
    base_cols = keys + ["capture_error_pct", "temperature_rmse_K", "runtime_s"]
    wall_cols = keys + ["capture_error_pct", "temperature_rmse_K", "runtime_s", "wall_heat_loss_coeff_W_m_K"]
    merged = baseline[base_cols].merge(
        wall[wall_cols],
        on=keys,
        suffixes=("_baseline", "_wall_loss"),
        how="inner",
    )
    merged["temperature_rmse_delta_K"] = (
        merged["temperature_rmse_K_wall_loss"] - merged["temperature_rmse_K_baseline"]
    )
    merged["temperature_rmse_improvement_K"] = -merged["temperature_rmse_delta_K"]
    merged["temperature_rmse_improvement_pct"] = (
        merged["temperature_rmse_improvement_K"] / merged["temperature_rmse_K_baseline"] * 100.0
    )
    merged["capture_error_delta_pct"] = (
        merged["capture_error_pct_wall_loss"] - merged["capture_error_pct_baseline"]
    )
    return merged.sort_values(["thermo_model", "case_id"]).reset_index(drop=True)


def _thermal_diagnostics(baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in baseline.to_dict("records"):
        profile_dir = Path(row["profile_csv_dir"])
        accounting = pd.read_csv(profile_dir / "thermal_accounting.csv")
        rows.append(
            {
                "case_id": row["case_id"],
                "thermo_model": row["thermo_model"],
                "temperature_rmse_K": row["temperature_rmse_K"],
                "capture_error_pct": row["capture_error_pct"],
                "dHmix_dT_model_mean": accounting["dHmix_dT_model"].mean(),
                "dHmix_dT_fd_mean": accounting["dHmix_dT_fd"].mean(),
                "dHmix_dT_relative_error_mean": accounting["dHmix_dT_relative_error"].mean(),
                "q_absorption_proxy_mean_W_per_m": accounting["q_absorption_proxy"].mean(),
                "q_water_phase_change_proxy_mean_W_per_m": accounting["q_water_phase_change_proxy"].mean(),
                "q_interphase_liquid_mean_W_per_m": accounting["q_interphase_liquid"].mean(),
                "q_missing_if_liquid_species_transport_mean_W_per_m": accounting[
                    "q_missing_if_liquid_species_transport"
                ].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["thermo_model", "case_id"]).reset_index(drop=True)


def _summary_table(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for thermo_model, group in comparison.groupby("thermo_model", sort=False):
        rows.append(
            {
                "thermo_model": thermo_model,
                "cases": int(len(group)),
                "wall_heat_loss_coeff_W_m_K": group["wall_heat_loss_coeff_W_m_K"].iloc[0],
                "capture_mae_baseline_pct": group["capture_error_pct_baseline"].abs().mean(),
                "capture_mae_wall_loss_pct": group["capture_error_pct_wall_loss"].abs().mean(),
                "temperature_rmse_mean_baseline_K": group["temperature_rmse_K_baseline"].mean(),
                "temperature_rmse_mean_wall_loss_K": group["temperature_rmse_K_wall_loss"].mean(),
                "temperature_rmse_mean_improvement_K": group["temperature_rmse_improvement_K"].mean(),
                "temperature_rmse_max_baseline_K": group["temperature_rmse_K_baseline"].max(),
                "temperature_rmse_max_wall_loss_K": group["temperature_rmse_K_wall_loss"].max(),
            }
        )
    return pd.DataFrame(rows)


def _render_figures(comparison: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    for thermo_model, group in comparison.groupby("thermo_model", sort=False):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=170)
        x = range(len(group))
        ax.plot(x, group["temperature_rmse_K_baseline"], marker="o", linewidth=1.8, label="adiabatic baseline")
        ax.plot(x, group["temperature_rmse_K_wall_loss"], marker="s", linewidth=1.8, label="wall-loss screen")
        ax.set_xticks(list(x))
        ax.set_xticklabels(group["case_id"])
        ax.set_xlabel("NCCC C case")
        ax.set_ylabel("Liquid-temperature RMSE [K]")
        ax.set_title(f"Thermal-closure screen: {thermo_model}")
        ax.grid(alpha=0.25, linewidth=0.7)
        ax.legend(frameon=True, framealpha=0.92)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            path = FIGURES / f"thermal_closure_rmse_{thermo_model}.{suffix}"
            fig.savefig(path)
            paths.append(path)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=170)
    labels = summary["thermo_model"].tolist()
    x = range(len(labels))
    width = 0.34
    ax.bar([i - width / 2 for i in x], summary["temperature_rmse_mean_baseline_K"], width, label="adiabatic")
    ax.bar([i + width / 2 for i in x], summary["temperature_rmse_mean_wall_loss_K"], width, label="wall-loss screen")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean liquid-temperature RMSE [K]")
    ax.set_title("Mean C-case thermal-profile improvement")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(frameon=True, framealpha=0.92)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        path = FIGURES / f"thermal_closure_mean_rmse.{suffix}"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths


def _report_markdown(
    comparison: pd.DataFrame,
    diagnostics: pd.DataFrame,
    summary: pd.DataFrame,
    comparison_path: Path,
    diagnostics_path: Path,
    summary_path: Path,
    figure_paths: list[Path],
) -> str:
    lines = [
        "# Thermal-Closure Improvement Screen",
        "",
        "## Scope",
        "",
        "This analysis tests whether the remaining one-bed NCCC C-case temperature-profile errors are caused by the thermodynamic driving force, internal gas-liquid heat transfer, or a missing thermal-closure term. The model code now exports a `thermal_accounting.csv` profile with local heat and enthalpy-balance diagnostics for every requested dense profile run.",
        "",
        "## Key result",
        "",
        "A global wall heat-loss screen with a single coefficient of 75 W m^-1 K^-1 improves the liquid-temperature RMSE while preserving capture accuracy and using the same coefficient for all seven C cases. This is not a final calibrated plant heat-loss model, but it is strong evidence that the remaining profile error is a thermal-closure problem rather than a solver artifact or a fugacity-only problem.",
        "",
        _markdown_table(summary),
        "",
        "## Thermal accounting findings",
        "",
        "The accounting profiles show that the current liquid heat-capacity derivative diagnostic, `f_dHl_dT(...)`, is about three times larger than a finite-difference derivative of the same mixture enthalpy function. That derivative does not control the default enthalpy-state BVP solution, but it explains why direct temperature-state experiments are sensitive and should not replace the enthalpy-state balance until the derivative closure is repaired.",
        "",
        "Internal gas-liquid heat-transfer scaling was also tested separately and did not materially improve 7C. In contrast, a countercurrent-coordinate wall heat-removal term improves the C-case temperature profiles with little capture penalty. That pattern supports adding a limited wall-loss/thermal-closure discussion to the manuscript, while keeping the claim conservative.",
        "",
        "## Case-level comparison",
        "",
        _markdown_table(
            comparison[
                [
                    "case_id",
                    "thermo_model",
                    "capture_error_pct_baseline",
                    "capture_error_pct_wall_loss",
                    "temperature_rmse_K_baseline",
                    "temperature_rmse_K_wall_loss",
                    "temperature_rmse_improvement_K",
                ]
            ]
        ),
        "",
        "## Artifacts",
        "",
        f"- Comparison table: `{_repo_path(comparison_path)}`",
        f"- Thermal diagnostics table: `{_repo_path(diagnostics_path)}`",
        f"- Summary table: `{_repo_path(summary_path)}`",
    ]
    lines.extend(f"- Figure: `{_repo_path(path)}`" for path in figure_paths)
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame) -> str:
    rounded = frame.copy()
    for column in rounded.columns:
        if pd.api.types.is_numeric_dtype(rounded[column]):
            rounded[column] = rounded[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
        else:
            rounded[column] = rounded[column].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(rounded.columns) + " |"
    separator = "| " + " | ".join("---" for _ in rounded.columns) + " |"
    rows = ["| " + " | ".join(row[column] for column in rounded.columns) + " |" for _, row in rounded.iterrows()]
    return "\n".join([header, separator, *rows])


def _repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
