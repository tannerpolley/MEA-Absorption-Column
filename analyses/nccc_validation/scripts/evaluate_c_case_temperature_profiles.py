from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path(__file__).resolve().parents[1]
RUNS = ANALYSIS / "results" / "runs" / "c_case_campaign_temperature_gallery"
FINAL = ANALYSIS / "results" / "final"
TABLES = FINAL / "tables"
FIGURES = FINAL / "figures"
DOC_FIGURES = ROOT / "docs" / "latex" / "figures"


THERMO_STYLE = {
    "ideal_henry": {"color": "#2f5d8c", "label": "ideal Henry"},
    "epcsaft_ionic": {"color": "#8a4b2b", "label": "ePC-SAFT"},
}


def _tap_columns(case_row: pd.Series) -> list[str]:
    cols = [str(col) for col in case_row.index if _is_float_like(col)]
    cols.sort(key=lambda value: float(value))
    return cols


def _is_float_like(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _load_cases() -> pd.DataFrame:
    return pd.read_csv(ROOT / "src" / "mea_absorption_column" / "data" / "C_cases_campaign_inputs.csv")


def _load_results() -> pd.DataFrame:
    return pd.read_csv(RUNS / "benchmark_results.csv")


def _tap_and_profile_metrics(
    case_id: str,
    thermo_model: str,
    case_row: pd.Series,
    run_row: pd.Series,
) -> dict:
    profile_path = RUNS / "profiles" / "C_cases_campaign_inputs" / case_id / "scipy-bvp" / thermo_model / "T.csv"
    profile = pd.read_csv(profile_path)
    tap_columns = _tap_columns(case_row)
    tap_positions = np.array([float(col) for col in tap_columns], dtype=float)
    observed_taps = case_row[tap_columns].astype(float).to_numpy()
    model_taps = np.interp(tap_positions, profile["Position"].to_numpy(), profile["Tl"].to_numpy())
    residual = model_taps - observed_taps

    return {
        "case_id": case_id,
        "thermo_model": thermo_model,
        "capture_error_pct": float(run_row["capture_error_pct"]),
        "case_capture_pct": float(run_row["capture_pct"]),
        "target_capture_pct": float(case_row["CO2 %"]),
        "temperature_rmse_K": float(run_row["temperature_rmse_K"]),
        "tap_rmse_K": float(np.sqrt(np.mean(residual**2))),
        "tap_mae_K": float(np.mean(np.abs(residual))),
        "tap_max_abs_K": float(np.max(np.abs(residual))),
        "tap_bias_K": float(np.mean(residual)),
        "runtime_s": float(run_row["runtime_s"]),
    }


def _evaluate_all_cases() -> pd.DataFrame:
    case_inputs = _load_cases()
    benchmark = _load_results()
    case_inputs = case_inputs.set_index("Case")
    rows: list[dict] = []

    for case_id in sorted(case_inputs.index):
        case_row = case_inputs.loc[case_id]
        subset = benchmark[
            (benchmark["case_id"] == case_id)
            & (benchmark["method"] == "scipy-bvp")
            & (benchmark["success"])
            & (benchmark["thermo_model"].isin(["ideal_henry", "epcsaft_ionic"]))
        ]
        for _, run_row in subset.iterrows():
            metric = _tap_and_profile_metrics(case_id, run_row["thermo_model"], case_row, run_row)
            rows.append(metric)

    return pd.DataFrame(rows)


def _choose_additional_cases(metrics: pd.DataFrame) -> list[str]:
    # Conservative inclusion criteria for manuscript figure:
    #  - capture error within ±10%
    #  - profile agreement at taps RMSE <= 9.0 K
    #  - exclude 3C because it is already the existing anchor case
    candidate_mask = (
        (metrics["capture_error_pct"].abs() <= 10.0)
        & (metrics["tap_rmse_K"] <= 9.0)
        & (metrics["case_id"] != "3C")
    )
    candidates = metrics.loc[candidate_mask]
    return sorted(candidates["case_id"].unique().tolist())


def _plot_recommended_profiles(metrics: pd.DataFrame, recommended_cases: list[str]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    case_inputs = _load_cases().set_index("Case")
    benchmark = _load_results()

    cases_for_plot = ["3C"] + recommended_cases
    if not cases_for_plot:
        return

    n = len(cases_for_plot)
    fig, axes = plt.subplots(n, 1, figsize=(8.0, 3.1 * n), sharex=True, sharey=False)
    if n == 1:
        axes = np.array([axes])

    for ax, case_id in zip(axes, cases_for_plot):
        case_row = case_inputs.loc[case_id]
        tap_columns = _tap_columns(case_row)
        tap_positions = np.array([float(col) for col in tap_columns], dtype=float)
        tap_temperatures = case_row[tap_columns].astype(float).to_numpy()

        for thermo_model in ["ideal_henry", "epcsaft_ionic"]:
            profile_path = RUNS / "profiles" / "C_cases_campaign_inputs" / case_id / "scipy-bvp" / thermo_model / "T.csv"
            if not profile_path.exists():
                continue
            profile = pd.read_csv(profile_path)
            style = THERMO_STYLE[thermo_model]
            run_row = benchmark[
                (benchmark["case_id"] == case_id)
                & (benchmark["method"] == "scipy-bvp")
                & (benchmark["thermo_model"] == thermo_model)
                & (benchmark["success"])
            ].iloc[0]

            label_prefix = style["label"]
            ax.plot(
                profile["Position"],
                profile["Tl"],
                color=style["color"],
                linewidth=1.8,
                label=f"{label_prefix} liquid",
            )
            ax.plot(
                profile["Position"],
                profile["Tv"],
                color=style["color"],
                linewidth=1.1,
                linestyle=":",
                alpha=0.75,
                label=f"{label_prefix} vapor",
            )

        ax.scatter(tap_positions, tap_temperatures, s=28, marker="o", color="k", edgecolors="white", label="NCCC taps")
        cap_err = float(
            benchmark.loc[
                (benchmark["case_id"] == case_id)
                & (benchmark["method"] == "scipy-bvp")
                & (benchmark["thermo_model"] == "epcsaft_ionic")
                & (benchmark["success"]),
            ].iloc[0]["capture_error_pct"]
        )
        ax.set_title(f"{case_id} | capture error (ePC-SAFT): {cap_err:+.2f}%")
        ax.set_ylabel("Temperature [K]")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(315, 355)

    axes[-1].set_xlabel("Normalized column position")

    fig.tight_layout()
    out_pdf = FIGURES / "c_case_temperature_profile_overlay_recommended.pdf"
    out_svg = FIGURES / "c_case_temperature_profile_overlay_recommended.svg"
    fig.savefig(out_pdf, dpi=220, bbox_inches="tight")
    fig.savefig(out_svg, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # This legacy diagnostic figure is intentionally not copied into
    # docs/latex/figures. The manuscript-facing C-case overlay is generated
    # from corrected campaign inputs by render_c_case_campaign_temperature_gallery.py
    # and synced through docs/latex/scripts/sync_latex_figures.ps1.


def _write_recommendation_report(metrics: pd.DataFrame, recommendations: list[str]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TABLES / "c_case_profile_tap_metrics.csv", index=False)

    report_path = FINAL / "reports" / "c_case_temperature_profile_recommendations.md"
    best_case_rows = []
    for case_id in sorted(metrics["case_id"].unique()):
        best = metrics[metrics["case_id"] == case_id].sort_values("tap_rmse_K").iloc[0]
        best_case_rows.append(
            {
                "case_id": best["case_id"],
                "best_thermo_model": str(best["thermo_model"]),
                "capture_error_pct": best["capture_error_pct"],
                "tap_rmse_K": best["tap_rmse_K"],
                "tap_mae_K": best["tap_mae_K"],
                "temperature_rmse_K": best["temperature_rmse_K"],
            }
        )

    best_df = pd.DataFrame(best_case_rows)
    report = [
        "# C-Case Temperature Profile Recommendation",
        "",
        "Conservative inclusion criteria used for profile-based figure selection:",
        "- |capture_error_pct| <= 10",
        "- tap-based liquid-temperature RMSE <= 9 K",
        "",
        "Case metrics (best model by tap RMSE):",
        "",
    ]
    report.append(best_df.to_markdown(index=False))
    report.append("")
    report.append("## Recommended additional cases beyond 3C")
    if recommendations:
        report.append("- " + ", ".join(recommendations))
    else:
        report.append("- None that satisfy the strict criteria.")
    report.append("")
    report.append("## Caveats")
    report.append("- The campaign overlay figure remains the paper-facing profile summary for the 1C--7C set.")
    report.append("- Case 7C remains the hardest thermal-shape case, while the capture errors stay within the campaign validation gate.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    metrics = _evaluate_all_cases()
    additional_cases = _choose_additional_cases(metrics)
    _write_recommendation_report(metrics, additional_cases)
    _plot_recommended_profiles(metrics, additional_cases)


if __name__ == "__main__":
    main()
