from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from generate_accuracy_credibility_artifacts import main as generate_accuracy_credibility_artifacts


ANALYSIS = Path(__file__).resolve().parents[1]
TABLES = ANALYSIS / "results" / "final" / "tables"
FIGURES = ANALYSIS / "results" / "final" / "figures"


def _label_thermo(value: str) -> str:
    if value == "ideal_henry":
        return "Henry"
    if value == "epcsaft_neutral":
        return "PC-SAFT"
    return value


def _strip_svg_trailing_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    c_cases = _load_c_case_results()
    c_cases["thermo_label"] = c_cases["thermo_model"].map(_label_thermo)
    c_cases["abs_capture_error_pct"] = c_cases["capture_error_pct"].abs()
    c_cases.to_csv(TABLES / "plot_c_case_thermo_benchmark.csv", index=False)

    summary = (
        c_cases.groupby("thermo_label", sort=False)
        .agg(
            capture_mae_pct=("abs_capture_error_pct", "mean"),
            temperature_rmse_K=("temperature_rmse_K", "mean"),
            runtime_median_s=("runtime_s", "median"),
            success_count=("success", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(TABLES / "plot_c_case_thermo_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), constrained_layout=True)
    colors = {"Henry": "#2f5d8c", "PC-SAFT": "#8a4b2b"}

    for label, group in c_cases.groupby("thermo_label", sort=False):
        axes[0].plot(
            group["case_id"],
            group["capture_error_pct"],
            marker="o",
            linewidth=1.8,
            label=label,
            color=colors.get(label),
        )
    axes[0].axhline(0, color="0.35", linewidth=0.8)
    axes[0].set_ylabel("Capture error (%)")
    axes[0].set_xlabel("NCCC one-bed C case")
    axes[0].set_title("Capture validation", pad=8)
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(
        summary["thermo_label"],
        summary["runtime_median_s"],
        color=[colors.get(label) for label in summary["thermo_label"]],
        width=0.55,
    )
    axes[1].set_ylabel("Median runtime (s)")
    axes[1].set_title("Runtime", pad=8)
    axes[1].tick_params(axis="x", rotation=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    c_case_svg = FIGURES / "c_case_thermo_benchmark.svg"
    fig.savefig(c_case_svg, bbox_inches="tight")
    fig.savefig(FIGURES / "c_case_thermo_benchmark.pdf", bbox_inches="tight")
    _strip_svg_trailing_whitespace(c_case_svg)
    plt.close(fig)

    generate_accuracy_credibility_artifacts()


def _load_c_case_results() -> pd.DataFrame:
    campaign_metrics = TABLES / "c_case_campaign_temperature_overlay_metrics.csv"
    if campaign_metrics.exists():
        c_cases = pd.read_csv(campaign_metrics)
        c_cases["success"] = True
        c_cases["source_dataset"] = "campaign"
        if "plot_png" in c_cases.columns:
            c_cases["plot_png"] = c_cases["case_id"].map(
                lambda case_id: (
                    "analyses/nccc_validation/results/final/figures/"
                    f"c_case_campaign_temperature_overlays/{case_id}_temperature_overlay.png"
                )
            )
        return c_cases

    c_cases = pd.read_csv(TABLES / "raw_c_case_thermo_benchmark.csv")
    c_cases["source_dataset"] = "legacy"
    return c_cases


if __name__ == "__main__":
    main()
