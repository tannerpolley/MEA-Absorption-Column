from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUT = Path(__file__).resolve().parent / "out"


def _label_thermo(value: str) -> str:
    if value == "ideal_henry":
        return "Henry"
    if value == "epcsaft_neutral":
        return "neutral ePC-SAFT"
    return value


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c_cases = pd.read_csv(OUT / "raw_c_case_thermo_benchmark.csv")
    c_cases["thermo_label"] = c_cases["thermo_model"].map(_label_thermo)
    c_cases["abs_capture_error_pct"] = c_cases["capture_error_pct"].abs()
    c_cases.to_csv(OUT / "plot_c_case_thermo_benchmark.csv", index=False)

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
    summary.to_csv(OUT / "plot_c_case_thermo_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    colors = {"Henry": "#2f5d8c", "neutral ePC-SAFT": "#8a4b2b"}

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
    axes[0].set_title("Capture validation")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(
        summary["thermo_label"],
        summary["runtime_median_s"],
        color=[colors.get(label) for label in summary["thermo_label"]],
        width=0.55,
    )
    axes[1].set_ylabel("Median runtime (s)")
    axes[1].set_title("Runtime")
    axes[1].tick_params(axis="x", rotation=15)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.savefig(OUT / "c_case_thermo_benchmark.svg", bbox_inches="tight")
    fig.savefig(OUT / "c_case_thermo_benchmark.pdf", bbox_inches="tight")
    plt.close(fig)

    staged = pd.read_csv(OUT / "raw_staged_kcase_benchmark.csv")
    staged["abs_capture_error_pct"] = staged["capture_error_pct"].abs()
    staged.to_csv(OUT / "plot_staged_kcase_benchmark.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.2, 3.0), constrained_layout=True)
    ax.bar(staged["case_id"], staged["capture_error_pct"], color="#4f6f52", width=0.55)
    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.set_ylabel("Capture error (%)")
    ax.set_xlabel("Intercooled NCCC case")
    ax.set_title("Verified staged-bed Henry runs")
    fig.savefig(OUT / "staged_kcase_capture_error.svg", bbox_inches="tight")
    fig.savefig(OUT / "staged_kcase_capture_error.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
