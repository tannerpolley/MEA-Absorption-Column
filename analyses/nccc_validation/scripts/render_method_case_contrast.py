from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TABLE = ROOT / "analyses" / "nccc_validation" / "results" / "final" / "tables" / "method_case_contrast.csv"
ANALYSIS_FIGURE_DIR = ROOT / "analyses" / "nccc_validation" / "results" / "final" / "figures"
LATEX_FIGURE_DIR = ROOT / "docs" / "latex" / "figures"


def _method_colors(method: str) -> str:
    return {
        "Shooting": "#5B8FF9",
        "SciPy BVP": "#2F9E44",
        "Finite difference": "#D97706",
    }.get(method, "#6B7280")


def render() -> None:
    df = pd.read_csv(TABLE)
    scenarios = list(dict.fromkeys(df["scenario"]))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, scenario in zip(axes, scenarios):
        sub = df[df["scenario"] == scenario].reset_index(drop=True)
        labels = list(sub["method"])
        colors = [_method_colors(method) if ok else "#B8B8B8" for method, ok in zip(sub["method"], sub["success"])]
        bars = ax.bar(labels, sub["runtime_s"], color=colors, edgecolor="#333333", linewidth=0.7)
        for bar, row in zip(bars, sub.to_dict("records")):
            runtime = float(row["runtime_s"])
            status = "ok" if bool(row["success"]) else "failed"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                runtime + max(1.0, 0.03 * runtime),
                f"{runtime:.1f}s\n{status}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(scenario, fontsize=10)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=24)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Runtime (s)")
    fig.suptitle("Solver behavior depends on case conditioning", fontsize=11)
    fig.text(
        0.5,
        0.01,
        "SRP-LG7 is a favorable one-bed method case; NCCC Case 3C includes measured temperature taps and a stronger thermal pinch.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    ANALYSIS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for target in (
        ANALYSIS_FIGURE_DIR / "method_case_solver_contrast.svg",
        ANALYSIS_FIGURE_DIR / "method_case_solver_contrast.pdf",
        LATEX_FIGURE_DIR / "method-case-solver-contrast.pdf",
    ):
        fig.savefig(target, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    render()
