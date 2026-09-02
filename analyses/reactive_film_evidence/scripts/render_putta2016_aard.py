from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
PLOT_DATA = ROOT / "analyses/reactive_film_evidence/results/final/tables/putta2016_table4_aard_plot.csv"
FIGURE = ROOT / "analyses/reactive_film_evidence/results/final/figures/putta2016_table4_aard"
MODELS = [
    "Aboudheir2003 concentration",
    "Luo2015 concentration",
    "Luo2015 activity",
    "Present concentration",
    "Present activity",
]
DATA_GROUPS = ["SDC Luo", "WWC Luo", "WWC Puxty", "laminar jet Aboudheir"]


def main() -> None:
    with PLOT_DATA.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 20 or {row["model"] for row in rows} != set(MODELS):
        raise AssertionError("plot CSV must retain all 20 Table 4 cells")
    if any(row["evidence_scope"] != "aggregate_model_comparison_not_row_level_validation" for row in rows):
        raise AssertionError("aggregate evidence must not be relabeled as row-level validation")
    by_key = {(row["model"], row["apparatus_or_dataset"]): float(row["AARD_percent"]) for row in rows}
    matrix = [[by_key[(model, data_group)] for data_group in DATA_GROUPS] for model in MODELS]

    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    fig.subplots_adjust(left=0.29, right=0.88, bottom=0.30, top=0.90)
    image = ax.imshow(matrix, cmap="viridis_r", vmin=0, vmax=40, aspect="auto")
    ax.set_xticks(range(len(DATA_GROUPS)), DATA_GROUPS, rotation=20, ha="right")
    ax.set_yticks(range(len(MODELS)), MODELS)
    ax.set_title("Putta2016 aggregate errors do not establish row-level film validation")
    ax.set_xlabel("Apparatus/data grouping reported in Table 4")
    ax.set_ylabel("Rate-model basis")
    for y, values in enumerate(matrix):
        for x, value in enumerate(values):
            ax.text(x, y, f"{value:.1f}%", ha="center", va="center", color="white" if value > 22 else "black")
    colorbar = fig.colorbar(image, ax=ax, label="Aggregate AARD (%) — lower is better")
    colorbar.ax.tick_params(labelsize=9)
    fig.text(
        0.5,
        0.025,
        "Source: Putta et al. (2016), printed p. 349, Table 4. Reported groups only; raw flux rows and uncertainties are not retained.",
        ha="center",
        fontsize=8,
    )
    fig.savefig(FIGURE.with_suffix(".png"), dpi=240)
    fig.savefig(FIGURE.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
