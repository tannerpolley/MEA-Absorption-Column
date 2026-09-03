from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TABLE = ROOT / "analyses/reactive_film_evidence/results/final/tables/column_film_capture_comparison.csv"
FIGURE = ROOT / "analyses/reactive_film_evidence/results/final/figures/column_film_capture_comparison"


def main() -> None:
    data = pd.read_csv(TABLE)
    required = {"case_id", "observed_capture_pct", "baseline_capture_pct", "film_capture_pct"}
    if not required.issubset(data.columns) or len(data) != 8:
        raise AssertionError("retained comparison must contain the eight declared cases")
    x = range(len(data))
    fig, (capture, change) = plt.subplots(2, 1, figsize=(8.4, 7.0), sharex=True)
    capture.scatter(x, data["observed_capture_pct"], marker="x", s=55, color="black", label="Observed")
    capture.scatter(x, data["baseline_capture_pct"], marker="o", s=45, facecolors="none", edgecolors="#4477AA", label="Enhancement factor")
    converged = data["outer_iteration_converged"].astype(bool)
    capture.scatter(data.index[converged], data.loc[converged, "film_capture_pct"], marker="s", s=42, color="#CC6677", label="Reactive film")
    capture.scatter(data.index[~converged], data.loc[~converged, "film_capture_pct"], marker="s", s=42, facecolors="none", edgecolors="#CC6677", label="Film (outer limit)")
    capture.set_ylabel("CO$_2$ capture (%)")
    capture.legend(ncol=3, frameon=False, loc="lower left")
    capture.grid(axis="y", alpha=0.25)
    delta = data["film_capture_pct"] - data["baseline_capture_pct"]
    change.axhline(0.0, color="black", linewidth=0.8)
    change.scatter(data.index[converged], delta[converged], marker="s", s=42, color="#CC6677")
    change.scatter(data.index[~converged], delta[~converged], marker="s", s=42, facecolors="none", edgecolors="#CC6677")
    change.set_ylabel("Film − enhancement\n(percentage points)")
    change.set_xticks(list(x), data["case_id"])
    change.set_xlabel("NCCC one-bed case")
    change.grid(axis="y", alpha=0.25)
    fig.suptitle("Reactive-film closure changes the predicted capture campaign", y=0.995)
    fig.text(0.5, 0.955, "Fixed-input candidate; open squares did not meet the outer-field stopping rule", ha="center", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE.with_suffix(".png"), dpi=240)
    fig.savefig(FIGURE.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
