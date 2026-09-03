from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TABLE = ROOT / "analyses/reactive_film_evidence/results/final/tables/column_film_capture_comparison.csv"
PROFILES = ROOT / "analyses/reactive_film_evidence/results/final/tables/column_film_axial_profiles.csv"
CASE_INPUTS = ROOT / "src/mea_absorption_column/data/C_cases_campaign_inputs.csv"
FIGURE = ROOT / "analyses/reactive_film_evidence/results/final/figures/column_film_capture_comparison"
TEMPERATURE_FIGURE = ROOT / "analyses/reactive_film_evidence/results/final/figures/column_film_temperature_overlay"
CASE_3C_FIGURE = ROOT / "analyses/reactive_film_evidence/results/final/figures/column_film_3c_temperature"
TEMPERATURE_METRICS = ROOT / "analyses/reactive_film_evidence/results/final/tables/column_film_temperature_metrics.csv"


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
    _render_temperature_figures(data)


def _render_temperature_figures(comparison: pd.DataFrame) -> None:
    profiles = pd.read_csv(PROFILES)
    case_inputs = pd.read_csv(CASE_INPUTS).set_index("Case")
    cases = tuple(f"{index}C" for index in range(1, 7))
    metrics = []
    fig, axes = plt.subplots(3, 2, figsize=(10.2, 10.0), sharex=True, sharey=True)
    for ax, case_id in zip(axes.flat, cases):
        case_profile = profiles[profiles["case_id"] == case_id].sort_values("position")
        taps = case_inputs.loc[case_id, ["0", "0.2", "0.4", "0.6", "0.8"]].astype(float)
        positions = taps.index.astype(float).to_numpy()
        modeled = np.interp(positions, case_profile["position"], case_profile["liquid_temperature_K"])
        residual = modeled - taps.to_numpy()
        row = comparison.loc[comparison["case_id"] == case_id].iloc[0]
        metrics.append({
            "case_id": case_id,
            "film_capture_pct": row["film_capture_pct"],
            "film_capture_error_pp": row["film_capture_error_pp"],
            "tap_rmse_K": float(np.sqrt(np.mean(residual**2))),
            "tap_bias_K": float(np.mean(residual)),
            "tap_max_abs_K": float(np.max(np.abs(residual))),
            "outer_iteration_converged": bool(row["outer_iteration_converged"]),
        })
        status = "" if row["outer_iteration_converged"] else "; outer field not converged"
        ax.plot(case_profile["position"], case_profile["liquid_temperature_K"], color="#CC6677", linewidth=2.0,
                label=f"film: {row['film_capture_pct']:.1f}%; capture error {row['film_capture_error_pp']:+.1f} pp{status}")
        ax.scatter(positions, taps, marker="x", s=38, color="black", label="NCCC liquid taps")
        ax.set_title(f"{case_id} reactive-film temperature profile")
        ax.grid(alpha=0.24)
        ax.legend(frameon=False, fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Temperature (K)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Normalized column position")
    fig.suptitle("Reactive-film liquid-temperature profiles for the 2017 C cases", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(TEMPERATURE_FIGURE.with_suffix(".png"), dpi=240)
    fig.savefig(TEMPERATURE_FIGURE.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(TEMPERATURE_METRICS, index=False)
    case_id = "3C"
    case_profile = profiles[profiles["case_id"] == case_id].sort_values("position")
    taps = case_inputs.loc[case_id, ["0", "0.2", "0.4", "0.6", "0.8"]].astype(float)
    row = metrics_frame.loc[metrics_frame["case_id"] == case_id].iloc[0]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(case_profile["position"], case_profile["liquid_temperature_K"], color="#CC6677", linewidth=2.15,
            label=f"film: {row['film_capture_pct']:.1f}%; capture error {row['film_capture_error_pp']:+.1f} pp")
    ax.scatter(taps.index.astype(float), taps, marker="x", s=48, color="black", label="NCCC liquid taps")
    ax.set(title="3C reactive-film temperature profile", xlabel="Normalized column position", ylabel="Temperature (K)")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(CASE_3C_FIGURE.with_suffix(".png"), dpi=240)
    fig.savefig(CASE_3C_FIGURE.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
