from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TABLE = ROOT / "analyses/bvp_derivative_trials/results/final/tables/issue19_scipy_bvp_candidate_rows.csv"
FIGURE = ROOT / "analyses/bvp_derivative_trials/results/final/figures/issue19_scipy_bvp_summary.png"


def main() -> None:
    rows = pd.read_csv(TABLE)
    if len(rows) != 12:
        raise RuntimeError(f"Expected 12 retained Issue 19 rows, found {len(rows)}")

    colors = {"epcsaft_ionic": "#0072B2", "ideal_henry": "#E69F00"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)

    initialization = rows.loc[rows.setting_id.isin(["baseline", "init_25", "init_60"])].sort_values("capture_guess_pct")
    ax = axes[0, 0]
    ax.scatter(initialization.capture_guess_pct, initialization.capture_pct, color="#0072B2", s=52, zorder=3)
    for row in initialization.itertuples():
        ax.annotate(row.initialization_id.replace("capture_", "") + "% guess", (row.capture_guess_pct, row.capture_pct), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.axhspan(initialization.capture_pct.min(), initialization.capture_pct.max(), color="#56B4E9", alpha=0.18)
    ax.set_xlabel("Initial capture guess (%)")
    ax.set_ylabel("Calculated capture (%) — tight scale")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
    ax.set_title("A  Three initializations form one scalar capture cluster", loc="left", fontsize=10)
    ax.grid(True, alpha=0.25)

    setting_order = ["mesh_11", "baseline", "mesh_41", "tol_1", "tol_0p25"]
    labels = ["mesh\n11", "mesh\n21", "mesh\n41", "tol\n1.0", "tol\n0.25"]
    settings = rows.set_index("setting_id").loc[setting_order].reset_index()
    ax = axes[0, 1]
    for index, row in settings.iterrows():
        color = "#D55E00" if not row.certificate_pass else "#009E73"
        marker = "X" if not row.certificate_pass else "o"
        ax.scatter(index, row.capture_delta_from_reference_pct, color=color, marker=marker, s=58, zorder=3)
        offset = 7 if row.capture_delta_from_reference_pct < -0.04 else -13
        ax.annotate(f"r={row.dense_ode_residual_max:.3g}", (index, row.capture_delta_from_reference_pct), xytext=(0, offset), textcoords="offset points", ha="center", fontsize=7)
    ax.axhline(0, color="0.35", linewidth=0.9)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Capture change from baseline (percentage points)")
    ax.set_ylim(-0.095, 0.008)
    ax.set_title("B  Capture shifts <0.09 pp; tol=1 fails certificate", loc="left", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)

    evaluated = rows.loc[rows.solver_rhs_node_evaluations.notna()]
    ax = axes[1, 0]
    for model, group in evaluated.groupby("thermo_model"):
        ax.scatter(group.solver_rhs_node_evaluations, group.runtime_s, color=colors[model], label=model, s=42, alpha=0.9)
    for setting in ("baseline", "mesh_41", "tol_0p25", "nccc_henry"):
        row = rows.loc[rows.setting_id.eq(setting)].iloc[0]
        ax.annotate(setting, (row.solver_rhs_node_evaluations, row.runtime_s), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("RHS node evaluations (count)")
    ax.set_ylabel("Cold isolated subprocess runtime (s)")
    ax.set_title("C  Evaluated rows retain runtime and solver-work counters", loc="left", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    outcome_order = ["evaluated", "certificate_failure", "campaign_timeout"]
    outcome_labels = ["Evaluated", "Certificate failure", "Campaign timeout"]
    counts = rows.outcome.value_counts().reindex(outcome_order, fill_value=0)
    ax = axes[1, 1]
    bars = ax.barh(outcome_labels, counts, color=["#009E73", "#D55E00", "#CC79A7"])
    ax.bar_label(bars, padding=3)
    ax.set_xlim(0, max(counts) + 1)
    ax.set_xlabel("Retained attempts (count)")
    ax.set_title("D  Nine evaluated; three remain bounded non-results", loc="left", fontsize=10)
    ax.grid(True, axis="x", alpha=0.25)

    fig.suptitle("Issue 19 retained SciPy BVP numerical evidence — fixed chemistry", fontsize=13, fontweight="bold")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE,
        dpi=220,
        metadata={"Description": "Summary derived only from retained Issue 19 candidate rows."},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
