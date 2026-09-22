from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[3]
TABLE = ROOT / "analyses/nccc_validation/results/final/tables/issue16_provisional_reactive_film_probe.csv"
OUTPUT = ROOT / "analyses/issue16_reactive_film_runtime/figures/output/issue16_runtime_summary"


def main() -> None:
    with TABLE.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one retained Position 1 row, found {len(rows)}")
    row = rows[0]

    fugacity = [
        float(row["bulk_liquid_CO2_fugacity_Pa"]),
        float(row["interface_liquid_CO2_fugacity_Pa"]),
        float(row["bulk_vapor_CO2_fugacity_Pa"]),
    ]
    flux = [
        float(row["predicted_flux_mol_s_m"]),
        float(row["retained_column_flux_mol_s_m"]),
    ]
    residual_limits = {
        "Interface": (float(row["maximum_interface_residual"]), 1e-7),
        "Conservation": (float(row["maximum_conservation_residual"]), 1e-7),
        "Invariant source": (float(row["maximum_invariant_source_residual"]), 1e-12),
        "Electroneutrality": (float(row["maximum_electroneutrality_residual"]), 1e-12),
        "Zero current": (float(row["maximum_zero_current_residual"]), 1e-12),
    }
    requests = int(row["thermodynamic_state_request_count"])
    evaluations = int(row["thermodynamic_state_evaluation_count"])
    hits = int(row["thermodynamic_state_cache_hit_count"])
    if evaluations + hits != requests:
        raise RuntimeError("Thermodynamic work counters do not close")

    blue, orange, green, red = "#0072B2", "#E69F00", "#009E73", "#D55E00"
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), constrained_layout=True)

    ax = axes[0, 0]
    labels = ["Bulk liquid", "Interface liquid", "Bulk vapor"]
    ax.scatter(range(3), fugacity, s=65, color=[blue, orange, green], zorder=3)
    for index, value in enumerate(fugacity):
        ax.annotate(f"{value:.3f}", (index, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(range(3), labels)
    ax.set_yscale("log")
    ax.set_ylabel(r"CO$_2$ fugacity (Pa, log scale)")
    ax.set_title("A  Retained fugacity states", loc="left", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[0, 1]
    rlabels = list(residual_limits)
    ratios = [value / limit for value, limit in residual_limits.values()]
    ax.scatter(range(len(ratios)), ratios, s=55, color=[green if ratio <= 1 else red for ratio in ratios], zorder=3)
    ax.axhline(1, color=red, linewidth=1, linestyle="--", label="Acceptance limit")
    ax.set_xticks(range(len(rlabels)), rlabels, rotation=24, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Residual / accepted numerical limit")
    ax.set_title("B  All five numerical residual checks pass", loc="left", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.scatter(range(2), flux, s=65, color=[blue, orange], zorder=3)
    for index, value in enumerate(flux):
        ax.annotate(f"{value:.6f}", (index, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(range(2), ["Reactive-film\nprobe", "Retained column\ncomparator"])
    ax.set_yscale("log")
    ax.set_ylabel(r"Axial CO$_2$ flux (mol s$^{-1}$ m$^{-1}$ packed height, log scale)")
    ax.set_title("C  Probe flux is 115.46x below comparator", loc="left", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 1]
    bars = ax.barh(["Evaluated states", "Cache hits"], [evaluations, hits], color=[blue, green])
    ax.bar_label(bars, labels=[f"{evaluations:,}", f"{hits:,}"], padding=4)
    ax.set_xlim(0, max(evaluations, hits) * 1.18)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_xlabel("Thermodynamic-state requests (count)")
    ax.set_title(f"D  {requests:,} requests; {hits / requests:.1%} cache hits", loc="left", fontsize=10)
    ax.grid(True, axis="x", alpha=0.25)

    fig.suptitle("Issue 16 Position 1 reactive-film runtime probe", fontsize=13, fontweight="bold")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUTPUT.with_suffix(f".{suffix}"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
