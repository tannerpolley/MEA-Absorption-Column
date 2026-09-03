from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "analyses/reactive_film_evidence/results/final/tables"
FIGURES = ROOT / "analyses/reactive_film_evidence/results/final/figures"


def _read(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise AssertionError(f"empty plotted data: {name}")
    return rows


def _panel_a(ax, rows: list[dict[str, str]], trace: list[dict[str, str]]) -> None:
    if len(rows) != 4 or any(row["claim_scope"] != "thermodynamic_force_isolation_non_predictive" for row in rows):
        raise AssertionError("chemical-potential panel must retain four non-predictive rows")
    if len(trace) != 2:
        raise AssertionError("chemical-potential trace must retain integrated and local definitions")
    by_id = {row["definition_id"]: row for row in trace}
    integrated = float(by_id["integrated_boundary_response"]["relative_delta_percent"])
    local = float(by_id["local_projected_CO2_same_gradient"]["relative_delta_percent"])
    values = [integrated, local]
    positions = [0, 1]
    ax.barh(positions, values, color=["#0072B2", "#D55E00"], height=0.58)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xscale("symlog", linthresh=1.0e-3)
    ax.set_yticks(positions, ["integrated CO2\nboundary response", "local projected CO2\nsame-gradient diagnostic"])
    ax.set_xlabel("ePC-SAFT vs ideal-log reference\nrelative delta [%]")
    ax.set_title("(a) Thermodynamic-factor diagnostics")
    for y, value in zip(positions, values):
        ax.annotate(f"{value:.3g}%", (value, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)


def _panel_b(ax, diffusion_rows: list[dict[str, str]], viscosity_rows: list[dict[str, str]]) -> None:
    if len(diffusion_rows) != 9:
        raise AssertionError("diffusion-anchor panel must retain nine exact/assumption rows")
    if len(viscosity_rows) != 25:
        raise AssertionError("viscosity panel must retain 25 exact Ramezani S4 rows")
    for loading in sorted({row["co2_loading_mol_per_mol_mea"] for row in viscosity_rows}, key=float):
        series = sorted(
            (row for row in viscosity_rows if row["co2_loading_mol_per_mol_mea"] == loading),
            key=lambda row: float(row["sugar_wt_pct"]),
        )
        ax.plot(
            [float(row["sugar_wt_pct"]) for row in series],
            [float(row["normalized_inverse_viscosity"]) for row in series],
            "o-",
            linewidth=1.3,
            markersize=3.5,
            label=f"loading {loading}",
        )
    ax.axhline(1.0, color="0.35", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Sugar perturbation [wt%]")
    ax.set_ylabel("Viscosity proxy: μ(no sugar) / μ(sugar)")
    ax.set_title("(b) Fixed-chemistry mobility/viscosity perturbation")
    ax.legend(fontsize=6, ncol=2, loc="upper left")


def _panel_c(ax, rows: list[dict[str, str]]) -> None:
    if len(rows) != 50 or sum(row["kg_prime_reported"] == "false" for row in rows) != 1:
        raise AssertionError("Dugas Table 1 panel must retain 50 rows and one unreported kg-prime cell")
    plotted = [row for row in rows if row["kg_prime_reported"] == "true"]
    colors = {"7": "#0072B2", "9": "#009E73", "11": "#E69F00", "13": "#D55E00"}
    markers = {"40": "o", "60": "s", "80": "^", "100": "D"}
    for row in plotted:
        ax.scatter(
            float(row["co2_loading_mol_per_mol_mea"]),
            float(row["kg_prime_mol_per_Pa_m2"]),
            color=colors[row["mea_molal"]],
            marker=markers[row["T_C"]],
            s=24,
        )
    for molal, color in colors.items():
        ax.scatter([], [], color=color, marker="o", label=f"{molal} molal")
    for temperature, marker in markers.items():
        ax.scatter([], [], color="0.35", marker=marker, label=f"{temperature} deg C")
    values = [float(row["kg_prime_mol_per_Pa_m2"]) for row in plotted]
    ratio = max(values) / min(values)
    ax.set_yscale("log")
    ax.set_xlabel("CO2 loading [mol CO2 mol$^{-1}$ MEA]")
    ax.set_ylabel("Reactive $k_G'$ [mol Pa$^{-1}$ m$^{-2}$ s$^{-1}$]")
    ax.set_title("(c) Dugas Table 1 dimensional film evidence")
    ax.text(
        0.03,
        0.04,
        f"Exact local rows; max/min = {ratio:.1f}x (~30x)\n"
        "kg-prime unreported for 9 molal, 40 deg C, loading 0.231.",
        transform=ax.transAxes,
        fontsize=7,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    ax.legend(fontsize=6, ncol=2, loc="upper right")


def main() -> None:
    panel_a = _read("chemical_potential_isolation_plot.csv")
    trace = _read("chemical_potential_definition_trace.csv")
    diffusion = _read("diffusion_anchor_plot.csv")
    viscosity = _read("ramezani_viscosity_sensitivity_plot.csv")
    panel_c = _read("dugas2011_table1_mea_plot.csv")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), constrained_layout=True)
    _panel_a(axes[0], panel_a, trace)
    _panel_b(axes[1], diffusion, viscosity)
    _panel_c(axes[2], panel_c)
    fig.suptitle("Reactive-film evidence synthesis — non-predictive and non-validation", fontsize=15)
    fig.savefig(FIGURES / "reactive_film_evidence_panels.png", dpi=220)
    fig.savefig(FIGURES / "reactive_film_evidence_panels.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print("reactive-film evidence figures rendered")


if __name__ == "__main__":
    main()
