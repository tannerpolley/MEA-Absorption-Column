from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TABLE = ROOT / "analyses/nccc_validation/results/final/tables/retained_reactive_case3c_enhancement_plot.csv"
COMPARISON = ROOT / "analyses/nccc_validation/results/final/tables/retained_reactive_case3c_enhancement_film_comparison.csv"
OUTPUT = ROOT / "analyses/nccc_validation/results/final/figures/retained_reactive_case3c_enhancement_consistency"

STYLES = {
    "idaes_explicit_luo_current": ("Current explicit, Luo", "#333333", "-"),
    "idaes_explicit_luo_uncorrected": ("Explicit, Luo, no divisor", "#0072B2", "--"),
    "idaes_explicit_putta_uncorrected": ("Explicit, Putta, no divisor", "#E69F00", "-."),
    "gaspar_implicit_luo_legacy_henry": ("Gaspar implicit, legacy Henry", "#CC79A7", ":"),
    "gaspar_implicit_luo_epcsaft_local": ("Gaspar implicit, local ePC-SAFT", "#009E73", "-"),
}


def main() -> None:
    table = pd.read_csv(TABLE)
    comparison = pd.read_csv(COMPARISON)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True, constrained_layout=True)
    for formulation, group in table.groupby("formulation", sort=False):
        label, color, linestyle = STYLES[formulation]
        axes[0].plot(group.height_m, group.E, label=label, color=color, linestyle=linestyle, linewidth=1.8)
        axes[1].plot(
            group.height_m,
            group.flux_ratio_to_current,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
        )
    film = comparison.loc[comparison.method_class.eq("mechanistic_reaction_diffusion")].iloc[0]
    axes[1].scatter(
        film.height_m,
        film.flux_ratio_to_current,
        marker="D",
        s=42,
        color="#D55E00",
        label="Nonlinear reactive film (one state)",
        zorder=5,
    )
    axes[0].set_ylabel("Enhancement factor, E")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].axhline(1.0, color="0.55", linewidth=0.9)
    axes[1].set_xlabel("Packing height (m)")
    axes[1].set_ylabel("CO$_2$ flux / current flux")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(OUTPUT.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
