from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TABLE = ROOT / "analyses/nccc_validation/results/final/tables/issue17_fugacity_only_enhancement_formulations.csv"
FIGURES = ROOT / "analyses/nccc_validation/results/final/figures"
STYLES = {
    "EF-GF-IMPLICIT": ("Gaspar implicit", "#0072B2", "-", "o"),
    "EF-AOP-78-PUBLISHED-MEA": ("Published Eq. 78", "#D55E00", "--", "s"),
    "EF-AOP-73-CORRECTED-MEA": ("Corrected Eq. 73 MEA", "#009E73", "-.", "^"),
    "EF-CURRENT": ("Current expression", "#CC79A7", ":", "D"),
}


def _series(table: pd.DataFrame):
    for formulation, (label, color, linestyle, marker) in STYLES.items():
        yield table.loc[table.formulation.eq(formulation)].sort_values("height_m"), label, color, linestyle, marker


def main() -> None:
    table = pd.read_csv(TABLE)
    if len(table) != 84:
        raise RuntimeError(f"Expected 84 retained rows, found {len(table)}")
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.17, top=0.97)
    for group, label, color, linestyle, marker in _series(table):
        ax.plot(
            group.height_m,
            group.E,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=4,
            linewidth=1.8,
            markersize=4,
        )
    ax.axhline(1.0, color="0.35", linewidth=0.9, label="Nonreactive limit")
    ax.set_yscale("log")
    ax.set_xlabel("Packed height (m)", labelpad=8)
    ax.set_ylabel("Enhancement factor, E")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(FIGURES / "issue17_axial_enhancement.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.17, top=0.97)
    for group, label, color, linestyle, marker in _series(table):
        ax.plot(
            group.height_m,
            group.predicted_flux_mol_s_m,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=4,
            linewidth=1.8,
            markersize=4,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Packed height (m)", labelpad=8)
    ax.set_ylabel(r"CO$_2$ flux (mol s$^{-1}$ m$^{-1}$ packed height)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(FIGURES / "issue17_axial_flux.pdf")
    plt.close(fig)

    implicit = table.loc[table.formulation.eq("EF-GF-IMPLICIT"), ["Position", "E"]].rename(
        columns={"E": "implicit_E"}
    )
    parity = table.merge(implicit, on="Position", validate="many_to_one")
    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    for formulation, (label, color, _, marker) in STYLES.items():
        if formulation == "EF-GF-IMPLICIT":
            continue
        group = parity.loc[parity.formulation.eq(formulation)]
        ax.scatter(group.implicit_E, group.E, label=label, color=color, marker=marker, s=28)
    limits = np.array([min(parity.implicit_E.min(), parity.E.min()), max(parity.implicit_E.max(), parity.E.max())])
    ax.plot(limits, limits, color="0.3", linewidth=1.0, label="Parity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Gaspar implicit enhancement, E")
    ax.set_ylabel("Compared enhancement, E")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGURES / "issue17_parity_to_gaspar_implicit.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
