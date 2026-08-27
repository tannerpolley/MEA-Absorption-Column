from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "analyses/nccc_validation/results/final/tables"
FIGURES = ROOT / "analyses/nccc_validation/results/final/figures"


def main() -> None:
    profile = pd.read_csv(TABLES / "retained_reactive_case3c_profile_comparison.csv")
    sensitivity = pd.read_csv(TABLES / "retained_reactive_case3c_sensitivity.csv")
    FIGURES.mkdir(parents=True, exist_ok=True)
    blue, orange = "#0072B2", "#D55E00"
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(profile.Position, profile.prior_fl_CO2, color=blue, linestyle="--", label="Prior fugacity-only")
    ax.plot(profile.Position, profile.retained_fl_CO2, color=orange, label="Retained reactive")
    ax.set_yscale("log")
    ax.set_xlabel("Normalized column position")
    ax.set_ylabel(r"Liquid CO$_2$ fugacity (Pa)")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(profile.Position, profile.prior_E, color=blue, linestyle="--", label="Prior fugacity-only")
    ax.plot(profile.Position, profile.retained_E, color=orange, label="Retained reactive")
    ax.set_xlabel("Normalized column position")
    ax.set_ylabel("Enhancement factor, E")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    blend = sensitivity.query("family == 'fugacity_blend'").sort_values("setting")
    ax.plot(blend.setting, blend.capture_pct, color=orange, marker="o", label="Reactive column")
    parameter = sensitivity.query("family == 'parameter_structure'").iloc[0]
    ax.scatter([1.0], [parameter.capture_pct], marker="s", color="#009E73", label="No CO$_2$--water T adjustment")
    ax.axhline(89.5, color="black", linestyle=":", label="Observed")
    ax.set_xlabel("ePC-SAFT fugacity blend")
    ax.set_ylabel("CO$_2$ capture (%)")
    ax.set_xlim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    eta = sensitivity.query("family == 'eta_psi'").sort_values("setting")
    ax.plot(eta.setting, eta.capture_pct, color=orange, marker="o", label="Reactive column")
    ax.axhline(89.5, color="black", linestyle=":", label="Observed")
    ax.set_xlabel(r"Transfer multiplier, $\eta_\Psi$")
    ax.set_ylabel("CO$_2$ capture (%)")
    ax.legend(frameon=False)

    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat, strict=True):
        ax.text(0.01, 0.98, label, transform=ax.transAxes, va="top", fontweight="bold")
        ax.grid(alpha=0.2)

    stem = FIGURES / "retained_reactive_case3c_diagnosis"
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
