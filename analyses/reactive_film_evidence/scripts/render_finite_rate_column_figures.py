"""Render the checkpoint 4 comparison from certified column records."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
CLOSURES = (
    ("enhancement", "Enhancement factor", "#4477AA", "o"),
    ("manifold", "Equilibrium manifold", "#AA3377", "s"),
    ("n2o", "Finite rate · N₂O analogy", "#007D65", "^"),
    ("epcsaft", "Finite rate · ePC-SAFT", "#A34A00", "D"),
)
RESISTANCES = (
    ("gas_resistance", "Gas film", "#4477AA", "o"),
    ("kinetic_resistance", "Kinetic", "#AA3377", "s"),
    ("manifold_resistance", "Manifold", "#007D65", "^"),
)


def render(output):
    cases = json.loads((output / "column_comparison.json").read_text())["cases"]
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    fig, panels = plt.subplots(2, 2, figsize=(7.2, 6.7), sharex=True, sharey=True, layout="constrained")
    for ax, (key, label, color, marker) in zip(panels.flat, CLOSURES, strict=True):
        ax.axline((0, 0), slope=1, ls="--", color="0.45", lw=0.9, label="Parity")
        for case in cases.values():
            run = case["closures"][key]
            if "capture_percent" in run:
                ax.errorbar(case["measured_capture_percent"], run["capture_percent"], xerr=case["capture_sd_pp"],
                            fmt=marker, color=color, markerfacecolor="white", capsize=2.5, ms=5.5, lw=1)
        ax.set(title=label, xlim=(56, 102), ylim=(56, 102), xticks=range(60, 101, 10), aspect="equal")
        ax.grid(alpha=0.2)
    fig.supxlabel("Measured CO₂ capture (%)")
    fig.supylabel("Predicted CO₂ capture (%)")
    fig.savefig(figures / "capture_parity.png", dpi=240)
    plt.close(fig)

    fig, panels = plt.subplots(1, 2, figsize=(8, 3.7), sharey=True, layout="constrained")
    for ax, (key, label, _, _) in zip(panels, CLOSURES[2:], strict=True):
        path = ROOT / cases["3C"]["closures"][key]["path"]
        run = json.loads(path.read_text())
        z = run["native_profile"]["grid"]
        diagnostics = run["diagnostics"]["finite_rate_film"]
        for field, name, color, marker in RESISTANCES:
            y = [node[field] for node in diagnostics]
            ax.plot(z, y, label=name, color=color, marker=marker, markersize=3.5)
        ax.set(title=label, xlabel="Height above gas inlet (m)", yscale="log")
        ax.grid(alpha=0.2)
    panels[0].set_ylabel("CO₂ transfer resistance (Pa m² s/mol)")
    panels[1].legend(frameon=False)
    fig.savefig(figures / "3c_film_resistances.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    render(Path(sys.argv[1]))
