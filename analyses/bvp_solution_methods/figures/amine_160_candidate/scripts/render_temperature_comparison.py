"""Render the retained discrete candidate-temperature and source-tap comparison."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[5]
FIGURE = Path(__file__).resolve().parents[1]
OUT = FIGURE / "output"
DATA = OUT / "temperature-comparison.csv"
PROVENANCE = OUT / "temperature-comparison-provenance.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    metadata = json.loads(PROVENANCE.read_text())
    if sha256(DATA) != metadata["data_sha256"]:
        raise ValueError("The plotted table does not match its retained provenance")
    with DATA.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    model = [row for row in rows if row["series"] == "candidate_liquid_temperature_native_node"]
    taps = [row for row in rows if row["series"] == "source_reported_tap_phase_ambiguous"]
    if len(model) != 65 or len(taps) != 5:
        raise ValueError("The plotted table must contain 65 native predictions and five source taps")
    model_z = [float(row["position_m"]) for row in model]
    model_t = [float(row["temperature_K"]) for row in model]
    tap_z = [float(row["position_m"]) for row in taps]
    tap_t = [float(row["temperature_K"]) for row in taps]

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.hashsalt": "issue160-density-candidate",
    })
    fig, axis = plt.subplots(figsize=(7.2, 4.0), layout="constrained")
    axis.scatter(model_z, model_t, s=19, marker="o", color="#0072B2",
                 label="Candidate liquid temperature (native nodes)", zorder=2)
    axis.scatter(tap_z, tap_t, s=62, marker="x", linewidths=1.7, color="#D55E00",
                 label="Source-reported taps (phase ambiguous)", zorder=3)
    axis.set(xlim=(0.0, 6.0), xlabel="Height from gas inlet, z (m)", ylabel="Temperature (K)")
    axis.set_xticks(range(7))
    low, high = min(model_t + tap_t), max(model_t + tap_t)
    margin = 0.06 * (high - low)
    axis.set_ylim(low - margin, high + margin)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, loc="lower right", fontsize=8)
    axis.set_title("3C candidate temperatures and source taps")
    figure = OUT / "temperature-comparison.svg"
    preview = OUT / "temperature-comparison.png"
    fig.savefig(figure, metadata={"Date": None})
    fig.savefig(preview, dpi=180)
    plt.close(fig)

    record = {
        "figure_path": str(figure.relative_to(ROOT)),
        "figure_sha256": sha256(figure),
        "preview_path": str(preview.relative_to(ROOT)),
        "preview_sha256": sha256(preview),
        "data_path": str(DATA.relative_to(ROOT)),
        "data_sha256": sha256(DATA),
        "provenance_path": str(PROVENANCE.relative_to(ROOT)),
        "provenance_sha256": sha256(PROVENANCE),
        "renderer_path": str(Path(__file__).relative_to(ROOT)),
        "renderer_sha256": sha256(Path(__file__)),
        "claim": "The accepted candidate has a sampled liquid-temperature maximum of 348.328 K at z=5.223 m; the retained source tap values are shown without assigning a phase.",
        "model_series": "65 unconnected native-node points; no curve, fit, or new model evaluation",
        "source_series": "Five reported 3C temperatures; source tap phase is ambiguous",
        "uncertainty": "No tap uncertainty intervals were retained; none are inferred here.",
    }
    (OUT / "temperature-comparison-render.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
