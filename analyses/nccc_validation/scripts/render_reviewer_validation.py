"""Summarize retained energy-corrected attempts; never run the column."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[3]
    results = root / "analyses/nccc_validation/results"
    output = results / "reviewer_energy"
    output.mkdir(exist_ok=True)
    runs = ["energy_corrected_3c_reference", "energy_corrected_3c_refined",
            "energy_corrected_additional_henry", "energy_corrected_3c_method_attempts",
            "energy_corrected_3c_reactive_diagnostic"]
    tables, inputs, profiles = [], [], {}
    for name in runs:
        path = results / "runs" / name / "benchmark_results.csv"
        inputs.extend([path, path.with_name("run_identity.json")])
        table = pd.read_csv(path)
        table.insert(0, "run", name)
        for i, row in table.iterrows():
            if not row["success"]:
                continue
            profile = root / row["profile_csv_dir"]
            inputs.extend(profile / f"{sheet}.csv" for sheet in ("T", "Hl", "Hv"))
            t, hl, hv = [pd.read_csv(profile / f"{sheet}.csv") for sheet in ("T", "Hl", "Hv")]
            profiles[(name, row["case_id"])] = t
            net = hv["Hvf"] - hl["Hlf"]
            table.loc[i, "peak_liquid_temperature_K"] = t["Tl"].max()
            table.loc[i, "net_energy_range_W"] = net.max() - net.min()
        tables.append(table)
    summary = pd.concat(tables, ignore_index=True)
    summary.to_csv(output / "summary.csv", index=False)
    # Retain plotted values independently of disposable run directories.
    pd.concat([p.assign(run=k[0], case_id=k[1]) for k, p in profiles.items()]).to_csv(
        output / "temperature_profiles.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for name, label, style, color in zip(runs[:2], ["21 initial nodes; tol 0.5", "41 initial nodes; tol 0.05"], ["-", "--"], ["#0072B2", "#D55E00"]):
        t = profiles[(name, "3C")]
        axes[0].plot(t["Position"], t["Tl"], style, color=color, label=label)
    axes[0].set(xlabel="Normalized height, bottom to top", ylabel="Liquid temperature (K)",
                title="Case 3C: refinement changes peak by 0.004 K")
    axes[0].legend(fontsize=8)
    cases = summary[(summary["method"] == "scipy-bvp") & summary["success"]
                    & (summary["run"] != runs[0])].sort_values("case_id")
    positions = np.arange(len(cases))
    axes[1].plot(positions, cases["capture_pct"], "o", color="#0072B2", label="Corrected Henry model")
    axes[1].plot(positions, cases["capture_pct"] - cases["capture_error_pct"], "x", color="#D55E00", label="Retained NCCC observation")
    axes[1].set(xticks=positions, xticklabels=cases["case_id"], xlabel="NCCC 2017 case",
                ylabel="CO₂ capture (%)", title="Four cases; no parameter adjustment")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)
    (output / "provenance.json").write_text(json.dumps({
        "input_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Working reviewer evidence; not promoted manuscript results. No observation uncertainty supplied."
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
