from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "analyses/reactive_film_evidence/results/runs"
FINAL = ROOT / "analyses/reactive_film_evidence/results/final/tables"
BUNDLE = ROOT / "src/mea_absorption_column/data/epcsaft_datasets/MEA_reactive_epcsaft_bundle/bundle.json"
CASES = ("K18", "K19", "1C", "2C", "3C", "4C", "5C", "6C")


def main() -> None:
    comparisons, nodes, profiles, provenance = [], [], [], []
    for case_id in CASES:
        run = RUNS / f"current_bundle_campaign_{case_id}"
        comparisons.append(pd.read_csv(run / "column_comparison.csv"))
        nodes.append(pd.read_csv(run / "film_nodes.csv"))
        profiles.append(pd.read_csv(run / "axial_profiles.csv"))
        provenance.append(json.loads((run / "run_provenance.json").read_text()))
    comparison = pd.concat(comparisons, ignore_index=True)
    node_table = pd.concat(nodes, ignore_index=True)
    capture_changes = {}
    for case_id, case_nodes in node_table.groupby("case_id", sort=False):
        captures = case_nodes.groupby("outer_iteration")["column_capture_pct"].first()
        capture_changes[case_id] = abs(captures.iloc[-1] - captures.iloc[-2])
    comparison["final_capture_change_pp"] = comparison["case_id"].map(capture_changes)
    comparison["outer_iteration_converged"] = (
        (comparison["final_capture_change_pp"] < 0.05)
        & (comparison["final_conductance_change_relative"] < 0.02)
        & (comparison["final_bulk_fugacity_change_relative"] < 0.02)
    )
    if tuple(comparison["case_id"]) != CASES:
        raise AssertionError("campaign must contain each declared case exactly once and in order")
    commands = [item["command"] for item in provenance]
    common = {key: value for key, value in provenance[0].items() if key != "command"}
    if any(
        {key: value for key, value in item.items() if key != "command"} != common
        for item in provenance[1:]
    ):
        raise AssertionError("campaign provenance differs between cases")
    FINAL.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(FINAL / "column_film_capture_comparison.csv", index=False)
    node_table.to_csv(FINAL / "column_film_nodes.csv", index=False)
    pd.concat(profiles, ignore_index=True).to_csv(FINAL / "column_film_axial_profiles.csv", index=False)
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    (FINAL / "column_film_run_provenance.json").write_text(
        json.dumps({
            **common,
            "parameter_source_commit": bundle["parameter_source_commit"],
            "engine_source_commit": bundle["engine_source_commit"],
            "engine_wheel_sha256": bundle["engine_wheel_sha256"],
            "commands": commands,
            "case_ids": list(CASES),
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
