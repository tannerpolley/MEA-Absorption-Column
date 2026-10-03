"""Extract native-node liquid temperatures and the retained 3C source tap row."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
FIGURE = Path(__file__).resolve().parents[1]
OUT = FIGURE / "output"
ATTEMPT = ROOT / "analyses/bvp_solution_methods/results/amine_160_candidate/upwind_cos_n65_parameter_transfer/attempt.json"
TAPS = ROOT / "src/mea_absorption_column/data/NCCC_2017_absorber_temperature_profiles.csv"
PARAMETERS = ROOT / "src/mea_absorption_column/data/epcsaft_datasets/MEA_amine160_density_candidate/parameters.json"
HEIGHT_M = 6.0
LIQUID_TEMPERATURE_STATE = 4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    attempt = json.loads(ATTEMPT.read_text())
    if (attempt["execution"]["status"] != "completed" or attempt["numerical_acceptance"] != "accepted"
            or attempt["physical_certification"]["accepted"] is not True):
        raise ValueError("The retained attempt must pass numerical and physical certification")
    profile = attempt["native_profile"]
    grid = [float(z) for z in profile["grid"]]
    state_matrix = profile["state_matrix"]
    temperatures = [float(t) for t in state_matrix[LIQUID_TEMPERATURE_STATE]]
    if len(grid) != 65 or len(temperatures) != len(grid):
        raise ValueError("Expected the accepted 65-node candidate profile")

    with TAPS.open(newline="") as stream:
        tap_row = next(row for row in csv.DictReader(stream) if row["case_no"] == "3C")
    source_taps = []
    for field, value_c in tap_row.items():
        if not field.startswith("position_") or not field.endswith("_C"):
            continue
        fraction_from_top = float(field.removeprefix("position_").removesuffix("_C"))
        position_m = (1.0 - fraction_from_top) * HEIGHT_M
        source_taps.append({
            "series": "source_reported_tap_phase_ambiguous",
            "position_m": position_m,
            "temperature_K": float(value_c) + 273.15,
            "tap_source_field": field,
            "fraction_from_top": fraction_from_top,
            "source_temperature_C": float(value_c),
        })
    source_taps.sort(key=lambda row: row["position_m"])
    if len(source_taps) != 5:
        raise ValueError("Expected the five reported source temperature taps for 3C")

    model_points = [
        {
            "series": "candidate_liquid_temperature_native_node",
            "position_m": z,
            "temperature_K": temperature,
            "tap_source_field": "",
            "fraction_from_top": "",
            "source_temperature_C": "",
        }
        for z, temperature in zip(grid, temperatures)
    ]
    rows = model_points + source_taps
    OUT.mkdir(parents=True, exist_ok=True)
    data_path = OUT / "temperature-comparison.csv"
    with data_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    parameter_record = json.loads(PARAMETERS.read_text())
    metadata = {
        "question": "How do the candidate's retained liquid-temperature nodes compare with the reported 3C source taps?",
        "intended_use": "Display the fixed-grid candidate values and the source tap values without implying a continuous model curve or resolving tap phase.",
        "axes": {"x": "height from gas inlet, m", "y": "temperature, K"},
        "series": {
            "candidate_liquid_temperature_native_node": {
                "role": "model prediction",
                "rows": len(model_points),
                "support": "65 retained native grid nodes; shown as points without a connecting line",
                "state_row": LIQUID_TEMPERATURE_STATE,
            },
            "source_reported_tap_phase_ambiguous": {
                "role": "source-reported observation",
                "rows": len(source_taps),
                "phase": "not resolved by the source record",
                "uncertainty": "no intervals retained in the source row",
            },
        },
        "coordinate_transform": "source position fraction x is measured from the top; z=(1-x)*6.0 m from the gas inlet",
        "temperature_transform": "source tap Celsius values converted to kelvin by adding 273.15; model temperatures are retained in kelvin",
        "inputs": {
            "attempt_path": str(ATTEMPT.relative_to(ROOT)),
            "attempt_sha256": sha256(ATTEMPT),
            "attempt_id": attempt["attempt_id"],
            "source_taps_path": str(TAPS.relative_to(ROOT)),
            "source_taps_sha256": sha256(TAPS),
            "candidate_parameters_path": str(PARAMETERS.relative_to(ROOT)),
            "candidate_parameters_sha256": sha256(PARAMETERS),
            "candidate_document_id": parameter_record["document_id"],
            "engine_commit": attempt["engine"]["commit"],
            "engine_wheel_sha256": attempt["engine"]["actual_sha256"],
            "case": "corrected published 2017 3C",
        },
        "source_tap_rows": source_taps,
        "data_path": str(data_path.relative_to(ROOT)),
        "data_sha256": sha256(data_path),
        "producer_path": str(Path(__file__).relative_to(ROOT)),
        "producer_sha256": sha256(Path(__file__)),
    }
    (OUT / "temperature-comparison-provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "data": str(data_path.relative_to(ROOT)),
                      "sha256": metadata["data_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
