from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.parse import unquote, urlparse

from mea_absorption_column.Thermodynamics.Chemical_Equilibrium import SPECIES_9
from mea_absorption_column.Thermodynamics.epcsaft_v02 import (
    certify_homogeneous_reactive_liquid_state,
)


CHARGES = (0, 0, 0, 1, -1, -1, -2, 1, -1)
COMPONENT_IDS = (
    "carbon-dioxide",
    "monoethanolamine",
    "water",
    "protonated-monoethanolamine",
    "carbamate-anion",
    "bicarbonate-anion",
    "carbonate-anion",
    "hydronium-cation",
    "hydroxide-anion",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluate(payload):
    observation, source, dataset_text = payload
    started = time.perf_counter()
    identity = observation["identity"]
    try:
        result = certify_homogeneous_reactive_liquid_state(
            observation["request"], dataset_text, SPECIES_9
        )
        temperature_k = float(result["temperature_k"])
        pressure_pa = float(result["pressure_pa"])
        composition = tuple(float(value) for value in result["composition"])
        phi_co2 = float(result["fugacity_coefficients"][0])
        return {
            "state_id": identity,
            "temperature_k": temperature_k,
            "pressure_pa": pressure_pa,
            "mea_mass_fraction": float(source["mea_mass_fraction"]),
            "loading": float(source["loading"]),
            "status": "evaluated",
            "solver_status": "certified_homogeneous_continuation_reference",
            "runtime_s": time.perf_counter() - started,
            "parameter_fingerprint": result["parameter_fingerprint"],
            "certificate_fingerprint": result["certificate_fingerprint"],
            "mole_fraction_sum_error": abs(math.fsum(composition) - 1.0),
            "charge_residual": math.fsum(
                charge * value for charge, value in zip(CHARGES, composition, strict=True)
            ),
            "minimum_mole_fraction": min(composition),
            "phi_co2": phi_co2,
            "fugacity_co2_pa": composition[0] * phi_co2 * pressure_pa,
            **{
                f"x_{component_id}": value
                for component_id, value in zip(COMPONENT_IDS, composition, strict=True)
            },
        }
    except Exception as error:
        return {
            "state_id": identity,
            "temperature_k": source["temperature_k"],
            "pressure_pa": "",
            "mea_mass_fraction": source["mea_mass_fraction"],
            "loading": source["loading"],
            "status": "failed",
            "solver_status": f"{type(error).__name__}: {error}",
            "runtime_s": time.perf_counter() - started,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-input", type=Path, required=True)
    parser.add_argument("--state-table", type=Path, required=True)
    parser.add_argument("--parameter-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    fit_input = json.loads(args.fit_input.read_text(encoding="utf-8"))
    with args.state_table.open(newline="", encoding="utf-8") as handle:
        sources = {row["state_id"]: row for row in csv.DictReader(handle)}
    observations = fit_input["observations"]
    if len(observations) != 44 or set(sources) != {
        observation["identity"] for observation in observations
    }:
        raise RuntimeError("Expected the complete 44-state MEA speciation definition")

    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(
            pool.map(
                _evaluate,
                (
                    (observation, sources[observation["identity"]], str(args.parameter_dataset))
                    for observation in observations
                ),
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for row in rows:
        for fieldname in fieldnames:
            row.setdefault(fieldname, "")
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    direct_url = json.loads(
        importlib.metadata.distribution("epcsaft").read_text("direct_url.json") or "{}"
    ).get("url", "")
    wheel = Path(unquote(urlparse(direct_url).path))
    failures = [row for row in rows if row["status"] != "evaluated"]
    summary = {
        "state_count": len(rows),
        "evaluated_count": len(rows) - len(failures),
        "failed_count": len(failures),
        "elapsed_seconds": time.perf_counter() - started,
        "fit_input_sha256": _sha256(args.fit_input),
        "state_table_sha256": _sha256(args.state_table),
        "parameter_document_sha256": _sha256(args.parameter_dataset / "parameters.json"),
        "temperature_adjustments_sha256": _sha256(
            args.parameter_dataset / "temperature_adjustments.json"
        ),
        "engine_wheel_sha256": _sha256(wheel),
        "engine_wheel_path": str(wheel),
        "maximum_mole_fraction_sum_error": max(
            float(row.get("mole_fraction_sum_error") or 0.0) for row in rows
        ),
        "maximum_absolute_charge_residual": max(
            abs(float(row.get("charge_residual") or 0.0)) for row in rows
        ),
        "failures": failures,
    }
    summary_path = args.output.with_name(f"{args.output.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(rows)} states failed")


if __name__ == "__main__":
    main()
