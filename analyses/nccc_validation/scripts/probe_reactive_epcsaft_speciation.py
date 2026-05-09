from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mea_absorption_column.misc.Convert_Data import convert_data
from mea_absorption_column.Thermodynamics.Chemical_Equilibrium import chemical_equilibrium
from mea_absorption_column.Thermodynamics.thermo_models import (
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
    ensure_epcsaft_importable,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = MEA_THERMODYNAMICS_EPCSAFT_DATASET
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "analyses"
    / "nccc_validation"
    / "results"
    / "runs"
    / "reactive_epcsaft_speciation_probe"
)

SPECIES_6 = ("CO2", "MEA", "H2O", "MEAH+", "MEACOO-", "HCO3-")
REACTIONS_6 = (
    {"CO2": -1.0, "MEA": -2.0, "MEAH+": 1.0, "MEACOO-": 1.0},
    {"CO2": -1.0, "MEA": -1.0, "H2O": -1.0, "MEAH+": 1.0, "HCO3-": 1.0},
)
REACTION_NAMES_6 = ("carbamate", "bicarbonate")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    return value


def legacy_log_constants(temperature_K: float) -> tuple[float, float]:
    a1, b1, c1, d1 = 233.4, -3410.0, -36.8, 0.0
    a2, b2, c2, d2 = 176.72, -2909.0, -28.46, 0.0
    T = float(temperature_K)
    return (
        float(a1 + b1 / T + c1 * np.log(T) + d1 * T),
        float(a2 + b2 / T + c2 * np.log(T) + d2 * T),
    )


def log_k_from_activity_state(mixture, temperature_K: float, pressure_Pa: float, x: np.ndarray) -> tuple[float, float]:
    state = mixture.state(T=float(temperature_K), P=float(pressure_Pa), x=x, phase="liq")
    gamma = state.activity_coefficient(species=list(SPECIES_6))
    values: list[float] = []
    for reaction in REACTIONS_6:
        total = 0.0
        for species, coefficient in reaction.items():
            idx = SPECIES_6.index(species)
            activity = max(float(x[idx]) * float(gamma[species]), 1.0e-300)
            total += float(coefficient) * math.log(activity)
        values.append(float(total))
    return tuple(values)


def build_mixture(epcsaft, dataset: Path, temperature_K: float, x: np.ndarray):
    options_path = dataset / "user_options.json"
    user_options = None
    if options_path.exists():
        user_options = json.loads(options_path.read_text(encoding="utf-8"))
    return epcsaft.ePCSAFTMixture.from_dataset(
        str(dataset),
        list(SPECIES_6),
        np.asarray(x, dtype=float),
        float(temperature_K),
        user_options=user_options,
    )


def solve_mode(
    epcsaft,
    mixture,
    mode: str,
    temperature_K: float,
    pressure_Pa: float,
    initial_x: np.ndarray,
    apparent_x: np.ndarray,
    log_k_values: tuple[float, float],
    standard_state: str,
) -> dict[str, Any]:
    balances = {
        "amine_total": {"MEA": 1.0, "MEAH+": 1.0, "MEACOO-": 1.0},
        "carbon_total": {"CO2": 1.0, "MEACOO-": 1.0, "HCO3-": 1.0},
        "water_total": {"H2O": 1.0},
    }
    totals = {
        "amine_total": float(apparent_x[1]),
        "carbon_total": float(apparent_x[0]),
        "water_total": float(apparent_x[2]),
    }
    reactions = [
        epcsaft.ReactionDefinition(
            reaction,
            value,
            name=name,
            standard_state=standard_state,
        )
        for reaction, value, name in zip(REACTIONS_6, log_k_values, REACTION_NAMES_6)
    ]
    started = time.perf_counter()
    row: dict[str, Any] = {
        "mode": mode,
        "standard_state": standard_state,
        "success": False,
        "runtime_s": None,
        "message": "",
    }
    try:
        result = epcsaft.solve_reactive_speciation(
            species=list(SPECIES_6),
            mixture_factory=lambda x, T, P: mixture,
            T=float(temperature_K),
            P=float(pressure_Pa),
            balances=balances,
            totals=totals,
            reactions=reactions,
            initial_x=np.asarray(initial_x, dtype=float),
            options=epcsaft.ReactiveSpeciationOptions(
                max_iterations=60,
                tolerance=1.0e-8,
                mass_tolerance=1.0e-7,
                charge_tolerance=1.0e-7,
                reaction_tolerance=1.0e-7,
                damping=0.7,
                return_best_effort=True,
            ),
        )
        row.update(
            {
                "success": bool(result.success),
                "runtime_s": time.perf_counter() - started,
                "message": getattr(result, "message", ""),
                "max_mass_balance_residual": max(
                    (abs(float(value)) for value in result.mass_balance_residuals.values()),
                    default=float("nan"),
                ),
                "max_reaction_residual": max(
                    (abs(float(value)) for value in result.reaction_residuals),
                    default=float("nan"),
                ),
                "charge_residual": float(result.charge_residual),
                "state_failure_count": int(result.state_failure_count),
                "activity_model": result.diagnostics.get("activity_model"),
                "activity_basis": result.diagnostics.get("activity_basis"),
                "native_entrypoint": result.diagnostics.get("native_entrypoint"),
            }
        )
        for species in SPECIES_6:
            row[f"x_{species}"] = float(result.x[species])
            row[f"gamma_{species}"] = float(result.activity_coefficients[species])
        return row
    except Exception as exc:
        row.update(
            {
                "runtime_s": time.perf_counter() - started,
                "message": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
            }
        )
        return row


def run_probe(case_id: str, dataset: Path, output_root: Path) -> Path:
    ensure_epcsaft_importable()
    import epcsaft

    data_path = REPO_ROOT / "src" / "mea_absorption_column" / "data" / "C_cases_data.csv"
    frame = pd.read_csv(data_path, index_col=0)
    if case_id not in frame.index:
        raise ValueError(f"Case {case_id!r} was not found in {data_path}.")
    run_index = int(frame.index.get_loc(case_id))
    inputs, _, metadata = convert_data(frame, run=run_index, type="mole", return_metadata=True)
    liquid_flows, _, liquid_temperature_K, _, _, _, _, pressure_Pa, _ = inputs
    apparent_x = np.asarray(liquid_flows, dtype=float)
    apparent_x = apparent_x / float(apparent_x.sum())
    legacy_concentrations, legacy_x = chemical_equilibrium(list(liquid_flows), float(liquid_temperature_K))
    legacy_x = np.asarray(legacy_x, dtype=float)
    mixture = build_mixture(epcsaft, dataset, float(liquid_temperature_K), legacy_x)

    legacy_k = legacy_log_constants(float(liquid_temperature_K))
    activity_k = log_k_from_activity_state(
        mixture,
        float(liquid_temperature_K),
        float(pressure_Pa),
        legacy_x,
    )
    modes = [
        (
            "legacy_concentration_constants",
            legacy_k,
            "concentration",
        ),
        (
            "legacy_constants_as_activity_basis",
            legacy_k,
            "mole_fraction_activity",
        ),
        (
            "activity_constants_calibrated_to_legacy_state",
            activity_k,
            "mole_fraction_activity",
        ),
    ]
    rows = [
        solve_mode(
            epcsaft,
            mixture,
            mode,
            float(liquid_temperature_K),
            float(pressure_Pa),
            legacy_x,
            apparent_x,
            log_k_values,
            standard_state,
        )
        for mode, log_k_values, standard_state in modes
    ]
    for row in rows:
        row["case_id"] = case_id
        row["temperature_K"] = float(liquid_temperature_K)
        row["pressure_Pa"] = float(pressure_Pa)
        row["epcsaft_version"] = getattr(epcsaft, "__version__", "")
        row["epcsaft_file"] = getattr(epcsaft, "__file__", "")
        row["dataset"] = str(dataset)
        row["beds"] = metadata["beds"]
        row["intercoolers"] = metadata["intercoolers"]
        for idx, species in enumerate(SPECIES_6):
            row[f"legacy_x_{species}"] = float(legacy_x[idx])
            row[f"apparent_x_{species}"] = float(apparent_x[idx]) if idx < len(apparent_x) else 0.0

    out_dir = output_root / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "reactive_epcsaft_speciation_probe.csv"
    json_path = out_dir / "reactive_epcsaft_speciation_probe.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(_jsonable({"rows": rows}), indent=2) + "\n", encoding="utf-8")
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="3C")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = run_probe(args.case_id, args.dataset, args.output_root)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
