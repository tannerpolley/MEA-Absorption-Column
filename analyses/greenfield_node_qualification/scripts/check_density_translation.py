"""Check Amine #160's density-only translation and time one exact column-node Jacobian."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import casadi as ca
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analyses/bvp_solution_methods/scripts"))

from mea_absorption_column.Thermodynamics.casadi_reactive import EnginePhase
from mea_absorption_column.Thermodynamics.reactive_bundle import engine_liquid, ideal_gas_record
from mea_absorption_column.column import _prepare_conserved_column_in_process
from mea_absorption_column.config.column import resolve_column_config
import run_case


DATASET = ROOT / "src/mea_absorption_column/data/epcsaft_datasets/MEA_amine160_density_candidate"
OUTPUT = ROOT / "analyses/greenfield_node_qualification/results/amine_160_candidate/density_translation_probe.json"
RETAINED_PROFILE = ROOT / "analyses/bvp_solution_methods/results/replay_149/upwind_cos_n65/attempt.json"
FIRST_ACTION_RTOL = 5e-6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    dataset = dataset.resolve()
    probe_start = time.perf_counter()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    ideal = dataset / "ideal-gas-thermochemistry.json"
    parameters_path = dataset / "parameters.json"
    mapping = json.loads(parameters_path.read_text())
    correction = mapping["empirical_density_correction"]
    coefficient_m3_per_mol = 1e-6 * float(correction["c_cm3_mol"])
    case_path = ROOT / run_case.CASE
    case = json.loads(case_path.read_text())
    policy = case["physical_inputs"]["liquid_branch_policy"]
    candidate = engine_liquid(dataset, ideal, "translated_density_probe", loading_policy=policy)
    ids = candidate.component_ids
    untranslated = EnginePhase(
        "raw_density_probe", mapping, ideal_gas_record(ideal, ids), kind="liquid", feed_ids=ids[:3],
        molar_masses=candidate.molar_masses, reactions=candidate.reactions,
        neutral_reference=candidate.neutral_reference, loading_policy=policy,
    )
    point = np.asarray([case["physical_inputs"]["liquid_temperature_k"],
                        case["physical_inputs"]["bottom_pressure_pa"],
                        *case["physical_inputs"]["liquid_feed_mol_s"]], dtype=float)
    translated_values = np.asarray(candidate(point)).ravel()
    raw_values = np.asarray(untranslated(point)).ravel()
    total_amount = float(raw_values[:len(ids)].sum())
    ion_amount = float(sum(raw_values[ids.index(component)] for component in correction["component_ids"]))
    expected_density = total_amount / (total_amount / raw_values[len(ids) + 2]
                                       + coefficient_m3_per_mol * ion_amount)
    density_defect = abs(translated_values[len(ids) + 2] - expected_density) / expected_density
    other_indices = [i for i, name in enumerate(candidate.output_ids) if name != "molar-density"]
    other_values = translated_values[other_indices] - raw_values[other_indices]
    other_scale = np.maximum(np.abs(raw_values[other_indices]), 1e-12)
    other_max_relative_defect = float(np.max(np.abs(other_values) / other_scale, initial=0.0))
    other_max_absolute_difference = float(np.max(np.abs(other_values), initial=0.0))

    symbol = ca.MX.sym("liquid_inputs", 5)
    density = candidate(ca.vertcat(*(symbol[i] for i in range(5))))[len(ids) + 2]
    density_and_jacobian = ca.Function("density_and_jacobian", [symbol],
                                       [density, ca.jacobian(density, symbol)])
    exact_density, exact_gradient = density_and_jacobian(point)
    exact_feed_derivative = float(np.asarray(exact_gradient).ravel()[2])
    steps = point[2] * np.asarray((1e-2, 5e-3, 2.5e-3))

    def density_at_co2_feed(feed):
        changed = point.copy()
        changed[2] = feed
        return float(np.asarray(candidate(changed)).ravel()[len(ids) + 2])

    differences = [(density_at_co2_feed(point[2] + step)
                    - density_at_co2_feed(point[2] - step)) / (2 * step) for step in steps]
    coarse = (4 * differences[1] - differences[0]) / 3
    fine = (4 * differences[2] - differences[1]) / 3
    derivative_resolved = (abs(coarse - fine) <= 0.1 * abs(fine)
                           and abs(2 * steps[2] * differences[2]) >= 1e-8 * abs(float(exact_density)))
    derivative_defect = abs(exact_feed_derivative - fine) / max(abs(exact_feed_derivative), abs(fine), 1e-300)

    retained = json.loads(RETAINED_PROFILE.read_text())
    grid = np.asarray(retained["native_profile"]["grid"], dtype=float)
    states = np.asarray(retained["native_profile"]["state_matrix"], dtype=float)
    index = int(np.argmin(abs(grid - 3.0)))
    request = run_case.request("trapezoidal", 11, 9, wall=600.0, dataset=dataset)
    prepared = _prepare_conserved_column_in_process(resolve_column_config(request))
    node = prepared["assembly"]["node"]
    node_symbol, height = ca.MX.sym("node_state", 12), ca.MX(float(grid[index]))
    node_outputs = node.call([height, node_symbol], True, False)
    node_jacobian = ca.Function("translated_density_node_jacobian", [node_symbol],
                                [ca.jacobian(ca.vertcat(*node_outputs), node_symbol)])
    node_start = time.perf_counter()
    node_failure = None
    try:
        node_jacobian_value = np.asarray(node_jacobian(states[:, index]))
        node_status = "completed"
    except Exception as exc:
        node_jacobian_value = None
        node_status = "failed"
        node_failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    node_wall_s = time.perf_counter() - node_start

    contract = json.loads((ROOT / "integration/epcsaft_contract.json").read_text())["final_identity"]
    result = {
        "engine_commit": contract["engine_commit"],
        "engine_wheel_sha256": contract["wheel_sha256"],
        "inputs_sha256": {
            "dataset_parameters.json": sha256(parameters_path),
            "dataset_engine-reactions.json": sha256(dataset / "engine-reactions.json"),
            "dataset_ideal-gas-thermochemistry.json": sha256(ideal),
            "case_3c.json": sha256(case_path),
            "retained_profile_used_only_for_node_state.json": sha256(RETAINED_PROFILE),
            "casadi_reactive.py": sha256(ROOT / "src/mea_absorption_column/Thermodynamics/casadi_reactive.py"),
            "reactive_bundle.py": sha256(ROOT / "src/mea_absorption_column/Thermodynamics/reactive_bundle.py"),
            "Coupled_Column.py": sha256(ROOT / "src/mea_absorption_column/BVP/Coupled_Column.py"),
            "run_case.py": sha256(ROOT / "analyses/bvp_solution_methods/scripts/run_case.py"),
        },
        "density_translation": {
            "c_cm3_mol": correction["c_cm3_mol"],
            "c_m3_per_mol": coefficient_m3_per_mol,
            "input_state": point.tolist(),
            "component_ids": correction["component_ids"],
            "native_molar_density_mol_m3": float(raw_values[len(ids) + 2]),
            "translated_molar_density_mol_m3": float(translated_values[len(ids) + 2]),
            "equation_density_mol_m3": float(expected_density),
            "relative_equation_defect": density_defect,
            "criterion": 1e-12,
            "passed": density_defect <= 1e-12,
            "non_density_outputs_max_relative_difference": other_max_relative_defect,
            "non_density_outputs_max_absolute_difference": other_max_absolute_difference,
            "non_density_outputs_equal_within_1e-12": other_max_relative_defect <= 1e-12,
        },
        "probe_wall_s": time.perf_counter() - probe_start,
        "thread_limits": {"OMP_NUM_THREADS": 1, "OPENBLAS_NUM_THREADS": 1, "MKL_NUM_THREADS": 1},
        "external_timeout_s": 900,
        "density_feed_derivative": {
            "input_direction": "CO2 apparent feed at fixed T, P, MEA feed and water feed",
            "casadi_exact_mol_m3_per_mol_s": exact_feed_derivative,
            "centred_differences_mol_m3_per_mol_s": differences,
            "richardson_fine_mol_m3_per_mol_s": fine,
            "richardson_resolved": bool(derivative_resolved),
            "relative_defect": float(derivative_defect),
            "criterion": FIRST_ACTION_RTOL,
            "passed": bool(derivative_resolved and derivative_defect <= FIRST_ACTION_RTOL),
        },
        "node_jacobian": {
            "status": node_status,
            "state_source": "nearest 3 m node of accepted #149 upwind cosine 65-node profile; used only as an evaluation point",
            "height_m": float(grid[index]),
            "jacobian_shape": None if node_jacobian_value is None else list(node_jacobian_value.shape),
            "wall_s": node_wall_s,
            "failure": node_failure,
            "method": "existing trapezoidal twelve-state node, 11-node request and nine film points",
        },
        "historical_full_attempt": {
            "path": str(RETAINED_PROFILE.relative_to(ROOT)),
            "wall_s": retained["execution"]["runtime_s"],
            "nlp_jac_g_calls": retained["result"]["solver_statistics"]["n_call_nlp_jac_g"],
            "nlp_jac_g_wall_s": retained["result"]["solver_statistics"]["t_wall_nlp_jac_g"],
            "native_liquid_actions": retained["native_calls"]["liquid_actions"],
        },
        "claim_limit": "One candidate density evaluation, one exact feed derivative and one node Jacobian timing; no candidate column profile was solved.",
        "command": ["timeout", "900", "env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
                    "MKL_NUM_THREADS=1", "PYTHONPATH=src", "uv", "run", "--frozen", "--no-sync",
                    "python", str(Path(__file__).relative_to(ROOT))],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run_case.clean(result)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    if node_status != "completed" or not result["density_translation"]["passed"] or not result["density_feed_derivative"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
