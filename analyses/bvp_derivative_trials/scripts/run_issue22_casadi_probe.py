"""Probe an equation-preserving CasADi route without adding a runtime path.

The probe deliberately wraps the live absorber RHS as a CasADi callback.  A
direct-collocation candidate is admissible only if CasADi can obtain a checked
Jacobian for that callback; enabling CasADi finite differences would violate
Issue #22's derivative gate and is therefore not used.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import casadi as ca

import mea_absorption_column.Run_Model as run_model_module
from mea_absorption_column.BVP.Methods.Scipy_BVP_Solve import (
    _column_rhs,
    _physical_rhs_to_solver_rhs,
)
from mea_absorption_column.Thermodynamics import thermo_models


ROOT = Path(__file__).parents[3]
RESULT = ROOT / "analyses/bvp_derivative_trials/results/final/tables/issue22_casadi_probe.json"
CAMPAIGN_TABLE = ROOT / "analyses/reactive_film_evidence/results/final/tables/column_film_capture_comparison.csv"
CAMPAIGN_NODES = ROOT / "analyses/reactive_film_evidence/results/final/tables/column_film_nodes.csv"
CAMPAIGN_PROVENANCE = ROOT / "analyses/reactive_film_evidence/results/final/tables/column_film_run_provenance.json"
CASE_FILES = {
    "K": (ROOT / "src/mea_absorption_column/data/NCCC_2014_model_inputs_mass.csv", "mass"),
    "C": (ROOT / "src/mea_absorption_column/data/C_cases_campaign_inputs.csv", "mole"),
}


def _read_campaign_rows(case_ids: list[str]) -> dict[str, dict[str, object]]:
    with CAMPAIGN_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["case_id"]: {
            key: _number(value)
            for key, value in row.items()
            if key not in {"case_id", "message", "boundary_residual_components"}
        }
        for row in rows
        if row["case_id"] in case_ids
    }


def _read_campaign_nodes(case_ids: list[str]) -> dict[str, list[dict[str, object]]]:
    with CAMPAIGN_NODES.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        case_id: [
            {key: _number(value) for key, value in row.items()}
            for row in rows
            if row["case_id"] == case_id
        ]
        for case_id in case_ids
    }


def _number(value: str) -> object:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


class _LiveColumnRhs(ca.Callback):
    """CasADi value callback for the unchanged scaled SciPy column RHS."""

    def __init__(self, parameters, transform_mode: str, guard_rhs: bool):
        self.parameters = parameters
        self.transform_mode = transform_mode
        self.guard_rhs = guard_rhs
        self.evaluations = 0
        super().__init__()
        self.construct("live_column_rhs", {})

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, _index):
        return ca.Sparsity.dense(8, 1)

    def get_sparsity_out(self, _index):
        return ca.Sparsity.dense(7, 1)

    def eval(self, arguments):
        self.evaluations += 1
        payload = np.asarray(arguments[0], dtype=float).reshape(-1)
        z = float(payload[0])
        y_solver = payload[1:]
        rhs = _column_rhs(
            z,
            y_solver,
            self.parameters,
            transform_mode=self.transform_mode,
            guard_rhs=self.guard_rhs,
        )
        return [ca.DM(_physical_rhs_to_solver_rhs(y_solver, rhs, self.transform_mode))]


def _capture_scipy_run(dataframe, data_type: str, case_id: str) -> dict[str, object]:
    captured: dict[str, object] = {}
    original = run_model_module.scipy_BVP_solve

    def wrapped(y_a, y_b, z, parameters, settings=None):
        captured.update(
            parameters=parameters,
            y_a=np.asarray(y_a, dtype=float).copy(),
            z=np.asarray(z, dtype=float).copy(),
            settings=dict(settings or {}),
        )
        return original(y_a, y_b, z, parameters, settings=settings)

    run_model_module.scipy_BVP_solve = wrapped
    started = time.perf_counter()
    try:
        result = run_model_module.run_model(
            dataframe,
            method="scipy-bvp",
            thermo_model="epcsaft_ionic",
            data_type=data_type,
            run=list(dataframe.index).index(case_id),
            return_details=True,
            solver_settings={
                "mesh_points": 21,
                "tol": 0.1,
                "bc_tol": 0.001,
                "max_nodes": 1000,
            },
        )
    finally:
        run_model_module.scipy_BVP_solve = original
    result["probe_wall_runtime_s"] = time.perf_counter() - started
    result.pop("_profiles", None)
    result.pop("_raw_solution_scaled", None)
    return result | {"_captured": captured}


def _casadi_attempt(run: dict[str, object]) -> dict[str, object]:
    captured = run.pop("_captured")
    parameters = captured["parameters"]
    settings = captured["settings"]
    callback = _LiveColumnRhs(
        parameters,
        str(settings.get("transform_mode", "bounded_guarded_raw_state")),
        bool(settings.get("guard_rhs", True)),
    )
    y0 = np.asarray(captured["y_a"], dtype=float)
    z0 = np.asarray(captured["z"], dtype=float)
    started = time.perf_counter()
    value = np.asarray(callback(ca.DM(np.r_[z0[0], y0])), dtype=float).reshape(-1)
    callback_eval_s = time.perf_counter() - started

    grid = np.linspace(float(z0[0]), float(z0[-1]), 5)
    states = ca.MX.sym("X", 7, grid.size)
    constraints = []
    for index in range(grid.size - 1):
        left = callback(ca.vertcat(float(grid[index]), states[:, index]))
        right = callback(ca.vertcat(float(grid[index + 1]), states[:, index + 1]))
        step = float(grid[index + 1] - grid[index])
        constraints.append(states[:, index + 1] - states[:, index] - 0.5 * step * (left + right))
    defect = ca.vertcat(*constraints)
    jacobian_error = None
    try:
        ca.jacobian(defect, ca.vec(states))
    except Exception as exc:  # typed negative result required by Issue #22
        jacobian_error = f"{type(exc).__name__}: {exc}".splitlines()[0]
    if jacobian_error is None:
        raise RuntimeError("unexpectedly obtained an unchecked live RHS Jacobian")

    return {
        "callback_value_status": "evaluated",
        "callback_value_inf_norm": float(np.linalg.norm(value, ord=np.inf)),
        "callback_eval_runtime_s": callback_eval_s,
        "callback_has_jacobian": bool(callback.has_jacobian()),
        "callback_evaluations_after_transcription": callback.evaluations,
        "direct_collocation_grid_points": int(grid.size),
        "direct_collocation_state_dimension": 7,
        "direct_collocation_defect_count": int(defect.shape[0]),
        "ipopt_available": bool(ca.has_nlpsol("ipopt")),
        "jacobian_status": "missing_checked_derivative",
        "jacobian_error": jacobian_error,
        "ipopt_status": "not_attempted_missing_derivative",
    }


def _eos_derivative_probe() -> dict[str, object]:
    composition = np.asarray((1.0, 20.0, 70.0, 3.0, 2.0, 0.5, 0.25, 0.5, 0.5), dtype=float)
    composition /= composition.sum()
    state = thermo_models.epcsaft_liquid_transport_state(318.15, 109500.0, composition)
    derivative = state.fixed_other_concentrations_log_fugacity_derivative(0)
    return {
        "status": "available",
        "composition_dimension": int(state.composition.size),
        "log_composition_basis_shape": list(state.log_composition_basis.shape),
        "chemical_potential_derivative_shape": list(state.chemical_potential_derivatives_over_rt.shape),
        "co2_fixed_other_log_fugacity_derivative": float(derivative),
        "artifact_fingerprint": state.artifact_fingerprint,
        "parameter_fingerprint": state.parameter_fingerprint,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-ids", nargs="+", default=["K18", "1C", "5C"])
    parser.add_argument("--output", type=Path, default=RESULT)
    args = parser.parse_args()
    campaign_rows = _read_campaign_rows(args.case_ids)
    campaign_nodes = _read_campaign_nodes(args.case_ids)
    cases = []
    for case_id in args.case_ids:
        case_file, data_type = CASE_FILES["K" if case_id.startswith("K") else "C"]
        import pandas as pd

        dataframe = pd.read_csv(case_file, index_col=0)
        run = _capture_scipy_run(dataframe, data_type, case_id)
        cases.append(
            {
                "case_id": case_id,
                "scipy_reference": {
                    key: value
                    for key, value in run.items()
                    if key != "_captured"
                },
                "retained_outer_fixed_point": campaign_rows.get(case_id),
                "retained_outer_node_diagnostics": campaign_nodes.get(case_id, []),
                "casadi_probe": _casadi_attempt(run),
            }
        )

    provenance = json.loads(CAMPAIGN_PROVENANCE.read_text(encoding="utf-8"))
    wheel = importlib.metadata.distribution("epcsaft")
    payload = {
        "status": "supported_negative",
        "claim_boundary": "Equation-preserving CasADi/IPOPT comparison only; no physical-model, thermodynamic, transport, or capture claim.",
        "adoption_decision": "reject_casadi_runtime_path",
        "adoption_rule_test": "CasADi must preserve all checks and either resolve an admitted SciPy failure or reduce median total runtime by at least 2x; missing checked absorber derivatives fail the candidate before IPOPT.",
        "equation_boundary": "The live abs_column RHS and current equilibrium-manifold outer fixed-point route were called unchanged. No ePC-SAFT equation was copied or differentiated downstream.",
        "external_derivative_boundary": "The public ePC-SAFT fixed-T,P chemical-potential tangent is available, but it does not provide the total derivative of the Python absorber RHS through chemistry, transport, enthalpy, and the interpolated outer film nodes.",
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "engine_source_commit": provenance["engine_source_commit"],
        "engine_wheel_sha256": provenance["engine_wheel_sha256"],
        "engine_wheel_direct_url": wheel.read_text("direct_url.json"),
        "dataset_parameter_document_sha256": provenance["parameter_document_sha256"],
        "dataset_reaction_system_sha256": provenance["reaction_system_sha256"],
        "casadi_version": ca.__version__,
        "ipopt_available": bool(ca.has_nlpsol("ipopt")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "campaign_provenance": provenance,
        "eos_derivative_probe": _eos_derivative_probe(),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output), "cases": args.case_ids}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
