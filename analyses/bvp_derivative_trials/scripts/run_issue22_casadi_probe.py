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
from mea_absorption_column.BVP.robust_core import (
    POSITIVE_SOLVER_IDXS,
    POSITIVE_TRANSFORM_CEILING,
    POSITIVE_TRANSFORM_FLOOR,
    scaled_physical_to_solver,
    solver_to_scaled_physical,
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


def _solver_to_scaled_physical_casadi(y_solver, transform_mode: str):
    if transform_mode in {None, "", "none", "bounded_guarded_raw_state", "raw"}:
        return y_solver
    if transform_mode != "positive_flow_pressure":
        raise ValueError(f"Unknown transform_mode: {transform_mode}")
    values = [y_solver[index] for index in range(7)]
    for index in POSITIVE_SOLVER_IDXS:
        values[int(index)] = POSITIVE_TRANSFORM_FLOOR + (
            POSITIVE_TRANSFORM_CEILING - POSITIVE_TRANSFORM_FLOOR
        ) / (1.0 + ca.exp(-y_solver[int(index)]))
    return ca.vertcat(*values)


class _LiveColumnRhs(ca.Callback):
    """CasADi value callback for the unchanged scaled SciPy column RHS."""

    def __init__(self, parameters, transform_mode: str, guard_rhs: bool, diagnostic_fd: bool = False):
        self.parameters = parameters
        self.transform_mode = transform_mode
        self.guard_rhs = guard_rhs
        self.diagnostic_fd = diagnostic_fd
        self.evaluations = 0
        self.jacobian_evaluations = 0
        self.derivative_callbacks = []
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
        return [ca.DM(self.evaluate_payload(payload))]

    def evaluate_payload(self, payload):
        payload = np.asarray(payload, dtype=float).reshape(-1)
        z = float(payload[0])
        y_solver = payload[1:]
        rhs = _column_rhs(
            z,
            y_solver,
            self.parameters,
            transform_mode=self.transform_mode,
            guard_rhs=self.guard_rhs,
        )
        return _physical_rhs_to_solver_rhs(y_solver, rhs, self.transform_mode)

    def has_jacobian(self):
        return self.diagnostic_fd

    def get_jacobian(self, name, _inames, _onames, opts):
        derivative = _FiniteDifferenceJacobian(self, name, opts)
        self.derivative_callbacks.append(derivative)
        return derivative


class _FiniteDifferenceJacobian(ca.Callback):
    """Diagnostic-only Callback Jacobian; Issue #22 does not admit this."""

    def __init__(self, parent, name, opts):
        self.parent = parent
        super().__init__()
        self.construct(f"{name}_diagnostic_fd", opts)

    def get_n_in(self):
        return 2  # nominal callback input and nominal callback output

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, index):
        return self.parent.get_sparsity_in(0) if index == 0 else self.parent.get_sparsity_out(0)

    def get_sparsity_out(self, _index):
        return ca.Sparsity.dense(7, 8)

    def eval(self, arguments):
        self.parent.jacobian_evaluations += 1
        payload = np.asarray(arguments[0], dtype=float).reshape(-1)
        jacobian = np.empty((7, 8), dtype=float)
        step = np.sqrt(np.finfo(float).eps) * (1.0 + np.abs(payload))
        for index, h in enumerate(step):
            plus = payload.copy()
            minus = payload.copy()
            plus[index] += h
            minus[index] -= h
            jacobian[:, index] = (
                self.parent.evaluate_payload(plus) - self.parent.evaluate_payload(minus)
            ) / (2.0 * h)
        return [ca.DM(jacobian)]


def _capture_scipy_run(dataframe, data_type: str, case_id: str) -> dict[str, object]:
    captured: dict[str, object] = {}
    original = run_model_module.scipy_BVP_solve

    def wrapped(y_a, y_b, z, parameters, settings=None):
        captured.update(
            parameters=parameters,
            y_a=np.asarray(y_a, dtype=float).copy(),
            y_b=np.asarray(y_b, dtype=float).copy(),
            z=np.asarray(z, dtype=float).copy(),
            settings=dict(settings or {}),
        )
        solved = original(y_a, y_b, z, parameters, settings=settings)
        captured["reference_profile_scaled"] = np.asarray(solved[0], dtype=float).copy()
        captured["reference_z"] = np.asarray(solved[1], dtype=float).copy()
        return solved

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
                "return_internal_profile": True,
            },
        )
    finally:
        run_model_module.scipy_BVP_solve = original
    result["probe_wall_runtime_s"] = time.perf_counter() - started
    result.pop("_profiles", None)
    result.pop("_raw_solution_scaled", None)
    return result | {"_captured": captured}


def _diagnostic_fd_ipopt(captured: dict[str, object], callback: _LiveColumnRhs) -> dict[str, object]:
    parameters = captured["parameters"]
    settings = captured["settings"]
    transform_mode = str(settings.get("transform_mode", "bounded_guarded_raw_state"))
    # scipy_BVP_solve returns its profile sampled on the original requested z;
    # the adaptive mesh is retained separately in its internal diagnostics.
    z_reference = np.asarray(captured["z"], dtype=float)
    profile_reference = np.asarray(captured["reference_profile_scaled"], dtype=float)
    grid = np.linspace(float(z_reference[0]), float(z_reference[-1]), 5)
    profile = np.vstack([
        np.interp(grid, z_reference, profile_reference[row])
        for row in range(profile_reference.shape[0])
    ])
    initial = np.column_stack([
        scaled_physical_to_solver(profile[:, index], transform_mode=transform_mode)
        for index in range(grid.size)
    ])

    states = ca.MX.sym("X", 7, grid.size)
    constraints = []
    for index in range(grid.size - 1):
        left = callback(ca.vertcat(float(grid[index]), states[:, index]))
        right = callback(ca.vertcat(float(grid[index + 1]), states[:, index + 1]))
        step = float(grid[index + 1] - grid[index])
        constraints.append(states[:, index + 1] - states[:, index] - 0.5 * step * (left + right))
    defect = ca.vertcat(*constraints)

    y_a = np.asarray(captured["y_a"], dtype=float)
    y_b = np.asarray(captured["y_b"], dtype=float)
    scales = np.asarray(parameters[0], dtype=float)
    bottom = _solver_to_scaled_physical_casadi(states[:, 0], transform_mode)
    top = _solver_to_scaled_physical_casadi(states[:, -1], transform_mode)
    boundary = ca.vertcat(
        (top[0] - y_b[0]) / scales[0],
        (top[1] - y_b[1]) / scales[1],
        (bottom[2] - y_a[2]) / scales[2],
        (bottom[3] - y_a[3]) / scales[3],
        (top[4] - y_b[4]) / scales[4],
        (bottom[5] - y_a[5]) / scales[5],
        (bottom[6] - y_a[6]) / scales[6],
    )
    nlp = {"x": ca.vec(states), "f": ca.DM(0), "g": ca.vertcat(defect, boundary)}
    solver = ca.nlpsol(
        "issue22_fd_diagnostic",
        "ipopt",
        nlp,
        {
            "ipopt.print_level": 0,
            "print_time": False,
            "ipopt.max_iter": 80,
            "ipopt.tol": 1.0e-7,
            "ipopt.acceptable_tol": 1.0e-5,
            # The disposable RHS callback supplies first derivatives only.
            # An admissible production path must either expose second-order
            # information or make this explicit Hessian choice part of its
            # solver contract.
            "ipopt.hessian_approximation": "limited-memory",
        },
    )
    started = time.perf_counter()
    try:
        solved = solver(
            x0=ca.vec(ca.DM(initial)),
            lbg=0.0,
            ubg=0.0,
        )
        stats = solver.stats()
        status = "solved" if bool(stats.get("success", False)) else "failed"
        solution = np.asarray(solved["x"], dtype=float).reshape(7, grid.size, order="F")
        residual = np.asarray(solved["g"], dtype=float).reshape(-1)
        return {
            "status": status,
            "runtime_s": time.perf_counter() - started,
            "ipopt_return_status": stats.get("return_status", ""),
            "ipopt_iterations": int(stats.get("iter_count", -1)),
            "primal_infeasibility_inf": float(np.linalg.norm(residual, ord=np.inf)),
            "defect_inf": float(np.linalg.norm(residual[: defect.shape[0]], ord=np.inf)),
            "boundary_inf": float(np.linalg.norm(residual[defect.shape[0] :], ord=np.inf)),
            "reference_initial_defect_inf": float(
                np.linalg.norm(np.asarray(ca.Function("defect", [states], [defect])(ca.DM(initial))).reshape(-1), ord=np.inf)
            ),
            "callback_evaluations": callback.evaluations,
            "callback_jacobian_evaluations": callback.jacobian_evaluations,
            "solution_state_inf_norm": float(np.linalg.norm(solution, ord=np.inf)),
            "derivative_status": "diagnostic_finite_difference_only",
            "admissible": False,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "runtime_s": time.perf_counter() - started,
            "ipopt_return_status": f"{type(exc).__name__}: {exc}".splitlines()[0],
            "ipopt_exception": str(exc),
            "callback_evaluations": callback.evaluations,
            "callback_jacobian_evaluations": callback.jacobian_evaluations,
            "derivative_status": "diagnostic_finite_difference_only",
            "admissible": False,
        }


def _casadi_attempt(run: dict[str, object], diagnostic_fd_ipopt: bool = False) -> dict[str, object]:
    captured = run.pop("_captured")
    parameters = captured["parameters"]
    settings = captured["settings"]
    callback = _LiveColumnRhs(
        parameters,
        str(settings.get("transform_mode", "bounded_guarded_raw_state")),
        bool(settings.get("guard_rhs", True)),
        diagnostic_fd=diagnostic_fd_ipopt,
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
    jacobian_status = "diagnostic_fd_checked" if jacobian_error is None else "missing_checked_derivative"

    result = {
        "callback_value_status": "evaluated",
        "callback_value_inf_norm": float(np.linalg.norm(value, ord=np.inf)),
        "callback_eval_runtime_s": callback_eval_s,
        "callback_has_jacobian": bool(callback.has_jacobian()),
        "callback_evaluations_after_transcription": callback.evaluations,
        "direct_collocation_grid_points": int(grid.size),
        "direct_collocation_state_dimension": 7,
        "direct_collocation_defect_count": int(defect.shape[0]),
        "ipopt_available": bool(ca.has_nlpsol("ipopt")),
        "jacobian_status": jacobian_status,
        "jacobian_error": jacobian_error,
        "ipopt_status": "not_attempted_missing_derivative" if jacobian_error else "available_diagnostic_only",
    }
    if diagnostic_fd_ipopt:
        result["diagnostic_fd_ipopt"] = _diagnostic_fd_ipopt(captured, callback)
    return result


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
    parser.add_argument(
        "--diagnostic-fd-ipopt",
        action="store_true",
        help="Run a non-admissible finite-difference Callback Jacobian through IPOPT.",
    )
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
                "casadi_probe": _casadi_attempt(run, diagnostic_fd_ipopt=args.diagnostic_fd_ipopt),
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
