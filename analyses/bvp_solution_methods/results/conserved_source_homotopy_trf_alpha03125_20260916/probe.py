"""Bounded exact-Jacobian trust-region probe for one artificial source stage.

This diagnostic owns no production behavior.  It starts from the retained
alpha=0.3046875 profile and evaluates the production trapezoidal graph at
alpha=0.3125 with SciPy's trust-region least-squares wrapper.  The retained
claim is numerical and artificial (alpha < 1) only.
"""
from __future__ import annotations

import json
import math
import signal
import time
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.optimize import least_squares

from mea_absorption_column.column import _prepare_conserved_column_in_process
from mea_absorption_column.config.column import resolve_column_config


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "analyses/bvp_solution_methods/results/conserved_source_homotopy_binary_barrier_20260916/accepted_alpha_0.304687500_profile.json"
SOURCE_RUN = ROOT / "analyses/bvp_solution_methods/results/conserved_source_homotopy_binary_barrier_20260916/stage_00_alpha_0.304687500/run.json"
TARGET = 0.3125
TOLERANCE = 1.0e-7


def clean(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else ("nan" if math.isnan(value) else ("inf" if value > 0 else "-inf"))
    return value


def save(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    source = json.loads(SOURCE.read_text())
    source_run = json.loads(SOURCE_RUN.read_text())
    settings = source_run["settings"]
    profile = np.asarray(source["profile"], dtype=float)
    grid = np.asarray(source["grid"], dtype=float)
    state_scale = np.asarray(settings["state_scale"], dtype=float)
    balance_scale = np.asarray(settings["balance_scale"], dtype=float)
    algebraic_scale = np.asarray(settings["algebraic_scale"], dtype=float)
    boundary_scale = np.asarray(settings["boundary_scale"], dtype=float)
    lower = np.asarray([float(v) for v in settings["lower"]], dtype=float)
    upper = np.asarray([float(v) for v in settings["upper"]], dtype=float)
    request = {
        "preset": "twelve_state_conserved",
        "case": {"physical_input_file": "analyses/bvp_solution_methods/input/case_3c.json"},
        "numerics": {"method": "trapezoidal", "nodes": 3},
    }
    prepared = _prepare_conserved_column_in_process(resolve_column_config(request))
    assembly = prepared["assembly"]
    node, boundary = assembly["node"], assembly["boundary"]
    native = {}

    def instrument(owner, name, label):
        original = getattr(owner, name)
        native[label] = {"started": 0, "returned": 0, "failed": 0, "wall_s": 0.0}

        def measured(*args, **kwargs):
            native[label]["started"] += 1
            started = time.perf_counter()
            try:
                value = original(*args, **kwargs)
                native[label]["returned"] += 1
                return value
            except Exception:
                native[label]["failed"] += 1
                raise
            finally:
                native[label]["wall_s"] += time.perf_counter() - started

        setattr(owner, name, measured)

    instrument(assembly["reactive_liquid"], "solve", "liquid_value_A1")
    instrument(assembly["reactive_liquid"], "solve_actions", "liquid_A2")
    instrument(assembly["vapor"], "_state", "vapor_value_A1_H2")
    n, nodes, m, q = 12, len(grid), 7, 5
    flat = ca.MX.sym("scaled_nodes_flat", n * nodes)
    variables = ca.reshape(flat, n, nodes)
    physical = ca.repmat(ca.DM(state_scale), 1, nodes) * variables
    evaluated = [node.call([ca.MX(float(z)), physical[:, k]], True, False) for k, z in enumerate(grid)]
    defects = ca.horzcat(*[
        evaluated[k + 1][0] - evaluated[k][0] - 0.5 * h * TARGET
        * (evaluated[k + 1][1] + evaluated[k][1])
        for k, h in enumerate(np.diff(grid))
    ])
    algebraic = ca.horzcat(*[value[2] for value in evaluated])
    boundary_residual = boundary(physical[:, 0], physical[:, -1])
    scaled = ca.vertcat(
        ca.vec(defects / (balance_scale[:, None] * np.diff(grid)[None, :])),
        ca.vec(algebraic / ca.repmat(ca.DM(algebraic_scale), 1, nodes)),
        boundary_residual / ca.DM(boundary_scale),
    )
    values = ca.Function("trf_source_values", [flat], [defects, algebraic, boundary_residual, scaled], {"cse": True})
    jacobian = ca.Function("trf_source_jacobian", [flat], [ca.jacobian(scaled, flat)], {"cse": True})
    x0 = (profile / state_scale[:, None]).reshape(-1, order="F")
    bounds = (np.tile(lower / state_scale, nodes), np.tile(upper / state_scale, nodes))
    checkpoint = {
        "schema_version": 1,
        "probe_kind": "retained_exact_jacobian_trf_source_stage",
        "source_profile": str(SOURCE),
        "source_profile_source_multiplier": 0.3046875,
        "target_source_multiplier": TARGET,
        "settings": {"method": "trapezoidal", "nodes": nodes, "film_points": 3,
                      "state_scale": state_scale, "balance_scale": balance_scale,
                      "algebraic_scale": algebraic_scale, "boundary_scale": boundary_scale,
                      "lower": lower, "upper": upper, "tolerance": TOLERANCE,
                      "scipy_method": "trf", "max_nfev": 20, "jacobian": "exact CasADi",
                      "equations": "production scaled trapezoidal residual at target alpha"},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evaluations": [], "native_calls": native, "status": "running",
        "best": {"residual_inf": float("inf"), "x": x0},
    }
    save(OUT / "checkpoint.json", checkpoint)

    def record(kind, x, value=None, error=None):
        row = {"kind": kind, "elapsed_s": time.perf_counter() - started,
               "native_calls": native.copy(), "error": error}
        if value is not None:
            residual = np.asarray(value, dtype=float).ravel()
            row.update(residual_inf=float(np.max(np.abs(residual))),
                       residual_l2=float(np.linalg.norm(residual)),
                       top_rows=sorted(({"index": int(i), "value": float(residual[i])}
                                        for i in range(residual.size)),
                                       key=lambda item: abs(item["value"]), reverse=True)[:8])
            if row["residual_inf"] < checkpoint["best"]["residual_inf"]:
                checkpoint["best"] = {"residual_inf": row["residual_inf"], "x": np.asarray(x, dtype=float)}
        checkpoint["evaluations"].append(row)
        checkpoint["native_calls"] = native
        save(OUT / "checkpoint.json", checkpoint)

    def residual(x):
        try:
            value = np.asarray(values(x)[3], dtype=float).ravel()
            record("residual", x, value=value)
            return value
        except BaseException as error:
            record("residual", x, error=f"{type(error).__name__}: {error}")
            raise

    def jac(x):
        try:
            value = np.asarray(jacobian(x), dtype=float)
            row = {"kind": "jacobian", "elapsed_s": time.perf_counter() - started,
                   "shape": list(value.shape), "native_calls": native.copy(),
                   "jacobian_inf": float(np.max(np.abs(value)))}
            checkpoint["evaluations"].append(row)
            checkpoint["native_calls"] = native
            save(OUT / "checkpoint.json", checkpoint)
            return value
        except BaseException as error:
            record("jacobian", x, error=f"{type(error).__name__}: {error}")
            raise

    def interrupted(_signum, _frame):
        checkpoint["status"] = "interrupted"
        checkpoint["interrupted_at_s"] = time.perf_counter() - started
        save(OUT / "checkpoint.json", checkpoint)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupted)
    started = time.perf_counter()
    result = None
    termination = None
    try:
        result = least_squares(residual, x0, jac=jac, bounds=bounds, method="trf",
                               ftol=TOLERANCE, xtol=TOLERANCE, gtol=TOLERANCE,
                               max_nfev=20, verbose=0)
        termination = {"status": int(result.status), "message": result.message,
                       "success": bool(result.success), "nfev": int(result.nfev),
                       "njev": int(result.njev), "optimality": float(result.optimality),
                       "cost": float(result.cost)}
    except BaseException as error:
        termination = {"exception": f"{type(error).__name__}: {error}"}
    wall = time.perf_counter() - started
    if result is None:
        candidate_x = checkpoint["best"]["x"]
    else:
        candidate_x = np.asarray(result.x, dtype=float)
    candidate = candidate_x.reshape((n, nodes), order="F") * state_scale[:, None]
    raw_defect, raw_algebraic, raw_boundary, scaled_value = (np.asarray(value, dtype=float)
                                                              for value in values(candidate_x))
    violation = np.maximum(np.maximum(lower[:, None] - candidate, candidate - upper[:, None]), 0.0)
    scaled_inf = float(np.max(np.abs(scaled_value)))
    bound_inf = float(np.max(violation / state_scale[:, None]))
    final = {
        "schema_version": 1,
        "probe_kind": checkpoint["probe_kind"],
        "source_profile": str(SOURCE),
        "target_source_multiplier": TARGET,
        "claim_limit": "Artificial source-homotopy diagnostic only; alpha < 1; no full-physics thermodynamic, transfer, or certified column claim.",
        "settings": checkpoint["settings"],
        "termination": termination,
        "wall_s": wall,
        "native_calls": native,
        "evaluation_count": len(checkpoint["evaluations"]),
        "candidate_profile": candidate,
        "original_residuals": {"defects": raw_defect, "algebraic": raw_algebraic,
                                "boundary": raw_boundary, "scaled": scaled_value},
        "original_residual_inf": scaled_inf,
        "original_bound_violation_inf": bound_inf,
        "finite": bool(np.all(np.isfinite(candidate)) and np.all(np.isfinite(scaled_value))),
        "residual_pass": bool(np.isfinite(scaled_inf) and scaled_inf <= TOLERANCE),
        "bounds_pass": bool(np.isfinite(bound_inf) and bound_inf <= TOLERANCE),
        "accepted": bool(result is not None and result.success and np.isfinite(scaled_inf)
                          and scaled_inf <= TOLERANCE and np.isfinite(bound_inf)
                          and bound_inf <= TOLERANCE),
        "evaluation_trajectory": checkpoint["evaluations"],
    }
    checkpoint.update(status=("completed" if result is not None and "exception" not in termination
                              else "finalized_after_external_timeout"), wall_s=wall, termination=termination,
                      final_original_residual_inf=scaled_inf,
                      final_original_bound_violation_inf=bound_inf)
    save(OUT / "checkpoint.json", checkpoint)
    save(OUT / "attempt.json", final)
    print(json.dumps({"accepted": final["accepted"], "termination": termination,
                      "wall_s": wall, "original_residual_inf": scaled_inf,
                      "original_bound_violation_inf": bound_inf,
                      "native_calls": native}, indent=2), flush=True)


if __name__ == "__main__":
    main()
