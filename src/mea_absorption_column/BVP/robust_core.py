from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


STATE_SIZE = 7
FLOW_IDXS = np.array([0, 1, 2, 3], dtype=int)
ENTHALPY_IDXS = np.array([4, 5], dtype=int)
PRESSURE_IDX = 6
POSITIVE_SOLVER_IDXS = np.array([0, 1, 2, 3, 6], dtype=int)
POSITIVE_TRANSFORM_FLOOR = 1.0e-12
POSITIVE_TRANSFORM_CEILING = 100.0


@dataclass(frozen=True)
class BoundedStateSettings:
    flow_floor: float = 1.0e-10
    pressure_floor: float = 1.0e3
    enthalpy_abs_limit: float = 1.0e10


def make_solver_diagnostics() -> dict:
    return {
        "invalid_state_count": 0,
        "last_invalid_state": "",
        "jacobian_status": "",
        "domain_guard_counts": {},
        "first_failed_domain": "",
    }


def bounded_to_unbounded_positive(values):
    arr = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("Positive bounded variables must be finite and greater than zero.")
    if np.any(arr >= POSITIVE_TRANSFORM_CEILING):
        raise ValueError("Positive bounded variables must be below the transform ceiling.")
    ratio = (arr - POSITIVE_TRANSFORM_FLOOR) / (POSITIVE_TRANSFORM_CEILING - POSITIVE_TRANSFORM_FLOOR)
    ratio = np.clip(ratio, 1.0e-15, 1.0 - 1.0e-15)
    return np.log(ratio / (1.0 - ratio))


def unbounded_to_positive(values):
    sigmoid = _stable_sigmoid(np.asarray(values, dtype=float))
    return POSITIVE_TRANSFORM_FLOOR + (POSITIVE_TRANSFORM_CEILING - POSITIVE_TRANSFORM_FLOOR) * sigmoid


def _stable_sigmoid(values):
    arr = np.asarray(values, dtype=float)
    out = np.empty_like(arr, dtype=float)
    positive = arr >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-arr[positive]))
    exp_values = np.exp(arr[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def _bounded_positive_derivative(values):
    sigmoid = _stable_sigmoid(values)
    return (POSITIVE_TRANSFORM_CEILING - POSITIVE_TRANSFORM_FLOOR) * sigmoid * (1.0 - sigmoid)


def scaled_physical_to_solver(y_scaled, transform_mode="bounded_guarded_raw_state"):
    y_scaled = np.asarray(y_scaled, dtype=float)
    if transform_mode in {None, "", "none", "bounded_guarded_raw_state", "raw"}:
        return y_scaled.copy()
    if transform_mode == "positive_flow_pressure":
        transformed = y_scaled.copy()
        transformed[POSITIVE_SOLVER_IDXS] = bounded_to_unbounded_positive(transformed[POSITIVE_SOLVER_IDXS])
        return transformed
    raise ValueError(f"Unknown transform_mode: {transform_mode}")


def solver_to_scaled_physical(y_solver, transform_mode="bounded_guarded_raw_state"):
    y_solver = np.asarray(y_solver, dtype=float)
    if transform_mode in {None, "", "none", "bounded_guarded_raw_state", "raw"}:
        return y_solver.copy()
    if transform_mode == "positive_flow_pressure":
        physical = y_solver.copy()
        physical[POSITIVE_SOLVER_IDXS] = unbounded_to_positive(physical[POSITIVE_SOLVER_IDXS])
        return physical
    raise ValueError(f"Unknown transform_mode: {transform_mode}")


def solver_to_scaled_physical_derivative(y_solver, transform_mode="bounded_guarded_raw_state"):
    y_solver = np.asarray(y_solver, dtype=float)
    derivative = np.ones_like(y_solver, dtype=float)
    if transform_mode in {None, "", "none", "bounded_guarded_raw_state", "raw"}:
        return derivative
    if transform_mode == "positive_flow_pressure":
        derivative[POSITIVE_SOLVER_IDXS] = np.maximum(
            _bounded_positive_derivative(y_solver[POSITIVE_SOLVER_IDXS]),
            POSITIVE_TRANSFORM_FLOOR,
        )
        return derivative
    raise ValueError(f"Unknown transform_mode: {transform_mode}")


def solver_profile_to_scaled_physical(profile, transform_mode="bounded_guarded_raw_state"):
    arr = np.asarray(profile, dtype=float)
    if arr.ndim == 1:
        return solver_to_scaled_physical(arr, transform_mode=transform_mode)
    return np.column_stack(
        [solver_to_scaled_physical(arr[:, i], transform_mode=transform_mode) for i in range(arr.shape[1])]
    )


def validate_scaled_state(y_scaled, scales, settings: BoundedStateSettings | None = None):
    settings = settings or BoundedStateSettings()
    state = np.asarray(y_scaled, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if state.shape != (STATE_SIZE,) or scales.shape != (STATE_SIZE,):
        raise ValueError(f"Expected {STATE_SIZE} state variables and scales")
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("State scales must be positive and finite")
    physical = state * scales
    if np.any(~np.isfinite(physical)):
        raise ValueError("Non-finite column state")
    if np.any(physical[FLOW_IDXS] <= settings.flow_floor):
        raise ValueError("Molar flow at or below its physical lower bound")
    if np.any(np.abs(physical[ENTHALPY_IDXS]) > settings.enthalpy_abs_limit):
        raise ValueError("Enthalpy flow outside its physical bounds")
    if physical[PRESSURE_IDX] <= settings.pressure_floor:
        raise ValueError("Pressure at or below its physical lower bound")
    return state.copy()


def record_invalid_state(diagnostics: dict | None, reason: str):
    if diagnostics is None:
        return
    diagnostics["invalid_state_count"] = int(diagnostics.get("invalid_state_count", 0)) + 1
    diagnostics["last_invalid_state"] = str(reason)


def record_domain_guard(diagnostics: dict | None, domain: str, reason: str):
    if diagnostics is None:
        return
    counts = diagnostics.setdefault("domain_guard_counts", {})
    counts[domain] = int(counts.get(domain, 0)) + 1
    if not diagnostics.get("first_failed_domain"):
        diagnostics["first_failed_domain"] = domain
    record_invalid_state(diagnostics, f"{domain}: {reason}")


def guard_column_rhs(
    zi: float,
    y_scaled,
    parameters,
    evaluator: Callable,
    run_type: str = "simulating",
    column_names: bool = False,
):
    scales = np.asarray(parameters[0], dtype=float)
    model_options = parameters[6] if len(parameters) > 6 else {}
    diagnostics = model_options.get("solver_diagnostics") if isinstance(model_options, dict) else None
    settings = model_options.get("bounded_state_settings", BoundedStateSettings()) if isinstance(model_options, dict) else BoundedStateSettings()

    try:
        state = validate_scaled_state(y_scaled, scales, settings)
        rhs = evaluator(zi, state, parameters, run_type=run_type, column_names=column_names)
        rhs_arr = np.asarray(rhs, dtype=float)
        if np.any(~np.isfinite(rhs_arr)):
            raise FloatingPointError("non-finite column RHS")
        return rhs
    except Exception as exc:
        record_invalid_state(diagnostics, str(exc))
        raise
