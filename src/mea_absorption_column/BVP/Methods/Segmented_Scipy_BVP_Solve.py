from __future__ import annotations

import numpy as np
import time
from scipy.integrate import solve_bvp

from mea_absorption_column.BVP.ABS_Column import abs_column
from mea_absorption_column.BVP.robust_core import (
    guard_column_rhs,
    scaled_physical_to_solver,
    solver_profile_to_scaled_physical,
    solver_to_scaled_physical,
    solver_to_scaled_physical_derivative,
)
from mea_absorption_column.BVP.Methods.Scipy_BVP_Solve import DEFAULT_SCIPY_BVP_SETTINGS
from mea_absorption_column.misc.Polynomial_Fit import polynomial_fit
from mea_absorption_column.Properties.Thermophysical_Properties import enthalpy
from mea_absorption_column.intercooling import (
    BedStackSpec,
    liquid_enthalpy_after_intercooler,
)


STATE_SIZE = 7
LIQUID_IDXS = np.array([0, 1, 4])
VAPOR_IDXS = np.array([2, 3, 5, 6])
BOUNDED_CASE_IDXS = np.array([0, 1, 2, 3, 6])
BOUNDED_CASE_WITH_TEMPERATURE_IDXS = np.array([0, 1, 2, 3, 4, 5, 6])


def _slice_bed(vector: np.ndarray, bed_index: int) -> np.ndarray:
    start = bed_index * STATE_SIZE
    return vector[start:start + STATE_SIZE]


def _intercooler_for_upper_bed(stack_spec: BedStackSpec, upper_bed_index: int):
    for cooler in stack_spec.intercoolers:
        if cooler.below_upper_bed_index == upper_bed_index:
            return cooler
    return None


def stacked_boundary_conditions(
    bottom_scaled,
    top_scaled,
    y_bottom_target_scaled,
    y_top_target_scaled,
    scales,
    fl_mea,
    stack_spec: BedStackSpec,
    thermal_state_mode="enthalpy",
):
    bottom_scaled = np.asarray(bottom_scaled, dtype=float)
    top_scaled = np.asarray(top_scaled, dtype=float)
    y_bottom_target_scaled = np.asarray(y_bottom_target_scaled, dtype=float)
    y_top_target_scaled = np.asarray(y_top_target_scaled, dtype=float)
    scales = np.asarray(scales, dtype=float)

    residuals = []

    bottom_bed_bottom = _slice_bed(bottom_scaled, 0)
    top_bed_top = _slice_bed(top_scaled, stack_spec.beds - 1)
    residuals.extend(bottom_bed_bottom[VAPOR_IDXS] - y_bottom_target_scaled[VAPOR_IDXS])
    residuals.extend(top_bed_top[LIQUID_IDXS] - y_top_target_scaled[LIQUID_IDXS])

    for lower_bed in range(stack_spec.beds - 1):
        upper_bed = lower_bed + 1
        lower_top = _slice_bed(top_scaled, lower_bed)
        upper_bottom = _slice_bed(bottom_scaled, upper_bed)

        residuals.extend(lower_top[VAPOR_IDXS] - upper_bottom[VAPOR_IDXS])

        liquid_from_upper = upper_bottom.copy() * scales
        cooler = _intercooler_for_upper_bed(stack_spec, upper_bed)
        if cooler is not None:
            if thermal_state_mode == "temperature":
                liquid_from_upper[4] = (
                    (1.0 - cooler.strength) * liquid_from_upper[4]
                    + cooler.strength * cooler.target_temperature_K
                )
            else:
                cooled = liquid_enthalpy_after_intercooler(
                    liquid_from_upper,
                    fl_mea=fl_mea,
                    target_temperature_K=cooler.target_temperature_K,
                )
                liquid_from_upper[4] = (
                    (1.0 - cooler.strength) * liquid_from_upper[4]
                    + cooler.strength * cooled[4]
                )
        liquid_from_upper_scaled = liquid_from_upper / scales
        residuals.extend(lower_top[LIQUID_IDXS] - liquid_from_upper_scaled[LIQUID_IDXS])

    return np.asarray(residuals, dtype=float)


def stacked_boundary_jacobian(bottom_scaled, scales, fl_mea, stack_spec, thermal_state_mode="enthalpy"):
    """Exact derivatives of the existing stream/cooler boundary equations."""
    scales = np.asarray(scales, dtype=float)
    left = np.zeros((STATE_SIZE * stack_spec.beds,) * 2)
    right = np.zeros_like(left)
    left[np.arange(4), VAPOR_IDXS] = 1.0
    right[np.arange(4, 7), (stack_spec.beds - 1) * STATE_SIZE + LIQUID_IDXS] = 1.0
    for lower_bed in range(stack_spec.beds - 1):
        lower = lower_bed * STATE_SIZE
        upper = lower + STATE_SIZE
        rows = STATE_SIZE + lower
        right[rows + np.arange(4), lower + VAPOR_IDXS] = 1.0
        left[rows + np.arange(4), upper + VAPOR_IDXS] = -1.0
        right[rows + np.arange(4, 7), lower + LIQUID_IDXS] = 1.0
        left[rows + np.arange(4, 7), upper + LIQUID_IDXS] = -1.0
        cooler = _intercooler_for_upper_bed(stack_spec, lower_bed + 1)
        if cooler is not None:
            left[rows + 6, upper + 4] = -(1.0 - cooler.strength)
            if thermal_state_mode != "temperature":
                state = _slice_bed(bottom_scaled, lower_bed + 1) * scales
                flows = np.array([state[0], fl_mea, state[1]])
                partial_h, _ = enthalpy(cooler.target_temperature_K, flows / flows.sum(), phase="liquid")
                left[rows + 6, upper + np.array([0, 1])] = (
                    -cooler.strength * partial_h[[0, 2]] * scales[[0, 1]] / scales[4]
                )
    return left, right


def _stack_initial_guess(
    Y_a_scaled,
    Y_b_scaled,
    mesh_points,
    beds,
    explicit_initial_guess=None,
    scales=None,
    thermal_state_mode="enthalpy",
    initial_guess_z=None,
):
    expected_shape = (STATE_SIZE * int(beds), int(mesh_points))
    if explicit_initial_guess is not None:
        guess = np.asarray(explicit_initial_guess, dtype=float)
        if guess.ndim != 2 or guess.shape[1] < 2 or not np.all(np.isfinite(guess)):
            raise ValueError("explicit_initial_guess must be a finite state profile")
        old_grid = np.asarray(
            np.linspace(0.0, 1.0, guess.shape[1]) if initial_guess_z is None else initial_guess_z,
            dtype=float,
        )
        if (old_grid.shape != (guess.shape[1],) or not np.all(np.isfinite(old_grid))
                or np.any(np.diff(old_grid) <= 0.0) or old_grid[0] > 0.0 or old_grid[-1] < 1.0):
            raise ValueError("initial_guess_z must strictly increase and cover [0, 1]")
        if guess.shape == expected_shape and initial_guess_z is None:
            return guess
        if guess.shape[0] == STATE_SIZE and guess.shape[1] > 1:
            blocks = []
            for bed_index in range(int(beds)):
                new_grid = np.linspace(
                    bed_index / int(beds),
                    (bed_index + 1) / int(beds),
                    expected_shape[1],
                )
                blocks.append(np.vstack([np.interp(new_grid, old_grid, row) for row in guess]))
            return np.vstack(blocks)
        if guess.shape[0] == expected_shape[0] and guess.shape[1] > 1:
            new_grid = np.linspace(0.0, 1.0, expected_shape[1])
            return np.vstack([np.interp(new_grid, old_grid, row) for row in guess])
        raise ValueError("explicit_initial_guess has the wrong shape.")

    Y_a_scaled = np.asarray(Y_a_scaled, dtype=float)
    Y_b_scaled = np.asarray(Y_b_scaled, dtype=float)
    if int(beds) == 1 and scales is not None and thermal_state_mode != "temperature":
        z_values = np.linspace(0.0, 1.0, int(mesh_points))
        scales = np.asarray(scales, dtype=float)
        return np.asarray(
            [
                np.asarray(polynomial_fit(z_values, Y_a_scaled[i] * scales[i], i), dtype=float) / scales[i]
                for i in range(STATE_SIZE)
            ]
        )
    blocks = []
    for bed_index in range(int(beds)):
        global_z = np.linspace(
            bed_index / int(beds),
            (bed_index + 1) / int(beds),
            int(mesh_points),
        )
        block = _linear_profile(Y_a_scaled, Y_b_scaled, global_z)
        blocks.append(block)
    return np.vstack(blocks)


def _linear_profile(Y_a_scaled, Y_b_scaled, z_values):
    return np.vstack(
        [
            Y_a_scaled[i] + (Y_b_scaled[i] - Y_a_scaled[i]) * z_values
            for i in range(STATE_SIZE)
        ]
    )


def external_profile_from_stacked_solution(stacked_y, beds):
    stacked_y = np.asarray(stacked_y, dtype=float)
    beds = int(beds)
    if beds == 1:
        return stacked_y
    return np.hstack(
        [
            stacked_y[bed_index * STATE_SIZE:(bed_index + 1) * STATE_SIZE, :]
            for bed_index in range(beds)
        ]
    )


def segmented_scipy_BVP_solve(
    Y_a_scaled,
    Y_b_scaled,
    z,
    parameters,
    stack_spec: BedStackSpec,
    settings=None,
):
    settings = {**DEFAULT_SCIPY_BVP_SETTINGS, **(settings or {})}
    scales, eq_scales, const_flow, H, A, packing, model_options = parameters
    fl_mea = const_flow[0]
    thermal_state_mode = model_options.get("thermal_state_mode", "enthalpy") if isinstance(model_options, dict) else "enthalpy"
    bed_parameters = (
        scales,
        eq_scales,
        const_flow,
        stack_spec.single_bed_height_m,
        A,
        packing,
        model_options,
    )

    mesh_points = int(settings["mesh_points"])
    max_runtime_s = settings.get("max_runtime_s")
    start_time = time.monotonic()
    transform_mode = settings.get("transform_mode", "bounded_guarded_raw_state")
    case_bounds = (
        _case_state_bounds_scaled(
            Y_a_scaled,
            Y_b_scaled,
            scales,
            thermal_state_mode=thermal_state_mode,
            co2_vapor_upper_factor=float(settings.get("co2_vapor_upper_factor", 1.0)),
        )
        if transform_mode == "case_bounded_flow_pressure"
        else None
    )
    z_mesh = np.linspace(z[0], z[-1], mesh_points)
    y_guess = _stack_initial_guess(
        Y_a_scaled,
        Y_b_scaled,
        mesh_points,
        stack_spec.beds,
        explicit_initial_guess=settings.get("initial_guess_scaled"),
        scales=scales,
        thermal_state_mode=thermal_state_mode,
        initial_guess_z=settings.get("initial_guess_z"),
    )
    y_guess_solver = _stacked_profile_to_solver(y_guess, stack_spec.beds, transform_mode, bounds=case_bounds)

    def check_runtime():
        if max_runtime_s is not None and time.monotonic() - start_time > float(max_runtime_s):
            raise TimeoutError(f"Segmented SciPy BVP exceeded max_runtime_s={float(max_runtime_s):g}")

    def column_odes(z_values, stacked_y):
        check_runtime()
        blocks = []
        for bed_index in range(stack_spec.beds):
            start = bed_index * STATE_SIZE
            bed_y_solver = stacked_y[start:start + STATE_SIZE, :]
            differentials = [
                _physical_rhs_to_solver_rhs(
                    bed_y_solver[:, i],
                    guard_column_rhs(
                        z_values[i],
                        _vector_to_scaled_physical(bed_y_solver[:, i], transform_mode=transform_mode, bounds=case_bounds),
                        bed_parameters,
                        evaluator=abs_column,
                    ),
                    transform_mode,
                    bounds=case_bounds,
                )
                for i in range(bed_y_solver.shape[1])
            ]
            blocks.append(np.asarray(differentials).T)
        return np.vstack(blocks)

    def boundary(bottom, top):
        check_runtime()
        return stacked_boundary_conditions(
            bottom_scaled=_stacked_vector_to_physical(bottom, stack_spec.beds, transform_mode, bounds=case_bounds),
            top_scaled=_stacked_vector_to_physical(top, stack_spec.beds, transform_mode, bounds=case_bounds),
            y_bottom_target_scaled=np.asarray(Y_a_scaled, dtype=float),
            y_top_target_scaled=np.asarray(Y_b_scaled, dtype=float),
            scales=scales,
            fl_mea=fl_mea,
            stack_spec=stack_spec,
            thermal_state_mode=thermal_state_mode,
        )

    def boundary_jacobian(bottom, top):
        physical_bottom = _stacked_vector_to_physical(bottom, stack_spec.beds, transform_mode, bounds=case_bounds)
        left, right = stacked_boundary_jacobian(
            physical_bottom, scales, fl_mea, stack_spec, thermal_state_mode,
        )
        for matrix, state in ((left, bottom), (right, top)):
            derivative = np.concatenate([
                _vector_derivative(_slice_bed(state, bed), transform_mode, bounds=case_bounds)
                for bed in range(stack_spec.beds)
            ])
            matrix *= derivative[np.newaxis, :]
        return left, right

    if 'use_finite_jacobian' in settings:
        raise ValueError("use_finite_jacobian was removed; SciPy owns numerical Jacobian estimation")
    try:
        sol = solve_bvp(
            column_odes,
            boundary,
            z_mesh,
            y_guess_solver,
            max_nodes=int(settings["max_nodes"]),
            tol=float(settings["tol"]),
            bc_tol=float(settings["bc_tol"]),
            verbose=int(settings["verbose"]),
            bc_jac=boundary_jacobian,
        )
    except TimeoutError as exc:
        if isinstance(model_options, dict):
            model_options.get("solver_diagnostics", {})["jacobian_status"] = "timeout"
        return y_guess, z_mesh, "Segmented SciPy collocation-style BVP", False, str(exc)
    if isinstance(model_options, dict):
        model_options.get("solver_diagnostics", {})["jacobian_status"] = str(sol.status)

    return _stacked_profile_to_physical(sol.sol(z), stack_spec.beds, transform_mode, bounds=case_bounds), sol.x, "Segmented SciPy collocation-style BVP", sol.success, sol.message


def _physical_rhs_to_solver_rhs(y_solver, rhs_physical, transform_mode, bounds=None):
    derivative = _vector_derivative(y_solver, transform_mode=transform_mode, bounds=bounds)
    if np.any(~np.isfinite(derivative)) or np.any(derivative <= 0.0):
        raise FloatingPointError("singular state transform: trial reached a coordinate bound")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        rhs = np.asarray(rhs_physical, dtype=float) / derivative
    if np.any(~np.isfinite(rhs)):
        raise FloatingPointError("non-finite transformed column RHS")
    return rhs


def _stacked_vector_to_physical(vector, beds, transform_mode, bounds=None):
    return np.concatenate(
        [
            _vector_to_scaled_physical(_slice_bed(vector, bed_index), transform_mode=transform_mode, bounds=bounds)
            for bed_index in range(int(beds))
        ]
    )


def _stacked_profile_to_physical(profile, beds, transform_mode, bounds=None):
    return np.vstack(
        [
            _profile_to_scaled_physical(
                profile[bed_index * STATE_SIZE:(bed_index + 1) * STATE_SIZE, :],
                transform_mode=transform_mode,
                bounds=bounds,
            )
            for bed_index in range(int(beds))
        ]
    )


def _stacked_profile_to_solver(profile, beds, transform_mode, bounds=None):
    return np.vstack(
        [
            np.column_stack(
                [
                    _scaled_physical_to_solver(
                        profile[bed_index * STATE_SIZE:(bed_index + 1) * STATE_SIZE, i],
                        transform_mode=transform_mode,
                        bounds=bounds,
                    )
                    for i in range(profile.shape[1])
                ]
            )
            for bed_index in range(int(beds))
        ]
    )


def _case_state_bounds_scaled(
    Y_a_scaled,
    Y_b_scaled,
    scales,
    thermal_state_mode="enthalpy",
    co2_vapor_upper_factor=1.0,
):
    y_a = np.asarray(Y_a_scaled, dtype=float)
    y_b = np.asarray(Y_b_scaled, dtype=float)
    scales = np.asarray(scales, dtype=float)
    bounded_idxs = BOUNDED_CASE_WITH_TEMPERATURE_IDXS if thermal_state_mode == "temperature" else BOUNDED_CASE_IDXS
    lower = np.full(STATE_SIZE, -np.inf, dtype=float)
    upper = np.full(STATE_SIZE, np.inf, dtype=float)

    floor = np.maximum(1.0e-10 / np.maximum(scales, 1.0e-30), 1.0e-12)
    lower[BOUNDED_CASE_IDXS] = floor[BOUNDED_CASE_IDXS]

    fl_co2_upper_unscaled = max(y_a[0] * scales[0], y_b[0] * scales[0]) + 1.2 * y_a[2] * scales[2]
    fl_h2o_upper_unscaled = max(y_a[1] * scales[1], y_b[1] * scales[1]) + 3.0 * max(y_a[3] * scales[3], y_b[3] * scales[3])
    fv_co2_upper_unscaled = max(float(co2_vapor_upper_factor), 1.0) * y_a[2] * scales[2]
    fv_h2o_upper_unscaled = 3.0 * max(y_a[3] * scales[3], y_b[3] * scales[3])
    pressure_upper_unscaled = 1.2 * max(y_a[6] * scales[6], y_b[6] * scales[6])

    upper[0] = max(fl_co2_upper_unscaled / scales[0], y_a[0], y_b[0], floor[0] * 10.0)
    upper[1] = max(fl_h2o_upper_unscaled / scales[1], y_a[1], y_b[1], floor[1] * 10.0)
    upper[2] = max(fv_co2_upper_unscaled / scales[2], y_a[2], y_b[2], floor[2] * 10.0)
    upper[3] = max(fv_h2o_upper_unscaled / scales[3], y_a[3], y_b[3], floor[3] * 10.0)
    upper[6] = max(pressure_upper_unscaled / scales[6], y_a[6], y_b[6], floor[6] * 10.0)
    if thermal_state_mode == "temperature":
        lower[4] = 250.0 / scales[4]
        lower[5] = 250.0 / scales[5]
        upper[4] = 500.0 / scales[4]
        upper[5] = 500.0 / scales[5]

    return lower, upper, bounded_idxs


def _scaled_physical_to_solver(y_scaled, transform_mode, bounds=None):
    if transform_mode == "case_bounded_flow_pressure":
        return _bounded_scaled_physical_to_solver(y_scaled, bounds)
    return scaled_physical_to_solver(y_scaled, transform_mode=transform_mode)


def _vector_to_scaled_physical(y_solver, transform_mode, bounds=None):
    if transform_mode == "case_bounded_flow_pressure":
        return _bounded_solver_to_scaled_physical(y_solver, bounds)
    return solver_to_scaled_physical(y_solver, transform_mode=transform_mode)


def _vector_derivative(y_solver, transform_mode, bounds=None):
    if transform_mode == "case_bounded_flow_pressure":
        return _bounded_solver_derivative(y_solver, bounds)
    return solver_to_scaled_physical_derivative(y_solver, transform_mode=transform_mode)


def _profile_to_scaled_physical(profile, transform_mode, bounds=None):
    if transform_mode == "case_bounded_flow_pressure":
        return np.column_stack(
            [_bounded_solver_to_scaled_physical(profile[:, i], bounds) for i in range(profile.shape[1])]
        )
    return solver_profile_to_scaled_physical(profile, transform_mode=transform_mode)


def _bounded_scaled_physical_to_solver(y_scaled, bounds):
    lower, upper, bounded_idxs = _bounds_parts(bounds)
    y_scaled = np.asarray(y_scaled, dtype=float)
    solver = y_scaled.copy()
    span = upper - lower
    ratio = (solver[bounded_idxs] - lower[bounded_idxs]) / span[bounded_idxs]
    ratio = np.clip(ratio, 1.0e-12, 1.0 - 1.0e-12)
    solver[bounded_idxs] = np.log(ratio / (1.0 - ratio))
    return solver


def _bounded_solver_to_scaled_physical(y_solver, bounds):
    lower, upper, bounded_idxs = _bounds_parts(bounds)
    y_solver = np.asarray(y_solver, dtype=float)
    physical = y_solver.copy()
    sigmoid = _stable_sigmoid(physical[bounded_idxs])
    physical[bounded_idxs] = lower[bounded_idxs] + (upper[bounded_idxs] - lower[bounded_idxs]) * sigmoid
    return physical


def _bounded_solver_derivative(y_solver, bounds):
    lower, upper, bounded_idxs = _bounds_parts(bounds)
    derivative = np.ones_like(np.asarray(y_solver, dtype=float))
    sigmoid = _stable_sigmoid(np.asarray(y_solver, dtype=float)[bounded_idxs])
    derivative[bounded_idxs] = (upper[bounded_idxs] - lower[bounded_idxs]) * sigmoid * (1.0 - sigmoid)
    return derivative


def _bounds_parts(bounds):
    if len(bounds) == 2:
        lower, upper = bounds
        bounded_idxs = BOUNDED_CASE_IDXS
    else:
        lower, upper, bounded_idxs = bounds
    return lower, upper, bounded_idxs


def _stable_sigmoid(values):
    arr = np.asarray(values, dtype=float)
    out = np.empty_like(arr, dtype=float)
    positive = arr >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-arr[positive]))
    exp_values = np.exp(arr[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out
