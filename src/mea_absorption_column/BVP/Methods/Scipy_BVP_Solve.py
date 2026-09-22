import numpy as np
import time
from scipy.integrate import solve_bvp
from ...BVP.ABS_Column import abs_column
from ...BVP.robust_core import (
    POSITIVE_SOLVER_IDXS,
    POSITIVE_TRANSFORM_CEILING,
    POSITIVE_TRANSFORM_FLOOR,
    guard_column_rhs,
    scaled_physical_to_solver,
    solver_profile_to_scaled_physical,
    solver_to_scaled_physical,
    solver_to_scaled_physical_derivative,
)
from ...misc.Polynomial_Fit import polynomial_fit
from ...Thermodynamics.Chemical_Equilibrium import chemical_equilibrium
EPS = np.finfo(float).eps


DEFAULT_SCIPY_BVP_SETTINGS = {
    'mesh_points': 51,
    'max_nodes': 1000,
    'tol': 5e-1,
    'bc_tol': 1e-3,
    'verbose': 0,
    'use_finite_jacobian': False,
}


def native_state_layout(scales, *, beds=1, thermal_state_mode="enthalpy", coordinate="normalized_height", representation="adaptive_collocation"):
    thermal_states = ("T_L", "T_V") if thermal_state_mode == "temperature" else ("H_L", "H_V")
    state_names = ("F_L_CO2", "F_L_H2O", "F_V_CO2", "F_V_H2O", *thermal_states, "P")
    scales = np.asarray(scales, dtype=float).copy()
    beds = int(beds)
    bed_states = [
        {
            "bed": bed_index + 1,
            "state_order": tuple(f"bed_{bed_index + 1}.{name}" for name in state_names),
            "state_scale": scales.copy(),
            "coordinate_sides": {"start": "bottom", "end": "top"},
        }
        for bed_index in range(beds)
    ]
    if beds == 1 and "stacked" not in representation:
        state_order = state_names
        state_scale = scales
    else:
        state_order = tuple(
            name
            for bed in bed_states
            for name in bed["state_order"]
        )
        state_scale = np.tile(scales, beds)
    return {
        "representation": representation,
        "coordinate": coordinate,
        "coordinate_sides": "bottom_to_top_per_bed" if beds > 1 else "bottom_to_top",
        "beds": beds,
        "state_order": state_order,
        "state_scale": state_scale,
        "bed_states": bed_states,
        "interface_sides": [
            {
                "interface": interface_index + 1,
                "lower": {"bed": interface_index + 1, "side": "top"},
                "upper": {"bed": interface_index + 2, "side": "bottom"},
            }
            for interface_index in range(max(0, beds - 1))
        ],
    }


def scipy_BVP_solve(Y_a_scaled, Y_b_scaled, z, parameters, settings=None):
    settings = {**DEFAULT_SCIPY_BVP_SETTINGS, **(settings or {})}
    Fl_CO2_a_guess, Fl_H2O_a_guess, Fv_CO2_a, Fv_H2O_a, Hlf_a_guess, Hvf_a, P_a = Y_a_scaled
    Fl_CO2_b, Fl_H2O_b, Fv_CO2_b_guess, Fv_H2O_b_guess, Hlf_b, Hvf_b_guess, P_b = Y_b_scaled

    scales = parameters[0]
    transform_mode = settings.get('transform_mode', 'bounded_guarded_raw_state')
    guard_rhs = bool(settings.get('guard_rhs', True))
    native_jacobian = None
    if settings.get('jacobian_mode') == 'native':
        from ..reactive_jacobian import ReactiveColumnJacobian
        native_jacobian = ReactiveColumnJacobian(parameters, transform_mode)
    started = time.perf_counter()
    evaluations = {'rhs_batches': 0, 'jacobian_batches': 0}

    def progress(kind, nodes):
        evaluations[kind] += 1
        if settings.get('verbose', 0):
            liquid = parameters[6].get('reactive_liquid')
            print(f"column {kind}={evaluations[kind]} nodes={nodes} elapsed={time.perf_counter()-started:.2f}s "
                  f"thermodynamics={liquid.stats if liquid is not None else {}}", flush=True)

    bcs_1 = np.array([Fl_CO2_b, Fl_H2O_b, Fv_CO2_a, Fv_H2O_a, Hlf_b, Hvf_a, P_a]) / scales

    # Define the system of differential equations for the absorption column
    def column_odes(z, w):
        progress('rhs_batches', w.shape[1])
        differentials = [
            _physical_rhs_to_solver_rhs(
                w[:, i],
                _column_rhs(
                    z[i],
                    w[:, i],
                    parameters,
                    transform_mode=transform_mode,
                    guard_rhs=guard_rhs,
                ),
                transform_mode,
            )
            for i in range(np.shape(w)[1])
        ]
        if hasattr(chemical_equilibrium, "cache"):
            del chemical_equilibrium.cache
        return np.array(differentials).T

    # Define the boundary conditions
    def boundary_conditions(bottom, top):
        # Enforce the boundary conditions at the bottom (vapor) and top (liquid)
        bottom = solver_to_scaled_physical(bottom, transform_mode=transform_mode)
        top = solver_to_scaled_physical(top, transform_mode=transform_mode)
        Fl_CO2_a_bc, Fl_H2O_a_bc, Fv_CO2_a_bc, Fv_H2O_a_bc, Hlf_a_bc, Hvf_a_bc, P_a_bc = bottom
        Fl_CO2_b_bc, Fl_H2O_b_bc, Fv_CO2_b_bc, Fv_H2O_b_bc, Hlf_b_bc, Hvf_b_bc, P_b_bc = top

        bcs_2 = np.array([Fl_CO2_b_bc, Fl_H2O_b_bc, Fv_CO2_a_bc, Fv_H2O_a_bc, Hlf_b_bc, Hvf_a_bc, P_a_bc]) / scales

        # Boundary conditions at the bottom for vapor and at the top for liquid
        return bcs_1 - bcs_2

    def fun_jac(x, y):
        progress('jacobian_batches', y.shape[1])
        if native_jacobian is not None:
            return np.stack([native_jacobian(x[i], y[:, i]) for i in range(y.shape[1])], axis=2)
        fun = column_odes
        n, m = y.shape

        dtype = y.dtype

        df_dy = np.empty((n, n, m), dtype=dtype)
        h = EPS ** 0.5 * (1 + np.abs(y))
        for i in range(n):
            y_new = y.copy()
            y_new2 = y.copy()
            y_new[i] += h[i]
            y_new2[i] -= h[i]
            hi = y_new[i] - y[i]
            f_new = fun(x, y_new)
            f_new2 = fun(x, y_new2)

            df_dy[:, i, :] = (f_new - f_new2) / (2*hi)

        return df_dy


    # Initial guess for the solution (constant profiles as initial guess)

    m = len(Y_a_scaled)
    n = int(settings['mesh_points'])
    z_2 = np.linspace(z[0], z[-1], n)
    w_guess_scaled = _initial_guess_profile(settings, z, z_2, Y_a_scaled, scales, m)
    if settings.get('thermal_state_mode') == 'temperature' and 'initial_guess_scaled' not in settings:
        # The retained polynomial coefficients describe enthalpy, not kelvin.
        for i in (4, 5):
            w_guess_scaled[i] = np.linspace(Y_a_scaled[i], Y_b_scaled[i], n)
    if transform_mode == "positive_flow_pressure":
        w_guess_scaled[POSITIVE_SOLVER_IDXS, :] = np.clip(
            w_guess_scaled[POSITIVE_SOLVER_IDXS, :],
            POSITIVE_TRANSFORM_FLOOR * 10.0,
            POSITIVE_TRANSFORM_CEILING * (1.0 - 1.0e-12),
        )
    w_guess_solver = np.column_stack(
        [scaled_physical_to_solver(w_guess_scaled[:, i], transform_mode=transform_mode) for i in range(w_guess_scaled.shape[1])]
    )


    # Solve the BVP

    jacobian_kwargs = {'fun_jac': fun_jac} if native_jacobian is not None or settings.get('use_finite_jacobian', False) else {}
    legacy_grid = np.asarray(z, dtype=float).copy()
    sol = solve_bvp(column_odes, boundary_conditions, z_2, w_guess_solver,
                    max_nodes=int(settings['max_nodes']),
                    tol=float(settings['tol']),
                    bc_tol=float(settings['bc_tol']),
                    verbose=int(settings['verbose']),
                    **jacobian_kwargs,
                    )
    native_grid = np.asarray(sol.x, dtype=float)
    native_state = solver_profile_to_scaled_physical(sol.sol(native_grid), transform_mode=transform_mode)
    if native_state.shape[1] != native_grid.size:
        raise RuntimeError("Adaptive native profile grid/state lengths disagree")
    native_profile = {
        "grid": native_grid,
        "state_matrix_scaled": np.asarray(native_state, dtype=float),
        "layout": native_state_layout(
            scales,
            thermal_state_mode=settings.get("thermal_state_mode", "enthalpy"),
        ),
    }
    if len(parameters) > 6 and isinstance(parameters[6], dict):
        diagnostics = parameters[6].setdefault("solver_diagnostics", {})
        diagnostics.setdefault("stage_status", {})["outer"] = {
            "status": "converged" if bool(getattr(sol, "success", False)) else "failed",
            "success": bool(getattr(sol, "success", False)),
            "message": str(getattr(sol, "message", "")),
            "status_code": int(getattr(sol, "status", 0)),
            "iterations": int(getattr(sol, "niter", 0)),
            "grid_points": int(native_grid.size),
        }
    Y_scaled = solver_profile_to_scaled_physical(sol.sol(legacy_grid), transform_mode=transform_mode)

    success = sol.success
    message = sol.message
    if len(parameters) > 6 and isinstance(parameters[6], dict):
        parameters[6].get("solver_diagnostics", {}).update(
            jacobian_status=str(sol.status),
            solver_iterations=int(sol.niter),
            final_mesh_nodes=int(native_grid.size),
            max_rms_residual=float(np.max(getattr(sol, "rms_residuals", [np.nan]))),
            max_scaled_boundary_residual=float(np.max(np.abs(boundary_conditions(sol.y[:, 0], sol.y[:, -1])))),
            legacy_grid=legacy_grid,
            native_profile=native_profile,
        )

    return Y_scaled, legacy_grid, 'SciPy collocation-style BVP', success, message


def _physical_rhs_to_solver_rhs(y_solver, rhs_physical, transform_mode):
    derivative = solver_to_scaled_physical_derivative(y_solver, transform_mode=transform_mode)
    return np.asarray(rhs_physical, dtype=float) / derivative


def _column_rhs(zi, y_solver, parameters, transform_mode, guard_rhs):
    y_scaled = solver_to_scaled_physical(y_solver, transform_mode=transform_mode)
    if guard_rhs:
        return guard_column_rhs(zi, y_scaled, parameters, evaluator=abs_column)
    return abs_column(zi, y_scaled, parameters)


def _initial_guess_profile(settings, z_source, z_target, Y_a_scaled, scales, m):
    explicit = settings.get("initial_guess_scaled")
    if explicit is not None:
        profile = np.asarray(explicit, dtype=float)
        if profile.ndim == 2 and profile.shape[0] == m and profile.shape[1] >= 2:
            source_grid = np.asarray(settings.get("initial_guess_z", np.linspace(z_source[0], z_source[-1], profile.shape[1])), dtype=float)
            if source_grid.shape[0] == profile.shape[1] and np.all(np.isfinite(profile)):
                return np.vstack([
                    np.interp(z_target, source_grid, profile[i])
                    for i in range(m)
                ])
    return np.array([polynomial_fit(z_target, Y_a_scaled[i] * scales[i], i) / scales[i] for i in range(m)])
