import numpy as np
from scipy.integrate import solve_bvp
from ...BVP.ABS_Column import abs_column
from ...BVP.robust_core import (
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


def scipy_BVP_solve(Y_a_scaled, Y_b_scaled, z, parameters, settings=None):
    settings = {**DEFAULT_SCIPY_BVP_SETTINGS, **(settings or {})}
    Fl_CO2_a_guess, Fl_H2O_a_guess, Fv_CO2_a, Fv_H2O_a, Hlf_a_guess, Hvf_a, P_a = Y_a_scaled
    Fl_CO2_b, Fl_H2O_b, Fv_CO2_b_guess, Fv_H2O_b_guess, Hlf_b, Hvf_b_guess, P_b = Y_b_scaled

    scales = parameters[0]
    transform_mode = settings.get('transform_mode', 'bounded_guarded_raw_state')

    bcs_1 = np.array([Fl_CO2_b, Fl_H2O_b, Fv_CO2_a, Fv_H2O_a, Hlf_b, Hvf_a, P_a]) / scales

    # Define the system of differential equations for the absorption column
    def column_odes(z, w):

        differentials = [
            _physical_rhs_to_solver_rhs(
                w[:, i],
                guard_column_rhs(
                    z[i],
                    solver_to_scaled_physical(w[:, i], transform_mode=transform_mode),
                    parameters,
                    evaluator=abs_column,
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
    w_guess_scaled = np.array([polynomial_fit(z_2, Y_a_scaled[i] * scales[i], i) / scales[i] for i in range(m)])
    w_guess_solver = np.column_stack(
        [scaled_physical_to_solver(w_guess_scaled[:, i], transform_mode=transform_mode) for i in range(w_guess_scaled.shape[1])]
    )


    # Solve the BVP

    jacobian_kwargs = {'fun_jac': fun_jac} if settings.get('use_finite_jacobian', False) else {}
    sol = solve_bvp(column_odes, boundary_conditions, z_2, w_guess_solver,
                    max_nodes=int(settings['max_nodes']),
                    tol=float(settings['tol']),
                    bc_tol=float(settings['bc_tol']),
                    verbose=int(settings['verbose']),
                    **jacobian_kwargs,
                    )
    Y_scaled = solver_profile_to_scaled_physical(sol.sol(z), transform_mode=transform_mode)
    z = sol.x

    success = sol.success
    message = sol.message
    if len(parameters) > 6 and isinstance(parameters[6], dict):
        parameters[6].get("solver_diagnostics", {})["jacobian_status"] = str(sol.status)

    return Y_scaled, z, 'SciPy collocation-style BVP', success, message


def _physical_rhs_to_solver_rhs(y_solver, rhs_physical, transform_mode):
    derivative = solver_to_scaled_physical_derivative(y_solver, transform_mode=transform_mode)
    return np.asarray(rhs_physical, dtype=float) / derivative

