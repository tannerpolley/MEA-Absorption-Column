import numpy as np
from scipy.optimize import root
import time

from ...BVP.ABS_Column import abs_column
from ...BVP.robust_core import guard_column_rhs


DEFAULT_FINITE_DIFFERENCE_SETTINGS = {
    'maxfev': 500,
    'tol': 1e-4,
}


def _initial_profile(Y_a_scaled, Y_b_scaled, n_points):
    guess = np.zeros((len(Y_a_scaled), n_points), dtype=float)
    for idx, (left, right) in enumerate(zip(Y_a_scaled, Y_b_scaled)):
        guess[idx] = np.linspace(left, right, n_points)
    guess[2] = Y_a_scaled[2]
    guess[3] = Y_a_scaled[3]
    guess[5] = Y_a_scaled[5]
    guess[6] = Y_a_scaled[6]
    return guess


def finite_difference_solve(Y_a_scaled, Y_b_scaled, z, parameters, settings=None):
    settings = {**DEFAULT_FINITE_DIFFERENCE_SETTINGS, **(settings or {})}
    start_time = time.monotonic()
    Y_a_scaled = np.asarray(Y_a_scaled, dtype=float)
    Y_b_scaled = np.asarray(Y_b_scaled, dtype=float)
    z = np.asarray(z, dtype=float)
    n_vars = len(Y_a_scaled)
    requested_points = settings.get("mesh_points")
    if requested_points is None:
        z_work = z
    else:
        z_work = np.linspace(float(z[0]), float(z[-1]), int(requested_points))
    n_points = len(z_work)

    if n_points < 3:
        raise ValueError("Finite difference solve requires at least three mesh points.")

    guess = _initial_profile(Y_a_scaled, Y_b_scaled, n_points)

    def residual(flat):
        _raise_if_timed_out(settings, start_time)
        w = flat.reshape(n_vars, n_points)
        eqs = np.zeros_like(w)

        for i in range(n_points):
            if i == 0:
                derivative = (w[:, 1] - w[:, 0]) / (z_work[1] - z_work[0])
            elif i == n_points - 1:
                derivative = (w[:, -1] - w[:, -2]) / (z_work[-1] - z_work[-2])
            else:
                derivative = (w[:, i + 1] - w[:, i - 1]) / (z_work[i + 1] - z_work[i - 1])
            rhs = guard_column_rhs(z_work[i], w[:, i], parameters, evaluator=abs_column)
            eqs[:, i] = derivative - rhs

        eqs[2, 0] = w[2, 0] - Y_a_scaled[2]
        eqs[3, 0] = w[3, 0] - Y_a_scaled[3]
        eqs[5, 0] = w[5, 0] - Y_a_scaled[5]
        eqs[6, 0] = w[6, 0] - Y_a_scaled[6]
        eqs[0, -1] = w[0, -1] - Y_b_scaled[0]
        eqs[1, -1] = w[1, -1] - Y_b_scaled[1]
        eqs[4, -1] = w[4, -1] - Y_b_scaled[4]

        return eqs.ravel()

    solution = root(
        residual,
        guess.ravel(),
        method='hybr',
        tol=float(settings['tol']),
        options={'maxfev': int(settings['maxfev'])},
    )

    Y_scaled = solution.x.reshape(n_vars, n_points)
    if len(parameters) > 6 and isinstance(parameters[6], dict):
        parameters[6].get("solver_diagnostics", {})["jacobian_status"] = str(solution.status)
    return Y_scaled, z_work, 'Finite difference BVP', bool(solution.success), str(solution.message)


def _raise_if_timed_out(settings, start_time):
    max_runtime = settings.get("max_runtime_s")
    if max_runtime is None:
        return
    if time.monotonic() - start_time > float(max_runtime):
        raise TimeoutError(f"Finite difference solve exceeded max_runtime_s={float(max_runtime):g}")
