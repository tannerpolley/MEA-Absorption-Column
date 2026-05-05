import numpy as np
from scipy.optimize import root

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
    Y_a_scaled = np.asarray(Y_a_scaled, dtype=float)
    Y_b_scaled = np.asarray(Y_b_scaled, dtype=float)
    z = np.asarray(z, dtype=float)
    n_vars = len(Y_a_scaled)
    n_points = len(z)

    if n_points < 3:
        raise ValueError("Finite difference solve requires at least three mesh points.")

    guess = _initial_profile(Y_a_scaled, Y_b_scaled, n_points)

    def residual(flat):
        w = flat.reshape(n_vars, n_points)
        eqs = np.zeros_like(w)
        dz = np.gradient(z)

        for i in range(n_points):
            if i == 0:
                derivative = (w[:, 1] - w[:, 0]) / (z[1] - z[0])
            elif i == n_points - 1:
                derivative = (w[:, -1] - w[:, -2]) / (z[-1] - z[-2])
            else:
                derivative = (w[:, i + 1] - w[:, i - 1]) / (z[i + 1] - z[i - 1])
            eqs[:, i] = derivative - guard_column_rhs(z[i], w[:, i], parameters, evaluator=abs_column)

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
    return Y_scaled, z, 'Finite difference BVP', bool(solution.success), str(solution.message)
