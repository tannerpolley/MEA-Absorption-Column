from ..Properties.Amine_Properties import resolve_amine_properties
from ..Properties.Thermophysical_Properties import enthalpy
from scipy.optimize import root
from scipy.optimize import least_squares
import numpy as np

TEMPERATURE_BOUNDS_K = (250.0, 500.0)


def get_liquid_temperature(x, Hl_T_given, amine_properties=None):
    amine_properties = resolve_amine_properties(amine_properties)

    def solve(Tl):
        _, Hl_T = amine_properties.enthalpy(Tl, x)

        return Hl_T - Hl_T_given

    return _solve_temperature(solve, initial_guess=333.0)


def get_vapor_temperature(y, Hv_T_given):
    def solve(Tv):
        _, Hv_T = enthalpy(Tv, y, phase='vapor')

        return Hv_T - Hv_T_given

    return _solve_temperature(solve, initial_guess=320.0)


def _solve_temperature(residual, initial_guess):
    lower, upper = TEMPERATURE_BOUNDS_K
    safe_residual = _safe_temperature_residual(residual)
    root_result = root(safe_residual, np.array([initial_guess]))
    candidate = float(np.asarray(root_result.x).ravel()[0])
    if root_result.success and lower <= candidate <= upper and np.isfinite(candidate):
        return candidate

    bounded = least_squares(
        safe_residual,
        x0=np.array([np.clip(initial_guess, lower, upper)], dtype=float),
        bounds=(np.array([lower]), np.array([upper])),
        max_nfev=100,
    )
    return float(np.clip(bounded.x[0], lower, upper))


def _safe_temperature_residual(residual):
    def wrapped(T):
        try:
            values = np.asarray(residual(T), dtype=float).ravel()
        except Exception:
            return np.array([1.0e12], dtype=float)
        if values.size == 0 or np.any(~np.isfinite(values)):
            return np.full(max(values.size, 1), 1.0e12, dtype=float)
        return values

    return wrapped


def get_liquid_enthalpy(Fl, Tl, amine_properties=None):
    amine_properties = resolve_amine_properties(amine_properties)
    x = [Fl[i] / sum(Fl) for i in range(len(Fl))]
    _, Hl_T = amine_properties.enthalpy(Tl, x)
    return Hl_T


def get_vapor_enthalpy(Fv, Tv):
    y = [Fv[i] / sum(Fv) for i in range(len(Fv))]
    _, Hv_T = enthalpy(Tv, y, phase='vapor')
    return Hv_T




