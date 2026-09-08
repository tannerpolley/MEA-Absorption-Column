from ..Properties.Thermophysical_Properties import enthalpy, enthalpy_temperature_derivative
from scipy.optimize import root
import numpy as np

TEMPERATURE_BOUNDS_K = (250.0, 500.0)


def get_liquid_temperature(x, Hl_T_given):
    def solve(Tl):
        _, Hl_T = enthalpy(Tl, x, phase='liquid')

        return Hl_T - Hl_T_given

    return _solve_temperature(
        solve, lambda T: enthalpy_temperature_derivative(T, x, phase='liquid'),
        initial_guess=333.0,
    )


def get_vapor_temperature(y, Hv_T_given):
    def solve(Tv):
        _, Hv_T = enthalpy(Tv, y, phase='vapor')

        return Hv_T - Hv_T_given

    return _solve_temperature(
        solve, lambda T: enthalpy_temperature_derivative(T, y, phase='vapor'),
        initial_guess=320.0,
    )


def _solve_temperature(residual, derivative, initial_guess):
    def checked_residual(temperature):
        values = np.asarray(residual(temperature), dtype=float).ravel()
        if values.size != 1 or np.any(~np.isfinite(values)):
            raise ValueError("Temperature inversion returned a non-finite enthalpy residual")
        return values

    def jacobian(temperature):
        value = float(derivative(temperature))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Temperature inversion requires a positive finite dh/dT")
        return np.array([[value]])

    result = root(checked_residual, np.array([initial_guess]), jac=jacobian)
    candidate = float(np.asarray(result.x).ravel()[0])
    lower, upper = TEMPERATURE_BOUNDS_K
    if not result.success:
        raise RuntimeError(f"Temperature inversion failed: {result.message}")
    if not np.isfinite(candidate) or not lower <= candidate <= upper:
        raise ValueError(f"Temperature inversion outside [{lower}, {upper}] K: {candidate}")
    return candidate


def get_liquid_enthalpy(Fl, Tl):
    x = [Fl[i] / sum(Fl) for i in range(len(Fl))]
    _, Hl_T = enthalpy(Tl, x, phase='liquid')
    return Hl_T


def get_vapor_enthalpy(Fv, Tv):
    y = [Fv[i] / sum(Fv) for i in range(len(Fv))]
    _, Hv_T = enthalpy(Tv, y, phase='vapor')
    return Hv_T




