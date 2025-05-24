from ..Properties.Thermophysical_Properties import enthalpy
from scipy.optimize import root
import numpy as np


def get_liquid_temperature(x, Hl_T_given):
    def solve(Tl):
        _, Hl_T = enthalpy(Tl, x, phase='liquid')

        return Hl_T - Hl_T_given

    Tl = root(solve, np.array([333.0])).x[0]
    return Tl


def get_vapor_temperature(y, Hv_T_given):
    def solve(Tv):
        _, Hv_T = enthalpy(Tv, y, phase='vapor')

        return Hv_T - Hv_T_given

    Tv = root(solve, np.array([320.0])).x[0]

    return Tv


def get_liquid_enthalpy(Fl, Tl):
    x = [Fl[i] / sum(Fl) for i in range(len(Fl))]
    _, Hl_T = enthalpy(Tl, x, phase='liquid')
    return Hl_T


def get_vapor_enthalpy(Fv, Tv):
    y = [Fv[i] / sum(Fv) for i in range(len(Fv))]
    _, Hv_T = enthalpy(Tv, y, phase='vapor')
    return Hv_T





