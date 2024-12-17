from MEA_Absorption_Column.Thermodynamics.Enthalpy import liquid_enthalpy, vapor_enthalpy
from MEA_Absorption_Column.Properties.Density import liquid_density
from scipy.optimize import root
import numpy as np


def get_liquid_temperature(x, Hl_T_given):

    def solve(Tl):
        Hl_i = liquid_enthalpy(Tl, x)
        Hl_T = sum([x[i]*Hl_i[i] for i in range(len(x))])

        return Hl_T - Hl_T_given

    Tl = root(solve, np.array([333.0])).x[0]

    return Tl


def get_vapor_temperature(y, Hv_T_given):
    def solve(Tv):
        Hv_i = vapor_enthalpy(Tv)

        Hv_T = sum([y[i] * Hv_i[i] for i in range(len(y))])

        return Hv_T - Hv_T_given

    Tv = root(solve, np.array([320.0])).x[0]

    return Tv


def get_liquid_enthalpy(Fl, Tl):
    x = [Fl[i] / sum(Fl) for i in range(len(Fl))]
    Hl_i = liquid_enthalpy(Tl, x)
    Hl_T = sum([x[i] * Hl_i[i] for i in range(len(x))])
    return Hl_T


def get_vapor_enthalpy(Fv, Tv):
    y = [Fv[i] / sum(Fv) for i in range(len(Fv))]
    Hv_i = vapor_enthalpy(Tv)
    Hv_T = sum([y[i] * Hv_i[i] for i in range(len(y))])
    return Hv_T





