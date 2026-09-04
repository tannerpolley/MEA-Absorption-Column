import numpy as np
from ..Properties.Thermophysical_Properties import heat_capacity


def finite_difference(f, x, h):

    m = len(f(x))
    n = len(x)
    J = np.zeros((m, n))
    for i in range(m):
        f_eval_0 = f(x.copy())[i]
        for j in range(n):
            x_new = x.copy()
            Δx = h * (1 + np.abs(x_new[j]))
            x_new[j] += Δx
            f_eval_1 = f(x_new)[i]
            J[i, j] = (f_eval_1 - f_eval_0) / Δx
            # x[j] = x[j] - Δx

    return J

def f_dHl_dT(Tl, x):
    """Fixed-composition derivative of the implemented empirical enthalpy."""
    _, Cpl_MEA, Cpl_H2O = heat_capacity(Tl, x, phase='liquid')[0]
    # CO2 absorption enthalpy and solvent reference offsets are constants.
    return x[1]*Cpl_MEA + x[2]*Cpl_H2O
