import numpy as np


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

def dHvap_dT(T, species):

    coefficients = {'CO2': np.array([21730000, 0.382, -0.4339, 0.42213, 304.21]),
                    'MEA': np.array([82393000, 0.59045, - 0.43602, 0.37843, 678.2]),
                    'H2O': np.array([56600000, 0.612041, -0.625697, 0.398804, 647.096])
                    }
    A, B, C, D, Tc = coefficients[species]
    return (A * ((-T + Tc) / Tc) ** ((B * Tc ** 2 + C * T * Tc + D * T ** 2) / Tc ** 2) * (
                B * Tc ** 2 + C * T * Tc + D * T ** 2 + (T - Tc) * (C * Tc + 2 * D * T) * np.log((-T + Tc) / Tc)) / (
                Tc ** 2 * (T - Tc)))/1000


def f_dHl_dT(Tl, x):
    Cpl_CO2, Cpl_MEA, Cpl_H2O = heat_capacity(Tl, x, phase='liquid')[0]
    c, d = -8.598, -.012

    return -R*(c + 2*d*Tl) + Cpl_MEA + dHvap_dT(Tl, 'MEA') + Cpl_H2O + dHvap_dT(Tl, 'H2O')