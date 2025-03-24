import numpy as np
from MEA_Absorption_Column.Parameters import R
from MEA_Absorption_Column.Properties.Thermophysical_Properties import heat_capacity, heat_of_vaporization, enthalpy
EPS = np.finfo(float).eps

def finite_difference(f, x, *args, h=1e-5):

    x_forward = np.copy(x)
    x_backward = np.copy(x)

    x_forward += h  # Increment the i-th component
    x_backward -= h  # Decrement the i-th component

    # Central difference formula for the partial derivative

    grad = (f(x_forward, *args) - f(x, *args)) / (h)

    return grad


def jac(f, x, y):

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
        f_new = f(x, y_new)
        f_new2 = f(x, y_new2)

        df_dy[:, i, :] = (f_new - f_new2) / (2*hi)

    return df_dy


def complex_step(f, x, *args, h=1e-200):

    return np.real(np.imag(f(x + 1j * h, *args)) / h)


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

    # print(-R*(c + 2*d*Tl), Cpl_MEA - dHvap_dT(Tl, 'MEA'), Cpl_H2O - dHvap_dT(Tl, 'H2O'))
    return -R*(c + 2*d*Tl) + Cpl_MEA + dHvap_dT(Tl, 'MEA') + Cpl_H2O + dHvap_dT(Tl, 'H2O')
