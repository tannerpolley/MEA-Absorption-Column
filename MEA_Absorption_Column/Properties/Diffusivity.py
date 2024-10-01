import numpy as np


def liquid_diffusivity(Tl, C_MEA, df_param):

    C_MEA = C_MEA * 1e-3

    # Get Diffusivity of Liquid
    a, b, c, d, e = 0.00000235, 2.98E-08, -9.71E-09, -2119, -20.132

    Dl_CO2 = (a + b * C_MEA + c * C_MEA**2) * np.exp((d + (e * C_MEA))/Tl)

    return Dl_CO2


def vapor_diffusivity(Tv, y, P,  df_param):

    coefficients = np.array([0.000087, 0.00012, 0.000095, 0.000116])

    Dv = (coefficients * Tv ** 1.75) / P

    Dv_T = np.sum([y[i] * Dv[i] for i in range(len(y))])

    Dv_CO2, Dv_H2O, Dv_N2, Dv_O2 = Dv

    return Dv_CO2, Dv_H2O, Dv_N2, Dv_O2, Dv_T
