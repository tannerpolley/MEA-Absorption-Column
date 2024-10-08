import numpy as np


def liquid_diffusivity(Tl, C_MEA, mul_mix):

    C_MEA_scaled = C_MEA * 1e-3

    # Get Diffusivity of Liquid
    a, b, c, d, e = 0.00000235, 2.98E-08, -9.71E-09, -2119, -20.132

    Dl_CO2 = (a + b * C_MEA_scaled + c * C_MEA_scaled**2) * np.exp((d + (e * C_MEA_scaled))/Tl)

    a, b, c = -13.275, -2198.3, -7.8142e-5
    Dl_MEA = np.exp(a + b/Tl + c*C_MEA)

    a, b, c = -22.64, -1000, -.7
    Dl_ion = np.exp(a + b/Tl + c*np.log(mul_mix))

    return Dl_CO2, Dl_MEA, Dl_ion


def vapor_diffusivity(Tv, y, P,  df_param):

    coefficients = np.array([0.000087, 0.00012, 0.000095, 0.000116])

    Dv = (coefficients * Tv ** 1.75) / P

    Dv_T = np.sum([y[i] * Dv[i] for i in range(len(y))])

    Dv_CO2, Dv_H2O, Dv_N2, Dv_O2 = Dv

    return Dv_CO2, Dv_H2O, Dv_N2, Dv_O2, Dv_T
