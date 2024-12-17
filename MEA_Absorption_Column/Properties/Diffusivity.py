import numpy as np
from MEA_Absorption_Column.Parameters import MWs_v


def liquid_diffusivity(Tl, C_MEA, mul_mix):

    C_MEA_scaled = C_MEA * 1e-3

    # Get Diffusivity of Liquid
    a, b, c, d, e = 2.35e-6, 2.9837E-08, -9.7078e-9, -2119, -20.132

    Dl_CO2 = (a + b * C_MEA_scaled + c * C_MEA_scaled**2) * np.exp((d + (e * C_MEA_scaled))/Tl)

    a, b, c = -13.275, -2198.3, -7.8142e-5
    Dl_MEA = np.exp(a + b/Tl + c*C_MEA)

    a, b, c = -22.64, -1000, -.7
    Dl_ion = np.exp(a + b/Tl + c*np.log(mul_mix))

    return Dl_CO2, Dl_MEA, Dl_ion


def vapor_diffusivity(Tv, y, P):

    params = 26.7, 13.1, 18.5, 16.3

    binary_set = []
    for i in range(4):
        for j in range(4):
            if i != j and (j, i) not in binary_set:
                binary_set.append((i, j))

    def binary(i, j):
        if i == j:
            return 1
        else:
            return 1.013e-2*Tv**1.75/P*np.sqrt(1e-3 * (1/MWs_v[i] + 1/MWs_v[j]))/(params[i]**(1/3) + params[j]**(1/3))**2

    Dv = []
    for i in range(4):
        sum1 = 0
        sum2 = 0
        for j in range(4):
            if (i, j) in binary_set:
                sum1 += y[j]/binary(i, j)
            if (j, i) in binary_set:
                sum2 += y[j]/binary(j, i)
        Dv_i = (1-y[i])/(sum1 + sum2)
        Dv.append(Dv_i)

    Dv_T = np.sum([y[i] * Dv[i] for i in range(len(y))])

    Dv_CO2, Dv_H2O, Dv_N2, Dv_O2 = Dv

    return Dv_CO2, Dv_H2O, Dv_N2, Dv_O2, Dv_T
