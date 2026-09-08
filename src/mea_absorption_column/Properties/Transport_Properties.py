import numpy as np
import casadi as ca
from mea_absorption_column.config.Constants import MWs_v


def viscosity(T, z, w_MEA, w_H2O, phase='liquid'):
    """Shared numerical/symbolic correlations; composition is apparent liquid
    or CO2/H2O/N2/O2 vapor. No EOS viscosity model is implied.
    """

    if phase == 'liquid':
        Tl = T
        x = z
        alpha = x[0]/x[1]
        r = (w_MEA / (w_MEA + w_H2O)) * 100
        A, B, C, D = 1.002e-3, 1.3272, .001053, 168.15
        mul_H2O = A * 10 ** ((B * (293.15 - Tl - C * (Tl - 293.15) ** 2)) / (Tl - D))
        a, b, c, d, e, f, g = (-0.0854041877181552, 2.72913373574306, 35.1158892542595,
                               1805.52759876533, 0.00716025669867574, 0.0106488402285381, -0.0854041877181552)

        exponent = r * (Tl * (a * r + b) + c * r + d) * (alpha * (e * r + f * Tl + g) + 1) / Tl ** 2
        deviation = ca.exp(exponent) if isinstance(exponent, (ca.MX, ca.SX, ca.DM)) else np.exp(exponent)
        mul_mix = mul_H2O * deviation

        return mul_mix, mul_H2O

    elif phase == 'vapor':
        Tv = T
        y = z

        # Get Viscosity Vapor
        muv_CO2 = 2.148e-6 * Tv ** .46 / (1 + 290 / Tv)
        muv_H2O = 1.7096e-8 * Tv ** 1.1146
        muv_N2 = 0.01781e-3 * (300.55 + 111) / (Tv + 111) * (Tv / 300.55) ** 1.5
        muv_O2 = 0.02018e-3 * (292.25 + 127) / (Tv + 127) * (Tv / 292.25) ** 1.5

        muv = [muv_CO2, muv_H2O, muv_N2, muv_O2]
        theta = [[(1 + (muv[i] / muv[j]) ** .5 * (MWs_v[j] / MWs_v[i]) ** .25) ** 2
                  / (8 * (1 + MWs_v[i] / MWs_v[j])) ** .5
                  for j in range(4)] for i in range(4)]
        muv_mix = sum(y[i] * muv[i] / sum(y[j] * theta[i][j] for j in range(4)) for i in range(4))
        return muv_mix, muv
    return None


def diffusivity(T, z, P, mul_mix, rho_mol_l, phase='liquid'):

    if phase == 'liquid':
        Tl = T
        x = z
        Cl_MEA = rho_mol_l*x[1]
        C_MEA_scaled = Cl_MEA * 1e-3

        # Get Diffusivity of Liquid
        a, b, c, d, e = 2.35e-6, 2.9837E-08, -9.7078e-9, -2119, -20.132

        symbolic = any(isinstance(v, (ca.MX, ca.SX, ca.DM)) for v in (T, Cl_MEA, mul_mix))
        exp, log = (ca.exp, ca.log) if symbolic else (np.exp, np.log)
        Dl_CO2 = (a + b * C_MEA_scaled + c * C_MEA_scaled ** 2) * exp((d + (e * C_MEA_scaled)) / Tl)

        a, b, c = -13.275, -2198.3, -7.8142e-5
        Dl_MEA = exp(a + b / Tl + c * Cl_MEA)

        a, b, c = -22.64, -1000, -.7
        Dl_ion = exp(a + b / Tl + c * log(mul_mix))

        return Dl_CO2, Dl_MEA, Dl_ion

    elif phase == 'vapor':
        Tv = T
        y = z

        params = 26.7, 13.1, 18.5, 16.3

        def binary(i, j):
            return 1.013e-2 * Tv ** 1.75 / P * np.sqrt(1e-3 * (1 / MWs_v[i] + 1 / MWs_v[j])) / (
                    params[i] ** (1 / 3) + params[j] ** (1 / 3)) ** 2

        Dv = [(1 - y[i]) / sum(y[j] / binary(i, j) for j in range(4) if i != j) for i in range(4)]
        Dv_T = sum(y[i] * Dv[i] for i in range(4))

        Dv_CO2, Dv_H2O, Dv_N2, Dv_O2 = Dv

        return Dv_CO2, Dv_H2O, Dv_N2, Dv_O2, Dv_T
    return None
