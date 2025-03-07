import numpy as np
from MEA_Absorption_Column.Parameters import g, MWs_l, MWs_v


def viscosity(T, z, w_MEA, w_H2O, phase='liquid'):

    if phase == 'liquid':
        Tl = T
        x = z
        alpha = x[0]/x[1]
        r = (w_MEA / (w_MEA + w_H2O)) * 100
        A, B, C, D = 1.002e-3, 1.3272, .001053, 168.15
        mul_H2O = A * 10 ** ((B * (293.15 - Tl - C * (Tl - 293.15) ** 2)) / (Tl - D))
        a, b, c, d, e, f, g = (-0.0854041877181552, 2.72913373574306, 35.1158892542595,
                               1805.52759876533, 0.00716025669867574, 0.0106488402285381, -0.0854041877181552)

        # print(f'From Viscosity: {a}')

        deviation = np.exp(r * (Tl * (a * r + b) + c * r + d) * (alpha * (e * r + f * Tl + g) + 1) / Tl ** 2)
        mul_mix = mul_H2O * deviation

        return mul_mix, mul_H2O

    elif phase == 'vapor':
        Tv = T
        y = z
        y_CO2, y_H2O, y_N2, y_O2 = y

        # Get Viscosity Vapor
        muv_CO2 = 2.148e-6 * Tv ** .46 / (1 + 290 / Tv)
        muv_H2O = 1.7096e-8 * Tv ** 1.1146
        muv_N2 = 0.01781e-3 * (300.55 + 111) / (Tv + 111) * (Tv / 300.55) ** 1.5
        muv_O2 = 0.02018e-3 * (292.25 + 127) / (Tv + 127) * (Tv / 292.25) ** 1.5

        muv = [muv_CO2, muv_H2O, muv_N2, muv_O2]
        # print(muv)
        # muv = [0.0000161853801328463, 0.000010739452610202,
        #        0.0000188550342242103, 0.0000218920854587055]
        # print(muv)
        theta_ij = np.zeros((4, 4))

        for i in range(len(muv)):
            for j in range(i + 1, len(muv)):
                theta_ij[i, j] = (
                                         1
                                         + 2
                                         * np.sqrt(muv[i] / muv[j])
                                         * (MWs_v[j] / MWs_v[i]) ** 0.25
                                         + muv[i]
                                         / muv[j]
                                         * (MWs_v[j] / MWs_v[i]) ** 0.5
                                 ) / (8 + 8 * MWs_v[i] / MWs_v[j]) ** 0.5

                theta_ij[j, i] = (
                        muv[j]
                        / muv[i]
                        * MWs_v[i]
                        / MWs_v[j]
                        * theta_ij[i, j]
                )
        for i in range(len(muv)):
            for j in range(len(muv)):
                if i == j:
                    theta_ij[i, j] = 1

        muv_mix = sum([y[i] * muv[i] / sum([y[j] * theta_ij[i, j] for j in range(len(muv))]) for i in range(len(muv))])
        return muv_mix, muv


def diffusivity(T, z, P, mul_mix, rho_mol_l, phase='liquid'):

    if phase == 'liquid':
        Tl = T
        x = z
        Cl_MEA = rho_mol_l*x[1]
        C_MEA_scaled = Cl_MEA * 1e-3

        # Get Diffusivity of Liquid
        a, b, c, d, e = 2.35e-6, 2.9837E-08, -9.7078e-9, -2119, -20.132

        Dl_CO2 = (a + b * C_MEA_scaled + c * C_MEA_scaled ** 2) * np.exp((d + (e * C_MEA_scaled)) / Tl)

        a, b, c = -13.275, -2198.3, -7.8142e-5
        Dl_MEA = np.exp(a + b / Tl + c * Cl_MEA)

        a, b, c = -22.64, -1000, -.7
        Dl_ion = np.exp(a + b / Tl + c * np.log(mul_mix))

        return Dl_CO2, Dl_MEA, Dl_ion

    elif phase == 'vapor':
        Tv = T
        y = z

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
                return 1.013e-2 * Tv ** 1.75 / P * np.sqrt(1e-3 * (1 / MWs_v[i] + 1 / MWs_v[j])) / (
                        params[i] ** (1 / 3) + params[j] ** (1 / 3)) ** 2

        Dv = []
        for i in range(4):
            sum1 = 0
            sum2 = 0
            for j in range(4):
                if (i, j) in binary_set:
                    sum1 += y[j] / binary(i, j)
                if (j, i) in binary_set:
                    sum2 += y[j] / binary(j, i)
            Dv_i = (1 - y[i]) / (sum1 + sum2)
            Dv.append(Dv_i)

        Dv_T = np.sum([y[i] * Dv[i] for i in range(len(y))])

        Dv_CO2, Dv_H2O, Dv_N2, Dv_O2 = Dv

        return Dv_CO2, Dv_H2O, Dv_N2, Dv_O2, Dv_T


