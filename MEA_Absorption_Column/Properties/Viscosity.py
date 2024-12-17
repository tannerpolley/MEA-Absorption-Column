import numpy as np
from MEA_Absorption_Column.Parameters import MWs_v


def viscosity(Tl, Tv, y, w_MEA, w_H2O, alpha):

    r = (w_MEA/(w_MEA + w_H2O))*100

    mul_H2O = 1.002e-3 * 10 ** ((1.3272*(293.15 - Tl - .001053 * (Tl - 293.15) ** 2))/(Tl - 168.15))
    a, b, c, d, e, f, g = (-0.0854041877181552, 2.72913373574306, 35.1158892542595,
                           1805.52759876533, 0.00716025669867574, 0.0106488402285381, -0.0854041877181552)

    # print(f'From Viscosity: {a}')

    deviation = np.exp(r * (Tl*(a*r + b) + c*r + d) * (alpha * (e * r + f * Tl + g) + 1)/Tl**2)
    mul_mix = mul_H2O * deviation

    y_CO2, y_H2O, y_N2, y_O2 = y

    # Get Viscosity Vapor
    muv_CO2 = 2.148e-6*Tv**.46/(1 + 290/Tv)
    muv_H2O = 1.7096e-8*Tv**1.1146
    muv_N2 = 0.01781e-3*(300.55 + 111)/(Tv + 111)*(Tv / 300.55) ** 1.5
    muv_O2 = 0.02018e-3*(292.25 + 127)/(Tv + 127)*(Tv / 292.25) ** 1.5
    
    muv = [muv_CO2, muv_H2O, muv_N2, muv_O2]
    # print(muv)
    # muv = [0.0000161853801328463, 0.000010739452610202,
    #        0.0000188550342242103, 0.0000218920854587055]
    # print(muv)
    theta_ij = np.zeros((4, 4))

    for i in range(len(muv)):
        for j in range(i+1, len(muv)):
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
    # muv_mix = 0.0000178308980604464
    return muv_mix, mul_mix, mul_H2O, muv
