import numpy as np


def viscosity(Tl, Tv, y, w_MEA, alpha, df_param):

    W_MEA = w_MEA*100

    mul_H2O = 1.002 * 10 ** ((1.3272*(293.15 - Tl - .001053 * (Tl - 293.15) ** 2))/(Tl - 168.15))
    a, b, c, d, e, f, g = df_param['viscosity'].values()

    # print(f'From Viscosity: {a}')

    deviation = np.exp((((a*W_MEA +b)*Tl + c*W_MEA + d) * (alpha*(e*W_MEA + f*Tl + g) + 1)*W_MEA)/(Tl**2))
    mul_mix = mul_H2O * deviation

    mul_mix = mul_mix/1000
    mul_H2O = mul_H2O/1000

    y_CO2, y_H2O, y_N2, y_O2 = y

    # Get Viscosity Vapor
    muv_CO2 = 2.148e-6*Tv**.46/(1 + 290/Tv)
    muv_H2O = 1.7096e-8*Tv**1.1146
    muv_N2 = 6.5592e-7*Tv**.6081/(1 + 54.714/Tv)
    muv_O2 = 5.5462e-8*Tv**.8825/(1 + 73.316/Tv)

    muv_mix = np.exp(np.log(muv_CO2)*y_CO2 + np.log(muv_H2O)*y_H2O + np.log(muv_N2)*y_N2 + np.log(muv_O2)*y_O2)

    if mul_mix == 0 or mul_mix < 0:
        mul_mix = .0014

    return muv_mix, mul_mix, mul_H2O
