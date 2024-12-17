from MEA_Absorption_Column.Parameters import MWs_l


def surface_tension(Tl, x, w_MEA, w_H2O, alpha):
    x_CO2, x_MEA, x_H2O = x

    r = w_MEA / (w_MEA + w_H2O)

    S1, S2, S3, S4, S5, S6 = -0.00589934906112609, 0.00175020536428591, 0.129650182728177, 0.0000126444768126308, -5.73954817199691E-06, -0.00018969005534195,

    c1_MEA, c2_MEA, c3_MEA, c4_MEA, Tc_MEA = 0.09945, 1.067, 0, 0, 614.45
    c1_H2O, c2_H2O, c3_H2O, c4_H2O, Tc_H2O = 0.18548, 2.717, -3.554, 2.047, 647.13

    a, b, c, d, e, f, g, h, i, j = (1070.65668317975,-2578.78134208703,3399.24113311222,-2352.47410135319,2960.24753687833,
                                    3.06684894924048,-1.79435372759593,-7.2124219075848,2.97502322396621,-10.5738529301824)

    sigma_CO2 = S1 * r ** 2 + S2 * r + S3 + Tl * (S4 * r ** 2 + S5 * r + S6)

    sigma_MEA = c1_MEA * (1 - Tl / Tc_MEA) ** (c2_MEA + c3_MEA * (Tl / Tc_MEA) + c4_MEA * (Tl / Tc_MEA) ** 2)
    sigma_H2O = c1_H2O * (1 - Tl / Tc_H2O) ** (c2_H2O + c3_H2O * (Tl / Tc_H2O) + c4_H2O * (Tl / Tc_H2O) ** 2)

    fxn_f = a + b * alpha + c * alpha ** 2 + d * r + e * r ** 2
    fxn_g = f + g * alpha + h * alpha ** 2 + i * r + j * r ** 2

    sigma_l = sigma_H2O + (sigma_CO2 - sigma_H2O) * fxn_f * x_CO2 + (sigma_MEA - sigma_H2O) * fxn_g * x_MEA

    return sigma_l
