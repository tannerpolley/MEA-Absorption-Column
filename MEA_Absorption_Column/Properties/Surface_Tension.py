from MEA_Absorption_Column.Parameters import MWs_l


def surface_tension(Tl, x, w_MEA, alpha, df_param):

    x_CO2, x_MEA, x_H2O = x

    m_MEA = x_MEA * MWs_l[1]
    m_H2O = x_H2O * MWs_l[2]
    wt_MEA = m_MEA / (m_MEA + m_H2O)
    wt_H2O = m_H2O / (m_MEA + m_H2O)

    r =  wt_MEA/( wt_MEA + wt_H2O)

    S1, S2, S3, S4, S5, S6 = list(df_param['surface_tension'].values())[:6]

    S6 = float(S6)

    c1_MEA, c2_MEA, c3_MEA, c4_MEA, Tc_MEA = 0.09945, 1.067, 0, 0, 614.45
    c1_H2O, c2_H2O, c3_H2O, c4_H2O, Tc_H2O = 0.18548, 2.717, -3.554, 2.047, 647.13

    a, b, c, d, e, f, g, h, i, j = list(df_param['surface_tension'].values())[6:]

    sigma_CO2 = S1 * r**2 + S2 * r + S3 + Tl*(S4 * r**2 + S5 * r + S6)

    sigma_MEA = c1_MEA*(1 - Tl/Tc_MEA)**(c2_MEA + c3_MEA*(Tl/Tc_MEA) + c4_MEA*(Tl/Tc_MEA)**2)
    sigma_H2O = c1_H2O * (1 - Tl / Tc_H2O) ** (c2_H2O + c3_H2O * (Tl / Tc_H2O) + c4_H2O * (Tl / Tc_H2O) ** 2)

    fxn_f =  a + b * alpha + c * alpha ** 2 + d * w_MEA + e * w_MEA ** 2
    fxn_g =  f + g * alpha + h * alpha ** 2 + i * w_MEA + j * w_MEA ** 2

    sigma_l = sigma_H2O + (sigma_CO2 - sigma_H2O)*fxn_f*x_CO2 + (sigma_MEA - sigma_H2O)*fxn_g*x_MEA

    return sigma_l
