from numpy import log, exp
from MEA_Absorption_Column.Parameters import MWs_l


def henrys_law(x, Tl, df_param):

    x_CO2, x_MEA, x_H2O = x

    m_MEA = x_MEA * MWs_l[1]
    m_H2O = x_H2O * MWs_l[2]
    wt_MEA = m_MEA / (m_MEA + m_H2O)
    wt_H2O = m_H2O / (m_MEA + m_H2O)

    H_N2O_MEA = 2.448e5 * exp(-1348 / Tl)
    H_CO2_H2O = 3.52e6 * exp(-2113 / Tl)
    H_N2O_H2O = 8.449e6 * exp(-2283 / Tl)
    H_CO2_MEA = H_N2O_MEA * (H_CO2_H2O / H_N2O_H2O)

    a1, a2, a3, b = (df_param['VLE']['lwm_coeff_1'], df_param['VLE']['lwm_coeff_2'],
                     df_param['VLE']['lwm_coeff_3'], df_param['VLE']['lwm_coeff_4'])
    lwm = (a1 + a2 * (Tl-273.15) + a3 * (Tl-273.15) ** 2 + b * wt_H2O)

    Ra = lwm * wt_MEA * wt_H2O

    sigma = wt_MEA * log(H_CO2_MEA) + wt_H2O * log(H_CO2_H2O)

    log_H_CO2_mix = Ra + sigma

    H_CO2_mix = exp(log_H_CO2_mix)

    return H_CO2_mix
