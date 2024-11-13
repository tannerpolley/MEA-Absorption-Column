import numpy as np

from MEA_Absorption_Column.Parameters import MWs_l, MWs_v, column_params, n


def convert_NCCC_data(X, case='18'):
    m_T_l, m_T_v, alpha, w_MEA, y_CO2, Tl_z, Tv_0, P, beds = X[:9]

    D = column_params['NCCC']['D']
    H = column_params['NCCC']['H']*beds

    A = np.pi * D ** 2 / 4
    z = np.linspace(0, H, n)

    # Molecular Weights
    MW_CO2 = MWs_l[0]
    MW_MEA = MWs_l[1]
    MW_H2O = MWs_l[2]
    MW_N2 = MWs_v[2]
    MW_O2 = MWs_v[3]

    alpha_O2_N2 = 0.08485753604
    alpha_H2O_CO2 = 0.7626010166

    # Liquid Calculations

    # Find Liquid Mass Flow Rates
    m_MEA_l = w_MEA * m_T_l  # kg/s
    m_CO2_l = m_MEA_l * alpha / MW_MEA * MW_CO2  # kg/s
    m_H2O_l = m_T_l - m_MEA_l - m_CO2_l  # kg/s

    # Find Liquid Molar Flow Rates
    Fl_CO2 = m_CO2_l / MW_CO2  # mole/s
    Fl_MEA = m_MEA_l / MW_MEA  # mole/s
    Fl_H2O = m_H2O_l / MW_H2O  # mole/s

    # Vapor Calculations

    # Find Vapor Mole Fractions
    y_H2O = y_CO2 * alpha_H2O_CO2
    y_N2 = (1 - y_CO2 - y_H2O) / (1 + alpha_O2_N2)
    y_O2 = y_N2 * alpha_O2_N2
    sigma = y_N2 * MW_N2 + y_O2 * MW_O2 + y_CO2 * MW_CO2 + y_H2O * MW_H2O

    # Find Vapor Mass Flow Rates

    w_CO2_v = y_CO2 * MW_CO2 / sigma
    w_H2O_v = y_H2O * MW_H2O / sigma
    w_N2_v = y_N2 * MW_N2 / sigma
    w_O2_v = y_O2 * MW_O2 / sigma

    m_CO2_v = w_CO2_v * m_T_v
    m_H2O_v = w_H2O_v * m_T_v
    m_N2_v = w_N2_v * m_T_v
    m_O2_v = w_O2_v * m_T_v

    # Find Vapor Molar Flow Rates
    Fv_CO2 = m_CO2_v / MW_CO2  # mole/s
    Fv_H2O = m_H2O_v / MW_H2O  # mole/s
    Fv_N2 = m_N2_v / MW_N2  # mole/s
    Fv_O2 = m_O2_v / MW_O2  # mole/s

    Fl = [Fl_CO2, Fl_MEA, Fl_H2O]
    Fv = [Fv_CO2, Fv_H2O, Fv_N2, Fv_O2]

    if case == '18':
        Fl_T = 80.6281332727093
        Fl_CO2 = 0.0210508190313722 * Fl_T  # mole/s
        Fl_MEA = 0.113282040368075 * Fl_T  # mole/s
        Fl_H2O = 0.865667140600553 * Fl_T  # mole/s

        Fl = [Fl_CO2, Fl_MEA, Fl_H2O]

        Fv_T = 21.7367507327159
        Fv_CO2 = 0.101947863634366 * Fv_T  # mole/s
        Fv_H2O = 0.0912918913073196 * Fv_T  # mole/s
        Fv_N2 = 0.734006946649283 * Fv_T  # mole/s
        Fv_O2 = 0.072753298409031 * Fv_T  # mole/s
        Fv = [Fv_CO2, Fv_H2O, Fv_N2, Fv_O2]

    return Fl, Fv, Tl_z, Tv_0, z, A, P
