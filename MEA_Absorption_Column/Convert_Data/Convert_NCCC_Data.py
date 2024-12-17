import numpy as np
from scipy.optimize import root

from MEA_Absorption_Column.Parameters import MWs_l, MWs_v, column_params, packing_params, n


def convert_NCCC_data(X, case='18'):
    m_T_l, m_T_v, alpha, w_MEA_star, y_CO2, Tl_z, Tv_0, P, beds = X[:9]

    D = column_params['NCCC']['D']
    H = column_params['NCCC']['H'] * beds

    # P = 110000

    a_p = packing_params['MellapakPlus252Y']['a_e']
    ϵ = packing_params['MellapakPlus252Y']['eps']
    Clp = packing_params['MellapakPlus252Y']['Cl']
    Cvp = packing_params['MellapakPlus252Y']['Cv']
    Cs = packing_params['MellapakPlus252Y']['Cs']
    Cp_0 = packing_params['MellapakPlus252Y']['Cp_0']
    Ch = packing_params['MellapakPlus252Y']['Ch']

    packing = a_p, ϵ, Clp, Cvp, Cs, Cp_0, Ch

    A = np.pi * .25 * D ** 2
    z = np.linspace(0, H, n)

    # Molecular Weights
    MW_CO2 = MWs_l[0]
    MW_MEA = MWs_l[1]
    MW_H2O = MWs_l[2]
    MW_N2 = MWs_v[2]
    MW_O2 = MWs_v[3]

    alpha_O2_N2 = 0.08485753604
    alpha_H2O_CO2 = 0.9626010166
    # alpha_H2O_CO2 = 0.0626010166

    # Liquid Calculations

    # Find Liquid Mass Flow Rates

    w_CO2 = alpha*w_MEA_star*MW_CO2/MW_MEA/(1 + alpha*w_MEA_star*MW_CO2/MW_MEA)
    w_MEA = w_MEA_star*(1 - w_CO2)
    m_CO2_l = w_CO2 * m_T_l  # kg/s
    m_MEA_l = w_MEA * m_T_l  # kg/s
    m_H2O_l = m_T_l - m_CO2_l - m_MEA_l  # kg/s

    # Find Liquid Molar Flow Rates
    Fl_CO2_z = m_CO2_l / MW_CO2  # mole/s
    Fl_MEA_z = m_MEA_l / MW_MEA  # mole/s
    Fl_H2O_z = m_H2O_l / MW_H2O  # mole/s

    Fl_z = Fl_CO2_z + Fl_MEA_z + Fl_H2O_z

    x_CO2 = Fl_CO2_z / Fl_z
    x_MEA = Fl_MEA_z / Fl_z
    x_H2O = Fl_H2O_z / Fl_z

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

    Fl = [Fl_CO2_z, Fl_MEA_z, Fl_H2O_z]
    Fv = [Fv_CO2, Fv_H2O, Fv_N2, Fv_O2]

    if case == '18':
        Fl_T = 81.3551492717131
        x_CO2 = 0.0164615447958917
        x_MEA = 0.111188005429196
        x_H2O = 0.872350449774912
        Fl_CO2 = x_CO2 * Fl_T  # mole/s
        Fl_MEA = x_MEA * Fl_T  # mole/s
        Fl_H2O = x_H2O * Fl_T  # mole/s

        Fl = [Fl_CO2, Fl_MEA, Fl_H2O]

        # L_G_ratio = 8.7093001738919638

        Fv_T = 21.7367507327159
        y_CO2 = 0.101947864
        y_H2O = 0.091291891
        y_N2 = 0.734006947
        y_O2 = 0.072753298
        Tv_0 = 319.22

        # With explicit enhancement factor
        # Fv_T = 22.163104235595
        # y_CO2 = 0.095046837
        # y_H2O = 0.113712633
        # y_N2 = 0.719886793
        # y_O2 = 0.071353737
        # Tv_0 = 323.889022602209

        # With implicit enhancement factor
        # y_CO2 = 0.096679388
        # y_H2O = 0.110202318
        # y_N2 = 0.721595221
        # y_O2 = 0.071523073
        # Tv_0 = 323.889022602209

        Fv_CO2 = y_CO2 * Fv_T  # mole/s
        Fv_H2O = y_H2O * Fv_T  # mole/s
        Fv_N2 = y_N2 * Fv_T  # mole/s
        Fv_O2 = y_O2 * Fv_T  # mole/s
        Fv = [Fv_CO2, Fv_H2O, Fv_N2, Fv_O2]

    return Fl, Fv, Tl_z, Tv_0, z, A, P, packing
