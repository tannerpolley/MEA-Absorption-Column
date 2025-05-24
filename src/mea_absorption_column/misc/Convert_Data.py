import numpy as np
from scipy.optimize import root

from ..config.Constants import MWs_l, MWs_v, column_params, packing_params, n


def convert_data(df, run=0, type='mole'):

    X = df.iloc[run, :].to_numpy()

    if type == 'mole':
        L_G, Fv_T, alpha, w_MEA_unloaded, y_CO2, Tl_z, Tv_0, P, beds = X[:9]

        # Molecular Weights
        MW_CO2 = MWs_l[0]
        MW_MEA = MWs_l[1]
        MW_H2O = MWs_l[2]
        MW_N2 = MWs_v[2]
        MW_O2 = MWs_v[3]

        alpha_O2_N2 = 0.08485753604
        alpha_H2O_CO2 = 0.9626010166

        # Liquid Calculations
        Fl_T = L_G * Fv_T

        x_MEA_unloaded = w_MEA_unloaded / (MW_MEA / MW_H2O + w_MEA_unloaded * (1 - MW_MEA / MW_H2O))
        x_H2O_unloaded = 1 - x_MEA_unloaded

        Fl_MEA_b = Fl_T * x_MEA_unloaded
        Fl_H2O_b = Fl_T * x_H2O_unloaded

        Fl_CO2_b = Fl_MEA_b * alpha
        Fl = [Fl_CO2_b, Fl_MEA_b, Fl_H2O_b]

        # Vapor Calculations

        # Find Vapor Mole Fractions
        y_H2O = y_CO2 * alpha_H2O_CO2
        y_N2 = (1 - y_CO2 - y_H2O) / (1 + alpha_O2_N2)
        y_O2 = y_N2 * alpha_O2_N2

        # Find Vapor Molar Flow Rates
        Fv_CO2_a = y_CO2 * Fv_T  # mole/s
        Fv_H2O_a = y_H2O * Fv_T  # mole/s
        Fv_N2_a = y_N2 * Fv_T  # mole/s
        Fv_O2_a = y_O2 * Fv_T  # mole/s
        Fv = [Fv_CO2_a, Fv_H2O_a, Fv_N2_a, Fv_O2_a]

    elif type == 'mass':

        m_T_l, m_T_v, alpha, w_MEA_unloaded, y_CO2, Tl_z, Tv_0, P, beds = X[:9]

        D = column_params['NCCC']['D']
        H = column_params['NCCC']['H'] * beds

        a_p = packing_params['MellapakPlus252Y']['a_p']
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

        # Liquid Calculations

        # Find Liquid Mass Flow Rates

        w_H2O_unloaded = 1 - w_MEA_unloaded

        m_MEA_l = w_MEA_unloaded * m_T_l  # kg/s
        m_H2O_l = w_H2O_unloaded * m_T_l  # kg/s

        Fl_MEA_b = m_MEA_l/MW_MEA
        Fl_H2O_b = m_H2O_l/MW_H2O
        Fl_T_unloaded = Fl_MEA_b + Fl_H2O_b
        Fl_CO2_b = Fl_MEA_b * alpha
        Fl = [Fl_CO2_b, Fl_MEA_b, Fl_H2O_b]
        Fl_T = sum(Fl)

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

        Fv = [Fv_CO2, Fv_H2O, Fv_N2, Fv_O2]
        Fv_T = sum(Fv)

        L_G = Fl_T_unloaded/Fv_T

        X = L_G, Fv_T, alpha, w_MEA_unloaded, y_CO2, Tl_z, Tv_0, P, beds

    else:
        raise ValueError('Wrong Data Type')

    D = column_params['NCCC']['D']
    H = column_params['NCCC']['H'] * beds

    a_p = packing_params['MellapakPlus252Y']['a_p']
    ϵ = packing_params['MellapakPlus252Y']['eps']
    Clp = packing_params['MellapakPlus252Y']['Cl']
    Cvp = packing_params['MellapakPlus252Y']['Cv']
    Cs = packing_params['MellapakPlus252Y']['Cs']
    Cp_0 = packing_params['MellapakPlus252Y']['Cp_0']
    Ch = packing_params['MellapakPlus252Y']['Ch']

    packing = a_p, ϵ, Clp, Cvp, Cs, Cp_0, Ch

    A = np.pi * .25 * D ** 2
    z = np.linspace(0, 1, n)


    return [Fl, Fv, Tl_z, Tv_0, z, H, A, P, packing], X
