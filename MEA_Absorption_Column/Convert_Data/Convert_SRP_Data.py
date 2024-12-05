from MEA_Absorption_Column.Parameters import MWs_l, MWs_v, n, column_params, packing_params
import numpy as np


def convert_SRP_data(X, mass):

    MW_CO2 = MWs_l[0]
    MW_MEA = MWs_l[1]
    MW_H2O = MWs_l[2]
    MW_N2 = MWs_v[2]
    MW_O2 = MWs_v[3]

    alpha_O2_N2 = 0.085
    P = 109180
    D = column_params['SRP']['D']
    H = column_params['SRP']['H']
    # a_p = packing_params['IMTP-40']['a_e']
    # ϵ = packing_params['IMTP-40']['eps']
    # Clp = packing_params['IMTP-40']['Cl']
    # Cvp = packing_params['IMTP-40']['Cv']
    # S = packing_params['IMTP-40']['S']

    a_p = packing_params['MellapakPlus252Y']['a_e']
    ϵ = packing_params['MellapakPlus252Y']['eps']
    Clp = packing_params['MellapakPlus252Y']['Cl']
    Cvp = packing_params['MellapakPlus252Y']['Cv']
    Cs = packing_params['MellapakPlus252Y']['Cs']
    Cp_0 = packing_params['MellapakPlus252Y']['Cp_0']
    Ch = packing_params['MellapakPlus252Y']['Ch']

    packing = a_p, ϵ, Clp, Cvp, Cs, Cp_0, Ch

    # packing = a_p, ϵ, Clp, Cvp, S

    A = np.pi * D ** 2 / 4
    z = np.linspace(0, H, n)

    if mass:
        Tl_z, Tv_0, ml_z, mv_0, alpha, w_MEA, y_H2O, y_CO2 = X

        # Liquid Calculations

        # Find Liquid Mass Flow Rates
        ml_MEA_z = w_MEA * ml_z  # kg/s
        ml_CO2_z = ml_MEA_z * alpha / MW_MEA * MW_CO2  # kg/s
        ml_H2O_z = ml_z - ml_MEA_z - ml_CO2_z  # kg/s

        # print(ml_CO2_z, ml_MEA_z, ml_H2O_z)

        # Find Liquid Molar Flow Rates
        Fl_CO2_z = ml_CO2_z / MW_CO2  # mole/s
        Fl_MEA_z = ml_MEA_z / MW_MEA  # mole/s
        Fl_H2O_z = ml_H2O_z / MW_H2O  # mole/s

        # Vapor Calculations

        # Find Vapor Mole Fractions
        y_N2 = (1 - y_CO2 - y_H2O) / (1 + alpha_O2_N2)
        y_O2 = y_N2 * alpha_O2_N2
        sigma = y_N2 * MW_N2 + y_O2 * MW_O2 + y_CO2 * MW_CO2 + y_H2O * MW_H2O

        # Find Vapor Mass Flow Rates

        w_CO2_v = y_CO2 * MW_CO2 / sigma
        w_H2O_v = y_H2O * MW_H2O / sigma
        w_N2_v = y_N2 * MW_N2 / sigma
        w_O2_v = y_O2 * MW_O2 / sigma

        m_CO2_v = w_CO2_v * mv_0
        m_H2O_v = w_H2O_v * mv_0
        m_N2_v = w_N2_v * mv_0
        m_O2_v = w_O2_v * mv_0

        # Find Vapor Molar Flow Rates
        Fv_CO2 = m_CO2_v / MW_CO2  # mole/s
        Fv_H2O = m_H2O_v / MW_H2O  # mole/s
        Fv_N2 = m_N2_v / MW_N2  # mole/s
        Fv_O2 = m_O2_v / MW_O2  # mole/s

        Fl = [Fl_CO2_z, Fl_MEA_z, Fl_H2O_z]
        Fv = [Fv_CO2, Fv_H2O, Fv_N2, Fv_O2]

        Fl_T = sum(Fl)
        Fv_T = sum(Fv)

    else:
        Tl_z, Tv_0, Fl_z, Fv_0, alpha, w_MEA, y_H2O, y_CO2 = X

        # x_MEA = ((1 + alpha + (MW_MEA/MW_H2O))*(1-w_MEA)/w_MEA)**-1
        # x_CO2 = x_MEA*alpha
        # x_H2O = 1 - x_CO2 - x_MEA

        # Fl_CO2_z = x_CO2*Fl_z
        # Fl_MEA_z = x_MEA*Fl_z
        # Fl_H2O_z = x_H2O*Fl_z

        x_MEA_unloaded = w_MEA / (MW_MEA / MW_H2O + w_MEA * (1 - MW_MEA / MW_H2O))
        x_H2O_unloaded = 1 - x_MEA_unloaded

        Fl_MEA_z = Fl_z * x_MEA_unloaded
        Fl_H2O_z = Fl_z * x_H2O_unloaded

        Fl_CO2_z = Fl_MEA_z*alpha

        Fl_z = Fl_CO2_z + Fl_MEA_z + Fl_H2O_z

        x_CO2 = Fl_CO2_z/Fl_z
        x_MEA = Fl_MEA_z/Fl_z
        x_H2O = Fl_H2O_z/Fl_z

        ml_CO2_z = Fl_CO2_z * MW_CO2
        ml_MEA_z = Fl_MEA_z * MW_MEA
        ml_H2O_z = Fl_H2O_z * MW_H2O

        # print(ml_CO2_z, ml_MEA_z, ml_H2O_z)

        m_T_l = ml_CO2_z + ml_MEA_z +  ml_H2O_z

        y_N2 = (1 - y_CO2 - y_H2O)/(1 + alpha_O2_N2)
        y_O2 = alpha_O2_N2*y_N2

        Fv_CO2_0 = Fv_0 * y_CO2
        Fv_H2O_0 = Fv_0 * y_H2O
        Fv_N2_0 = Fv_0 * y_N2
        Fv_O2_0 = Fv_0 * y_O2

        mv_CO2_0 = Fv_CO2_0 * MW_CO2
        mv_H2O_0 = Fv_H2O_0 * MW_H2O
        mv_N2_0 = Fv_N2_0 * MW_N2
        mv_O2_0 = Fv_O2_0 * MW_O2

        m_T_v = mv_CO2_0 + mv_H2O_0 + mv_N2_0 + mv_O2_0

        # print(m_T_l, m_T_v)

        Fl = [Fl_CO2_z, Fl_MEA_z, Fl_H2O_z]
        Fv = [Fv_CO2_0, Fv_H2O_0, Fv_N2_0, Fv_O2_0]

    return Fl, Fv, Tl_z, Tv_0, z, A, P, packing



