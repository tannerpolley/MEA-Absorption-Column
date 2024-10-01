from MEA_Absorption_Column.Transport.Heat_Transfer import heat_transfer


def solve_temperature_change(Cl, Cv, Tl, Tv, α, w_MEA, N_CO2, N_H2O, kv_T, Dv_T, df_param):

    # Heat from Vaporization
    # ΔH_vap = heat_of_vaporization(Tl, 'H2O')
    ΔH_vap = 48000
    q_vap = -N_H2O * ΔH_vap

    # Heat from Reaction
    # alpha_st = (Cl[0] + Cl[6] + Cl[7]) / (Cl[1] + Cl[5] + Cl[6])
    # ΔH_rxn = -heat_of_reaction(alpha_st)
    ΔH_rxn = 82000
    q_rxn = N_CO2 * ΔH_rxn

    # Heat from Transfer
    UT, Cl_Cp_sum, Cv_Cp_sum = heat_transfer(Cl, Cv, Tl, Tv, α, w_MEA, kv_T, Dv_T, df_param)

    q_trn = -UT * (Tl - Tv)

    ql = q_trn + q_vap + q_rxn
    qv = q_trn

    return ql, qv, q_vap, q_rxn, q_trn, Cl_Cp_sum, Cv_Cp_sum
