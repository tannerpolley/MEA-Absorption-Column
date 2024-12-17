from numpy import log, exp


def heat_transfer(P, kv_CO2, kt_vap, Cpv_T, rho_mol_v, Dv_CO2, a_eA):

    # Compute Heat Transfer Coefficient
    # Ackmann_factor = Cpv[0]*N_CO2 + Cpv[1]*N_H2O
    # UT = ((kv_CO2*P)**3*kt_vap**2*Cpv_T/(rho_mol_v*Dv_CO2)**2)**(1/3) # J/(s*K*m^2)
    # UT = UT * a_eA # J/(s*K*m)
    log_UT = (3*(log(kv_CO2) + log(P)) + 2*log(kt_vap) + log(Cpv_T) - 2*(log(rho_mol_v) + log(Dv_CO2)))/3
    UT_base = exp(log_UT)
    UT = UT_base*a_eA
    # UT = Ackmann_factor/(1 - exp(-Ackmann_factor/(UT*a_e*A)))/(a_e*A)
    # UT = 7740.18346803192
    return UT_base, UT





