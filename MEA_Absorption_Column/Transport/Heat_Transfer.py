from numpy import log, exp


def heat_transfer(P, kv_CO2, kt_CO2, Cpv_T, rho_mol_v, Dv_CO2, N_CO2, N_H2O, Cpv, a_e, A):

    # Compute Heat Transfer Coefficient
    Ackmann_factor = Cpv[0]*N_CO2 + Cpv[1]*N_H2O
    # UT = ((kv_CO2*P)**3*kt_CO2**2*Cpv_T/(rho_mol_v**2*Dv_CO2))/3
    UT = kv_CO2*P*Cpv_T*(kt_CO2/(rho_mol_v*Cpv_T*Dv_CO2))**(2/3)
    UT = Ackmann_factor/(1 - exp(-Ackmann_factor/(UT*a_e*A)))/(a_e*A)

    return UT





