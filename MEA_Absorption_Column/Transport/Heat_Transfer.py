from numpy import log, exp


def heat_transfer(P, kv_CO2, kt_CO2, Cpv_T, rho_mol_v, Dv_CO2):

    # Compute Heat Transfer Coefficient

    UT = exp((3 * (log(kv_CO2) + log(P)) + 2 * log(kt_CO2) + log(Cpv_T) - 2*(log(rho_mol_v) + log(Dv_CO2)))/3)

    return UT





