import numpy as np
from numpy import log, exp
from ..config.Constants import R


def mass_transfer_coeff(h_L, h_V, rho_mass_v, muv_mix, Dl_CO2, Dv_CO2, Dv_H2O, Dv_T, A, Tv, ul, uv, packing):

    a_p, ϵ, Clp, Cvp, Cs, Cp_0, Ch = packing

    d_h = 4 * ϵ / a_p
    Lp = A * a_p / ϵ

    def f_kl(Dl):
        kl = Clp * (12 ** (1 / 6)) * ((ul / h_L) ** .5) * ((Dl / d_h) ** .5)  # m/s
        return kl

    def f_kv(Dv):
        kv = Cvp / R / Tv * np.sqrt(a_p / d_h / h_V) * Dv ** (2 / 3) * (muv_mix / rho_mass_v) ** (1 / 3) * (
                uv * rho_mass_v / a_p / muv_mix) ** (3 / 4)  # m/s
        return kv

    kl_CO2 = f_kl(Dl_CO2)
    kv_CO2 = f_kv(Dv_CO2)
    kv_H2O = f_kv(Dv_H2O)
    kv_T = f_kv(Dv_T) * (R * Tv)

    return kl_CO2, kv_CO2, kv_H2O, kv_T, [Clp, Cvp, ϵ, a_p, A, Lp, d_h]


def heat_transfer_coeff(P, kv_CO2, kt_vap, Cpv_T, rho_mol_v, Dv_CO2, a_eA):

    # Compute Heat Transfer Coefficient
    # Ackmann_factor = Cpv[0]*N_CO2 + Cpv[1]*N_H2O
    UT = ((kv_CO2*P)**3*kt_vap**2*Cpv_T/(rho_mol_v*Dv_CO2)**2)**(1/3) # J/(s*K*m^2)
    # UT = UT * a_eA # J/(s*K*m)
    log_UT = (3*(log(kv_CO2) + log(P)) + 2*log(kt_vap) + log(Cpv_T) - 2*(log(rho_mol_v) + log(Dv_CO2)))/3
    UT = exp(log_UT)
    # UT = Ackmann_factor/(1 - exp(-Ackmann_factor/(UT*a_e*A)))/(a_e*A)
    # UT = 7740.18346803192
    return UT
