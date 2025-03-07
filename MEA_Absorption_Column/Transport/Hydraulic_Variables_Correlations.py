import numpy as np
from numpy import log, exp
from MEA_Absorption_Column.Parameters import R, g


def velocity(rho_mol_l, rho_mol_v, A, Fl_T, Fv_T):
    ul = Fl_T / (A * rho_mol_l)
    uv = Fv_T / (A * rho_mol_v)

    return ul, uv

def interfacial_area(rho_mass_l, sigma, ul, A, packing):

    a_p, ϵ, Clp, Cvp, Cs, Cp_0, Ch = packing

    Lp = A * a_p / ϵ

    # Compute interfacial area
    A1 = 1.43914
    A2 = .12
    # a_e = a_p * A1 * (rho_mass_l / sigma * g ** 1/3 * (ul * A / Lp) ** (4 / 3)) ** A2
    a_e = np.log(a_p) + np.log(A1) + A2 * (
                np.log(rho_mass_l) - np.log(sigma) + 1 / 3 * np.log(g) + 4 / 3 * (np.log(ul) + np.log(A) - np.log(Lp)))
    a_e = np.exp(a_e)
    a_eA = a_e * A  # Combining cross-sectional area and interfacial area

    return a_e, a_eA


def holdup(ul, mul_mix, rho_mass_l, packing):

    a_p, ϵ, Clp, Cvp, Cs, Cp_0, Ch = packing

    # - Liquid Hold Up
    # -- Regressed Parameters from Chinen 2018
    # -- Correlation from Tsai 2010

    A_param = 11.4474
    B_param = .6471
    alpha = 3.185966

    h_L = A_param * ((ul * alpha) * (mul_mix / rho_mass_l) ** (1 / 3)) ** B_param
    h_V = ϵ - h_L

    # if h_V < 0:
    #     print('Error: Flooding as occurred')
    #     raise TypeError

    return h_L, h_V




def flooding_fraction(rho_mass_l, rho_mass_v, mul_mix, mul_H2O, Fl_T, Fv_T, uv, packing):

    a_p, ϵ, Clp, Cvp, Cs, Cp_0, Ch = packing

    # Flooding
    H = (Fl_T / Fv_T) * (rho_mass_v / rho_mass_l) ** (1 / 2)
    uv_FL = ((g * ϵ ** 3 / a_p) * (rho_mass_l / rho_mass_v) * (mul_mix / mul_H2O) ** (-.2) * np.exp(-4 * H ** .25)) ** .5
    flood_fraction = uv / uv_FL

    return flood_fraction


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


def pressure_drop(h_L, rho_mass_l, rho_mass_v, mul_mix, muv_mix, A, ul, uv, packing):

    a_p, ϵ, Clp, Cvp, Cs, Cp_0, Ch = packing
    D = (A * 4 / np.pi) ** 1 / 2

    # Liquid Hold up at loading point
    Re = ul * rho_mass_l / (a_p * mul_mix)
    if Re < 5:
        a_h_a_p = Ch * Re ** .15 * (uv ** 2 * a_p / g) ** .1
    elif Re >= 5:
        a_h_a_p = .85 * Ch * Re ** .25 * (uv ** 2 * a_p / g) ** .1
    else:
        a_h_a_p = 0

    h_Ls = (12 / g * mul_mix / rho_mass_l * ul * a_p ** 2) ** (1 / 3) * a_h_a_p ** (2 / 3)

    νv = muv_mix / rho_mass_v
    Fv = uv / rho_mass_v ** 1 / 2
    ds = D
    dp = 6 * (1 - ϵ) / a_p
    K = (1 + 2 / 3 * (1 / (1 - ϵ)) * dp / ds) ** -1
    Re_v = uv * dp / ((1 - ϵ) * νv) * K
    C1 = 13300 / (a_p ** 3 / 2)
    Fr_L = ul ** 2 * a_p / g
    Ψ_L = Cp_0 * (64 / Re_v + 1.8 / Re_v ** .08) * ((ϵ - h_L) / ϵ) ** 1.5 * (h_L / h_Ls) ** .3 * np.exp(
        C1 * np.sqrt(Fr_L))
    ΔP_H = Ψ_L * a_p / (ϵ - h_L) ** 3 ** Fv ** 2 / 2 * 1 / K

    return ΔP_H


def enhancement_factor(Tl, Cl_true, y_CO2, P,
                       H_CO2_mix, kl_CO2, kv_CO2,
                       Dl_CO2, Dl_MEA, Dl_ion, E_type='explicit'):
    enable_enhancement_factor = True

    Cl_CO2_true, Cl_MEA_true, Cl_H2O_true, Cl_MEAH_true, Cl_MEACOO_true, Cl_HCO3_true = Cl_true
    Cl_CO2_true = Cl_CO2_true / 1.04542981654115
    Dl_MEAH = Dl_ion
    Dl_MEACOO = Dl_ion

    k_MEA = 2.003e4 * exp(-4742.0 / Tl)
    k_H2O = 4.147 * exp(-3110 / Tl)
    k2 = (k_MEA * Cl_MEA_true + k_H2O * Cl_H2O_true)  # Pseudo Second Order from IDAES Luo

    Ha = (k2 * Cl_MEA_true * Dl_CO2) ** .5 / kl_CO2

    if enable_enhancement_factor:

        if E_type == 'implicit':

            def solve(x):

                E, Υ_MEA_int = x
                KH = E * kl_CO2 / kv_CO2 / (E * kl_CO2 / kv_CO2 + H_CO2_mix)
                Cl_CO2_int = (y_CO2 * P / KH + Cl_CO2_true) / (H_CO2_mix / KH + 1)
                Υ_CO2_bulk = Cl_CO2_true / Cl_CO2_int

                Υ_MEAH = 1 + Dl_MEA * Cl_MEA_true * ((1 - Υ_MEA_int) / (2 * Dl_MEAH * Cl_MEAH_true))
                Υ_MEACOO = 1 + Dl_MEA * Cl_MEA_true * ((1 - Υ_MEA_int) / (2 * Dl_MEACOO * Cl_MEACOO_true))
                Υ_CO2_int = Υ_CO2_bulk * Υ_MEAH * Υ_MEACOO / Υ_MEA_int ** 2

                E_inst = 1 + Dl_MEA * Cl_MEA_true / (2 * Dl_CO2 * Cl_CO2_int)

                eq1 = E - Ha * Υ_MEA_int ** (1 / 2) * (1 - Υ_CO2_int) / (1 - Υ_CO2_bulk)
                eq2 = E - (1 + (E_inst - 1) * (1 - Υ_MEA_int) / (1 - Υ_CO2_bulk))

                return eq1, eq2

            E, Cl_MEA_int = root(solve, array((65., .9))).x

        elif E_type == 'explicit':

            R_plus = (Dl_MEA * Cl_MEA_true) / (2 * Dl_MEAH * Cl_MEAH_true)
            R_minus = (Dl_MEA * Cl_MEA_true) / (2 * Dl_MEACOO * Cl_MEACOO_true)
            E_hat = (Dl_MEA * Cl_MEA_true) / (2 * Dl_CO2 * Cl_CO2_true)
            E = 1 + (Ha - 1) / (Ha * (R_plus + R_minus + 2) / E_hat + 1)

        else:
            raise ValueError('E_type must be explicit or explicit')

    else:
        E = Ha

    Psi = E * kl_CO2 / kv_CO2
    Psi_H = Psi / (Psi + H_CO2_mix)

    enhance_factor = [k2, Cl_MEA_true, Dl_CO2, kl_CO2, Ha, E, Psi_H, Psi]

    return E, Psi, Psi_H, enhance_factor