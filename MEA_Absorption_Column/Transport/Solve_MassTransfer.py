from fluids.numerics import epsilon

from MEA_Absorption_Column.Parameters import g, R
import numpy as np

log = np.log
exp = np.exp


def solve_masstransfer(rho_mass_l, rho_mass_v, mul_mix, muv_mix, mul_H2O, sigma, Dl_CO2, Dv_CO2, Dv_H2O, Dv_T,
                       A, Tv, ul, uv, Fl_T, Fv_T, packing):
    a_p, ϵ, Clp, Cvp, Cs, Cp_0, Ch = packing
    D = (A * 4 / np.pi) ** 1 / 2

    # Liquid Hold Up
    # Regressed Parameters from Chinen 2018
    # Correlation from Tsai 2010
    A_param = 11.4474
    B_param = .6471
    alpha = 3.185966
    h_L = A_param * ((ul * alpha) * (mul_mix / rho_mass_l) ** (1 / 3)) ** B_param
    # h_L = 0.0563472715819975
    h_V = ϵ - h_L

    Re = ul * rho_mass_l / (a_p * mul_mix)
    if Re < 5:
        a_h_a_p = Ch * Re ** .15 * (uv ** 2 * a_p / g) ** .1
    elif Re >= 5:
        a_h_a_p = .85 * Ch * Re ** .25 * (uv ** 2 * a_p / g) ** .1
    else:
        a_h_a_p = 0

    h_Ls = (12 / g * mul_mix / rho_mass_l * ul * a_p ** 2) ** (1 / 3) * a_h_a_p ** (2 / 3)

    # Flooding
    H = (Fl_T / Fv_T) * (rho_mass_v / rho_mass_l) ** (1 / 2)
    uv_FL = ((g * ϵ ** 3 / a_p) * (rho_mass_l / rho_mass_v) * (mul_mix / mul_H2O) ** (-.2) * np.exp(
        -4 * H ** .25)) ** .5
    flood_fraction = uv / uv_FL

    # print(uv, uv_FL, flood_fraction)

    # if h_V < 0:
    #     print('Error: Flooding as occurred')
    #     raise TypeError

    d_h = 4 * ϵ / a_p
    Lp = A * a_p / ϵ

    # Compute Pressure Drop
    Re = ul * rho_mass_l / (a_p * mul_mix)

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

    # Compute interfacial area
    A1 = 1.43914
    A2 = .12
    # a_e = a_p * A1 * (rho_mass_l / sigma * g ** 1/3 * (ul * A / Lp) ** (4 / 3)) ** A2
    a_e = np.log(a_p) + np.log(A1) + A2 * (np.log(rho_mass_l) - np.log(sigma) + 1/3*np.log(g) + 4/3*(np.log(ul) + np.log(A) - np.log(Lp)))
    a_e = np.exp(a_e)
    def f_kl(Dl):
        kl = Clp * (12 ** (1 / 6)) * ((ul / h_L) ** .5) * ((Dl / d_h) ** .5)  # m/s
        return kl

    def f_kv(Dv):
        kv = Cvp / R / Tv * np.sqrt(a_p / d_h / h_V) * Dv ** (2 / 3) * (muv_mix / rho_mass_v) ** (1 / 3) * (
                uv * rho_mass_v / a_p / muv_mix) ** (3 / 4)  # m/s
        return kv

    kl_CO2 = f_kl(Dl_CO2)
    # kl_CO2 = 0.000105585744900102
    kv_CO2 = f_kv(Dv_CO2)
    kv_H2O = f_kv(Dv_H2O)
    kv_T = f_kv(Dv_T) * (R * Tv)

    return (
        ΔP_H, kl_CO2, kv_CO2, kv_H2O, kv_T, [kl_CO2, kv_CO2, kv_H2O], uv, a_e,
        [ul, uv, uv_FL, h_L, h_V, a_e, flood_fraction],
        [Clp, Cvp, ϵ, a_p, A, Lp, d_h])
