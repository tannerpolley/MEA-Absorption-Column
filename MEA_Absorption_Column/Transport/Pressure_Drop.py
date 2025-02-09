import numpy as np
from MEA_Absorption_Column.Parameters import g


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
