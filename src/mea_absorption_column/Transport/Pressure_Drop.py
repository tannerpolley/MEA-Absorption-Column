import numpy as np
from ..config.Constants import g
from .domain_guards import require_fraction_between, require_positive


def pressure_drop(h_L, rho_mass_l, rho_mass_v, mul_mix, muv_mix, A, ul, uv, packing, diagnostics=None):

    a_p, ϵ, Clp, Cvp, Cs, Cp_0, Ch = packing
    require_positive(
        "pressure_drop",
        diagnostics,
        rho_mass_l=rho_mass_l,
        rho_mass_v=rho_mass_v,
        mul_mix=mul_mix,
        muv_mix=muv_mix,
        A=A,
        ul=ul,
        uv=uv,
        a_p=a_p,
        eps=ϵ,
        Cp_0=Cp_0,
        Ch=Ch,
    )
    require_fraction_between("pressure_drop", "h_L", h_L, 0.0, ϵ, diagnostics)
    D = (A * 4 / np.pi) ** 0.5

    # Liquid Hold up at loading point
    Re = ul * rho_mass_l / (a_p * mul_mix)
    if Re < 5:
        a_h_a_p = Ch * Re ** .15 * (uv ** 2 * a_p / g) ** .1
    elif Re >= 5:
        a_h_a_p = .85 * Ch * Re ** .25 * (uv ** 2 * a_p / g) ** .1
    else:
        a_h_a_p = 0

    h_Ls = (12 / g * mul_mix / rho_mass_l * ul * a_p ** 2) ** (1 / 3) * a_h_a_p ** (2 / 3)
    require_positive("pressure_drop", diagnostics, h_Ls=h_Ls)

    νv = muv_mix / rho_mass_v
    Fv = uv * rho_mass_v ** 0.5
    ds = D
    dp = 6 * (1 - ϵ) / a_p
    K = (1 + 2 / 3 * (1 / (1 - ϵ)) * dp / ds) ** -1
    Re_v = uv * dp / ((1 - ϵ) * νv) * K
    require_positive("pressure_drop", diagnostics, K=K, Re_v=Re_v)
    C1 = 13300 / (a_p ** (3 / 2))
    Fr_L = ul ** 2 * a_p / g
    Ψ_L = Cp_0 * (64 / Re_v + 1.8 / Re_v ** .08) * ((ϵ - h_L) / ϵ) ** 1.5 * (h_L / h_Ls) ** .3 * np.exp(
        C1 * np.sqrt(Fr_L))
    require_positive("pressure_drop", diagnostics, psi_l=Ψ_L)
    ΔP_H = Ψ_L * a_p / (ϵ - h_L) ** 3 * Fv ** 2 / (2 * K)
    require_positive("pressure_drop", diagnostics, pressure_drop=ΔP_H)

    return ΔP_H
