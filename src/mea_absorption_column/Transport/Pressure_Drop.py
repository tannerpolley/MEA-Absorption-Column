import numpy as np
import casadi as ca
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
    require_fraction_between("pressure_drop", "eps", ϵ, 0.0, 1.0, diagnostics)
    require_fraction_between("pressure_drop", "h_L", h_L, 0.0, ϵ, diagnostics)
    value = pressure_drop_expression(h_L, rho_mass_l, rho_mass_v, mul_mix, muv_mix, A, ul, uv, packing, diagnostics=diagnostics)
    require_positive("pressure_drop", diagnostics, pressure_drop=value)
    return value


def pressure_drop_expression(h_L, rho_mass_l, rho_mass_v, mul_mix, muv_mix, A, ul, uv, packing, *, diagnostics=None):
    """Shared Pa/m correlation; positive inputs, 0<h_L<eps<1 required.

    Retains the source's two Reynolds-number regimes. Derivatives are exact
    within each branch; no differentiability at Re=5 is asserted or invented.
    """
    a_p, ϵ, Clp, Cvp, Cs, Cp_0, Ch = packing
    symbolic = any(isinstance(v, (ca.MX, ca.SX, ca.DM))
                   for v in (h_L, rho_mass_l, rho_mass_v, mul_mix, muv_mix, A, ul, uv))
    D = (A * 4 / np.pi) ** 0.5

    # Liquid Hold up at loading point
    Re = ul * rho_mass_l / (a_p * mul_mix)
    if isinstance(Re, (ca.MX, ca.SX, ca.DM)):
        reynolds_factor = ca.if_else(Re < 5, Re ** .15, .85 * Re ** .25)
    else:
        reynolds_factor = Re ** .15 if Re < 5 else .85 * Re ** .25
    a_h_a_p = Ch * reynolds_factor * (uv ** 2 * a_p / g) ** .1

    h_Ls = (12 / g * mul_mix / rho_mass_l * ul * a_p ** 2) ** (1 / 3) * a_h_a_p ** (2 / 3)
    if not symbolic:
        require_positive("pressure_drop", diagnostics, h_Ls=h_Ls)

    νv = muv_mix / rho_mass_v
    Fv = uv * rho_mass_v ** 0.5
    ds = D
    dp = 6 * (1 - ϵ) / a_p
    K = (1 + 2 / 3 * (1 / (1 - ϵ)) * dp / ds) ** -1
    Re_v = uv * dp / ((1 - ϵ) * νv) * K
    if not symbolic:
        require_positive("pressure_drop", diagnostics, K=K, Re_v=Re_v)
    C1 = 13300 / (a_p ** (3 / 2))
    Fr_L = ul ** 2 * a_p / g
    exponent = C1 * Fr_L ** .5
    exp_term = ca.exp(exponent) if isinstance(exponent, (ca.MX, ca.SX, ca.DM)) else np.exp(exponent)
    Ψ_L = Cp_0 * (64 / Re_v + 1.8 / Re_v ** .08) * ((ϵ - h_L) / ϵ) ** 1.5 * (h_L / h_Ls) ** .3 * exp_term
    if not symbolic:
        require_positive("pressure_drop", diagnostics, psi_l=Ψ_L)
    ΔP_H = Ψ_L * a_p / (ϵ - h_L) ** 3 * Fv ** 2 / (2 * K)

    return ΔP_H
