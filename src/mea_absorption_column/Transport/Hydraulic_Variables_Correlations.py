import numpy as np
from ..config.Constants import g
from mea_absorption_column.BVP.robust_core import record_domain_guard
from .domain_guards import require_fraction_between, require_positive


def velocity(rho_mol_l, rho_mol_v, A, Fl_T, Fv_T, diagnostics=None):
    require_positive("hydraulics", diagnostics, rho_mol_l=rho_mol_l, rho_mol_v=rho_mol_v, A=A, Fl_T=Fl_T, Fv_T=Fv_T)
    ul = Fl_T / (A * rho_mol_l)
    uv = Fv_T / (A * rho_mol_v)
    require_positive("hydraulics", diagnostics, ul=ul, uv=uv)

    return ul, uv


def interfacial_area(rho_mass_l, sigma, ul, A, packing, diagnostics=None):

    a_p, ϵ, Clp, Cvp, Cs, Cp_0, Ch = packing
    require_positive("hydraulics", diagnostics, rho_mass_l=rho_mass_l, sigma=sigma, ul=ul, A=A, a_p=a_p, eps=ϵ)

    Lp = A * a_p / ϵ

    # Compute interfacial area
    A1 = 1.42
    A2 = .12
    # a_e = a_p * A1 * (rho_mass_l / sigma * g ** 1/3 * (ul * A / Lp) ** (4 / 3)) ** A2
    a_e = np.log(a_p) + np.log(A1) + A2 * (
                np.log(rho_mass_l) - np.log(sigma) + 1 / 3 * np.log(g) + 4 / 3 * (np.log(ul) + np.log(A) - np.log(Lp)))
    a_e = np.exp(a_e)
    a_eA = a_e * A  # Combining cross-sectional area and interfacial area
    require_positive("hydraulics", diagnostics, a_e=a_e, a_eA=a_eA)

    return a_e, a_eA


def holdup(ul, mul_mix, rho_mass_l, packing, diagnostics=None):

    a_p, ϵ, Clp, Cvp, Cs, Cp_0, Ch = packing
    require_positive("hydraulics", diagnostics, ul=ul, mul_mix=mul_mix, rho_mass_l=rho_mass_l, eps=ϵ)

    # - Liquid Hold Up
    # -- Regressed Parameters from Chinen 2018
    # -- Correlation from Tsai 2010

    A_param = 11.4474
    B_param = .6471
    alpha = 3.185966

    h_L_raw = A_param * ((ul * alpha) * (mul_mix / rho_mass_l) ** (1 / 3)) ** B_param
    if not np.isfinite(h_L_raw) or h_L_raw <= 0.0:
        record_domain_guard(diagnostics, "hydraulics", f"bounded liquid holdup from {h_L_raw!r}")
        h_L_raw = ϵ
    elif h_L_raw >= ϵ:
        record_domain_guard(diagnostics, "hydraulics", f"bounded liquid holdup from {h_L_raw!r}")
    h_L_raw = max(float(h_L_raw), ϵ * 1.0e-12)
    h_L = ϵ * h_L_raw / (ϵ + h_L_raw)
    h_V = ϵ - h_L
    require_fraction_between("hydraulics", "h_L", h_L, 0.0, ϵ, diagnostics)
    require_positive("hydraulics", diagnostics, h_V=h_V)

    return h_L, h_V




def flooding_fraction(rho_mass_l, rho_mass_v, mul_mix, mul_H2O, Fl_T, Fv_T, uv, packing, diagnostics=None):

    a_p, ϵ, Clp, Cvp, Cs, Cp_0, Ch = packing
    require_positive(
        "hydraulics",
        diagnostics,
        rho_mass_l=rho_mass_l,
        rho_mass_v=rho_mass_v,
        mul_mix=mul_mix,
        mul_H2O=mul_H2O,
        Fl_T=Fl_T,
        Fv_T=Fv_T,
        uv=uv,
        a_p=a_p,
        eps=ϵ,
    )

    # Flooding
    H = (Fl_T / Fv_T) * (rho_mass_v / rho_mass_l) ** (1 / 2)
    uv_FL = ((g * ϵ ** 3 / a_p) * (rho_mass_l / rho_mass_v) * (mul_mix / mul_H2O) ** (-.2) * np.exp(-4 * H ** .25)) ** .5
    flood_fraction = uv / uv_FL
    require_positive("hydraulics", diagnostics, uv_FL=uv_FL, flood_fraction=flood_fraction)

    return flood_fraction
