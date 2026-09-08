import numpy as np
from ..config.Constants import g
from .domain_guards import require_fraction_between, require_positive


def velocity(rho_mol_l, rho_mol_v, A, Fl_T, Fv_T, diagnostics=None):
    require_positive("hydraulics", diagnostics, rho_mol_l=rho_mol_l, rho_mol_v=rho_mol_v, A=A, Fl_T=Fl_T, Fv_T=Fv_T)
    ul = Fl_T / (A * rho_mol_l)
    uv = Fv_T / (A * rho_mol_v)
    require_positive("hydraulics", diagnostics, ul=ul, uv=uv)

    return ul, uv


def interfacial_area_expression(rho_mass_l, sigma, ul, A, packing):
    """Existing wetted-perimeter correlation, shared by numeric/CasADi paths."""
    a_p, eps, *_ = packing
    a_e = 1.42 * a_p * (rho_mass_l / sigma * g ** (1 / 3)
                        * (ul * eps / a_p) ** (4 / 3)) ** .12
    return a_e, a_e * A


def interfacial_area(rho_mass_l, sigma, ul, A, packing, diagnostics=None):

    a_p, ϵ, Clp, Cvp, Cs, Cp_0, Ch = packing
    require_positive("hydraulics", diagnostics, rho_mass_l=rho_mass_l, sigma=sigma, ul=ul, A=A, a_p=a_p, eps=ϵ)

    a_e, a_eA = interfacial_area_expression(rho_mass_l, sigma, ul, A, packing)
    require_positive("hydraulics", diagnostics, a_e=a_e, a_eA=a_eA)

    return a_e, a_eA


def holdup_expression(ul, mul_mix, rho_mass_l, packing):
    """Tsai/Chinen correlation; the caller must enforce 0 < h_L < epsilon."""
    h_L = 11.4474 * (ul * 3.185966 * (mul_mix / rho_mass_l) ** (1 / 3)) ** .6471
    return h_L, packing[1] - h_L


def holdup(ul, mul_mix, rho_mass_l, packing, diagnostics=None):

    a_p, ϵ, Clp, Cvp, Cs, Cp_0, Ch = packing
    require_positive("hydraulics", diagnostics, ul=ul, mul_mix=mul_mix, rho_mass_l=rho_mass_l, eps=ϵ)

    h_L, h_V = holdup_expression(ul, mul_mix, rho_mass_l, packing)
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
