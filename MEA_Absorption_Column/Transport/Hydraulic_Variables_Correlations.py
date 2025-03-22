import numpy as np
from MEA_Absorption_Column.Parameters import g


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