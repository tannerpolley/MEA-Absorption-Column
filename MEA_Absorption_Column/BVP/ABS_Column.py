import numpy as np
from numpy import sum

from MEA_Absorption_Column.Parameters import MWs_l

from MEA_Absorption_Column.Properties.Thermophysical_Properties import (density, surface_tension, heat_capacity,
                                                                        thermal_conductivity, henrys_law, enthalpy)
from MEA_Absorption_Column.Properties.Transport_Properties import viscosity, diffusivity

from MEA_Absorption_Column.Thermodynamics.Fugacity import fugacity
from MEA_Absorption_Column.Thermodynamics.Chemical_Equilibrium import chemical_equilibrium
from MEA_Absorption_Column.misc.Get_Temperature_Enthalpy import get_liquid_temperature, get_vapor_temperature

from MEA_Absorption_Column.Transport.Hydraulic_Variables_Correlations import (velocity, holdup, interfacial_area,
                                                                              flooding_fraction,
                                                                              mass_transfer_coeff, heat_transfer_coeff,
                                                                              pressure_drop,enhancement_factor)

def abs_column(zi, Y_scaled, parameters, run_type='simulating', column_names=False):

    # region - Unpack System Parameters
    scales, const_flow, A, packing = parameters
    Fl_MEA, Fv_N2, Fv_O2 = const_flow
    # endregion

    # region - Define System Variables
    Y = np.array(Y_scaled) * np.array(scales)
    Fl_CO2, Fl_H2O, Fv_CO2, Fv_H2O, Hlf, Hvf, P = Y

    Fl_T = Fl_CO2 + Fl_MEA + Fl_H2O
    Fv_T = Fv_CO2 + Fv_H2O + Fv_N2 + Fv_O2

    Fl = [Fl_CO2, Fl_MEA, Fl_H2O]
    Fv = [Fv_CO2, Fv_H2O, Fv_N2, Fv_O2]

    x = [Fl[i] / Fl_T for i in range(len(Fl))]
    y = [Fv[i] / Fv_T for i in range(len(Fv))]

    Hl = Hlf / Fl_T
    Hv = Hvf / Fv_T

    Tl = get_liquid_temperature(x, Hl)
    Tv = get_vapor_temperature(y, Hv)

    w = [MWs_l[i] * x[i] / sum([MWs_l[j] * x[j] for j in range(len(Fl))]) for i in range(len(Fl))]

    alpha = x[0] / x[1]
    w_MEA = w[1]
    w_H2O = w[2]
    # endregion

    # region - Properties

    # region -- Thermophysical Properties

    # region --- Henry's Law
    H_CO2_mix = henrys_law(Tl, x)
    # endregion

    # region --- Density
    rho_mol_l, rho_mass_l, volume = density(Tl, x, P, phase='liquid')
    rho_mol_v, rho_mass_v = density(Tv, y, P, phase='vapor')
    # endregion

    # region --- Surface Tension
    sigma = surface_tension(Tl, x, w_MEA, w_H2O)
    # endregion

    # region --- Heat Capacity
    Cpl, Cpl_T = heat_capacity(Tl, x, w, phase='liquid')
    Cpv, Cpv_T = heat_capacity(Tv, y, w, phase='vapor')
    # endregion

    # region --- Enthalpy
    Hl_CO2, Hl_MEA, Hl_H2O = enthalpy(Tl, x, phase='liquid')[0]  # J/mol
    Hv_CO2, Hv_H2O, Hv_N2, Hv_O2 = enthalpy(Tv, y, phase='vapor')[0]  # J/mol
    # endregion

    # endregion

    # region -- Transport Properties

    # region --- Viscosity
    mul_mix, mul_H2O = viscosity(Tl, x, w_MEA, w_H2O, phase='liquid')
    muv_mix, muv = viscosity(Tv, y, w_MEA, w_H2O, phase='vapor')
    # endregion

    # region --- Diffusivity
    Dl_CO2, Dl_MEA, Dl_ion = diffusivity(Tl, x, P, mul_mix, rho_mol_l,phase='liquid')
    Dv_CO2, Dv_H2O, Dv_N2, Dv_O2, Dv_T = diffusivity(Tv, y, P, mul_mix, rho_mol_l, phase='vapor')
    # endregion

    # region --- Thermal Conductivity
    kt_vap = thermal_conductivity(Tv, y, muv)
    # endregion

    # endregion

    # endregion

    # region - Thermodynamics

    # region -- Chemical Equilibrium
    Cl_true, x_true = chemical_equilibrium(Fl.copy(), Tl)

    Cl = [x[i] * rho_mol_l for i in range(len(x))]
    Cv = [y[i] * rho_mol_v for i in range(len(y))]

    Cl_true = [x_true[i] * rho_mol_l for i in range(len(x_true))]

    # endregion

    # region -- Vapor-Liquid Equilibrium

    fl_CO2, fv_CO2, fl_H2O, fv_H2O, CO2, H2O = fugacity(x, y, x_true, Cl_true, Tl, Tv, alpha, H_CO2_mix, P)

    # endregion

    # endregion

    # region - Transport

    # region -- Hydraulic Variables and Correlations

    # region --- Velocity
    ul, uv = velocity(rho_mol_l, rho_mol_v, A, Fl_T, Fv_T)
    # endregion

    # region --- Interfacial Area
    a_e, a_eA = interfacial_area(rho_mass_l, sigma, ul, A, packing)
    # endregion

    # region --- Holdup
    h_L, h_V = holdup(ul, mul_mix, rho_mass_l, packing)
    # endregion

    # region --- Flooding Fraction
    fl_frac = flooding_fraction(rho_mass_l, rho_mass_v, mul_mix, mul_H2O, Fl_T, Fv_T, uv, packing)
    # endregion

    # region --- Mass Transfer Coefficients

    kl_CO2, kv_CO2, kv_H2O, kv_T, const = mass_transfer_coeff(h_L, h_V, rho_mass_v, muv_mix, Dl_CO2, Dv_CO2, Dv_H2O,
                                                              Dv_T, A, Tv, ul, uv, packing)

    # endregion

    # region --- Heat Transfer Coefficient
    UT = heat_transfer_coeff(P, kv_CO2, kt_vap, Cpv_T, rho_mol_v, Dv_CO2, a_eA)  # J/(s*K*m) or W/(K*m)
    # endregion

    # region --- Pressure Drop

    ΔP_H = pressure_drop(h_L, rho_mass_l, rho_mass_v, mul_mix, muv_mix, A, ul, uv, packing)

    # endregion

    # region --- Enhancement Factor

    E, Psi, Psi_H, enhance_factor = enhancement_factor(Tl, Cl_true, y[0], P, H_CO2_mix, kl_CO2, kv_CO2,
                                                    Dl_CO2, Dl_MEA, Dl_ion, E_type='explicit')

    # endregion

    # endregion

    # region -- Flux

    # region --- Molar Flux

    # Liquid
    Nv_CO2 = -kv_CO2 * a_eA * (fv_CO2 - fl_CO2) * Psi_H  # mol/(s*m)
    Nv_H2O = -kv_H2O * a_eA * (fv_H2O - fl_H2O)  # mol/(s*m)

    # Vapor
    Nl_CO2 = -Nv_CO2  # mol/(s*m)
    Nl_H2O = -Nv_H2O  # mol/(s*m)

    # endregion

    # region --- Enthalpy Flux

    # region ---- Enthalpy Transfer

    Hl_CO2_trn = Nl_CO2 * Hl_CO2  # J/(s*m)
    Hl_H2O_trn = Nl_H2O * Hl_H2O  # J/(s*m)
    Hv_CO2_trn = Nv_CO2 * Hv_CO2  # J/(s*m)
    Hv_H2O_trn = Nv_H2O * Hv_H2O  # J/(s*m)

    Hv_trn = Nv_CO2 * Hv_CO2 + Nv_H2O * Hv_H2O  # J/(s*m)
    # Hl_trn = Nl_CO2 * Hl_CO2 + Nl_H2O * Hl_H2O  # J/(s*m)
    Hl_trn = -Hv_trn  # J/(s*m)

    # endregion

    # region ---- Heat Transfer
    qv = -UT * a_eA * (Tv - Tl)  # J/(s*m) =  J/(s*K*m) * K * m or W/(K*m) * K
    # ql =  UT * a_eA * (Tv - Tl)  # J/(s*m) =  J/(s*K*m) * K * m or W/(K*m) * K
    ql = -qv  # J/(s*m) =  J/(s*K*m) * K * m or W/(K*m) * K

    Hv_flux = Hv_trn + qv
    Hl_flux = Hl_trn + ql

    # endregion

    # endregion

    # endregion

    # endregion

    # region - Balance Equations

    # region -- Balance Equation Setup



    # endregion
    
    # end region

    # region -- Mass Balance
    dFl_CO2_dz = -Nl_CO2 + 1e-10  # mol/(s*m)
    dFl_H2O_dz = -Nl_H2O + 1e-10  # mol/(s*m)

    dFv_CO2_dz = Nv_CO2 + 1e-10  # mol/(s*m)
    dFv_H2O_dz = Nv_H2O + 1e-10  # mol/(s*m)
    # endregion

    # region -- Energy Balance
    dHlf_dz = Hl_flux
    dHvf_dz = Hv_flux + 1e-8

    dTl_dz = dHlf_dz / (Cpl_T * Fl_T)  # K/m
    dTv_dz = dHvf_dz / (Cpv_T * Fv_T)  # K/m
    # endregion

    # region -- Momentum Balance
    dP_dz = 0 / scales[6]  # Pa/m
    # endregion

    # endregion

    # region - Run Output

    if run_type == 'simulating':
        # Combine Differentials and Scale
        diffeqs = np.array([dFl_CO2_dz, dFl_H2O_dz, dFv_CO2_dz, dFv_H2O_dz, dHlf_dz, dHvf_dz, dP_dz])
        diffeqs_scaled = diffeqs / scales
        return diffeqs_scaled

    elif run_type == 'saving':
        Fl_true = [Cl_true[i] * ul * A for i in range(len(Cl_true))]

        Fl_CO2, Fl_MEA, Fl_H2O = Fl
        Fl_CO2_true, Fl_MEA_true, Fl_H2O_true, Fl_MEAH_true, Fl_MEACOO_true, Fl_HCO3_true = Fl_true
        Fv_CO2, Fv_H2O, Fv_N2, Fv_O2 = Fv
        Cl_CO2, Cl_MEA, Cl_H2O = Cl
        Cl_CO2_true, Cl_MEA_true, Cl_H2O_true, Cl_MEAH_true, Cl_MEACOO_true, Cl_HCO3_true = Cl_true
        x_CO2, x_MEA, x_H2O = x
        x_CO2_true, x_MEA_true, x_H2O_true, x_MEAH_true, x_MEACOO_true, x_HCO3_true = x_true
        y_CO2, y_H2O, y_N2, y_O2 = y
        Cv_CO2, Cv_H2O, Cv_N2, Cv_O2 = Cv
        DF_CO2, H_CO2_mix = CO2
        DF_H2O, Psat_H2O = H2O
        k2, Cl_MEA_true, Dl_CO2, kl_CO2, Ha, E, Psi_H, Psi = enhance_factor
        Cpl_CO2, Cpl_MEA, Cpl_H2O = Cpl
        Cpv_CO2, Cpv_H2O, Cpv_N2, Cpv_O2 = Cpv
        V_l, V_CO2, V_MEA, V_H2O = volume
        Hl_CO2 = Hl_CO2 + 1e-5
        Clp, Cvp, eps, a_p, A, Lp, d_h = const
        muv_CO2, muv_H2O, muv_N2, muv_O2 = muv

        output_dict = {
            'Fl': [Fl_CO2, Fl_MEA, Fl_H2O, Fl_T,
                   Fl_CO2_true, Fl_MEA_true, Fl_H2O_true, Fl_MEAH_true, Fl_MEACOO_true, Fl_HCO3_true],
            'Fv': [Fv_CO2, Fv_H2O, Fv_N2, Fv_O2, Fv_T],
            'Cl': [Cl_CO2, Cl_MEA, Cl_H2O,
                   Cl_CO2_true, Cl_MEA_true, Cl_H2O_true, Cl_MEAH_true, Cl_MEACOO_true, Cl_HCO3_true],
            'Cv': [Cv_CO2, Cv_H2O, Cv_N2, Cv_O2],
            'x': [x_CO2, x_MEA, x_H2O,
                  x_CO2_true, x_MEA_true, x_H2O_true, x_MEAH_true, x_MEACOO_true, x_HCO3_true],
            'y': [y_CO2, y_H2O, y_N2, y_O2],
            'T': [Tl, Tv],
            'Hl': [Tl, Hl_CO2, Hl_MEA, Hl_H2O, Hl, Hl_CO2_trn, Hl_H2O_trn, Hl_trn, ql, Hl_flux, dHlf_dz, Hlf],
            'Hv': [Tv, Hv_CO2, Hv_H2O, Hv_N2, Hv_O2, Hv, Hv_CO2_trn, Hv_H2O_trn, Hv_trn, qv, Hv_flux, dHvf_dz, Hvf],
            'CO2': [Nl_CO2, Nv_CO2, kv_CO2, a_eA, DF_CO2, fv_CO2, fl_CO2, Psi, H_CO2_mix],
            'H2O': [Nl_H2O, Nv_H2O, kv_H2O, a_eA, DF_H2O, fv_H2O, fl_H2O, Psat_H2O],
            'enhance_factor': [k2, Cl_MEA_true, Dl_CO2, kl_CO2, Ha, E, Psi, Psi_H],
            'transport': [kl_CO2, kv_CO2, kv_H2O, ul, uv, h_L, h_V, a_e, UT, P,
                          Clp, Cvp, eps, a_p, A, Lp, d_h],
            'Prop_l': [rho_mol_l, rho_mass_l, V_l, V_CO2, V_MEA, V_H2O, mul_mix, sigma, Dl_CO2, Dl_MEA,
                       Dl_ion, Cpl_CO2,
                       Cpl_MEA, Cpl_H2O],
            'Prop_v': [rho_mol_v, rho_mass_v, muv_CO2, muv_H2O, muv_N2, muv_O2, muv_mix, Dv_CO2, Dv_H2O,
                       Cpv_CO2, Cpv_H2O, Cpv_N2, Cpv_O2, kt_vap],
        }

        if zi == 0 and column_names:
            locals_dict = locals().items()
            keys_dict = {}
            for k, v in output_dict.items():
                key_list = []
                for vi in v:
                    for k2, v2 in locals_dict:
                        if isinstance(v2, float):
                            if vi == v2:
                                key_list.append(k2)
                                continue
                keys_dict[k] = key_list
        else:
            keys_dict = None
        return output_dict, keys_dict
    else:
        raise ValueError('Choose correct run type')
    # endregion
