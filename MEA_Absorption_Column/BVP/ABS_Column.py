import numpy as np
from numpy import sum
from MEA_Absorption_Column.Transport.Enhancement_Factor import enhancement_factor_implicit, enhancement_factor_explicit
from MEA_Absorption_Column.Transport.Solve_MassTransfer import solve_masstransfer
from MEA_Absorption_Column.Thermodynamics.Solve_Driving_Force import solve_driving_force
from MEA_Absorption_Column.Thermodynamics.Solve_ChemEQ import solve_ChemEQ
from MEA_Absorption_Column.Properties.Henrys_Law import henrys_law
from MEA_Absorption_Column.Properties.Density import liquid_density, vapor_density
from MEA_Absorption_Column.Properties.Viscosity import viscosity
from MEA_Absorption_Column.Properties.Surface_Tension import surface_tension
from MEA_Absorption_Column.Properties.Diffusivity import liquid_diffusivity, vapor_diffusivity
from MEA_Absorption_Column.Properties.Heat_Capacity import heat_capacity
from MEA_Absorption_Column.Properties.Thermal_Conductivity import thermal_conductivity
from MEA_Absorption_Column.Parameters import MWs_l
from MEA_Absorption_Column.Transport.Heat_Transfer import heat_transfer
from MEA_Absorption_Column.Thermodynamics.Enthalpy import liquid_enthalpy, vapor_enthalpy
from MEA_Absorption_Column.Thermodynamics.Get_Temperature_Enthalpy import get_liquid_temperature, get_vapor_temperature


def abs_column(zi, Y_scaled, parameters, run_type='simulating', column_names=False):

    scales, const_flow, A, packing = parameters
    Fl_MEA, Fv_N2, Fv_O2 = const_flow

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

    # ------------------------------ Chemical Equilibrium --------------------------------------
    Cl_true, x_true = solve_ChemEQ(Fl.copy(), Tl)

    # -------------------------------- Properties ---------------------------------------------

    # Density and Concentration
    rho_mol_l, rho_mass_l, volume = liquid_density(Tl, x)
    rho_mol_v, rho_mass_v = vapor_density(Tv, P, y)

    # Velocity
    ul = Fl_T / (A * rho_mol_l)
    uv = Fv_T / (A * rho_mol_v)

    Cl = [x[i] * rho_mol_l for i in range(len(x))]
    Cv = [y[i] * rho_mol_v for i in range(len(y))]

    Fl_true = [Cl_true[i] * ul * A for i in range(len(Cl_true))]
    Cl_true = [x_true[i] * rho_mol_l for i in range(len(x_true))]

    # Viscosity
    muv_mix, mul_mix, mul_H2O, muv = viscosity(Tl, Tv, y, w_MEA, w_H2O, alpha)

    # Surface Tension
    sigma = surface_tension(Tl, x, w_MEA, w_H2O, alpha)

    # Diffusion
    Dl_CO2, Dl_MEA, Dl_ion = liquid_diffusivity(Tl, rho_mol_l * x[1], mul_mix)
    Dv_CO2, Dv_H2O, Dv_N2, Dv_O2, Dv_T = vapor_diffusivity(Tv, y, P)

    # Heat Capacity
    Cpl = heat_capacity(Tl, 'liquid', x, w)
    Cpv = heat_capacity(Tv, 'vapor', x, w)

    Cpl_T = sum([Cpl[i] * x[i] for i in range(len(x))])
    Cpv_T = sum([Cpv[i] * y[i] for i in range(len(y))])

    Sigma_Fl_Cpl = Cpl_T * Fl_T
    Sigma_Fv_Cpv = Cpv_T * Fv_T

    # Thermal Conductivity
    kt_vap = thermal_conductivity(y, Tv, muv, 'vapor')

    # Henry's Law
    H_CO2_mix = henrys_law(x, Tl)

    # ------------------------------ Transport --------------------------------------
    # Mass Transfer Coefficients and Properties
    ΔP_H, kl_CO2, kv_CO2, kv_H2O, kv_T, k_mxs, uv, a_e, hydr, const = solve_masstransfer(rho_mass_l, rho_mass_v,
                                                                                         mul_mix, muv_mix, mul_H2O,
                                                                                         sigma, Dl_CO2, Dv_CO2, Dv_H2O,
                                                                                         Dv_T,
                                                                                         A, Tv,
                                                                                         ul, uv, Fl_T, Fv_T,
                                                                                         packing)
    # Enhancement Factor
    enhance_factor = 'explicit'
    if enhance_factor == 'explicit':

        E, Psi, enhance_factor = enhancement_factor_explicit(Tl, y[0], P, Cl_true, H_CO2_mix, kl_CO2, kv_CO2,
                                                             Dl_CO2, Dl_MEA, Dl_ion)
    elif enhance_factor == 'implicit':

        E, Psi, enhance_factor = enhancement_factor_implicit(Tl, y[0], P, Cl_true, H_CO2_mix, kl_CO2, kv_CO2,
                                                             Dl_CO2, Dl_MEA, Dl_ion)
    else:
        raise ValueError('enhance_factor must be either "explicit" or "implicit"')

    # ------------------------------ Thermodynamics --------------------------------------

    DF_CO2, DF_H2O, CO2, H2O = solve_driving_force(x, y, x_true, Cl_true, Tl, alpha, H_CO2_mix, P, Psi)

    # ------------------------------ Material and Energy Flux Setup --------------------------------------

    a_eA = a_e * A  # m

    # Molar Flux
    Nl_CO2 = kv_CO2 * DF_CO2 * a_eA  # mol/(s*m)
    Nl_H2O = kv_H2O * DF_H2O * a_eA  # mol/(s*m)

    Nv_CO2 = -Nl_CO2
    Nv_H2O = -Nl_H2O

    # Heat Transfer Coefficient
    UT_base, UT = heat_transfer(P, kv_CO2, kt_vap, Cpv_T, rho_mol_v, Dv_CO2, a_eA)  # J/(s*K*m) or W/(K*m)

    # Enthalpy Transfer
    Hl_CO2, Hl_MEA, Hl_H2O = liquid_enthalpy(Tl, x)
    Hv_CO2, Hv_H2O, Hv_N2, Hv_O2 = vapor_enthalpy(Tv)

    Hl_CO2_trn, Hl_H2O_trn = Nl_CO2 * Hl_CO2, Nl_H2O * Hl_H2O
    Hv_CO2_trn, Hv_H2O_trn = Nv_CO2 * Hv_CO2, Nv_H2O * Hv_H2O

    Hl_trn = Hl_CO2_trn + Hl_H2O_trn
    Hv_trn = Hv_CO2_trn + Hv_H2O_trn
    Hl_trn = -Hv_trn

    # Heat Transfer
    Ql_trn = UT * (Tv - Tl)  # J/(s*m) =  J/(s*K*m) * K * m or W/(K*m) * K
    Qv_trn = -UT * (Tv - Tl)

    # ------------------------------ Material Balance --------------------------------------

    dFl_CO2_dz = -Nl_CO2 + 1e-10  # mol/(s*m)
    dFl_H2O_dz = -Nl_H2O + 1e-10  # mol/(s*m)

    dFv_CO2_dz = Nv_CO2 + 1e-10  # mol/(s*m)
    dFv_H2O_dz = Nv_H2O + 1e-10  # mol/(s*m)

    # ------------------------------ Energy Balance --------------------------------------

    dHlf_dz = -(Hl_trn + Ql_trn)
    dHvf_dz = Hv_trn + Qv_trn + 1e-8

    dTl_dz = dHlf_dz / Sigma_Fl_Cpl  # K/m
    dTv_dz = dHvf_dz / Sigma_Fv_Cpv  # K/m

    # ------------------------------ Momentum Balance --------------------------------------

    dP_dz = 0 / scales[6]  # Pa/m

    diffeqs = np.array([dFl_CO2_dz, dFl_H2O_dz, dFv_CO2_dz, dFv_H2O_dz, dHlf_dz, dHvf_dz, dP_dz])
    diffeqs_scaled = diffeqs / scales

    if run_type == 'simulating':

        return diffeqs_scaled

    # ------------------------------ Run Output Code Setup --------------------------------------

    elif run_type == 'saving':
        Fl_CO2, Fl_MEA, Fl_H2O = Fl
        Fl_CO2_true, Fl_MEA_true, Fl_H2O_true, Fl_MEAH_true, Fl_MEACOO_true, Fl_HCO3_true = Fl_true
        Fv_CO2, Fv_H2O, Fv_N2, Fv_O2 = Fv
        Cl_CO2, Cl_MEA, Cl_H2O = Cl
        Cl_CO2_true, Cl_MEA_true, Cl_H2O_true, Cl_MEAH_true, Cl_MEACOO_true, Cl_HCO3_true = Cl_true
        x_CO2, x_MEA, x_H2O = x
        x_CO2_true, x_MEA_true, x_H2O_true, x_MEAH_true, x_MEACOO_true, x_HCO3_true = x_true
        y_CO2, y_H2O, y_N2, y_O2 = y
        Cv_CO2, Cv_H2O, Cv_N2, Cv_O2 = Cv
        DF_CO2, Pv_CO2, Pl_CO2, Psi, H_CO2_mix = CO2
        DF_H2O, Pv_H2O, Pl_H2O, Psat_H2O = H2O
        kl_CO2, kv_CO2, kv_H2O = k_mxs
        k2, Cl_MEA_true, Dl_CO2, kl_CO2, Ha, E, Psi = enhance_factor
        ul, uv, uv_FL, h_L, h_V, a_e, fld_frac = hydr
        Cpl_CO2, Cpl_MEA, Cpl_H2O = Cpl
        Cpv_CO2, Cpv_H2O, Cpv_N2, Cpv_O2 = Cpv
        V_l, V_CO2, V_MEA, V_H2O = volume
        Hl_CO2 = Hl_CO2 + 1e-5
        Clp, Cvp, eps, a_p, A, Lp, d_h = const
        muv_CO2, muv_H2O, muv_N2, muv_O2 = muv

        output_dict = {'Fl': [Fl_CO2, Fl_MEA, Fl_H2O, Fl_T,
                              Fl_CO2_true, Fl_MEA_true, Fl_H2O_true, Fl_MEAH_true, Fl_MEACOO_true, Fl_HCO3_true],
                       'Fv': [Fv_CO2, Fv_H2O, Fv_N2, Fv_O2, Fv_T],
                       'Cl': [Cl_CO2, Cl_MEA, Cl_H2O,
                              Cl_CO2_true, Cl_MEA_true, Cl_H2O_true, Cl_MEAH_true, Cl_MEACOO_true, Cl_HCO3_true],
                       'Cv': [Cv_CO2, Cv_H2O, Cv_N2, Cv_O2],
                       'x': [x_CO2, x_MEA, x_H2O,
                             x_CO2_true, x_MEA_true, x_H2O_true, x_MEAH_true, x_MEACOO_true, x_HCO3_true],
                       'y': [y_CO2, y_H2O, y_N2, y_O2],
                       'T': [Tl, Tv],
                       'ql': [Tl, Hl_CO2, Hl_MEA, Hl_H2O, Hl, Hl_CO2_trn, Hl_H2O_trn, Hl_trn, Ql_trn, dHlf_dz, Hlf],
                       'qv': [Tv, Hv_CO2, Hv_H2O, Hv_N2, Hv_O2, Hv, Hv_CO2_trn, Hv_H2O_trn, Hv_trn, Qv_trn, dHvf_dz, Hvf],
                       'CO2': [Nl_CO2, Nv_CO2, kv_CO2, a_eA, DF_CO2, Pv_CO2, Pl_CO2, Psi, H_CO2_mix],
                       'H2O': [Nl_H2O, Nv_H2O, kv_H2O, a_eA, DF_H2O, Pv_H2O, Pl_H2O, y_H2O, P, x_H2O_true, Psat_H2O],
                       'enhance_factor': [k2, Cl_MEA_true, Dl_CO2, kl_CO2, Ha, E, Psi],
                       'transport': [kl_CO2, kv_CO2, kv_H2O, ul, uv, uv_FL, fld_frac, h_L, h_V, a_e, UT_base, UT, P,
                                     Clp, Cvp, eps, a_p, A, Lp, d_h],
                       'Prop_l': [rho_mol_l, rho_mass_l, V_l, V_CO2, V_MEA, V_H2O, mul_mix, sigma, Dl_CO2, Dl_MEA, Dl_ion, Cpl_CO2,
                                  Cpl_MEA, Cpl_H2O, Sigma_Fl_Cpl],
                       'Prop_v': [rho_mol_v, rho_mass_v, muv_CO2, muv_H2O, muv_N2, muv_O2, muv_mix, Dv_CO2, Dv_H2O,
                                  Cpv_CO2, Cpv_H2O, Cpv_N2, Cpv_O2,
                                  Sigma_Fv_Cpv, kt_vap],
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
        raise ValueError('Choose correct runtype')


