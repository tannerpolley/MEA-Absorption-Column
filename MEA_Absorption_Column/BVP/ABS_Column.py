import numpy as np
from numpy import sum
from MEA_Absorption_Column.Kinetics.Enhancement_Factor import enhancement_factor, enhancement_factor_2
from MEA_Absorption_Column.Transport.Solve_MassTransfer import solve_masstransfer
from MEA_Absorption_Column.Thermodynamics.Solve_Driving_Force import solve_driving_force
from MEA_Absorption_Column.Thermodynamics.Solve_ChemEQ import solve_ChemEQ, solve_ChemEQ_2
from MEA_Absorption_Column.Properties.Henrys_Law import henrys_law
from MEA_Absorption_Column.Properties.Density import liquid_density, vapor_density
from MEA_Absorption_Column.Properties.Viscosity import viscosity
from MEA_Absorption_Column.Properties.Surface_Tension import surface_tension
from MEA_Absorption_Column.Properties.Diffusivity import liquid_diffusivity, vapor_diffusivity
from MEA_Absorption_Column.Properties.Heat_Capacity import heat_capacity
from MEA_Absorption_Column.Properties.Thermal_Conductivity import thermal_conductivity
from MEA_Absorption_Column.Parameters import MWs_l
from MEA_Absorption_Column.Transport.Heat_Transfer import heat_transfer


def abs_column(zi, Y, Fl_MEA, Fv_N2, Fv_O2, P, A, df_param, run_type):
    Fl_CO2, Fl_H2O, Fv_CO2, Fv_H2O, Tl, Tv = Y

    # if Tl >= 500 or Tl <= 280:
    #     raise AssertionError

    a0 = Fl_MEA / (Fl_MEA + Fl_H2O)

    Fl_T = Fl_CO2 + Fl_MEA + Fl_H2O
    Fv_T = Fv_CO2 + Fv_H2O + Fv_N2 + Fv_O2

    Fl = [Fl_CO2, Fl_MEA, Fl_H2O]
    Fv = [Fv_CO2, Fv_H2O, Fv_N2, Fv_O2]

    x = [Fl[i] / Fl_T for i in range(len(Fl))]
    y = [Fv[i] / Fv_T for i in range(len(Fv))]

    w = [MWs_l[i] * x[i] / sum([MWs_l[j] * x[j] for j in range(len(Fl))]) for i in range(len(Fl))]

    alpha = x[0] / x[1]
    w_MEA = w[1]

    # -------------------------------- Properties ---------------------------------------------

    # Density
    rho_mol_l, rho_mass_l = liquid_density(Tl, x, df_param)
    rho_mol_v, rho_mass_v = vapor_density(Tv, P, y)

    # Viscosity
    muv_mix, mul_mix, mul_H2O = viscosity(Tl, Tv, y, w_MEA, alpha, df_param)

    # Surface Tension
    sigma = surface_tension(Tl, x, w_MEA, alpha, df_param)

    # Diffusion
    Dl_CO2, Dl_MEA, Dl_ion = liquid_diffusivity(Tl, rho_mol_l * x[1], mul_mix)
    Dv_CO2, Dv_H2O, Dv_N2, Dv_O2, Dv_T = vapor_diffusivity(Tv, y, P, df_param)

    # Heat Capacity
    Cpl = heat_capacity(Tl, 'liquid', x, w)
    Cpv = heat_capacity(Tv, 'vapor', x, w)

    Cpl_T = sum([Cpl[i] * x[i] for i in range(len(x))])
    Cpv_T = sum([Cpv[i] * y[i] for i in range(len(y))])

    Sigma_Fl_Cpl = Cpl_T * Fl_T
    Sigma_Fv_Cpv = Cpv_T * Fv_T

    # Thermal Conductivity
    kt_CO2 = thermal_conductivity(Tv, 'CO2', 'vapor')

    # Henry's Law
    H_CO2_mix = henrys_law(x, Tl, df_param)

    # ------------------------------ Chemical Equilibrium --------------------------------------

    Cl = [x[i] * rho_mol_l for i in range(len(x))]
    Cv = [y[i] * rho_mol_v for i in range(len(y))]

    Cl_true = solve_ChemEQ_2(alpha, w_MEA, Tl)
    # Cl_true = solve_ChemEQ(Cl, Tl)
    x_true = Cl_true / (sum(Cl_true)).astype('float')

    Cl_MEA_true = Cl_true[1]
    # From Gabrielsen 2005
    # Cl_MEA_true = Cl[1]*(1 - 2*alpha)

    # ------------------------------ Transport --------------------------------------

    # Velocity
    ul = Fl_T / (A * rho_mol_l)
    uv = Fv_T / (A * rho_mol_v)

    # Mass Transfer Coefficients and Properties
    kl_CO2, kv_CO2, kv_H2O, kv_T, k_mxs, uv, a_e, hydr = solve_masstransfer(rho_mass_l, rho_mass_v, mul_mix,
                                                                            muv_mix,
                                                                            sigma, Dl_CO2, Dv_CO2, Dv_H2O, Dv_T, A,
                                                                            Tl, Tv,
                                                                            ul, uv, H_CO2_mix, Cl_MEA_true)

    # Heat Transfer Coefficient
    UT = heat_transfer(P, kv_CO2, kt_CO2, Cpv_T, rho_mol_v, Dv_CO2) * .7

    # ------------------------------ Kinetics -------------------------------------------
    # E, kinetics = enhancement_factor(Tl, Cl, Cl_MEA_true, kl_CO2, Dl_CO2)
    E, kinetics = enhancement_factor_2(Tl, y[0], P, Cl_true, H_CO2_mix, kl_CO2, kv_CO2,
                                       Dl_CO2, Dl_MEA, Dl_ion)

    # ------------------------------ Thermodynamics --------------------------------------

    DF_CO2, DF_H2O, PEQ = solve_driving_force(x, y, x_true, Cl_true, Tl, a0, alpha, H_CO2_mix, P, E, kl_CO2, kv_CO2)

    # ------------------------------ Material and Energy Flux Setup --------------------------------------

    # # # Molar Flux
    N_CO2 = kv_CO2 * DF_CO2
    N_H2O = kv_H2O * DF_H2O

    # Enthalpy Transfer
    Hl_CO2 = -82000
    Hl_H2O = -48000

    q_abs = N_CO2 * Hl_CO2
    q_vap = N_H2O * Hl_H2O

    # Heat Transfer
    q_trn = UT * (Tv - Tl)

    # Liquid and Vapor Energy Flux

    ql = q_trn - q_abs - q_vap
    qv = q_trn

    kE_l = a_e * A / Sigma_Fl_Cpl
    kE_v = a_e * A / Sigma_Fv_Cpv
    a_eA = a_e * A

    # ------------------------------ Material and Energy Balance --------------------------------------

    dFl_CO2_dz = -N_CO2 * a_e * A
    dFl_H2O_dz = -N_H2O * a_e * A

    dFv_CO2_dz = -N_CO2 * a_e * A
    dFv_H2O_dz = -N_H2O * a_e * A

    dTl_dz = -a_e * A / (Cpl_T * Fl_T) * (q_trn - q_vap - q_abs)
    dTv_dz = -a_e * A / (Cpv_T * Fv_T) * q_trn

    diffeqs = [dFl_CO2_dz, dFl_H2O_dz, dFv_CO2_dz, dFv_H2O_dz, dTl_dz, dTv_dz]

    if run_type == 'saving':

        Fl_CO2, Fl_MEA, Fl_H2O = Fl
        Fv_CO2, Fv_H2O, Fv_N2, Fv_O2 = Fv
        Cl_CO2, Cl_MEA, Cl_H2O = Cl
        Cl_CO2_true, Cl_MEA_true, Cl_H2O_true, Cl_MEAH_true, Cl_MEACOO_true, Cl_HCO3_true = Cl_true
        x_CO2, x_MEA, x_H2O = x
        x_CO2_true, x_MEA_true, x_H2O_true, x_MEAH_true, x_MEACOO_true, x_HCO3_true = x_true
        y_CO2, y_H2O, y_N2, y_O2 = y
        Cv_CO2, Cv_H2O, Cv_N2, Cv_O2 = Cv
        DF_CO2, Pv_CO2, Pl_CO2, H_CO2_mix, DF_H2O, Pv_H2O, Pl_H2O, Psat_H2O = PEQ
        kl_CO2, kv_CO2, kv_H2O = k_mxs
        k_rxn, Ha, E = kinetics
        ul, uv, h_L, a_e = hydr
        Cpl_CO2, Cpl_MEA, Cpl_H2O = Cpl
        Cpv_CO2, Cpv_H2O, Cpv_N2, Cpv_O2 = Cpv
        dTl_dz = -ql * kE_l
        dTv_dz = -qv * kE_v
        T_diff = Tl - Tv

        output_dict = {'Fl': [Fl_CO2, Fl_MEA, Fl_H2O],
                       'Fv': [Fv_CO2, Fv_H2O, Fv_N2, Fv_O2],
                       'Cl': [Cl_CO2, Cl_MEA, Cl_H2O,
                              Cl_CO2_true, Cl_MEA_true, Cl_H2O_true, Cl_MEAH_true, Cl_MEACOO_true, Cl_HCO3_true],
                       'Cv': [Cv_CO2, Cv_H2O, Cv_N2, Cv_O2],
                       'x': [x_CO2, x_MEA, x_H2O,
                             x_CO2_true, x_MEA_true, x_H2O_true, x_MEAH_true, x_MEACOO_true, x_HCO3_true],
                       'y': [y_CO2, y_H2O, y_N2, y_O2],
                       'T': [Tl, Tv],
                       'CO2': [N_CO2, DF_CO2, Pv_CO2, Pl_CO2, H_CO2_mix],
                       'H2O': [N_H2O, DF_H2O, Pv_H2O, Pl_H2O, Psat_H2O],
                       'k_mx': [kl_CO2, kv_CO2, kv_H2O],
                       'kinetics': [k_rxn, Ha, E],
                       'hydr': [ul, uv, h_L, a_e, UT],
                       'Prop_l': [rho_mol_l, rho_mass_l, mul_mix, sigma, Dl_CO2, Cpl_CO2, Cpl_MEA, Cpl_H2O],
                       'Prop_v': [rho_mol_v, rho_mass_v, muv_mix, Dv_CO2, Dv_H2O, Cpv_CO2, Cpv_H2O, Cpv_N2, Cpv_O2],
                       'ql': [Tl, q_abs, q_vap, dTl_dz],
                       'qv': [Tv, q_trn, dTv_dz],
                       }
        if zi == 0:

            keys_dict = {}
            for k, v in output_dict.items():
                key_list = []
                for i in range(len(v)):
                    for k2, v2 in locals().items():
                        if type(v2) == np.float64:
                            if v[i] == v2:
                                key_list.append(k2)
                keys_dict[k] = key_list
        else:
            keys_dict = None

        return output_dict, keys_dict

    return diffeqs
