import numpy as np

from ..Properties.Amine_Properties import resolve_amine_properties
from ..Properties.Thermophysical_Properties import density, heat_capacity, thermal_conductivity, enthalpy, vapor_pressure
from ..Properties.Transport_Properties import viscosity, diffusivity
from ..Thermodynamics.Fugacity import fugacity
from ..Transport.Hydraulic_Variables_Correlations import velocity, holdup, interfacial_area, flooding_fraction
from ..Transport.Transfer_Coefficients import mass_transfer_coeff, heat_transfer_coeff
from ..Transport.Pressure_Drop import pressure_drop
from ..Transport.Flux import molar_flux, enthalpy_flux
from ..misc.Get_Temperature_Enthalpy import get_liquid_temperature, get_vapor_temperature
from .robust_core import record_domain_guard


def abs_column(zi, Y_scaled, parameters, run_type='simulating', column_names=False):
    # region - Unpack System Parameters
    if len(parameters) == 6:
        scales, eq_scales, const_flow, H, A, packing = parameters
        model_options = {}
    else:
        scales, eq_scales, const_flow, H, A, packing, model_options = parameters
    thermo_model = model_options.get('thermo_model', 'ideal_henry')
    solver_diagnostics = model_options.get('solver_diagnostics')
    guard_invalid_states = model_options.get('guard_invalid_states', True)
    mass_transfer_factor = float(model_options.get('mass_transfer_factor', 1.0))
    heat_transfer_factor = float(model_options.get('heat_transfer_factor', 1.0))
    thermal_state_mode = model_options.get('thermal_state_mode', 'enthalpy')
    co2_flux_mode = model_options.get('co2_flux_mode', 'bidirectional')
    eta_psi = float(model_options.get('eta_psi', 1.0))
    epcsaft_fugacity_blend = float(model_options.get('epcsaft_fugacity_blend', 1.0))
    chemical_equilibrium_model = model_options.get('chemical_equilibrium_model', 'legacy')
    gas_velocity_area_exponent = float(model_options.get('gas_velocity_area_exponent', 0.0) or 0.0)
    gas_velocity_area_reference_m_s = model_options.get('gas_velocity_area_reference_m_s')
    gas_velocity_area_bounds = model_options.get('gas_velocity_area_bounds', (0.1, 3.0))
    amine_properties = resolve_amine_properties(model_options.get('amine_properties'))
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

    if thermal_state_mode == 'temperature':
        Tl = float(Hlf)
        Tv = float(Hvf)
    else:
        Hl = Hlf / Fl_T
        Hv = Hvf / Fv_T
        Tl = get_liquid_temperature(x, Hl, amine_properties)
        Tv = get_vapor_temperature(y, Hv)
    temperature_bounds = model_options.get('temperature_bounds_K', (250.0, 500.0))
    if guard_invalid_states and not _temperatures_in_bounds(Tl, Tv, temperature_bounds):
        record_domain_guard(
            solver_diagnostics,
            "thermal_state",
            f"temperature outside bounds {temperature_bounds}: Tl={Tl!r}, Tv={Tv!r}",
        )
        raise ValueError("thermal_state: temperature outside absorber correlation bounds")

    w = amine_properties.mass_fractions(x)

    alpha = x[0] / x[1]
    w_MEA = w[1]
    w_H2O = w[2]
    # endregion

    # region - Properties

    # region -- Thermophysical Properties

    # region --- Henry's Law
    H_CO2_mix = amine_properties.henry_co2(Tl, x)
    # endregion

    # region --- Density
    rho_mol_l, rho_mass_l, volume = amine_properties.density(Tl, x, P)
    rho_mol_v, rho_mass_v = density(Tv, y, P, phase='vapor')
    # endregion

    # region --- Surface Tension
    sigma = amine_properties.surface_tension(Tl, x, w_MEA, w_H2O)
    # endregion

    # region --- Heat Capacity
    Cpl, Cpl_T = amine_properties.heat_capacity(Tl, x)
    Cpv, Cpv_T = heat_capacity(Tv, y, phase='vapor')
    # endregion

    # region --- Enthalpy
    Hl, Hl_T = amine_properties.enthalpy(Tl, x) # J/mol
    Hl_CO2, Hl_MEA, Hl_H2O = Hl
    Hv, Hv_T = enthalpy(Tv, y, phase='vapor')  # J/mol
    Hv_CO2, Hv_H2O, Hv_N2, Hv_O2 = Hv
    if thermal_state_mode == 'temperature':
        Hlf = Hl_T * Fl_T
        Hvf = Hv_T * Fv_T
    # endregion

    #region --- Vapor Pressure

    P_sat_H2O = vapor_pressure(Tl)

    # endregion

    # endregion

    # region -- Transport Properties

    # region --- Viscosity
    mul_mix, mul_H2O = amine_properties.viscosity(Tl, x, w_MEA, w_H2O)
    muv_mix, muv = viscosity(Tv, y, w_MEA, w_H2O, phase='vapor')
    # endregion

    # region --- Diffusivity
    Dl_CO2, Dl_MEA, Dl_ion = amine_properties.diffusivity(Tl, x, P, mul_mix, rho_mol_l)
    Dv_CO2, Dv_H2O, Dv_N2, Dv_O2, Dv_T = diffusivity(Tv, y, P, mul_mix, rho_mol_l, phase='vapor')
    # endregion

    # region --- Thermal Conductivity
    kt_vap = thermal_conductivity(Tv, y, muv)
    # endregion

    # endregion

    # endregion

    # region - Thermodynamics

    # region -- Chemical Equilibrium
    Cl_true, x_true = amine_properties.chemical_equilibrium(
        Fl.copy(),
        Tl,
        model=chemical_equilibrium_model,
        pressure_Pa=P,
        diagnostics=solver_diagnostics,
        liquid_molar_density=rho_mol_l,
    )

    Cl = [x[i] * rho_mol_l for i in range(len(x))]
    Cv = [y[i] * rho_mol_v for i in range(len(y))]

    Cl_true = [x_true[i] * rho_mol_l for i in range(len(x_true))]

    # endregion

    # region -- Vapor-Liquid Equilibrium

    fl_CO2, fv_CO2, fl_H2O, fv_H2O, CO2, H2O = fugacity(
        x,
        y,
        x_true,
        Cl_true,
        Tl,
        Tv,
        alpha,
        H_CO2_mix,
        P,
        P_sat_H2O,
        thermo_model=thermo_model,
        epcsaft_fugacity_blend=epcsaft_fugacity_blend,
        diagnostics=solver_diagnostics,
        guard_invalid_states=guard_invalid_states,
    )

    # endregion

    # endregion

    # region - Transport

    # region -- Hydraulic Variables

    # region --- Velocity
    ul, uv = velocity(rho_mol_l, rho_mol_v, A, Fl_T, Fv_T, diagnostics=solver_diagnostics)
    # endregion

    # region --- Interfacial Area
    a_e, a_eA = interfacial_area(rho_mass_l, sigma, ul, A, packing, diagnostics=solver_diagnostics)
    # endregion

    # region --- Holdup
    h_L, h_V = holdup(ul, mul_mix, rho_mass_l, packing, diagnostics=solver_diagnostics)
    # endregion

    # region --- Flooding Fraction
    fl_frac = flooding_fraction(rho_mass_l, rho_mass_v, mul_mix, mul_H2O, Fl_T, Fv_T, uv, packing, diagnostics=solver_diagnostics)
    # endregion

    if gas_velocity_area_exponent != 0.0 and gas_velocity_area_reference_m_s is not None:
        area_factor = _gas_velocity_area_factor(
            uv,
            float(gas_velocity_area_reference_m_s),
            gas_velocity_area_exponent,
            gas_velocity_area_bounds,
        )
        a_e *= area_factor
        a_eA *= area_factor
    else:
        area_factor = 1.0

    # endregion

    # region -- Transfer Coefficients
    # region --- Mass Transfer Coefficients

    kl_CO2, kv_CO2, kv_H2O, kv_T, const = mass_transfer_coeff(
        h_L,
        h_V,
        rho_mass_l,
        rho_mass_v,
        mul_mix,
        muv_mix,
        Dl_CO2,
        Dv_CO2,
        Dv_H2O,
        Dv_T,
        A,
        Tv,
        ul,
        uv,
        packing,
        diagnostics=solver_diagnostics,
    )
    # endregion

    # region --- Heat Transfer Coefficient
    UT = heat_transfer_coeff(P, kv_CO2, kt_vap, Cpv_T, rho_mol_v, Dv_CO2, a_eA, diagnostics=solver_diagnostics)  # J/(s*K*m) or W/(K*m)
    # endregion
    # endregion

    # region -- Pressure Drop

    ΔP_H = pressure_drop(h_L, rho_mass_l, rho_mass_v, mul_mix, muv_mix, A, ul, uv, packing, diagnostics=solver_diagnostics)

    # endregion

    # region -- Enhancement Factor

    E, Psi, Psi_H, enhance_factor = amine_properties.enhancement_factor(
        Tl, Cl_true, y[0], P, H_CO2_mix, kl_CO2, kv_CO2,
        Dl_CO2, Dl_MEA, Dl_ion, E_type='explicit',
        diagnostics=solver_diagnostics, eta_psi=eta_psi,
    )

    # endregion

    # region -- Flux

    # region --- Molar Flux
    Nv_CO2, Nv_H2O, Nl_CO2, Nl_H2O = molar_flux(fl_CO2, fv_CO2, fl_H2O, fv_H2O, kv_CO2, kv_H2O, a_eA, Psi_H)
    if co2_flux_mode == 'absorption_only':
        Nv_CO2 = _smooth_absorption_only_vapor_flux(Nv_CO2)
        Nl_CO2 = -Nv_CO2
    Nv_CO2 *= mass_transfer_factor
    Nv_H2O *= mass_transfer_factor
    Nl_CO2 *= mass_transfer_factor
    Nl_H2O *= mass_transfer_factor

    # endregion

    # region --- Enthalpy Flux

    Hv_flux, Hl_flux, qv, ql, Hv_trn, Hl_trn, Hv_CO2_trn, Hv_H2O_trn, Hl_CO2_trn, Hl_H2O_trn = enthalpy_flux(Nl_CO2, Hl_CO2, Nl_H2O, Hl_H2O, Nv_CO2, Hv_CO2, Nv_H2O, Hv_H2O, UT * heat_transfer_factor, a_eA, Tv, Tl)

    # endregion

    # endregion

    # endregion

    # region - Balance Equations

    # region -- Mass Balance
    dFl_CO2_dz = -Nl_CO2 + 1e-10  # mol/(s*m)
    dFl_H2O_dz = -Nl_H2O + 1e-10  # mol/(s*m)

    dFv_CO2_dz = Nv_CO2 + 1e-10  # mol/(s*m)
    dFv_H2O_dz = Nv_H2O + 1e-10  # mol/(s*m)
    # endregion

    # region -- Energy Balance
    dHlf_dz = Hl_flux + 1e-10
    dHvf_dz = Hv_flux + 1e-10


    dHl_dT = amine_properties.enthalpy_temperature_derivative(Tl, x)
    dHv_dT = Cpv_T

    dTl_dz = H*(Hl_flux + Hl_T*(Nl_CO2 + Nl_H2O))/(Fl_T*dHl_dT) # K/m
    dTv_dz = H*(Hv_flux - Hv_T*(Nv_CO2 + Nv_H2O))/(Fv_T*dHv_dT)

    # endregion

    # region -- Momentum Balance
    dP_dz = 0  # Pa/m
    # endregion

    # endregion

    # region - Run Output

    if run_type == 'simulating':
        # Combine Differentials and Scale
        if thermal_state_mode == 'temperature':
            diffeqs_scaled = np.array([
                dFl_CO2_dz / scales[0] * H,
                dFl_H2O_dz / scales[1] * H,
                dFv_CO2_dz / scales[2] * H,
                dFv_H2O_dz / scales[3] * H,
                dTl_dz / scales[4],
                dTv_dz / scales[5],
                dP_dz / scales[6] * H,
            ])
        else:
            diffeqs = np.array([dFl_CO2_dz, dFl_H2O_dz, dFv_CO2_dz, dFv_H2O_dz, dHlf_dz, dHvf_dz, dP_dz])
            # eq_scales = np.array([1, 1, 50, 50, 200000, 200000, P])
            diffeqs_scaled = diffeqs / scales * H
        return diffeqs_scaled

    elif run_type == 'saving':
        Fl_true = [Cl_true[i] * ul * A for i in range(len(Cl_true))]
        Fl_true_report = list(Fl_true[:6])
        Cl_true_report = list(Cl_true[:6])
        x_true_report = list(x_true[:6])

        Fl_CO2, Fl_MEA, Fl_H2O = Fl
        Fl_CO2_true, Fl_MEA_true, Fl_H2O_true, Fl_MEAH_true, Fl_MEACOO_true, Fl_HCO3_true = Fl_true_report
        Fv_CO2, Fv_H2O, Fv_N2, Fv_O2 = Fv
        Cl_CO2, Cl_MEA, Cl_H2O = Cl
        Cl_CO2_true, Cl_MEA_true, Cl_H2O_true, Cl_MEAH_true, Cl_MEACOO_true, Cl_HCO3_true = Cl_true_report
        x_CO2, x_MEA, x_H2O = x
        x_CO2_true, x_MEA_true, x_H2O_true, x_MEAH_true, x_MEACOO_true, x_HCO3_true = x_true_report
        y_CO2, y_H2O, y_N2, y_O2 = y
        Cv_CO2, Cv_H2O, Cv_N2, Cv_O2 = Cv
        DF_CO2, H_CO2_mix = CO2
        DF_H2O, Psat_H2O = H2O
        k2, Cl_MEA_true, Dl_CO2, kl_CO2, Ha, E, Psi_H, Psi, eta_psi = enhance_factor
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
            'Hl': [Tl, Hl_CO2, Hl_MEA, Hl_H2O, Hl_T, Fl_T, Hlf, Hl_CO2_trn, Hl_H2O_trn, Hl_trn, ql, Hl_flux, dHlf_dz, dTl_dz, dHl_dT],
            'Hv': [Tv, Hv_CO2, Hv_H2O, Hv_N2, Hv_O2, Hv_T, Fv_T, Hvf, Hv_CO2_trn, Hv_H2O_trn, Hv_trn, qv, Hv_flux, dHvf_dz, dTv_dz, dHv_dT],
            'CO2': [Nl_CO2, Nv_CO2, kv_CO2, a_eA, DF_CO2, fv_CO2, fl_CO2, Psi, H_CO2_mix],
            'H2O': [Nl_H2O, Nv_H2O, kv_H2O, a_eA, DF_H2O, fv_H2O, fl_H2O, Psat_H2O],
            'enhance_factor': [k2, Cl_MEA_true, Dl_CO2, kl_CO2, Ha, E, Psi, Psi_H, eta_psi],
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
            keys_dict['enhance_factor'] = ['k2', 'Cl_MEA_true', 'Dl_CO2', 'kl_CO2', 'Ha', 'E', 'Psi', 'Psi_H', 'eta_psi']
            keys_dict['transport'] = [
                'kl_CO2',
                'kv_CO2',
                'kv_H2O',
                'ul',
                'uv',
                'h_L',
                'h_V',
                'a_e',
                'UT',
                'P',
                'Clp',
                'Cvp',
                'eps',
                'a_p',
                'A',
                'Lp',
                'd_h',
            ]
        else:
            keys_dict = None
        return output_dict, keys_dict
    else:
        raise ValueError('Choose correct run type')
    # endregion


def _temperatures_in_bounds(Tl, Tv, bounds):
    low, high = bounds
    return (
        np.isfinite(Tl)
        and np.isfinite(Tv)
        and low <= float(Tl) <= high
        and low <= float(Tv) <= high
    )


def _smooth_absorption_only_vapor_flux(nv_co2):
    # Vapor z-direction flux is negative for CO2 absorption.  This smooth cap
    # removes the desorption branch without introducing a hard kink at zero.
    smoothing = 1.0e-10
    x = -float(nv_co2) / smoothing
    return -smoothing * np.logaddexp(0.0, x)


def _gas_velocity_area_factor(uv, reference_m_s, exponent, bounds):
    if reference_m_s <= 0.0:
        raise ValueError("gas_velocity_area_reference_m_s must be positive")
    low, high = bounds
    factor = (float(uv) / reference_m_s) ** exponent
    return float(np.clip(factor, low, high))
