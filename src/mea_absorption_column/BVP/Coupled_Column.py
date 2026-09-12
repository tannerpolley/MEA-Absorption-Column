"""Conservative reactive-column balances, independent of interface closure."""
from __future__ import annotations

import casadi as ca
import numpy as np

from ..Properties.Thermophysical_Properties import surface_tension, thermal_conductivity
from ..Properties.Transport_Properties import viscosity, diffusivity
from ..Thermodynamics.casadi_reactive import build_loading_path_function, build_caloric_flow_function
from ..Transport.Hydraulic_Variables_Correlations import holdup_expression, interfacial_area_expression
from ..Transport.Pressure_Drop import pressure_drop_expression
from ..Transport.Enhancement_Factor import enhancement_reference_expression
from ..Transport.Reactive_Film import (
    onsager_mobility_expression, interface_balance_residuals, binary_diffusivities_from_species,
)
from ..Transport.Transfer_Coefficients import (
    liquid_mass_transfer_expression, gas_mass_transfer_expression, heat_transfer_expression,
)


def build_column_balance_functions(
    liquid, vapor, *, liquid_feed_mol_s, vapor_feed_mol_s,
    liquid_temperature_k, vapor_temperature_k, bottom_pressure_pa, area_m2, packing,
):
    """Build native balance(state, transfer) and boundary(bottom, top).

    State: apparent Fl_CO2, Fl_H2O, Fv_CO2, Fv_H2O [mol/s], Tl, Tv [K],
    P [Pa], h_L [bed-volume fraction]. MEA/N2/O2 flows are fixed by feeds.
    Feed order: CO2/MEA/water liquid; CO2/water/N2/O2 vapor.

    Transfer: CO2 and water flux [mol/(m² s)] and TOTAL interphase energy
    flux [W/m²], positive gas to liquid. Total energy flux includes transported
    enthalpy as well as conduction; do not add reaction heat to the balances.
    Flux closure is intentionally not supplied by this balance assembly.

    Seven conserved rows are four apparent flows, H_L, H_V [W], P [Pa].
    z increases upward in metres; both positive phase-flow magnitudes have
    the same negative transfer source. Pressure decreases upward. Kinetic
    and potential energy, axial conduction and external heat loss are omitted.

    Raw holdup is an algebraic equation, not a saturation transform. Bound
    the independent h_L in (0,epsilon); pressure drop uses this bounded state,
    so an infeasible raw holdup is a violated equation, not a fractional power
    of negative vapor void. Scalar bounds alone do not ensure EOS admissibility.

    Returned balance outputs: B, R, holdup residual, native liquid and vapor
    outputs, hydraulics [Q_L,Q_V (m³/s),rho_L,rho_V (kg/m³),raw h_L,a_e*A
    (m²/m),pressure-drop magnitude (Pa/m)]. The returned balance owns the native
    callbacks; retain it while evaluating any graph derived from it.
    Prescribed transfer gives 8 states, 7 balances and 1 algebraic equation;
    Adding 3 fluxes and an interface loading requires 4 interface equations.
    The seven inlet conditions fix pressure only at the gas inlet (bottom).
    """
    fl, fv = (np.asarray(v, dtype=float) for v in (liquid_feed_mol_s, vapor_feed_mol_s))
    packing = np.asarray(packing, dtype=float)
    if (fl.shape != (3,) or fv.shape != (4,) or packing.shape != (7,)
            or any(np.any(~np.isfinite(v)) or np.any(v <= 0.) for v in (fl, fv, packing))
            or not 0. < packing[1] < 1.
            or any(not np.isfinite(v) or v <= 0. for v in
                   (liquid_temperature_k, vapor_temperature_k, bottom_pressure_pa, area_m2))):
        raise ValueError("Column feeds, temperatures, pressure, area and packing must be finite positive; 0<epsilon<1")
    if (tuple(liquid.component_ids) != (
            "carbon-dioxide", "monoethanolamine", "water", "protonated-monoethanolamine",
            "carbamate-anion", "bicarbonate-anion", "carbonate-anion", "hydronium-cation", "hydroxide-anion")
            or tuple(vapor.component_ids) != ("carbon-dioxide", "water", "nitrogen", "oxygen")
            or liquid.output_ids[-1] != "total-enthalpy" or vapor.output_ids[-1] != "total-enthalpy"):
        raise ValueError("Column requires the caloric nine-species liquid and four-species vapor callbacks")

    state, transfer = ca.MX.sym("bulk", 8), ca.MX.sym("transfer", 3)
    liquid_flow = ca.vertcat(state[0], fl[1], state[1])
    vapor_flow = ca.vertcat(state[2], state[3], fv[2], fv[3])
    total_l, total_v = ca.sum1(liquid_flow), ca.sum1(vapor_flow)
    apparent_x, vapor_y = liquid_flow / total_l, vapor_flow / total_v
    liquid_values = liquid(ca.vertcat(state[4], state[6], liquid_flow))
    vapor_values = vapor(ca.vertcat(state[5], state[6], vapor_flow))
    true_amounts, rho_l, rho_v = liquid_values[:9], liquid_values[27], vapor_values[4]
    true_x = true_amounts / ca.sum1(true_amounts)
    # Reaction changes true mole count, but not mass or conserved components.
    volume_l = total_l * ca.sum1(true_amounts) / rho_l
    volume_v = total_v / rho_v
    mass_density_l = rho_l * ca.dot(true_x, ca.DM(liquid.molar_masses))
    mass_density_v = rho_v * ca.dot(vapor_y, ca.DM(vapor.molar_masses))
    ul, uv = volume_l / area_m2, volume_v / area_m2
    apparent_mass = liquid_flow * ca.DM(liquid.molar_masses[:3])
    mass_fraction = apparent_mass / ca.sum1(apparent_mass)
    mul, _ = viscosity(state[4], apparent_x, mass_fraction[1], mass_fraction[2])
    muv, _ = viscosity(state[5], vapor_y, mass_fraction[1], mass_fraction[2], phase="vapor")
    sigma = surface_tension(state[4], apparent_x, mass_fraction[1], mass_fraction[2])
    _, area_per_length = interfacial_area_expression(mass_density_l, sigma, ul, area_m2, packing)
    raw_holdup, _ = holdup_expression(ul, mul, mass_density_l, packing)
    gradient = pressure_drop_expression(state[7], mass_density_l, mass_density_v, mul, muv, area_m2, ul, uv, packing)
    conserved = ca.vertcat(state[:4], total_l * liquid_values[-1], total_v * vapor_values[-1], state[6])
    exchange = transfer * area_per_length
    sources = -ca.vertcat(exchange[0], exchange[1], exchange[0], exchange[1], exchange[2], exchange[2], gradient)
    balance = ca.Function("column_balances", [state, transfer], [conserved, sources,
        state[7] - raw_holdup, liquid_values, vapor_values,
        ca.vertcat(volume_l, volume_v, mass_density_l, mass_density_v, raw_holdup, area_per_length, gradient)],
        ["state", "transfer"], ["conserved", "sources", "holdup_residual", "liquid", "vapor", "hydraulics"])
    # CasADi graphs do not own Python Callback lifetimes.
    balance._thermodynamic_callbacks = (liquid, vapor)

    bottom, top = ca.MX.sym("bottom", 8), ca.MX.sym("top", 8)
    residual = ca.vertcat(top[0] - fl[0], top[1] - fl[2], bottom[2] - fv[0], bottom[3] - fv[1],
                         top[4] - liquid_temperature_k, bottom[5] - vapor_temperature_k, bottom[6] - bottom_pressure_pa)
    boundary = ca.Function("column_inlets", [bottom, top], [residual])
    return balance, boundary


def build_coupled_column_functions(
    liquid, vapor, *, species_diffusivities, quadrature_points=9,
    co2_model="reactive_film", film_thickness_multiplier=1., **column_inputs,
):
    """Twelve-state native node, inlet residuals and physical diagnostics.

    State is the eight bulk variables above, j_CO2, j_water, total energy flux,
    and signed interface log-loading. Seven conservative balances and five
    algebraic equations include raw holdup and all four interface equations.
    Trapezoidal quadrature integrates from zero to the signed interface loading.

    species_diffusivities(T) must supply nine positive estimates [m²/s] in
    native liquid species order; no missing ionic mobilities are fabricated.
    Pair diffusivities use the existing harmonic-mean closure. Film thickness
    is D_CO2/k_L using the existing bulk correlation on the apparent basis.
    Water retains its separate gas-film law. Transfer enthalpies are the vapor
    partial total enthalpies at bulk Tv/P/y, as in the retained energy closure.

    co2_model="enhancement_reference" replaces only the liquid CO2 relation
    with E*k_L*(c_interface-c_bulk), using native molecular concentrations
    and the same nonlinear interface fugacity. It retains the declared
    reference kinetics, concentration divisor, common-property diffusivities
    and E bounds. Its conductance diagnostic is empty; integral is thickness
    times reference flux, and an enhancement_factor output is appended.
    The positive thickness multiplier applies only to reactive_film; species
    diffusivity changes do not alter the independently defined thickness.

    A1 suffices for evaluation. Exact node derivatives also require native
    second equilibrium AND caloric actions. This builder does not imply their
    availability, physical acceptance, or default-solver adoption. Runtime
    assertions reject nonpositive diffusivities, Cp and quadrature conductance,
    and holdup outside the strict void interval. EOS failures propagate.
    Retain node or diagnostics for callback ownership.
    """
    if isinstance(quadrature_points, bool) or int(quadrature_points) != quadrature_points or quadrature_points < 2:
        raise ValueError("Film quadrature requires an integer point count >= 2")
    if co2_model not in ("reactive_film", "enhancement_reference"):
        raise ValueError("CO2 model must be reactive_film or enhancement_reference")
    if not np.isfinite(film_thickness_multiplier) or film_thickness_multiplier <= 0.:
        raise ValueError("Film thickness multiplier must be finite positive")
    if co2_model == "enhancement_reference" and film_thickness_multiplier != 1.:
        raise ValueError("Film thickness perturbations apply only to reactive_film")
    balance, bulk_boundary = build_column_balance_functions(liquid, vapor, **column_inputs)
    loading_path = build_loading_path_function(liquid) if co2_model == "reactive_film" else None
    vapor_caloric = build_caloric_flow_function(vapor, partial_enthalpy_indices=(0, 1))
    state, z = ca.MX.sym("coupled", 12), ca.MX.sym("z")
    fl, fv = column_inputs["liquid_feed_mol_s"], column_inputs["vapor_feed_mol_s"]
    packing, area = column_inputs["packing"], column_inputs["area_m2"]
    liquid_flow = ca.vertcat(state[0], fl[1], state[1])
    vapor_flow = ca.vertcat(state[2], state[3], fv[2], fv[3])
    liquid_inputs = ca.vertcat(state[4], state[6], liquid_flow)
    bounded_bulk = state[:8].attachAssert(ca.logic_and(state[7] > 0., state[7] < packing[1]),
                                         "Column holdup must satisfy 0 < h_L < epsilon")
    conserved, sources, holdup, l, v, hydraulics = balance(bounded_bulk, state[8:11])
    x, y = liquid_flow / ca.sum1(liquid_flow), vapor_flow / ca.sum1(vapor_flow)
    mass = liquid_flow * ca.DM(liquid.molar_masses[:3])
    w = mass / ca.sum1(mass)
    mul, _ = viscosity(state[4], x, w[1], w[2])
    muv, species_viscosities = viscosity(state[5], y, w[1], w[2], phase="vapor")
    # Apparent concentration = apparent flow / actual reactive liquid volume.
    dl, dl_mea, dl_ion = diffusivity(state[4], x, state[6], mul, ca.sum1(liquid_flow) / hydraulics[0])
    dl = dl.attachAssert(dl > 0., "Bulk liquid CO2 diffusivity must be positive")
    dv = diffusivity(state[5], y, state[6], mul, l[27], phase="vapor")
    kl = liquid_mass_transfer_expression(dl, mul, hydraulics[2], hydraulics[0] / area, packing)
    kg = ca.vertcat(*[gas_mass_transfer_expression(
        d, muv, hydraulics[3], hydraulics[1] / area, packing[1] - state[7], state[5], packing,
    ) for d in dv[:2]])
    thickness = film_thickness_multiplier * dl / kl
    _, cp, partial_h = vapor_caloric(ca.vertcat(state[5], state[6], vapor_flow))
    cp = cp.attachAssert(cp > 0., "Native vapor Cp must be positive for heat transfer")
    heat = heat_transfer_expression(state[6], kg[0],
        thermal_conductivity(state[5], y, species_viscosities), cp, v[4], dv[0])
    conductances = []
    if co2_model == "reactive_film":
        species_d = ca.reshape(species_diffusivities(state[4]), 9, 1)
        species_d = ca.MX(species_d).attachAssert(
            ca.logic_and(ca.mmin(species_d) > 0., ca.mmax(species_d) < ca.inf),
            "Film species diffusivities must be finite positive")
        pairs = binary_diffusivities_from_species(species_d)
        # Conserved CO2, MEA, water and charge in the verified native species order.
        co2 = ca.DM([1, 0, 0, 0, 1, 1, 1, 0, 0])
        constraints = np.array([[0, 0, 0, 1, -1, -1, -2, 1, -1],
                                [0, 1, 0, 1, 1, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 1, 1, 1, 1]], dtype=float)
        for fraction in np.linspace(0., 1., int(quadrature_points)):
            at_loading, tangent = loading_path(liquid_inputs, fraction * state[11])
            mobility = onsager_mobility_expression(at_loading[:9] / ca.sum1(at_loading[:9]),
                at_loading[27], pairs, additional_flux_constraints=constraints)
            conductance = ca.dot(co2, mobility @ tangent)
            conductances.append(conductance.attachAssert(conductance > 0., "Native loading-path conductance must be positive"))
        integral = state[11] / (quadrature_points - 1) * (
            .5 * (conductances[0] + conductances[-1]) + sum(conductances[1:-1]))
    else:
        at_loading = liquid(ca.vertcat(liquid_inputs[:2], liquid_inputs[2] * ca.exp(state[11]), liquid_inputs[3:]))
        bulk_concentrations = l[27] * l[:9] / ca.sum1(l[:9])
        interface_co2_concentration = at_loading[27] * at_loading[0] / ca.sum1(at_loading[:9])
        enhancement = enhancement_reference_expression(state[4], bulk_concentrations, kl, ca.vertcat(dl, dl_mea, dl_ion))
        integral = thickness * enhancement * kl * (interface_co2_concentration - bulk_concentrations[0])
    interface = interface_balance_residuals(state[8:11], liquid_conductance_integral=integral,
        film_thickness_m=thickness, vapor_fugacities_pa=v[:2],
        interface_co2_fugacity_pa=at_loading[18], liquid_water_fugacity_pa=l[20],
        gas_coefficients_mol_m2_s_pa=kg, heat_coefficient_w_m2_k=heat,
        vapor_temperature_k=state[5], liquid_temperature_k=state[4], transfer_enthalpies_j_mol=partial_h[:2])
    node = ca.Function("coupled_column", [z, state], [conserved, sources, ca.vertcat(holdup, interface)])
    diagnostics = ca.Function("coupled_column_diagnostics", [state],
        [ca.vertcat(*conductances), integral, thickness, kg, heat, cp, partial_h, at_loading[18]]
        + ([enhancement] if co2_model == "enhancement_reference" else []),
        ["state"], ["conductances", "integral", "thickness", "gas_coefficients", "heat_coefficient",
                    "vapor_cp", "vapor_partial_enthalpies", "interface_fugacity"]
        + (["enhancement_factor"] if co2_model == "enhancement_reference" else []))
    for function in (node, diagnostics):
        function._thermodynamic_callbacks = (balance, loading_path, vapor_caloric, liquid, vapor)
    bottom, top = ca.MX.sym("bottom", 12), ca.MX.sym("top", 12)
    boundary = ca.Function("coupled_inlets", [bottom, top], [bulk_boundary(bottom[:8], top[:8])])
    return node, boundary, diagnostics
