"""Native thermodynamic first derivatives + AD of conventional film and energy.

Adapted from b6f5's isothermal assembly. This owner retains this manuscript's
EOS true concentrations, neutral-vapor closure and nonisothermal chain rule.
"""
from functools import partial

import casadi as ca
import numpy as np

from .ABS_Column import abs_column, temperature_gradients
from .robust_core import guard_column_rhs
from ..config.Constants import MWs_l
from ..Properties.Thermophysical_Properties import (
    density_expression, heat_capacity_expression, enthalpy_expression,
    thermal_conductivity_expression, surface_tension, vapor_pressure,
)
from ..Properties.Transport_Properties import viscosity, diffusivity
from ..Thermodynamics.reactive_bundle import state_input_jacobian
from ..Thermodynamics.thermo_models import COMPOSITION_FLOOR
from ..Transport.Hydraulic_Variables_Correlations import (
    velocity_expression, interfacial_area_expression, raw_holdup_expression,
    bounded_holdup_expression,
)
from ..Transport.Transfer_Coefficients import (
    liquid_mass_transfer_expression, gas_mass_transfer_expression, heat_transfer_expression,
)
from ..Transport.Enhancement_Factor import (
    hatta_expression, explicit_enhancement_expression, bounded_enhancement_expression,
    film_resistance_expression,
    CO2_CONCENTRATION_DIVISOR,
)
from ..Transport.Flux import molar_flux, enthalpy_flux


class ReactiveColumnJacobian:
    def __init__(self, parameters, transform_mode):
        self.parameters = parameters
        scales, _, fixed, height, area, packing, options = parameters
        if options.get('thermo_model') != 'epcsaft_reactive_nine':
            raise ValueError('Native reactive column Jacobian requires nine-species ePC-SAFT')
        if (transform_mode not in ('raw', 'bounded_guarded_raw_state', 'none', '', None)
                or options.get('thermal_state_mode') != 'temperature'
                or options.get('co2_flux_mode', 'bidirectional') != 'bidirectional'):
            raise ValueError('Native nonisothermal Jacobian requires raw temperature-state bidirectional equations')
        if options.get('gas_velocity_area_exponent', 0):
            raise ValueError('Native Jacobian does not cover gas-velocity area tuning')
        self.scales = np.asarray(scales)
        state = ca.SX.sym('physical', 7)
        native = ca.SX.sym('native', 12)  # amounts[9], density, liquid fCO2, neutral-vapor fCO2
        fl = [state[0], fixed[0], state[1]]
        fv = [state[2], state[3], fixed[1], fixed[2]]
        x, y = [v/sum(fl) for v in fl], [v/sum(fv) for v in fv]
        tl, tv, pressure = state[4], state[5], state[6]
        w = [MWs_l[i]*x[i]/sum(MWs_l[j]*x[j] for j in range(3)) for i in range(3)]
        rho_l, mass_l, _ = density_expression(tl, x, pressure)
        rho_v, mass_v = density_expression(tv, y, pressure, 'vapor')
        sigma = surface_tension(tl, x, w[1], w[2])
        mu_l, _ = viscosity(tl, x, w[1], w[2])
        mu_v, components_mu = viscosity(tv, y, w[1], w[2], 'vapor')
        dl, dmea, dion = diffusivity(tl, x, pressure, mu_l, rho_l)
        dv, dvwater, *_ = diffusivity(tv, y, pressure, mu_l, rho_l, 'vapor')
        ul, uv = velocity_expression(rho_l, rho_v, area, sum(fl), sum(fv))
        _, ae_area = interfacial_area_expression(mass_l, sigma, ul, area, packing)
        _, hv = bounded_holdup_expression(raw_holdup_expression(ul, mu_l, mass_l), packing[1], ca.fmax)
        kl = liquid_mass_transfer_expression(dl, mu_l, mass_l, ul, packing)
        kv = gas_mass_transfer_expression(dv, mu_v, mass_v, uv, hv, tv, packing)
        kvwater = gas_mass_transfer_expression(dvwater, mu_v, mass_v, uv, hv, tv, packing)
        true_x = [native[i]/ca.sum1(native[:9]) for i in range(9)]
        concentrations = [v*native[9] for v in true_x]
        _, ha = hatta_expression(tl, concentrations[1], concentrations[2], dl, kl)
        enhancement = explicit_enhancement_expression(
            ha, dmea, concentrations[1], dion, concentrations[3], dion, concentrations[4],
            dl, concentrations[0]/CO2_CONCENTRATION_DIVISOR, maximum=ca.fmax)
        enhancement = bounded_enhancement_expression(enhancement, ca.fmin, ca.fmax)
        _, psi_h = film_resistance_expression(enhancement, kl, kv,
                                              native[10]/concentrations[0], options.get('eta_psi', 1.))
        neutral = ca.vertcat(ca.fmax(y[0], COMPOSITION_FLOOR), COMPOSITION_FLOOR,
                            ca.fmax(y[1], COMPOSITION_FLOOR))
        neutral /= ca.sum1(neutral)
        self.vapor_map = ca.Function('vapor_composition', [state], [neutral, ca.jacobian(neutral, state)])
        co2, water, lco2, lwater = molar_flux(native[10], native[11]*y[0]/neutral[0],
                                             true_x[2]*vapor_pressure(tl), y[1]*pressure,
                                             kv, kvwater, ae_area, psi_h)
        factor = options.get('mass_transfer_factor', 1.)
        co2, water, lco2, lwater = [v*factor for v in (co2, water, lco2, lwater)]
        hl, _ = enthalpy_expression(tl, x)
        hv_components, _ = enthalpy_expression(tv, y, 'vapor')
        cp_l, _ = heat_capacity_expression(tl, x)
        _, cp_v = heat_capacity_expression(tv, y, 'vapor')
        kt = thermal_conductivity_expression(tv, y, components_mu)
        ut = heat_transfer_expression(pressure, kv, kt, cp_v, rho_v, dv)
        energy_v, energy_l, *_ = enthalpy_flux(lco2, hl[0], lwater, hl[2], co2,
            hv_components[0], water, hv_components[1], ut*options.get('heat_transfer_factor', 1.), ae_area, tv, tl)
        dlco2, dlwater, dvco2, dvwater = -lco2+1e-10, -lwater+1e-10, co2+1e-10, water+1e-10
        dtl, dtv = temperature_gradients(height, -energy_l, energy_v, hl[0], hl[2],
            hv_components[0], hv_components[1], dlco2, dlwater, dvco2, dvwater,
            sum(fl), sum(fv), x[1]*cp_l[1]+x[2]*cp_l[2], cp_v)
        rhs = ca.vertcat(dlco2*height, dlwater*height, dvco2*height, dvwater*height, dtl, dtv, 0)
        self.algebra = ca.Function('nonisothermal_partials', [state, native],
                                  [rhs, ca.jacobian(rhs, state), ca.jacobian(rhs, native)])

    def __call__(self, zi, scaled_state):
        def assemble(scaled, scaled_rhs, liquid, vapor_fugacity, vapor_block):
            physical = np.asarray(scaled)*self.scales
            apparent = np.array([physical[0], self.parameters[2][0], physical[1]])
            values = np.r_[liquid['amounts_mol'], liquid['density_mol_m3'],
                           liquid['fugacities_pa'][0], vapor_fugacity]
            rhs, direct, thermo = self.algebra(physical, values)
            # A value mismatch exposes an omitted term before any derivative reaches SciPy.
            np.testing.assert_allclose(np.asarray(rhs).ravel()/self.scales, scaled_rhs, rtol=2e-11, atol=1e-11)
            block = liquid['state_input_derivatives']
            feed_map = np.zeros((9, 3))
            feed_map[:3] = (np.eye(3)-liquid['feed_amounts_mol'][:3, None])/sum(apparent)
            output_ids = [f'amount:{s}' for s in block.component_ids]+['liquid-molar-density', 'fugacity:carbon-dioxide']
            expected_units = ['mole']*9+['mole / meter**3', 'pascal']
            if [block.output_units[block.output_identities.index(s)] for s in output_ids] != expected_units:
                raise ValueError('Liquid native output units changed')
            liquid_jac = state_input_jacobian(block, feed_map, output_ids)
            inputs = np.zeros((5, 7))
            inputs[[0,1,2,4], [4,6,0,1]] = 1.
            _, vapor_map = self.vapor_map(physical)
            vapor_jac = state_input_jacobian(vapor_block, np.asarray(vapor_map), ['fugacity:carbon-dioxide'])
            vapor_row = vapor_jac[0, 2:].copy()
            vapor_row[5] += vapor_jac[0, 0]
            vapor_row[6] += vapor_jac[0, 1]
            native_jac = np.vstack((liquid_jac @ inputs, vapor_row))
            jac = np.asarray(direct)+np.asarray(thermo) @ native_jac
            return jac*self.scales[None, :]/self.scales[:, None]
        return guard_column_rhs(zi, scaled_state, self.parameters, evaluator=partial(abs_column, jacobian=assemble))
