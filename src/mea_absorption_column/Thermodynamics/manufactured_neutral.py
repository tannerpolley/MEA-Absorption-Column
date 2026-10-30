"""Mathematical neutral equilibrium and column equations: #85 comment 6510746468.

One-second reference amounts map input mol/s to mol and output J to W.
These equations supply numerical verification, not an Engine thermodynamic model.
"""
import casadi as ca
import numpy as np

from .casadi_reactive import ActionCallback
from .solvent_inputs import StateDomainFailure


def neutral_amounts(f, a, b, water, k):
    coefficient = 1 + k * (a + b - f)
    s = 2 * f / (coefficient + ca.sqrt(coefficient**2 + 4*k*f))
    return dict(C=s, A=a/(1+k*s), B=b/(1+k*s), W=water,
                CA=a*k*s/(1+k*s), CB=b*k*s/(1+k*s))


class ManufacturedNeutralPhase:
    """Fixed-volume mathematical equilibrium; no Engine mixture or physical solvent."""
    def __init__(self, declaration):
        self.d = declaration
        self.name = 'manufactured_neutral'
        self.component_ids, self.feed_ids = declaration['species_order'], declaration['feed_order']
        self.co2_input = 2 + self.feed_ids.index('C_total')
        self.log_activities = tuple(range(len(self.component_ids)))
        self.u = ca.MX.sym('u', 2 + len(self.feed_ids))
        feed = dict(zip(self.feed_ids, ca.vertsplit(self.u[2:])))
        k, volume = declaration['K_per_mol'], declaration['volume_m3']
        f, a, b = (feed[s] for s in ('C_total', 'A_total', 'B_total'))
        amounts = neutral_amounts(f, a, b, feed['W_total'], k)
        n = ca.vertcat(*[amounts[s] for s in self.component_ids])
        mu = ca.log(n) - ca.DM([np.log(k) if s in ('CA', 'CB') else 0. for s in self.component_ids])
        self.chemical_potentials = mu
        self.action_functions = {}
        self.callback = ActionCallback(self.name + '_values', self, self.log_activities)

    def __call__(self, inputs):
        # Values and slopes both traverse the existing exact-action callback interface.
        mu = self.callback(inputs)
        n = ca.exp(mu + ca.DM([np.log(self.d['K_per_mol']) if s in ('CA', 'CB') else 0.
                              for s in self.component_ids]))
        return ca.vertcat(n, ca.sum1(n)/self.d['volume_m3'],
                          self.d['henry_pa_m3_mol']*n[self.component_ids.index('C')]/self.d['volume_m3'])

    def actions(self, inputs, observables, directions=()):
        if directions not in self.action_functions:
            expr = self.chemical_potentials
            for direction in directions:
                expr = ca.jacobian(expr, self.u)[:, direction]
            self.action_functions[directions] = ca.Function('analytic_action', [self.u], [expr])
        return np.asarray(self.action_functions[directions](inputs)).ravel()[list(observables)]


class MathematicalActionUnavailable(RuntimeError):
    kind = 'mathematical_action_unavailable'


class ManufacturedColumnPhase:
    """Declared S3-1/S3-2 values and exact actions; no native solve or result."""
    def __init__(self, solvent, kind):
        self.kind, self.name = kind, 'mathematical_' + kind
        t = solvent.thermo
        self.roles = solvent.roles
        self.component_ids = tuple(solvent.species if kind == 'liquid' else solvent.gas)
        self.feed_ids = tuple(solvent.feeds if kind == 'liquid' else solvent.gas)
        self.feed_index = {s: i for i, s in enumerate(self.feed_ids)}
        self.co2_input = 2 + self.feed_index[self.roles['co2']]
        self.charges = tuple(t['charge_numbers'][s] for s in self.component_ids)
        self.molar_masses = tuple(sum(t['atomic_masses_kg_mol'][e]*n for e, n in t['formulas'][s].items())
                                  for s in self.component_ids)
        self.components, self.film_constraints = solvent.components, solvent.constraints
        self.u = ca.MX.sym('phase_input', 2+len(self.feed_ids))
        feed = dict(zip(self.feed_ids, ca.vertsplit(self.u[2:])))
        if kind == 'liquid':
            amounts = neutral_amounts(feed['C'], feed['A'], feed['B'], feed['W'], t['K_per_mol'])
            n = ca.vertcat(*[amounts[s] for s in self.component_ids])
            mu = ca.log(n) - ca.DM([np.log(t['K_per_mol']) if s in ('CA', 'CB') else 0.
                                   for s in self.component_ids])
            fugacity = t['henry_pa_m3_mol']*amounts['C']/t['liquid_volume_m3']
            capacity = t['capacity_j_mol_k']*(feed['A']+feed['B']+feed['W'])
            density = ca.sum1(n)/t['liquid_volume_m3']
        else:
            n = ca.vertcat(*[feed[s] for s in self.component_ids])
            mu = ca.log(n)
            fugacity = t['gas_fugacity_pa_s_mol']*feed['C']
            capacity = t['capacity_j_mol_k']*feed['I']
            density = ca.sum1(n)/t['vapor_volume_m3']
        enthalpy = capacity*(self.u[0]-t['reference_temperature_k'])
        self.enthalpy = len(self.component_ids)
        self.log_activities = tuple(range(self.enthalpy))
        self.observables = ca.vertcat(mu, enthalpy/ca.sum1(n))
        values = ca.vertcat(n if kind == 'liquid' else ca.MX.zeros(0), fugacity,
                            t['water_fugacity_pa'], density, enthalpy)
        value_offset = self.observables.numel()
        self.observables = ca.vertcat(self.observables,values)
        self.values = ActionCallback(self.name+'_values',self,
                                     range(value_offset,value_offset+values.numel()))
        self.capacity = ca.Function(self.name+'_capacity', [self.u], [ca.jacobian(enthalpy, self.u)[0]])
        self.output_ids = (tuple('amount:'+s for s in self.component_ids) if kind == 'liquid' else ()) + (
            'fugacity:C', 'fugacity:W', 'molar-density', 'total-enthalpy')
        self.output_index = {s: i for i, s in enumerate(self.output_ids)}
        self.action_functions = {}

    def __call__(self, inputs):
        return self.values(inputs)

    def flow(self, co2, water, fixed):
        return ca.vertcat(*[co2 if s == 'C' else water if s == 'W' else fixed[i]
                            for i, s in enumerate(self.feed_ids)])

    def enthalpy_temperature_derivative(self, inputs):
        return float(self.capacity(self.check_domain(inputs)))

    def check_domain(self, inputs):
        inputs = np.asarray(inputs, dtype=float).ravel()
        if (not np.isfinite(inputs).all() or np.min(inputs) <= 0 or
                not 293.15 <= inputs[0] <= 393.15 or inputs[1] > 1e7):
            raise StateDomainFailure(f'{self.name}: invalid T [K], P [Pa], feed [mol/s]: {inputs}')
        return inputs

    def actions(self, inputs, observables, directions=()):
        inputs = self.check_domain(inputs)
        if directions not in self.action_functions:
            expr = self.observables
            for direction in directions:
                expr = ca.jacobian(expr, self.u)[:, direction]
            self.action_functions[directions] = ca.Function(self.name+'_action', [self.u], [expr])
        result = np.asarray(self.action_functions[directions](inputs)).ravel()[list(observables)]
        if not np.isfinite(result).all():
            raise MathematicalActionUnavailable(f'{self.name}: nonfinite actions {observables}, {directions}, inputs={inputs}')
        return result
