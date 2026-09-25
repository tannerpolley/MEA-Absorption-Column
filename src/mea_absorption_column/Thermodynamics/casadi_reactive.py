"""Native Engine values and exact actions in absorber coordinates.

Inputs are u = (T [K], P [Pa], absolute feed amounts [mol]); with an Amounts
feed every output is extensive and no unit-feed normalization exists. Product
and chain rules (H = N h, f = a R T rho0, the loading direction) are CasADi
expressions; the Engine supplies only solved-state directional actions.
"""
from __future__ import annotations

import math
from functools import lru_cache

import casadi as ca
import numpy as np

OUTPUT_IDS = ("fugacity:carbon-dioxide", "fugacity:water", "molar-density", "total-enthalpy")


class ActionUnavailable(RuntimeError):
    """Typed refusal of a consumed Engine action: no value, never zero or a difference estimate."""

    def __init__(self, phase, observable, directions, status, message):
        self.status = status
        super().__init__(f"{phase}: {observable} along input directions {directions} is unavailable: "
                         f"{status}: {message}")


class EnginePhase:
    """One Engine mixture in one declared phase, solved at exact inputs.

    Values [n_i (liquid only), f_CO2, f_water (Pa), rho (mol/m3), H (J)] come
    from calling the object on u; f = a R T rho0 with rho0 = 1 mol/m3. A failed
    cold solve with a loading policy follows the case's native continuation
    from the last accepted state (else its anchor CO2/MEA loading). Keep the object alive while CasADi
    graphs containing it are in use.
    """

    def __init__(self, name, mapping, thermochemistry, *, kind, feed_ids=None, molar_masses=None,
                 reactions=(), neutral_reference=None, loading_policy=None):
        import epcsaft
        from epcsaft import equilibrium

        self._q, self._gas_constant = equilibrium, epcsaft.GAS_CONSTANT_J_PER_MOL_K
        self.name, self.kind = name, kind
        self.model = epcsaft.Mixture(epcsaft.Parameters.from_mapping(mapping), thermochemistry=thermochemistry)
        self.component_ids = tuple(self.model.component_ids)
        self.feed_ids = self.component_ids if feed_ids is None else tuple(feed_ids)
        fixed = {c["component_id"]: c["fixed"] for c in mapping["components"]}
        if any(fixed[c]["molar_mass"]["value"]["unit"] not in ("kilogram / mole", "kg / mol") for c in self.component_ids):
            raise ValueError("Molar masses must be declared in kg/mol")
        self.molar_masses = tuple(molar_masses or (float(fixed[c]["molar_mass"]["value"]["magnitude"])
                                                   for c in self.component_ids))
        self.charges = tuple(float(fixed[c]["charge_number"]["value"]["magnitude"]) for c in self.component_ids)
        self.reactions, self.neutral_reference = tuple(reactions), neutral_reference
        self.loading_policy = loading_policy
        self.output_ids = (tuple(f"amount:{c}" for c in self.component_ids) if kind == "liquid" else ()) + OUTPUT_IDS
        kinds, index = equilibrium.SolvedStateObservableKind, self.component_ids.index
        enthalpy = epcsaft.PropertyObservable.TotalEnthalpy
        self.log_activities = tuple(equilibrium.SolvedStateObservable(kinds.PhaseLogActivity, 0, i)
                                    for i in range(len(self.component_ids)))
        self.enthalpy = equilibrium.SolvedStateObservable(kinds.PhaseProperty, 0, 0, enthalpy)
        self.state_observables = (
            tuple(equilibrium.SolvedStateObservable(kinds.PhaseComponentAmount, 0, i)
                  for i in range(len(self.component_ids)) if kind == "liquid")
            + (self.log_activities[index("carbon-dioxide")], self.log_activities[index("water")],
               equilibrium.SolvedStateObservable(kinds.PhaseMolarDensity, 0)))
        self.stats = dict(native_solves=0, continuations=0)
        self._accepted = None
        self.solve = lru_cache(maxsize=4096)(self._solve)
        # Separate callbacks keep H refusals out of Jacobians of H-free, inlined graph slices; any
        # Jacobian through H (or a dense e_T column) still refuses, typed and without a value.
        self._state = ActionCallback(name + "_state", self, self.state_observables)
        self._molar_enthalpy = ActionCallback(name + "_enthalpy", self, (self.enthalpy,))

    def __call__(self, inputs):
        inputs = inputs if isinstance(inputs, ca.MX) else ca.DM(inputs)
        y, count = self._state(inputs), len(self.state_observables) - 3
        amounts = y[:count]
        total = ca.sum1(amounts) if count else ca.sum1(inputs[2:])
        return ca.vertcat(amounts, ca.exp(y[count:count + 2]) * self._gas_constant * inputs[0],
                          y[count + 2], total * self._molar_enthalpy(inputs))

    def _problem(self, temperature, pressure, feed):
        q = self._q
        return q.Problem(phases=[q.Phase(self.kind, kind=self.kind)], T=temperature, P=pressure,
                         feed=q.Amounts(dict(zip(self.feed_ids, feed))), reactions=self.reactions,
                         neutral_reference=self.neutral_reference)

    def _solve(self, inputs):
        temperature, pressure, *feed = inputs
        if len(feed) != len(self.feed_ids) or not all(math.isfinite(v) and v > 0 for v in inputs):
            raise ValueError(f"{self.name} requires finite positive T, P and {len(self.feed_ids)} feed amounts")
        problem = self._problem(temperature, pressure, feed)
        self.stats["native_solves"] += 1
        result = self._q.solve_equilibrium(self.model, problem)
        if not result.success and self.loading_policy is not None:
            self.stats["continuations"] += 1
            result = self.continue_to(inputs)
        if not result.success:
            raise RuntimeError(f"{self.name} equilibrium failed at {inputs}: {result.message}")
        self._accepted = (inputs, result)
        return self._q.compile_problem(self.model, problem), result

    def continue_to(self, inputs):
        """Native continuation from the last accepted state, else the case's anchor loading at these T, P.

        The equilibrium is locally unique; the start changes the path, not the state. The
        case policy bounds the first log-feed step and the step budget.
        """
        temperature, pressure, *feed = inputs
        anchor, step, steps = (self.loading_policy[k] for k in ("loading_anchor", "max_log_loading_step", "max_loading_steps"))
        start, start_result = self._accepted or ((temperature, pressure, anchor * feed[1], *feed[1:]), None)
        distance = max(abs(math.log(a / b)) for a, b in zip(feed, start[2:]))
        options = self._q.Continuation(initial_step=min(1., step / max(distance, step)), minimum_step=1. / steps,
                                       max_steps=steps, max_retries_per_step=4)
        path = self._q.continue_equilibrium(self.model, self._problem(start[0], start[1], start[2:]),
                                            self._problem(temperature, pressure, feed), options, start_result)
        return path.states[-1] if path.success else path

    def _direction(self, index):
        values = np.zeros(2 + len(self.feed_ids))
        if index is not None:
            values[index] = 1.
        feed = [0.] * len(self.component_ids)
        for name, value in zip(self.feed_ids, values[2:]):
            feed[self.component_ids.index(name)] = float(value)
        return self._q.SolvedStateActionDirection(float(values[0]), float(values[1]), feed, [])

    def actions(self, inputs, observables, directions=()):
        """D^k O[e_d1, ..., e_dk] over input indices d (k = 0: values); refusal raises ActionUnavailable."""
        compiled, central = self.solve(tuple(float(v) for v in np.asarray(inputs).ravel()))
        request = self._q.SolvedStateActionRequest(
            list(observables), [self._direction(d) for d in directions] or [self._direction(None)])
        values = []
        for item in self._q.solved_state_actions(compiled, central, request).results:
            value = item.action if directions else item.value
            if item.status.name != "Available" or value is None or not math.isfinite(value):
                raise ActionUnavailable(self.name, item.observable.kind.name, tuple(directions),
                                        item.status.name, item.diagnostic_message)
            values.append(value)
        return np.asarray(values)

    def enthalpy_temperature_derivative(self, inputs):
        """(dH/dT)_{P,feed} [J/K] by the product rule on first actions."""
        observables = self.state_observables[:len(self.state_observables) - 3] + (self.enthalpy,)
        y, dy = (self.actions(inputs, observables, d) for d in ((), (0,)))
        total = y[:-1].sum() if len(y) > 1 else float(np.sum(np.asarray(inputs)[2:]))
        return float(dy[:-1].sum() * y[-1] + total * dy[-1])


class ActionCallback(ca.Callback):
    """Order-k Engine actions of fixed observables; the exact Jacobian adds one input direction.

    along=None returns values (Jacobian: first actions). along=(d,...) returns
    first actions D O[e_d] for each listed input index (Jacobian: second
    actions D^2 O[e_d, e_j]). No higher order is provided.
    """

    def __init__(self, name, phase, observables, along=None):
        self.phase, self.observables, self.along = phase, tuple(observables), along
        self._owned = []
        super().__init__()
        self.construct(name, {"enable_fd": False})

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, index):
        return ca.Sparsity.dense(2 + len(self.phase.feed_ids), 1)

    def get_sparsity_out(self, index):
        return ca.Sparsity.dense(len(self.observables) * (1 if self.along is None else len(self.along)), 1)

    def evaluate(self, inputs, extra=()):
        prefixes = [()] if self.along is None else [(d,) for d in self.along]
        return np.concatenate([self.phase.actions(inputs, self.observables, p + extra) for p in prefixes])

    def eval(self, arguments):
        return [ca.DM(self.evaluate(arguments[0]))]

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        jacobian = _ActionJacobian(name, self, inames, onames, opts)
        self._owned.append(jacobian)
        return jacobian


class _ActionJacobian(ca.Callback):
    def __init__(self, name, parent, inames, onames, opts):
        self.parent, self.inames, self.onames = parent, inames, onames
        super().__init__()
        self.construct(name, {**opts, "enable_fd": False})

    def get_n_in(self):
        return 2

    def get_n_out(self):
        return 1

    def get_name_in(self, index):
        return self.inames[index]

    def get_name_out(self, index):
        return self.onames[index]

    def get_sparsity_in(self, index):
        return self.parent.sparsity_in(0) if index == 0 else self.parent.sparsity_out(0)

    def get_sparsity_out(self, index):
        return ca.Sparsity.dense(self.parent.size1_out(0), self.parent.size1_in(0))

    def eval(self, arguments):
        count = self.parent.size1_in(0)
        return [ca.DM(np.column_stack([self.parent.evaluate(arguments[0], (j,)) for j in range(count)]))]


def build_loading_path_function(liquid):
    """Liquid values and loading tangent t = D ln a[v], v = u_CO2 e_CO2, at u_CO2 = F_CO2 exp(lambda).

    The tangent's outer derivative consumes second actions D^2 ln a[v, e]. Retain
    the returned function to own its callback.
    """
    bulk, loading = ca.MX.sym("bulk", 5), ca.MX.sym("loading")
    inputs = ca.vertcat(bulk[:2], bulk[2] * ca.exp(loading), bulk[3:])
    slope = ActionCallback(liquid.name + "_loading_slope", liquid, liquid.log_activities, along=(2,))
    path = ca.Function("native_loading_path", [bulk, loading], [liquid(inputs), inputs[2] * slope(inputs)])
    path._thermodynamic_callbacks = (liquid, slope)
    return path


def build_caloric_flow_function(vapor, *, partial_enthalpy_indices=(0, 1)):
    """Extensive H [W], molar Cp = dh/dT [J/(mol K)] and partial H_i = h + N dh/dF_i [J/mol] of the vapor.

    Fixed vapor composition makes (dH/dT)/N = dh/dT. Outer derivatives consume
    second enthalpy actions.
    """
    inputs = ca.MX.sym("caloric_inputs", 2 + len(vapor.feed_ids))
    slopes = ActionCallback(vapor.name + "_caloric", vapor, (vapor.enthalpy,),
                            along=(0, *(2 + i for i in partial_enthalpy_indices)))
    enthalpy = vapor(inputs)[-1]
    total, dh = ca.sum1(inputs[2:]), slopes(inputs)
    caloric = ca.Function("native_caloric_flow", [inputs], [enthalpy, dh[0], enthalpy / total + total * dh[1:]])
    caloric._thermodynamic_callbacks = (vapor, slopes)
    return caloric
