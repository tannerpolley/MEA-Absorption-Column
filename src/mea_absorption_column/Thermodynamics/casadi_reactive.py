"""Native equilibrium values and derivative actions in absorber coordinates."""
from __future__ import annotations

import casadi as ca
import numpy as np

from .reactive_bundle import ReactiveLiquid


def build_loading_path_function(liquid):
    """Native state and d(mu/RT)/dlambda on [F_CO2 exp(lambda),F_MEA,F_water].

    Inputs are bulk T/P/apparent flows and signed log-loading. Values require
    A1; differentiating the tangent requires native second equilibrium actions.
    Retain the returned function to own the underlying Python callback.
    """
    bulk, loading = ca.MX.sym("bulk", 5), ca.MX.sym("loading")
    inputs = ca.vertcat(bulk[:2], bulk[2] * ca.exp(loading), bulk[3:])
    tangent = _LoadingTangent(liquid.name() + "_loading_tangent", liquid, include_state=True)
    evaluated = tangent(inputs)
    size = len(liquid.output_ids)
    path = ca.Function("native_loading_path", [bulk, loading],
                       [evaluated[:size], evaluated[size:]])
    path._thermodynamic_callbacks = (liquid, tangent)
    return path


def build_caloric_flow_function(callback, *, partial_enthalpy_indices=None):
    """Extensive H [W], molar Cp [J/(mol K)], partial H [J/mol] from A1.

    The molar basis is apparent feed for reactive liquid and species feed for
    vapor. Product/normalization rules belong here; native caloric derivatives
    belong to Engine. Outer derivatives require native second caloric actions.
    """
    if callback.output_ids[-1] != "total-enthalpy":
        raise ValueError("Caloric flow requires explicit native reference thermochemistry")
    inputs = ca.MX.sym("caloric_inputs", callback.size1_in(0))
    count = callback.size1_in(0) - 2
    indices = tuple(range(count)) if partial_enthalpy_indices is None else tuple(partial_enthalpy_indices)
    if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < count for i in indices):
        raise ValueError("Partial enthalpy indices must be distinct species positions")
    total = ca.sum1(inputs[2:])
    enthalpy = total * callback(inputs)[-1]
    owners = (callback,)
    if isinstance(callback, FixedCompositionVaporCallback):
        derivative = _VaporCaloric(callback.name() + "_caloric", callback, indices)
        values = derivative(inputs)
        cp, partial = values[0], values[1:]
        owners += (derivative,)
    else:
        derivative = ca.jacobian(enthalpy, inputs)
        cp, partial = derivative[0] / total, derivative[[i + 2 for i in indices]].T
    caloric = ca.Function("native_caloric_flow", [inputs],
                          [enthalpy, cp, partial])
    caloric._thermodynamic_callbacks = owners
    return caloric


def _state_input_map(block, feed_jacobian):
    """Map physical T/P/feed changes through the complete native invariants."""
    invariant_jacobian = np.asarray(block.invariant_matrix) @ feed_jacobian
    invariant_rows = dict(zip(block.invariant_ids, invariant_jacobian, strict=True))
    expected_units = dict(zip(block.invariant_ids, block.invariant_units, strict=True))
    expected_units.update(temperature_k="kelvin", pressure_pa="pascal")
    mapping = np.zeros((len(block.input_identities), 2 + feed_jacobian.shape[1]))
    for row, (identity, unit) in enumerate(zip(block.input_identities, block.input_units, strict=True)):
        if unit != expected_units[identity]:
            raise ValueError(f"Engine derivative input unit changed: {identity}: {unit}")
        if identity in ("temperature_k", "pressure_pa"):
            mapping[row, 0 if identity == "temperature_k" else 1] = 1.0
        else:
            mapping[row, 2:] = invariant_rows[identity]
    active = np.flatnonzero(np.any(mapping != 0.0, axis=1))
    for index in active:
        if block.input_failures[index] is not None:
            raise RuntimeError(f"Native equilibrium derivative unavailable: {block.input_failures[index]}")
    return mapping, active


class ReactiveLiquidCallback(ca.Callback):
    """Input: T [K], P [Pa], apparent CO2/MEA/water amounts [mol].

    Output rows are nine true amounts per mole of apparent feed, nine mu/RT,
    nine fugacities [Pa], and molar density [mol/m³], in Engine species order.
    With explicit reference thermochemistry, the last row is total enthalpy
    [J] on the unit apparent-feed basis, not per mole of true liquid species.
    Multiply amounts by apparent feed flow to obtain true species flows;
    normalize separately to obtain true mole fractions. Keep this object alive
    while CasADi functions containing it are in use. No solved state is stored.
    """

    def __init__(self, name: str, liquid: ReactiveLiquid):
        self.liquid = liquid
        self.component_ids = tuple(liquid.model.component_ids)
        self.molar_masses = liquid.molar_masses
        self.output_ids = tuple(
            f"{quantity}:{species}"
            for quantity in ("amount", "mu", "fugacity")
            for species in self.component_ids
        ) + ("liquid-molar-density",)
        self.output_units = ("mole",) * 9 + ("dimensionless",) * 9 + ("pascal",) * 9 + ("mole / meter**3",)
        if liquid.thermochemistry is not None:
            self.output_ids += ("total-enthalpy",)
            self.output_units += ("joule",)
        self._derivative_callbacks = []
        super().__init__()
        self.construct(name, {"enable_fd": False})

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, index):
        return ca.Sparsity.dense(5, 1)

    def get_sparsity_out(self, index):
        return ca.Sparsity.dense(len(self.output_ids), 1)

    def _state(self, argument):
        inputs = np.asarray(argument, dtype=float).reshape(-1)
        if inputs.shape != (5,) or np.any(~np.isfinite(inputs)) or np.any(inputs <= 0.0):
            raise ValueError("Reactive liquid inputs must be finite positive T, P, CO2, MEA, water")
        return inputs, self.liquid.solve(*inputs[:2], inputs[2:], state_input_derivatives=True)

    def eval(self, arguments):
        _, state = self._state(arguments[0])
        return [ca.DM(self._values_from_state(state))]

    def _values_from_state(self, state):
        values = np.r_[state["amounts_mol"], state["chemical_potentials_over_rt"],
                       state["fugacities_pa"], state["density_mol_m3"]]
        if self.liquid.thermochemistry is not None:
            values = np.r_[values, state["total_enthalpy_j"]]
        if np.any(~np.isfinite(values)):
            raise RuntimeError("Native equilibrium returned non-finite CasADi outputs")
        return values

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        derivative = _EquilibriumJacobian(name, self, inames, onames, opts)
        self._derivative_callbacks.append(derivative)
        return derivative

    def input_jacobian(self, argument):
        """Compose A1 with the exact apparent-feed normalization/invariant map."""
        inputs, state = self._state(argument)
        return self._input_jacobian(inputs, state)

    def _input_jacobian(self, inputs, state):
        block = state["state_input_derivatives"]
        if tuple(block.component_ids) != self.component_ids:
            raise ValueError("Engine derivative species order changed")
        feed = state["feed_amounts_mol"]
        feed_jacobian = np.zeros((9, 3))
        feed_jacobian[:3] = (np.eye(3) - feed[:3, None]) / inputs[2:].sum()
        mapping, active = _state_input_map(block, feed_jacobian)
        rows = {identity: index for index, identity in enumerate(block.output_identities)}
        indices = [rows[identity] for identity in self.output_ids[:28]]
        if tuple(block.output_units[index] for index in indices) != self.output_units[:28]:
            raise ValueError("Engine derivative output units changed")
        native = np.asarray([[block.jacobian[row][col] for col in active] for row in indices], dtype=float)
        if self.liquid.thermochemistry is not None:
            if block.caloric_failure is not None or block.total_enthalpy_jacobian is None:
                raise RuntimeError(f"Native caloric derivative unavailable: {block.caloric_failure}")
            native = np.vstack((native, np.asarray(block.total_enthalpy_jacobian, dtype=float)[active]))
        if np.any(~np.isfinite(native)):
            raise RuntimeError("Native equilibrium derivative is unavailable or non-finite")
        result = native @ mapping[active]
        if np.any(~np.isfinite(result)):
            raise RuntimeError("Non-finite equilibrium derivative after feed mapping")
        return result


class FixedCompositionVaporCallback(ca.Callback):
    """Native vapor f/rho/H and exact first derivatives in T/P/species flows.

    GREPE fixes every species amount with identity balances and imposes no
    reactions or phase splitting. H is on a unit-total gas-feed basis; multiply
    by total gas flow for enthalpy flow. The caller supplies an admitted neutral
    parameter set, explicit caloric reference and physical packing interval.
    No solved state, phase-root anchor or derivative approximation is stored.
    """

    def __init__(self, name, parameters, thermochemistry, *, packing_interval):
        import epcsaft
        from epcsaft.records import SingleParameterRecord

        self.model = epcsaft.Mixture(parameters)
        self.component_ids = tuple(self.model.component_ids)
        thermochemistry.validate_component_order(self.component_ids)
        self.thermochemistry = thermochemistry
        self.packing_interval = tuple(packing_interval)
        records = [r for r in parameters.records if isinstance(r, SingleParameterRecord)]
        if any(float(r.value) != 0. for r in records if r.family == "charge_number"):
            raise ValueError("Fixed-composition vapor must contain neutral species only")
        masses = {r.component_id: float(r.value.to("kilogram / mole").magnitude)
                  for r in records if r.family == "molar_mass"}
        self.molar_masses = tuple(masses[name] for name in self.component_ids)
        self.output_ids = tuple(f"fugacity:{s}" for s in self.component_ids) + ("vapor-molar-density", "total-enthalpy")
        self.output_units = ("pascal",) * len(self.component_ids) + ("mole / meter**3", "joule")
        self._derivative_callbacks = []
        super().__init__()
        self.construct(name, {"enable_fd": False})

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, index):
        return ca.Sparsity.dense(2 + len(self.component_ids), 1)

    def get_sparsity_out(self, index):
        return ca.Sparsity.dense(len(self.output_ids), 1)

    def _state(self, argument, *, actions=()):
        import epcsaft
        from epcsaft import equilibrium as q

        inputs = np.asarray(argument, dtype=float).reshape(-1)
        n = len(self.component_ids)
        if inputs.shape != (n + 2,) or np.any(~np.isfinite(inputs)) or np.any(inputs <= 0.):
            raise ValueError("Vapor inputs must be finite positive T, P and species flows")
        feed = inputs[2:] / inputs[2:].sum()
        identity = tuple(map(tuple, np.eye(n)))
        units = epcsaft.unit_registry
        initial = self.model.state(T=inputs[0] * units.kelvin, P=inputs[1] * units.pascal,
                                   x=feed, phase="vapor")
        volume = 1. / float(initial.molar_density.to("mole / meter**3").magnitude)
        chemistry = q.ChemicalEquilibriumProblem(
            species_ids=self.component_ids, charges=(0,) * n,
            molar_masses_kg_per_mol=self.molar_masses,
            balance_matrix=identity, conserved_totals=tuple(feed), reaction_matrix=(),
            feed_amounts_mol=tuple(feed), equilibrium_constants=(),
            strict_interior_amount_floor_mol=1e-12,
        )
        phase = q.ReactivePhase("vapor", "vapor", "finite", q.AllComponents(),
            q.EosModel(self.packing_interval, "installed-eos"), q.FinitePhaseStart(tuple(feed), volume))
        outputs = tuple(q.EquilibriumOutput(self.output_ids[i], "phase.fugacity", "pascal",
            "true-species", "vapor", identity[i], support="positive") for i in range(n)) + (
            q.EquilibriumOutput("vapor-molar-density", "phase.molar_density", "mole / meter**3",
                                "true-species", "vapor", support="positive"),
        )
        result = q.solve(self.model, q.GeneralReactiveEquilibriumProblem(
            identity="absorber-fixed-composition-vapor", temperature=q.Fixed(inputs[0] * units.kelvin),
            pressure=q.Fixed(inputs[1] * units.pascal), phases=(phase,), reaction_system=chemistry,
            reaction_phase_ids=(), outputs=outputs, thermochemistry=self.thermochemistry,
            state_input_derivatives=True,
            state_input_actions=tuple(actions),
        ))
        if result.status != "evaluated" or result.numerical_status != "passed" or result.physical_status != "passed":
            raise RuntimeError(f"Native vapor failed: {result.failure}; evidence={dict(result.evidence)}")
        block, enthalpy = result.state_input_derivatives, result.total_enthalpy
        if block is None or isinstance(block, epcsaft.NonEvaluableTrial):
            raise RuntimeError(f"Native vapor derivatives unavailable: {block}")
        if (not isinstance(enthalpy, epcsaft.EquilibriumEnthalpy)
                or enthalpy.reference_fingerprint != self.thermochemistry.scientific_fingerprint
                or enthalpy.parameter_fingerprint != result.descriptor.parameter_fingerprint):
            raise RuntimeError(f"Native vapor enthalpy unavailable or identity changed: {enthalpy}")
        rows = {row.identity: row for row in result.rows}
        selected = [rows[name] for name in self.output_ids[:-1]]
        if any(row.status != "evaluated" or row.value is None for row in selected):
            raise RuntimeError("Native vapor did not certify all requested outputs")
        values = np.r_[[row.value for row in selected], float(enthalpy.value.to("joule").magnitude)]
        if np.any(~np.isfinite(values)):
            raise RuntimeError("Native vapor outputs are non-finite")
        return inputs, feed, block, values, result

    def eval(self, arguments):
        return [ca.DM(self._state(arguments[0])[3])]

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        derivative = _EquilibriumJacobian(name, self, inames, onames, opts)
        self._derivative_callbacks.append(derivative)
        return derivative

    def input_jacobian(self, argument):
        inputs, feed, block, _, _ = self._state(argument)
        if tuple(block.component_ids) != self.component_ids:
            raise ValueError("Native vapor derivative species order changed")
        mapping, active = _state_input_map(block, (np.eye(len(feed)) - feed[:, None]) / inputs[2:].sum())
        rows = {identity: i for i, identity in enumerate(block.output_identities)}
        indices = [rows[name] for name in self.output_ids[:-1]]
        if tuple(block.output_units[i] for i in indices) != self.output_units[:-1]:
            raise ValueError("Native vapor derivative output units changed")
        if block.caloric_failure is not None or block.total_enthalpy_jacobian is None:
            raise RuntimeError(f"Native vapor caloric derivative unavailable: {block.caloric_failure}")
        native = np.vstack((np.asarray(block.jacobian, dtype=float)[indices][:, active],
                            np.asarray(block.total_enthalpy_jacobian, dtype=float)[active]))
        result = native @ mapping[active]
        if np.any(~np.isfinite(result)):
            raise RuntimeError("Native vapor derivative is unavailable or non-finite")
        return result


class _EquilibriumJacobian(ca.Callback):
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
        return ca.Sparsity.dense(len(self.parent.output_ids), self.parent.size1_in(0))

    def eval(self, arguments):
        return [ca.DM(self.parent.input_jacobian(arguments[0]))]


class _VaporCaloric(ca.Callback):
    """Cp and selected partial H, with native H² and both flow chain rules."""

    def __init__(self, name, vapor, indices):
        self.vapor = vapor
        self.rows = (0, *(i + 2 for i in indices))
        self.output_ids = ("vapor-cp", *(f"partial-H:{vapor.component_ids[i]}" for i in indices))
        self._derivative_callbacks = []
        super().__init__()
        self.construct(name, {"enable_fd": False})

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, index):
        return self.vapor.sparsity_in(0)

    def get_sparsity_out(self, index):
        return ca.Sparsity.dense(len(self.rows), 1)

    def _data(self, argument):
        inputs, feed, block, values, _ = self.vapor._state(argument)
        if tuple(block.component_ids) != self.vapor.component_ids:
            raise ValueError("Native vapor derivative species order changed")
        if block.caloric_failure is not None or block.total_enthalpy_jacobian is None:
            raise RuntimeError(f"Native vapor caloric derivative unavailable: {block.caloric_failure}")
        mapping, active = _state_input_map(block, (np.eye(len(feed)) - feed[:, None]) / inputs[2:].sum())
        first = np.asarray(block.total_enthalpy_jacobian, dtype=float)[active]
        if np.any(~np.isfinite(first)):
            raise RuntimeError("Native vapor caloric first derivative is non-finite")
        return inputs, block, values[-1], mapping, active, first

    def eval(self, arguments):
        inputs, _, h, mapping, active, first = self._data(arguments[0])
        dh = first @ mapping[active]
        return [ca.DM([dh[0], *(h + inputs[2:].sum() * dh[i] for i in self.rows[1:])])]

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        derivative = _EquilibriumJacobian(name, self, inames, onames, opts)
        self._derivative_callbacks.append(derivative)
        return derivative

    def input_jacobian(self, argument):
        from epcsaft.equilibrium import EquilibriumStateInputAction

        inputs, block, _, mapping, active, first = self._data(argument)
        total, size = inputs[2:].sum(), len(inputs)
        actions = tuple(EquilibriumStateInputAction(
            f"vapor-H-{row}-{col}", 2, block.input_identities, block.input_units,
            (tuple(mapping[:, row]), tuple(mapping[:, col])), total_enthalpy=True,
        ) for row in self.rows for col in range(size))
        result = self.vapor._state(inputs, actions=actions)[4]
        if len(result.state_input_actions) != len(actions):
            raise RuntimeError("Native vapor enthalpy actions are missing")
        second = np.empty((len(self.rows), size))
        for index, (requested, action) in enumerate(zip(actions, result.state_input_actions, strict=True)):
            if (action.identity != requested.identity or action.order != 2
                    or action.input_identities != requested.input_identities
                    or action.input_units != requested.input_units or action.directions != requested.directions
                    or action.output_identities != self.vapor.output_ids[:-1]
                    or action.output_units != self.vapor.output_units[:-1]):
                raise ValueError("Native vapor enthalpy action identity, order, directions or units changed")
            # Ordinary-output failure is independent of the requested H² field.
            if action.caloric_failure is not None or action.total_enthalpy_action_j is None:
                raise RuntimeError(f"Native vapor enthalpy action unavailable: {action.caloric_failure}; evidence={dict(action.evidence)}; central={dict(result.evidence)}")
            second.flat[index] = action.total_enthalpy_action_j
        b = np.column_stack((np.zeros((size - 2, 2)), np.eye(size - 2)))
        ds = b.sum(axis=0)
        dh = first @ mapping[active]
        for index, row in enumerate(self.rows):
            a = b[:, row]
            curvature = -(a[:, None] * ds + b * a.sum()) / total**2
            curvature += 2 * inputs[2:, None] * a.sum() * ds / total**3
            invariant = dict(zip(block.invariant_ids, np.asarray(block.invariant_matrix) @ curvature, strict=True))
            native_curvature = np.asarray([np.zeros(size) if name in ("temperature_k", "pressure_pa")
                                           else invariant[name] for name in block.input_identities])
            # D²(S*h) includes normalization curvature and both product terms.
            second[index] = total * (second[index] + first @ native_curvature[active]) + ds[row] * dh + ds * dh[row]
        second[0] = second[0] / total - dh[0] * ds / total
        if np.any(~np.isfinite(second)):
            raise RuntimeError("Native vapor caloric second action is non-finite after normalization")
        return second


class _LoadingTangent(ca.Callback):
    """mu loading tangent and its exact A2 outer derivative, without H actions."""

    def __init__(self, name, liquid_callback, *, include_state=False):
        self.liquid_callback = liquid_callback
        self.include_state = include_state
        self.output_ids = (liquid_callback.output_ids if include_state else ()) + tuple(
            f"loading-tangent:{s}" for s in liquid_callback.component_ids)
        self._derivative_callbacks = []
        super().__init__()
        self.construct(name, {"enable_fd": False})

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, index):
        return ca.Sparsity.dense(5, 1)

    def get_sparsity_out(self, index):
        return ca.Sparsity.dense(len(self.output_ids), 1)

    def eval(self, arguments):
        callback = self.liquid_callback
        inputs, state = callback._state(arguments[0])
        tangent = callback._input_jacobian(inputs, state)[9:18, 2] * inputs[2]
        # Values and tangent consume this one evaluated state; none is retained.
        values = np.r_[callback._values_from_state(state), tangent] if self.include_state else tangent
        return [ca.DM(values)]

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        derivative = _EquilibriumJacobian(name, self, inames, onames, opts)
        self._derivative_callbacks.append(derivative)
        return derivative

    def input_jacobian(self, argument):
        from epcsaft.equilibrium import EquilibriumStateInputAction

        callback = self.liquid_callback
        inputs, state = callback._state(argument)
        block = state["state_input_derivatives"]
        if tuple(block.component_ids) != callback.component_ids:
            raise ValueError("Engine derivative species order changed")
        total = inputs[2:].sum()
        feed_jacobian = np.zeros((9, 3))
        feed_jacobian[:3] = (np.eye(3) - state["feed_amounts_mol"][:3, None]) / total
        mapping, active = _state_input_map(block, feed_jacobian)
        output_ids = callback.output_ids[9:18]
        rows = {identity: index for index, identity in enumerate(block.output_identities)}
        if tuple(block.output_units[rows[s]] for s in output_ids) != ("dimensionless",) * 9:
            raise ValueError("Engine chemical-potential derivative units changed")
        native_first = np.asarray([[block.jacobian[rows[s]][i] for i in active] for s in output_ids])
        if np.any(~np.isfinite(native_first)):
            raise RuntimeError("Native chemical-potential first derivative unavailable")
        actions = tuple(EquilibriumStateInputAction(
            f"loading-outer-{i}", 2, block.input_identities, block.input_units,
            (tuple(mapping[:, 2] * inputs[2]), tuple(mapping[:, i])),
        ) for i in range(5))
        result = callback.liquid.solve_actions(*inputs[:2], inputs[2:], actions, output_ids)
        returned = result.state_input_actions
        if len(returned) != len(actions):
            raise RuntimeError("Native loading actions are missing")
        second = np.empty((9, 5))
        for i, (requested, action) in enumerate(zip(actions, returned, strict=True)):
            if (action.identity != requested.identity or action.order != 2
                    or action.input_identities != requested.input_identities
                    or action.input_units != requested.input_units or action.directions != requested.directions
                    or action.output_identities != output_ids or action.output_units != ("dimensionless",) * 9):
                raise ValueError("Native loading action identity, order, directions or units changed")
            if action.failure is not None or any(value is None for value in action.values):
                raise RuntimeError(f"Native loading action unavailable: {action.failure}; evidence={dict(action.evidence)}; central={dict(result.evidence)}")
            second[:, i] = action.values
        # q(F/S) has curvature. The first direction is F_CO2 e_CO2;
        # additionally differentiate that direction's F_CO2 prefactor below.
        a = np.array([inputs[2], 0., 0.])
        b = np.column_stack((np.zeros((3, 2)), np.eye(3)))
        curvature = np.zeros((9, 5))
        curvature[:3] = -(a[:, None] * b.sum(axis=0) + b * a.sum()) / total**2
        curvature[:3] += 2 * inputs[2:, None] * a.sum() * b.sum(axis=0) / total**3
        invariant_curvature = dict(zip(block.invariant_ids, np.asarray(block.invariant_matrix) @ curvature, strict=True))
        native_curvature = np.asarray([np.zeros(5) if identity in ("temperature_k", "pressure_pa")
                                       else invariant_curvature[identity] for identity in block.input_identities])
        second += native_first @ native_curvature[active]
        second[:, 2] += native_first @ mapping[active, 2]
        if np.any(~np.isfinite(second)):
            raise RuntimeError("Native loading second action is non-finite after normalization")
        return np.vstack((callback._input_jacobian(inputs, state), second)) if self.include_state else second
