from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
from .epcsaft_v02 import parameter_fingerprint


_ATOMIC_MASS_KG_PER_MOL = {"C": 0.012011, "H": 0.001008, "N": 0.014007, "O": 0.015999}
_FORMULAS = {
    "carbon-dioxide": {"C": 1, "O": 2},
    "monoethanolamine": {"C": 2, "H": 7, "N": 1, "O": 1},
    "water": {"H": 2, "O": 1},
    "protonated-monoethanolamine": {"C": 2, "H": 8, "N": 1, "O": 1},
    "carbamate-anion": {"C": 3, "H": 6, "N": 1, "O": 3},
    "bicarbonate-anion": {"C": 1, "H": 1, "O": 3},
    "carbonate-anion": {"C": 1, "O": 3},
    "hydronium-cation": {"H": 3, "O": 1},
    "hydroxide-anion": {"H": 1, "O": 1},
}
_CHARGES = (0, 0, 0, 1, -1, -1, -2, 1, -1)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


_CORRELATION_FIELDS = ("a", "b", "c", "d", "reference_temperature", "temperature_min", "temperature_max",
                      "standard_state_id")
_ENGINE_REACTIONS = "engine-reactions.json"
_IDEAL_GAS = "ideal-gas-thermochemistry.json"


def ideal_gas_record(path: str | Path, component_ids):
    """Physical ideal-gas records in component order; absent species (ions) are reaction-completed by the Engine."""
    import epcsaft

    data = _json(Path(path))
    pressure, intervals = data.get("standard_pressure_pa", data.get("reference_pressure_pa")), []
    for name in component_ids:
        row = data["components"].get(name)
        if row is None:
            intervals.append([])
            continue
        form_name = row.get("form", row.get("representation"))
        if form_name == "shomate":
            form = epcsaft.IdealShomate(row["coefficients"], row["formation_enthalpy_j_per_mol"])
        elif form_name == "shomate_cp":
            a, b, c, d, e = row["coefficients"]
            # Inert-gas sensible properties use zero enthalpy and entropy at 298.15 K.
            t = 298.15 / 1000
            f = -(a*t + b*t**2/2 + c*t**3/3 + d*t**4/4 - e/t)
            g = -(a*math.log(t) + b*t + c*t**2/2 + d*t**3/3 - e/(2*t**2))
            form = epcsaft.IdealShomate([a, b, c, d, e, f, g, 0.], 0.)
        elif form_name == "cp_polynomial":
            form = epcsaft.IdealPolynomial(row["coefficients"], row["enthalpy_entropy_anchor_k"], 0., 0.)
        else:  # Cp = sum a_j T^j re-centred at T0, where h = s = 0 (moiety convention)
            t0, a = row["reference_temperature_k"], row["coefficients"]
            form = epcsaft.IdealPolynomial([math.fsum(c * math.comb(j, k) * t0 ** (j - k) for j, c in enumerate(a) if j >= k)
                                            for k in range(len(a))], t0, 0., 0.)
        intervals.append([epcsaft.IdealInterval(*row.get("interval_k", row.get("range_k")), True, True,
                                                epcsaft.IdealCorrelation(form, pressure))])
    return epcsaft.ThermochemistryRecord(pressure, intervals)


def _engine_reaction_record(parameter_path, reaction_path):
    record = _json(Path(reaction_path))
    if record["schema"] != "absorber-engine-reactions-v1":
        raise ValueError("Unsupported Engine reaction record")
    if hashlib.sha256(parameter_path.read_bytes()).hexdigest() != record["parameter_sha256"]:
        raise ValueError("Engine reaction records do not belong to this parameter record")
    return record


def _engine_reactions(record):
    from epcsaft import equilibrium as q

    reactions = [q.Reaction(
        row["stoichiometry"], name=row["name"],
        correlation=q.ReactionLogPolynomial(*(row["correlation"][key] for key in _CORRELATION_FIELDS)),
        reference=q.ReactionReference(getattr(q.ReferenceSourceBasis, row["reference"]["source_basis"]),
                                      reference_pressure_pa=row["reference"].get("reference_pressure_pa")),
    ) for row in record["reactions"]]
    return reactions, q.NeutralReference(**record["neutral_reference"])


def engine_liquid(dataset: str | Path, thermochemistry_path: str | Path, name="reactive_liquid", *, loading_policy=None, declarations=None, reaction_path=None, domain=None, parameter_path=None, row_tolerance=None):
    """Declared reactive liquid with native reactions, reference and absolute analytical feed."""
    from .casadi_reactive import EnginePhase
    from .solvent_inputs import formula_masses

    dataset = Path(dataset)
    parameter_path = Path(parameter_path or dataset / "parameters.json")
    record = _engine_reaction_record(parameter_path, reaction_path or dataset / _ENGINE_REACTIONS)
    mapping = _json(parameter_path)
    reactions, neutral_reference = _engine_reactions(record)
    ids = declarations["species_order"] if declarations is not None else [c["component_id"] for c in mapping["components"]]
    masses = formula_masses(declarations, ids, mapping) if declarations is not None else _molar_masses(dataset, ids)
    if declarations is None:
        # Undeclared callers retain the original fixed MEA basis, independent of analysis examples.
        declarations = {"liquid_feed_order": ids[:3], "liquid_support": ids,
                        "roles": {"co2": "carbon-dioxide", "water": "water", "amines": ["monoethanolamine"],
                                  "inert_gases": ["nitrogen", "oxygen"]},
                        "components": {"order": ["co2", "amine:monoethanolamine", "water"],
                                       "rows": [[1, 0, 0, 0, 1, 1, 1, 0, 0],
                                                [0, 1, 0, 1, 1, 0, 0, 0, 0],
                                                [0, 0, 1, 0, 0, 1, 1, 1, 1]]}}
    # Formula masses (checked against the declared ones) conserve mass exactly through the reactions.
    phase = EnginePhase(name, mapping, ideal_gas_record(thermochemistry_path, ids), kind="liquid",
                        feed_ids=declarations["liquid_feed_order"], molar_masses=masses,
                        reactions=reactions, neutral_reference=neutral_reference,
                        loading_policy=loading_policy, roles=declarations["roles"], support=declarations["liquid_support"], domain=domain,
                        component_order=ids, row_tolerance=row_tolerance)
    phase.components = dict(zip(declarations["components"]["order"], declarations["components"]["rows"], strict=True))
    phase.film_constraints = [phase.charges, *[row for key, row in phase.components.items() if key != "co2"]]
    return phase


def engine_vapor(parameters_path: str | Path, thermochemistry_path: str | Path, name="neutral_vapor", *, declarations=None, domain=None):
    """Nonreacting vapor at its species feed; the same ideal-gas records as the liquid."""
    from .casadi_reactive import EnginePhase

    mapping = _json(Path(parameters_path))
    ids = [c["component_id"] for c in mapping["components"]]
    return EnginePhase(name, mapping, ideal_gas_record(thermochemistry_path, ids), kind="vapor",
                       **({"feed_ids": declarations["vapor_feed_order"], "support": declarations["vapor_support"],
                           "roles": declarations["roles"], "domain": domain} if declarations else {}))


def _native_reactions(reactions):
    """Map the retained molality correlations to Engine raw mole-fraction references."""
    from epcsaft import equilibrium as q

    standard = reactions['source_standard_state']
    declarations = []
    for row in reactions['reactions']:
        c = row['coefficients']
        if row['ln_k_form'].startswith('-ln(10)'):
            a, b, ln_t, d = (-math.log(10.)*c['b'], -math.log(10.)*c['a_k'], 0., -math.log(10.)*c['c_per_k'])
        else:
            a, b, ln_t, d = c['a']+row.get('standard_state_offset', 0.), c['b_k'], c.get('c', 0.), c.get('d_per_k', 0.)
        # The old activity scale factors and the ln(T/Tref) recentering each enter once.
        a += sum(v*f for v, f in zip(row['stoichiometry'], standard['log_activity_scale_factors']))
        lo, hi = row['temperature_domain_k']
        correlation = q.ReactionLogPolynomial(a+ln_t*math.log(313.15), b, ln_t, d, 313.15, lo, hi,
                                              'aqueous-mole-fraction-infinite-dilution-water-v1')
        declarations.append(q.Reaction(dict(zip(reactions['species_ids'], row['stoichiometry'])),
            name=row['reaction_id'], correlation=correlation,
            reference=q.ReactionReference(q.ReferenceSourceBasis.RawMoleFractionInfiniteDilution,
                                           reference_pressure_pa=standard['reference_pressure_pa'])))
    return declarations


def _neutral_reference(reactions):
    from epcsaft import equilibrium as q

    standard = reactions['source_standard_state']
    ids = reactions['species_ids']
    ratios = np.array([1. if z == 0 else 0. for z in reactions['charges']])
    solvent = [ids[i] for i, v in enumerate(standard['solvent_composition']) if v == 1.]
    if len(solvent) != 1:
        raise ValueError('Reactive reference requires its declared pure solvent')
    ratios[ids.index(solvent[0])] = 0.
    for pair, coefficients in standard['ion_pairs']:
        for name, value in zip(pair, coefficients):
            ratios[ids.index(name)] += value
    return q.NeutralReference(solvent=solvent[0], solute_molality_ratios=dict(zip(ids, ratios)),
                              standard_molality_mol_per_kg=standard['standard_molality_mol_per_kg'])


def _action_directions(component_count, feed_count):
    from epcsaft import equilibrium as q

    directions = []
    for j in range(2+feed_count):
        feed = [0.] * component_count
        if j >= 2:
            feed[j-2] = 1.
        directions.append(q.SolvedStateActionDirection(float(j == 0), float(j == 1), feed, []))
    return directions


def _action_values(results, field):
    values = []
    for item in results:
        value = getattr(item, field)
        if item.status.name != 'Available' or value is None or not math.isfinite(value):
            raise RuntimeError(f'Native {item.observable.kind.name} {field} unavailable: '
                               f'{item.status}: {item.diagnostic_message}')
        values.append(value)
    return values


def _first_actions(model, problem, result, observables, feed_count):
    """Exact columns with respect to T [K], P [Pa] and neutral feed amounts [mol]."""
    from epcsaft import equilibrium as q

    compiled = q.compile_problem(model, problem)
    directions = _action_directions(len(model.component_ids), feed_count)
    batches = q.solved_state_first_actions(compiled, result, observables, directions)
    return np.array([_action_values(batch.results, 'action') for batch in batches]).T


def _system_enthalpy(compiled, result, observable, direction):
    from epcsaft import equilibrium as q

    request = q.SolvedStateActionRequest([observable], [direction])
    value, = _action_values(q.solved_state_actions(compiled, result, request).results, 'value')
    return value


def _enthalpy_rows(compiled, result, directions, indices, first, second):
    """Extensive enthalpy, its selected first actions, and optionally their Jacobian."""
    from epcsaft import equilibrium as q

    observable = q.SolvedStateObservable(q.SolvedStateObservableKind.SystemEnthalpy)
    value = _system_enthalpy(compiled, result, observable, directions[0])
    if not second:
        return value, first[list(indices)], None
    rows = np.empty((len(indices), len(directions)))
    evaluated = {}
    for row, i in enumerate(indices):
        for j in range(len(directions)):
            pair = tuple(sorted((i, j)))
            if pair not in evaluated:
                request = q.SolvedStateActionRequest([observable], [directions[k] for k in pair])
                evaluated[pair], = _action_values(q.solved_state_actions(compiled, result, request).results, 'action')
            rows[row, j] = evaluated[pair]
    return value, first[list(indices)], rows



class ReactiveLiquid:
    """Declared nine-species reacting liquid on one mole of apparent feed.

    A loading path keeps the selected liquid branch and its fixed reference
    pressure. Cached values and exact first actions share identical inputs.
    """

    def __init__(self, dataset, *, loading_anchor=None, max_log_loading_step=.1, max_loading_steps=32,
                 water_per_mea_anchor=None, reuse_states=False, warm_starts=True, kij_scale=None, reaction_scale=None):
        dataset = Path(dataset)
        self.dataset, self.kij_scale = str(dataset), None if kij_scale is None else tuple(kij_scale)
        record = reaction_system(self.dataset, reaction_scale)
        if record.get("schema") == "absorber-engine-reactions-v1":
            self._reactions, self._neutral_reference = _engine_reactions(record)
        else:
            self._reactions, self._neutral_reference = _native_reactions(record), _neutral_reference(record)
        self._temperature_domains = [
            (r.name, r.correlation.temperature_min, r.correlation.temperature_max) for r in self._reactions
        ]
        document = parameter_document(self.dataset, self.kij_scale)
        self.parameter_fingerprint = parameter_fingerprint(document)
        self.model = _mixture(dataset, document)
        charges = tuple(c["fixed"]["charge_number"]["value"]["magnitude"] for c in document["components"])
        if tuple(self.model.component_ids) != tuple(_FORMULAS) or charges != _CHARGES:
            raise ValueError("Reactive ePC-SAFT species order or charges changed")
        self.molar_masses = tuple(_molar_masses(dataset, self.model.component_ids))
        self.loading_anchor, self.max_log_loading_step = loading_anchor, max_log_loading_step
        self.max_loading_steps, self.water_per_mea_anchor = max_loading_steps, water_per_mea_anchor
        self.reuse_states, self.warm_starts = reuse_states, warm_starts
        self._accepted = self._native_accepted = None
        self._states = OrderedDict()
        self.stats = dict(queries=0, cache_hits=0, native_solves=0, native_seconds=0., warm_starts=0)

    def name(self):
        return 'reactive_liquid'

    def _problem(self, temperature, pressure, feed):
        from epcsaft import equilibrium as q

        return q.Problem(phases=[q.Phase('liquid', kind='liquid')], T=temperature, P=pressure,
            feed=q.Amounts(dict(zip(self.model.component_ids[:3], feed))),
            reactions=self._reactions, neutral_reference=self._neutral_reference)

    def _solve(self, problem, start=None):
        from epcsaft import equilibrium as q

        started = time.perf_counter()
        try:
            result = q.solve_equilibrium(self.model, problem, start)
        finally:
            self.stats['native_solves'] += 1
            self.stats['native_seconds'] += time.perf_counter()-started
        if not result.success:
            raise RuntimeError(f'Reactive equilibrium failed: {result.message}')
        return result

    def system_enthalpy(self, temperature, pressure, apparent_amounts):
        """Extensive enthalpy [J] for an absolute feed [mol], including zero carbon."""
        from epcsaft import equilibrium as q

        problem = self._problem(float(temperature), float(pressure), tuple(map(float, apparent_amounts)))
        result = self._solve(problem)
        compiled = q.compile_problem(self.model, problem)
        observable = q.SolvedStateObservable(q.SolvedStateObservableKind.SystemEnthalpy)
        direction = _action_directions(len(self.model.component_ids), len(apparent_amounts))[0]
        return _system_enthalpy(compiled, result, observable, direction), result.linear_solver

    def solve(self, temperature_k, pressure_pa, apparent_amounts, *, state_input_derivatives=False, caloric=False, water=False, caloric_jacobian=False):
        from epcsaft import equilibrium as q

        inputs = tuple(float(v) for v in (temperature_k, pressure_pa, *apparent_amounts))
        if len(inputs) != 5 or not all(math.isfinite(v) and v > 0 for v in inputs):
            raise ValueError('Reactive inputs require finite positive T, P and three apparent amounts')
        for name, lower, upper in self._temperature_domains:
            if not lower <= temperature_k <= upper:
                raise ValueError(f"{name} temperature {temperature_k} K is outside [{lower}, {upper}] K")
        key = (inputs, state_input_derivatives, caloric, water)
        self.stats['queries'] += 1
        if self.reuse_states and key in self._states:
            self.stats['cache_hits'] += 1
            self._states.move_to_end(key)
            resolved, compiled, result = self._states[key]
            if caloric_jacobian and resolved['enthalpy_jacobian'] is None:
                self._liquid_enthalpy_jacobian(resolved, compiled, result)
            return deepcopy(resolved)
        feed = np.array(inputs[2:])/sum(inputs[2:])
        problem = self._problem(temperature_k, pressure_pa, feed)
        start = None
        if self.reuse_states and self.warm_starts and self._accepted is not None:
            old_feed = self._accepted['feed_amounts_mol'][:3]
            distance = max(abs(math.log(feed[i]/feed[1])-math.log(old_feed[i]/old_feed[1])) for i in (0, 2))
            if self.loading_anchor is None or distance <= self.max_log_loading_step:
                start = self._native_accepted
                self.stats['warm_starts'] += 1
        if start is None and self.loading_anchor is not None:
            anchor = feed.copy()
            anchor[0] = self.loading_anchor*feed[1]
            if self.water_per_mea_anchor is not None:
                anchor[2] = self.water_per_mea_anchor*feed[1]
            distance = max(abs(math.log(feed[i]/anchor[i])) for i in (0, 2))
            steps = max(1, math.ceil(distance/self.max_log_loading_step))
            if steps > self.max_loading_steps:
                raise RuntimeError(f'Liquid loading path requires {steps} steps; declared budget is {self.max_loading_steps}')
            start = self._solve(self._problem(temperature_k, pressure_pa, anchor/anchor.sum()))
            for index in range(1, steps):
                intermediate = anchor*np.exp(np.log(feed/anchor)*index/steps)
                start = self._solve(self._problem(temperature_k, pressure_pa, intermediate/intermediate.sum()), start)
        result = self._solve(problem, start)
        composition, density = np.array(result.mole_fractions), result.molar_densities[0]
        root = self.model.state(temperature_k, P=pressure_pa, x=composition, phase='liquid')
        if not np.isclose(density, root.molar_density, rtol=1e-8, atol=0):
            raise RuntimeError('Reactive equilibrium changed the native liquid pressure root')
        state = self.model.state(temperature_k, rho=density, x=composition)
        fugacities = np.exp(state.log_fugacity_coefficient)*composition*pressure_pa
        amounts = composition*result.amounts[0]
        resolved = dict(composition=composition, amounts_mol=amounts, feed_amounts_mol=np.r_[feed, np.zeros(6)],
            density_mol_m3=density, fugacities_pa=fugacities, parameter_fingerprint=self.parameter_fingerprint,
            evidence=dict(iterations=result.iterations, maximum_residual=max(map(abs, result.residuals)),
                          reduced_hessian_minimum=result.reduced_hessian_minimum, linear_solver=result.linear_solver))
        compiled = None
        if state_input_derivatives or caloric:
            kind = q.SolvedStateObservableKind
            observables = []
            if state_input_derivatives:
                observables = [q.SolvedStateObservable(kind.PhaseComponentAmount, 0, i) for i in range(9)]
                observables += [q.SolvedStateObservable(kind.PhaseMolarDensity, 0),
                                q.SolvedStateObservable(kind.PhaseLogActivity, 0, 0)]
                if water:
                    observables.append(q.SolvedStateObservable(kind.PhaseLogActivity, 0, 2))
            if caloric:
                observables.append(q.SolvedStateObservable(kind.SystemEnthalpy))
            compiled = q.compile_problem(self.model, problem)
            directions = _action_directions(9, 3)
            batches = q.solved_state_first_actions(compiled, result, observables, directions)
            derivative = np.array([_action_values(batch.results, 'action') for batch in batches]).T
            if caloric:
                value, partials, _ = _enthalpy_rows(compiled, result, directions, (0, 2, 4), derivative[-1], False)
                resolved.update(enthalpy=value, enthalpy_partials=partials, enthalpy_jacobian=None)
                derivative = derivative[:-1]
                if caloric_jacobian:
                    self._liquid_enthalpy_jacobian(resolved, compiled, result)
            if state_input_derivatives:
                for row, i in ((10, 0), (11, 2)) if water else ((10, 0),):
                    derivative[row] *= fugacities[i]
                    derivative[row, 0] += fugacities[i]/temperature_k
                resolved['amount_density_fugacity_jacobian'] = derivative
        if self.reuse_states:
            if self.warm_starts:
                self._accepted, self._native_accepted = deepcopy(resolved), result
            self._states[key] = (deepcopy(resolved), compiled if caloric else None, result if caloric else None)
            if len(self._states) > 2048:
                self._states.popitem(last=False)
        return resolved

    @staticmethod
    def _liquid_enthalpy_jacobian(resolved, compiled, result):
        _, _, resolved['enthalpy_jacobian'] = _enthalpy_rows(
            compiled, result, _action_directions(9, 3), (0, 2, 4), np.zeros(5), True)


FIT_A = Path(__file__).resolve().parents[1] / "data/epcsaft_datasets/MEA_A_196faf11"
BULK_VAPOR = FIT_A.with_name("MEA_neutral_vapor") / "pc-saft.json"
CORRELATION_SOURCES = dict(liquid_density_source='correlation', energy_model='correlation',
                           water_transfer_source='correlation', inlet_humidity_source='correlation')
ENGINE_SOURCES = dict.fromkeys(CORRELATION_SOURCES, 'epcsaft')


def reactive_property_sources(dataset, settings):
    # Explicit pre-thermochemistry bundles remain usable with the correlation closures.
    defaults = ENGINE_SOURCES if (Path(dataset) / _IDEAL_GAS).is_file() else CORRELATION_SOURCES
    return {name: settings.get(name, default) for name, default in defaults.items()}


@lru_cache(maxsize=4)
def bulk_vapor_phase(dataset):
    return engine_vapor(BULK_VAPOR, Path(dataset) / _IDEAL_GAS, name="bulk_vapor")


@lru_cache(maxsize=2048)
def bulk_vapor_state(temperature, pressure, feed, dataset, caloric):
    """One four-species wet-gas state, with native first actions in absolute-feed coordinates."""
    from epcsaft import equilibrium as q

    phase = bulk_vapor_phase(dataset)
    compiled, result = phase.solve((temperature, pressure, *feed))
    composition = np.asarray(result.mole_fractions)
    density = result.molar_densities[0]
    state = phase.model.state(temperature, rho=density, x=composition)
    fugacities = np.exp(state.log_fugacity_coefficient)*composition*pressure
    observables = [phase.state_observables[-1], phase.log_activities[1]]
    if caloric:
        observables.append(q.SolvedStateObservable(q.SolvedStateObservableKind.SystemEnthalpy))
    directions = _action_directions(4, 4)
    batches = q.solved_state_first_actions(compiled, result, observables, directions)
    derivative = np.array([_action_values(batch.results, 'action') for batch in batches]).T
    derivative[1] *= fugacities[1]
    derivative[1, 0] += fugacities[1]/temperature
    resolved = dict(density=density, fugacities=fugacities, derivative=derivative[:2],
                    linear_solver=result.linear_solver)
    if caloric:
        value, partials, _ = _enthalpy_rows(compiled, result, directions, (0, 2, 3), derivative[2], False)
        resolved.update(enthalpy=value, enthalpy_partials=partials)
    return resolved


@lru_cache(maxsize=2048)
def bulk_vapor_enthalpy_jacobian(temperature, pressure, feed, dataset):
    compiled, result = bulk_vapor_phase(dataset).solve((temperature, pressure, *feed))
    return _enthalpy_rows(compiled, result, _action_directions(4, 4), (0, 2, 3), np.zeros(6), True)[2]


@lru_cache(maxsize=2)
def _pure_vapor_component(component, dataset=None):
    import epcsaft

    components = (component,)
    parameters = epcsaft.Parameters.from_mapping(_json(BULK_VAPOR), components=components)
    thermochemistry = ideal_gas_record(Path(dataset) / _IDEAL_GAS, components) if dataset else None
    return epcsaft.Pure(parameters, thermochemistry=thermochemistry)


def pure_vapor_enthalpy(temperature, pressure, component, dataset):
    """Molar enthalpy [J/mol] at the pure-gas reference for an apparent feed."""
    return _pure_vapor_component(component, dataset).state(temperature, P=pressure, phase='vapor').total_enthalpy


def saturated_water_fraction(temperature, pressure, dry_co2, dry_n2, dry_o2):
    """Water fugacity equality at inlet T/P, holding dry-gas proportions fixed."""
    from scipy.optimize import brentq

    liquid = _pure_vapor_component('water').state(temperature, P=pressure, x=[1.], phase='liquid')
    target = math.exp(liquid.log_fugacity_coefficient[0])*pressure
    model = bulk_vapor_phase(str(FIT_A)).model

    def residual(water):
        wet = 1.-water
        vapor = model.state(temperature, P=pressure, x=[wet*dry_co2, water, wet*dry_n2, wet*dry_o2],
                            phase='vapor')
        return water*pressure*math.exp(vapor.log_fugacity_coefficient[1])-target

    return brentq(residual, 1e-10, .95)


@lru_cache(maxsize=2048)
def neutral_vapor_state(temperature_k, pressure_pa, composition, dataset_text=None, kij_scale=None):
    """Neutral-vapor CO2 fugacity [Pa] and exact T/P/feed columns."""
    from epcsaft import equilibrium as q

    model = mixture(dataset_text or str(DATASET), ('carbon-dioxide', 'monoethanolamine', 'water'), kij_scale)
    problem = q.Problem(phases=[q.Phase('vapor', kind='vapor')], T=temperature_k, P=pressure_pa,
                        feed=q.Amounts(dict(zip(model.component_ids, composition))))
    result = q.solve_equilibrium(model, problem)
    if not result.success:
        raise RuntimeError(f'Neutral vapor equilibrium failed: {result.message}')
    state = model.state(temperature_k, rho=result.molar_densities[0], x=result.mole_fractions)
    fugacity = math.exp(state.log_fugacity_coefficient[0])*composition[0]*pressure_pa
    derivative = _first_actions(model, problem, result, [q.SolvedStateObservable(q.SolvedStateObservableKind.PhaseLogActivity, 0, 0)], 3)[0]
    derivative *= fugacity
    derivative[0] += fugacity/temperature_k
    return fugacity, derivative


def _bundle_file_hashes(bundle):
    return {
        Path(item["path"]).name: item["sha256"]
        for item in bundle["files"]
        if item["path"] in {"parameters/parameters.json", "chemistry/reaction-system.json",
                            "anchored-reference-thermochemistry.json", "adoption-receipt.json"}
    }


def dataset_input_paths(dataset: Path):
    if (dataset / _ENGINE_REACTIONS).is_file():
        return [dataset / name for name in ("parameters.json", _ENGINE_REACTIONS, _IDEAL_GAS)]
    bundle_path = dataset / "bundle.json"
    return [bundle_path, *(dataset / name for name in _bundle_file_hashes(_json(bundle_path)))]


def validate_reactive_bundle(dataset_text: str) -> dict:
    dataset = Path(dataset_text)
    bundle = _json(dataset / "bundle.json")
    expected = _bundle_file_hashes(bundle)
    if not {"parameters.json", "reaction-system.json"} <= expected.keys():
        raise ValueError("Reactive bundle must identify parameters and reaction-system files")
    for name, digest in expected.items():
        actual = hashlib.sha256((dataset / name).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(
                f"Reactive ePC-SAFT {name} SHA-256 mismatch: expected {digest}, got {actual}"
            )
    if bundle["parameter_document_sha256"] != expected["parameters.json"]:
        raise ValueError("Reactive bundle parameter metadata does not match verified parameters")
    if "adoption-receipt.json" in expected:
        receipt = _json(dataset / "adoption-receipt.json")
        if receipt["adopted_parameter_sha256"] != expected["parameters.json"]:
            raise ValueError("Adoption receipt does not match verified parameters")
    reactions = _json(dataset / "reaction-system.json")
    if tuple(reactions["species_ids"]) != tuple(_FORMULAS) or tuple(reactions["charges"]) != _CHARGES:
        raise RuntimeError("Reactive ePC-SAFT species order or charges changed")
    if reactions["reaction_sign_convention"] != "products_positive":
        raise RuntimeError("Reactive ePC-SAFT reaction sign convention changed")
    # Typed fitted correlations supersede source correlations as a whole.
    # In particular, their a already includes the molality conversion: adding
    # the historical source offset again would silently change equilibrium.
    document = _json(dataset / "parameters.json")
    units = {
        "ln-k-a-plus-b-over-t": {"a": "dimensionless", "b_k": "kelvin"},
        "ln-k-a-plus-b-over-t-plus-c-ln-t-plus-d-t": {
            "a": "dimensionless", "b_k": "kelvin", "c": "dimensionless", "d_per_k": "1 / kelvin"},
        "negative-log10-temperature-polynomial": {"a_k": "kelvin", "b": "dimensionless", "c_per_k": "1 / kelvin"},
    }
    for reaction in reactions["reactions"]:
        fitted = [row for row in document["reaction_correlations"] if row["reaction_id"] == reaction["reaction_id"]]
        if fitted:
            if len(fitted) != 1:
                raise ValueError("Duplicate fitted reaction correlation")
            row = fitted[0]
            expected = units[row["kind"]]
            if ({c["name"]: c["value"]["unit"] for c in row["coefficients"]} != expected
                    or len(row["coefficients"]) != len(expected)):
                raise ValueError(f"Incomplete fitted reaction correlation: {reaction['reaction_id']}")
            reaction["coefficients"] = {c["name"]: float(c["value"]["magnitude"]) for c in row["coefficients"]}
            reaction["standard_state_offset"] = 0.0
            reaction["ln_k_form"] = {
                "ln-k-a-plus-b-over-t": "a + b_k / T",
                "ln-k-a-plus-b-over-t-plus-c-ln-t-plus-d-t": "a + b_k / T + c * ln(T) + d_per_k * T",
                "negative-log10-temperature-polynomial": "-ln(10) * (a_k / T + b + c_per_k * T)",
            }[row["kind"]]
    return {"bundle": bundle, "reactions": reactions}


def reaction_system(dataset_text: str, reaction_scale=None):
    """Selected chemistry, with an explicit one-at-a-time K(T) study multiplier."""
    dataset = Path(dataset_text)
    native = (dataset / _ENGINE_REACTIONS).is_file()
    reactions = (_engine_reaction_record(dataset / "parameters.json", dataset / _ENGINE_REACTIONS)
                 if native else validate_reactive_bundle(dataset_text)["reactions"])
    if reaction_scale is None:
        return reactions
    identity, factor = reaction_scale
    if identity not in ('R4', 'R5') or not math.isfinite(factor) or factor <= 0:
        raise ValueError('Reaction sensitivity requires R4/R5 and a positive finite K multiplier')
    reactions = deepcopy(reactions)
    id_key = 'name' if native else 'reaction_id'
    row = next(r for r in reactions['reactions'] if r[id_key] == identity)
    if native:
        row['correlation']['a'] += math.log(factor)
    elif identity == 'R4' and row['ln_k_form'] == 'a + b_k / T':
        row['coefficients']['a'] += math.log(factor)
    elif identity == 'R5' and row['ln_k_form'] == '-ln(10) * (a_k / T + b + c_per_k * T)':
        row['coefficients']['b'] -= math.log(factor)/math.log(10.)
    else:
        raise ValueError('Selected reaction correlation changed; sensitivity mapping requires review')
    reactions['sensitivity'] = dict(reaction_id=identity, equilibrium_constant_multiplier=factor,
                                    meaning='Local K(T) perturbation; not a fit or uncertainty interval')
    return reactions


def _molar_masses(dataset: Path, species_ids: list[str]) -> list[float]:
    components = {
        component["component_id"]: component
        for component in _json(dataset / "parameters.json")["components"]
    }
    masses = [
        math.fsum(_ATOMIC_MASS_KG_PER_MOL[element] * count for element, count in _FORMULAS[species].items())
        for species in species_ids
    ]
    for species, mass in zip(species_ids, masses, strict=True):
        declared = float(components[species]["fixed"]["molar_mass"]["value"]["magnitude"])
        if abs(declared - mass) > 1.0e-5:
            raise RuntimeError(
                f"Reactive ePC-SAFT molar mass for {species} differs from its formula"
            )
    return masses


DATASET = Path(__file__).resolve().parents[1] / "data/epcsaft_datasets/MEA_reactive_epcsaft_bundle"
MODEL = "epcsaft_reactive_nine"


@lru_cache(maxsize=1)
def reactive_liquid():
    """Cache immutable model inputs only; every solve starts a fresh loading path."""
    return ReactiveLiquid(
        DATASET, loading_anchor=.25, max_log_loading_step=.1, max_loading_steps=32,
    )


def parameter_document(dataset_text: str, kij_scale=None):
    """Selected inputs or one declared multiplicative binary-interaction perturbation."""
    dataset = Path(dataset_text)
    native = (dataset / _ENGINE_REACTIONS).is_file()
    if not native:
        validate_reactive_bundle(dataset_text)
    document = _json(dataset / 'parameters.json')
    if kij_scale is None:
        return document
    identity, factor = kij_scale
    factor = float(factor)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('Binary-interaction scale must be finite and positive')
    matches = []
    for pair in document['pairs']:
        for coefficient in pair['coefficients']:
            if coefficient['family'] != 'k_ij':
                continue
            values = [coefficient]
            if native:
                law = coefficient.get('temperature_law')
                if law is not None:
                    values = [term['coefficient'] for term in law['terms']]
                    if 'reciprocal_temperature_slope' in law:
                        values.append(law['reciprocal_temperature_slope'])
                values.extend(c for c in pair['coefficients'] if c['family'] == 'k_ij_reciprocal_temperature_slope')
            if identity == coefficient['identity'] or any(c['identity'] == identity for c in values):
                matches.append((coefficient, values))
    if len(matches) != 1:
        raise ValueError('Expected exactly one existing k_ij law: '+str(identity))
    coefficient, values = matches[0]
    source_id = 'absorber-one-at-a-time-kij-screen'
    document['sources'].append(dict(source_id=source_id,
        citation='Declared multiplicative sensitivity perturbation of selected MEA parameters.',
        use_basis='Local sensitivity screen; not a fit or an uncertainty interval.'))
    for row in values:
        row['value']['magnitude'] *= factor
    meaning = 'all selected k_ij law coefficients' if native else 'selected coefficient'
    for row in (coefficient, *(c for c in values if c is not coefficient and 'provenance' in c)):
        row['provenance'] = dict(row['provenance'], source_id=source_id,
            locator=f'{identity}: {meaning} multiplied by {factor:.17g}')
    return document


def _mixture(dataset, document, components=None):
    import epcsaft

    parameters = epcsaft.Parameters.from_mapping(document, components=components)
    thermochemistry = (ideal_gas_record(dataset / _IDEAL_GAS, parameters.component_ids)
                       if (dataset / _ENGINE_REACTIONS).is_file() else None)
    return epcsaft.Mixture(parameters, thermochemistry=thermochemistry)


@lru_cache(maxsize=2)
def mixture(dataset_text: str, components=None, kij_scale=None):
    return _mixture(Path(dataset_text), parameter_document(dataset_text, kij_scale), components)


def reactive_fugacity(y, x_true, concentrations, Tl, Tv, P, P_sat_H2O, *, dataset_text=None, kij_scale=None):
    """Use the solved liquid density and the selected EOS; retain legacy water closure."""
    from .epcsaft_v02 import fugacity_coefficients, state, state_at_density
    from .thermo_models import neutral_vapor_composition

    composition = np.asarray(x_true, dtype=float)
    concentrations = np.asarray(concentrations, dtype=float)
    if (composition.shape != (9,) or concentrations.shape != (9,)
            or np.any(~np.isfinite(composition)) or np.any(composition <= 0)
            or np.any(~np.isfinite(concentrations)) or np.any(concentrations <= 0)):
        raise ValueError("Reactive fugacity requires nine positive finite species")
    if (not np.isclose(composition.sum(), 1.0, rtol=0, atol=1e-10)
            or not np.allclose(concentrations / concentrations.sum(), composition,
                               rtol=1e-10, atol=1e-14)):
        raise ValueError("Reactive fugacity requires normalized composition matching concentrations")
    dataset_text = str(dataset_text or DATASET)
    liquid = state_at_density(mixture(dataset_text, kij_scale=kij_scale), temperature_k=Tl,
                              density_mol_m3=float(concentrations.sum()), composition=composition)
    # Same neutral vapor approximation as the August fugacity formulation.
    vapor_x = neutral_vapor_composition(y)
    vapor = state(mixture(dataset_text, ("carbon-dioxide", "monoethanolamine", "water"), kij_scale),
                  temperature_k=Tv, pressure_pa=P, composition=vapor_x, phase="vapor")
    return (float(composition[0]) * fugacity_coefficients(liquid)[0] * P,
            float(y[0]) * fugacity_coefficients(vapor)[0] * P,
            float(composition[2]) * P_sat_H2O, float(y[1]) * P)
