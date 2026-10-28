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


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


_CORRELATION_FIELDS = ("a", "b", "c", "d", "reference_temperature", "temperature_min", "temperature_max",
                      "standard_state_id")


def ideal_gas_record(path: str | Path, component_ids):
    """Physical ideal-gas records in component order; absent species (ions) are reaction-completed by the Engine."""
    import epcsaft

    data = _json(Path(path))
    pressure, intervals = data["standard_pressure_pa"], []
    for name in component_ids:
        row = data["components"].get(name)
        if row is None:
            intervals.append([])
            continue
        if row["form"] == "shomate":
            form = epcsaft.IdealShomate(row["coefficients"], row["formation_enthalpy_j_per_mol"])
        else:  # Cp = sum a_j T^j re-centred at T0, where h = s = 0 (moiety convention)
            t0, a = row["reference_temperature_k"], row["coefficients"]
            form = epcsaft.IdealPolynomial([math.fsum(c * math.comb(j, k) * t0 ** (j - k) for j, c in enumerate(a) if j >= k)
                                            for k in range(len(a))], t0, 0., 0.)
        intervals.append([epcsaft.IdealInterval(*row["interval_k"], True, True,
                                                epcsaft.IdealCorrelation(form, pressure))])
    return epcsaft.ThermochemistryRecord(pressure, intervals)


def engine_liquid(dataset: str | Path, thermochemistry_path: str | Path, name="reactive_liquid", *, loading_policy=None):
    """Nine-species reactive liquid: MEA record, its Engine reactions and neutral reference, apparent CO2/MEA/water feed."""
    from epcsaft import equilibrium as q

    from .casadi_reactive import EnginePhase

    dataset = Path(dataset)
    record, mapping = _json(dataset / "engine-reactions.json"), _json(dataset / "parameters.json")
    if hashlib.sha256((dataset / "parameters.json").read_bytes()).hexdigest() != record["parameter_sha256"]:
        raise ValueError("Engine reaction records do not belong to this parameter record")
    reactions = [q.Reaction(
        row["stoichiometry"], name=row["name"],
        correlation=q.ReactionLogPolynomial(*(row["correlation"][key] for key in _CORRELATION_FIELDS)),
        reference=q.ReactionReference(getattr(q.ReferenceSourceBasis, row["reference"]["source_basis"]),
                                      reference_pressure_pa=row["reference"]["reference_pressure_pa"]),
    ) for row in record["reactions"]]
    ids = [c["component_id"] for c in mapping["components"]]
    # Formula masses (checked against the declared ones) conserve mass exactly through the reactions.
    return EnginePhase(name, mapping, ideal_gas_record(thermochemistry_path, ids), kind="liquid", feed_ids=ids[:3],
                       molar_masses=_molar_masses(dataset, ids), reactions=reactions, neutral_reference=q.NeutralReference(**record["neutral_reference"]),
                       loading_policy=loading_policy)


def engine_vapor(parameters_path: str | Path, thermochemistry_path: str | Path, name="neutral_vapor"):
    """Nonreacting vapor at its species feed; the same ideal-gas records as the liquid."""
    from .casadi_reactive import EnginePhase

    mapping = _json(Path(parameters_path))
    ids = [c["component_id"] for c in mapping["components"]]
    return EnginePhase(name, mapping, ideal_gas_record(thermochemistry_path, ids), kind="vapor")


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


def _first_actions(model, problem, result, observables, feed_count):
    """Exact columns with respect to T [K], P [Pa] and neutral feed amounts [mol]."""
    from epcsaft import equilibrium as q

    directions = []
    for j in range(2+feed_count):
        feed = [0.] * len(model.component_ids)
        if j >= 2:
            feed[j-2] = 1.
        directions.append(q.SolvedStateActionDirection(float(j == 0), float(j == 1), feed, []))
    batches = q.solved_state_first_actions(q.compile_problem(model, problem), result, observables, directions)
    for batch in batches:
        for item in batch.results:
            if item.status.name != 'Available' or item.action is None or not math.isfinite(item.action):
                raise RuntimeError(f'Native equilibrium action unavailable: {item.status}: {item.diagnostic_message}')
    return np.array([[item.action for item in batch.results] for batch in batches]).T


class ReactiveLiquid:
    """Declared nine-species reacting liquid on one mole of apparent feed.

    A loading path keeps the selected liquid branch and its fixed reference
    pressure. Cached values and exact first actions share identical inputs.
    """

    def __init__(self, dataset, *, loading_anchor=None, max_log_loading_step=.1, max_loading_steps=32,
                 water_per_mea_anchor=None, reuse_states=False, warm_starts=True, kij_scale=None, reaction_scale=None):
        import epcsaft

        self.dataset, self.kij_scale = str(dataset), None if kij_scale is None else tuple(kij_scale)
        self._reactions = reaction_system(self.dataset, reaction_scale)
        document = parameter_document(self.dataset, self.kij_scale)
        self.parameter_fingerprint = parameter_fingerprint(document)
        self.model = epcsaft.Mixture(epcsaft.Parameters.from_mapping(document))
        self.molar_masses = tuple(_molar_masses(Path(dataset), self.model.component_ids))
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
            reactions=_native_reactions(self._reactions), neutral_reference=_neutral_reference(self._reactions))

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

    def solve(self, temperature_k, pressure_pa, apparent_amounts, *, state_input_derivatives=False):
        from epcsaft import equilibrium as q

        inputs = tuple(float(v) for v in (temperature_k, pressure_pa, *apparent_amounts))
        if len(inputs) != 5 or not all(math.isfinite(v) and v > 0 for v in inputs):
            raise ValueError('Reactive inputs require finite positive T, P and three apparent amounts')
        for reaction in self._reactions['reactions']:
            lower, upper = reaction['temperature_domain_k']
            if not lower <= temperature_k <= upper:
                raise ValueError(f"{reaction['reaction_id']} temperature {temperature_k} K is outside [{lower}, {upper}] K")
        key = (inputs, state_input_derivatives)
        self.stats['queries'] += 1
        if self.reuse_states and key in self._states:
            self.stats['cache_hits'] += 1
            self._states.move_to_end(key)
            return deepcopy(self._states[key])
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
        if state_input_derivatives:
            kind = q.SolvedStateObservableKind
            observables = [q.SolvedStateObservable(kind.PhaseComponentAmount, 0, i) for i in range(9)]
            observables += [q.SolvedStateObservable(kind.PhaseMolarDensity, 0), q.SolvedStateObservable(kind.PhaseLogActivity, 0, 0)]
            derivative = _first_actions(self.model, problem, result, observables, 3)
            derivative[-1] *= fugacities[0]
            derivative[-1, 0] += fugacities[0]/temperature_k
            resolved['amount_density_fugacity_jacobian'] = derivative
        if self.reuse_states:
            if self.warm_starts:
                self._accepted, self._native_accepted = deepcopy(resolved), result
            self._states[key] = deepcopy(resolved)
            if len(self._states) > 2048:
                self._states.popitem(last=False)
        return resolved


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


def validate_reactive_bundle(dataset_text: str) -> dict:
    dataset = Path(dataset_text)
    bundle = _json(dataset / "bundle.json")
    expected = {
        Path(item["path"]).name: item["sha256"]
        for item in bundle["files"]
        if item["path"] in {"parameters/parameters.json", "chemistry/reaction-system.json",
                            "anchored-reference-thermochemistry.json", "adoption-receipt.json"}
    }
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
    if reactions["species_ids"] != [
        "carbon-dioxide",
        "monoethanolamine",
        "water",
        "protonated-monoethanolamine",
        "carbamate-anion",
        "bicarbonate-anion",
        "carbonate-anion",
        "hydronium-cation",
        "hydroxide-anion",
    ] or reactions["charges"] != [0, 0, 0, 1, -1, -1, -2, 1, -1]:
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
    reactions = validate_reactive_bundle(dataset_text)["reactions"]
    if reaction_scale is None:
        return reactions
    identity, factor = reaction_scale
    if identity not in ('R4', 'R5') or not math.isfinite(factor) or factor <= 0:
        raise ValueError('Reaction sensitivity requires R4/R5 and a positive finite K multiplier')
    reactions = deepcopy(reactions)
    row = next(r for r in reactions['reactions'] if r['reaction_id'] == identity)
    if identity == 'R4' and row['ln_k_form'] == 'a + b_k / T':
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
    validate_reactive_bundle(dataset_text)
    document = _json(Path(dataset_text) / 'parameters.json')
    if kij_scale is None:
        return document
    identity, factor = kij_scale
    factor = float(factor)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('Binary-interaction scale must be finite and positive')
    matches = [c for pair in document['pairs'] for c in pair['coefficients']
               if c['identity'] == identity and c['family'] == 'k_ij']
    if len(matches) != 1:
        raise ValueError('Expected exactly one existing k_ij coefficient: '+str(identity))
    coefficient = matches[0]
    coefficient['value']['magnitude'] *= factor
    source_id = 'absorber-one-at-a-time-kij-screen'
    document['sources'].append(dict(source_id=source_id,
        citation='Declared multiplicative sensitivity perturbation of selected MEA parameters.',
        use_basis='Local sensitivity screen; not a fit or an uncertainty interval.'))
    coefficient['provenance'] = dict(coefficient['provenance'], source_id=source_id,
        locator=f'{identity}: selected coefficient multiplied by {factor:.17g}')
    return document


@lru_cache(maxsize=2)
def mixture(dataset_text: str, components=None, kij_scale=None):
    import epcsaft
    parameters = epcsaft.Parameters.from_mapping(parameter_document(dataset_text, kij_scale), components=components)
    return epcsaft.Mixture(parameters)


def reactive_fugacity(y, x_true, concentrations, Tl, Tv, P, P_sat_H2O):
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
    liquid = state_at_density(mixture(str(DATASET)), temperature_k=Tl,
                              density_mol_m3=float(concentrations.sum()), composition=composition)
    # Same neutral vapor approximation as the August fugacity formulation.
    vapor_x = neutral_vapor_composition(y)
    vapor = state(mixture(str(DATASET), ("carbon-dioxide", "monoethanolamine", "water")),
                  temperature_k=Tv, pressure_pa=P, composition=vapor_x, phase="vapor")
    return (float(composition[0]) * fugacity_coefficients(liquid)[0] * P,
            float(y[0]) * fugacity_coefficients(vapor)[0] * P,
            float(composition[2]) * P_sat_H2O, float(y[1]) * P)
