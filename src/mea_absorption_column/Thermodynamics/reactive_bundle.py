from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np


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


def load_reference_thermochemistry(path: str | Path, *, liquid_reference=None):
    """Read retained liquid/vapor references; native types validate units/domain.

    Selection remains explicit: possessing a reference does not establish its
    physical accuracy or compatibility with a different parameter set.
    """
    import epcsaft

    data = _json(Path(path))
    fingerprint = data.pop("scientific_fingerprint")
    actual = "sha256:" + hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if fingerprint != actual or data["schema"] not in {
        "mea-anchored-reaction-consistent-reference-thermochemistry-v2",
        "absorber-neutral-reference-thermochemistry-v1",
    }:
        raise ValueError("Unsupported or modified MEA reference thermochemistry input")
    liquid = data["schema"] == "mea-anchored-reaction-consistent-reference-thermochemistry-v2"
    reference = epcsaft.ReferenceThermochemistry(
        "mea-anchored-r1-r5-reaction-consistent-gauge-v2" if liquid else data["identity"], fingerprint,
        tuple(data["component_ids"]), tuple(
            epcsaft.ComponentReferenceThermochemistry(
                row["component_id"], "mea-anchored-reaction-consistent-reference-v2" if liquid else row["reference_state_id"],
                data["reference_temperature_k"], row["reference_enthalpy_j_per_mol"],
                epcsaft.IdealHeatCapacityPolynomial(
                    row["component_id"] + "-anchored-cp", tuple(row["cp_coefficients_j_per_mol_k"]),
                    tuple(data["temperature_domain_k"]),
                ),
            ) for row in data["components"]
        ),
    )
    if not liquid:
        if liquid_reference is None:
            raise ValueError("Vapor reference requires its paired liquid reference")
        if data["inherited_liquid_reference_fingerprint"] != liquid_reference.scientific_fingerprint:
            raise ValueError("Vapor reference does not match the paired liquid fingerprint")
        shared = {row.component_id: row for row in liquid_reference.components}
        if any(row.component_id in shared and row != shared[row.component_id]
               for row in reference.components):
            raise ValueError("Shared vapor/liquid component reference thermochemistry differs")
    return reference


class ReactiveLiquid:
    """One immutable input set; optionally reuse exact certified states.

    Optional loading_anchor is CO2/MEA, not the bulk-relative film coordinate.
    Each query follows a fresh fixed-step path at its own T/P and MEA/water
    ratio. max_loading_steps bounds work, not the physical loading domain.
    """

    def __init__(self, dataset: str | Path, *, thermochemistry=None,
                 loading_anchor=None, max_log_loading_step=.1, max_loading_steps=32,
                 water_per_mea_anchor=None, reuse_states=False, kij_scale=None, reaction_scale=None):
        import epcsaft

        self.dataset = str(dataset)
        validated = validate_reactive_bundle(self.dataset)
        self.reaction_scale = None if reaction_scale is None else tuple(reaction_scale)
        self._reactions = reaction_system(self.dataset, self.reaction_scale)
        self.molar_masses = tuple(_molar_masses(Path(dataset), self._reactions["species_ids"]))
        self.kij_scale = None if kij_scale is None else tuple(kij_scale)
        self.model = epcsaft.Mixture(parameter_set(self.dataset, self.kij_scale))
        if thermochemistry is not None:
            thermochemistry.validate_component_order(tuple(self.model.component_ids))
            expected = validated["bundle"].get("reference_scientific_fingerprint")
            if expected is not None and thermochemistry.scientific_fingerprint != expected:
                raise ValueError("Reference thermochemistry does not match the adopted parameter bundle")
        self.thermochemistry = thermochemistry
        self.loading_anchor = loading_anchor
        self.max_log_loading_step = max_log_loading_step
        self.max_loading_steps = max_loading_steps
        self.water_per_mea_anchor = water_per_mea_anchor
        self.reuse_states = reuse_states
        self._states = OrderedDict()
        self._accepted = None
        self.stats = dict(queries=0, cache_hits=0, native_solves=0, native_seconds=0., warm_starts=0)

    def solve(self, temperature_k, pressure_pa, apparent_amounts, *, state_input_derivatives=False):
        inputs = tuple(float(v) for v in (temperature_k, pressure_pa, *apparent_amounts))
        if len(inputs) != 5 or not all(math.isfinite(v) and v > 0 for v in inputs):
            raise ValueError("Reactive inputs require finite positive T, P and three apparent amounts")
        # The instance owns one immutable native model, reaction set and reference.
        # Include every mutable numerical option; never quantize thermodynamic inputs.
        identity = (self.model.parameter_fingerprint,
                    json.dumps(self._reactions, sort_keys=True), self.molar_masses,
                    None if self.thermochemistry is None else self.thermochemistry.scientific_fingerprint)
        key = (identity, inputs, state_input_derivatives, self.loading_anchor,
               self.water_per_mea_anchor, self.max_log_loading_step, self.max_loading_steps)
        self.stats['queries'] += 1
        if self.reuse_states and key in self._states:
            self.stats['cache_hits'] += 1
            self._states.move_to_end(key)
            return deepcopy(self._states[key])
        start = None
        if self.reuse_states and self._accepted is not None:
            old = self._accepted
            if old['parameter_fingerprint'] != self.model.parameter_fingerprint:
                raise ValueError('Parameter identity changed within a reactive initialization sequence')
            new_feed = np.r_[np.asarray(inputs[2:]) / sum(inputs[2:]), np.zeros(6)]
            old_feed = old['feed_amounts_mol']
            distance = max(abs(math.log(new_feed[i]/new_feed[1])
                               - math.log(old_feed[i]/old_feed[1])) for i in (0, 2))
            # A collocation batch can jump from the lean end to a rich midpoint.
            # Use the declared loading path for jumps exceeding its own step bound.
            if self.loading_anchor is None or distance <= self.max_log_loading_step:
                start = _conservative_start(old['amounts_mol'], old_feed, new_feed,
                                            self._reactions, self.molar_masses)
                start = dict(amounts_mol=start.tolist(), molar_volume_m3_per_mol=1./old['density_mol_m3'])
                self.stats['warm_starts'] += 1
        result = solve_homogeneous_reactive_state(
            self.dataset, temperature_k, pressure_pa, apparent_amounts,
            model=self.model, reactions=self._reactions, molar_masses=self.molar_masses,
            state_input_derivatives=state_input_derivatives,
            thermochemistry=self.thermochemistry,
            loading_anchor=self.loading_anchor if start is None else None,
            water_per_mea_anchor=self.water_per_mea_anchor,
            max_log_loading_step=self.max_log_loading_step,
            max_loading_steps=self.max_loading_steps,
            _phase_start=start, _diagnostics=self.stats,
        )
        if self.reuse_states:
            # Exceptions and rejected roots never update the accepted seed or cache.
            self._accepted = deepcopy(result)
            self._states[key] = deepcopy(result)
            # ponytail: bounded per-column cache; eviction recomputes, never approximates.
            if len(self._states) > 2048:
                self._states.popitem(last=False)
        return result


def _conservative_start(old_amounts, old_feed, new_feed, reactions, molar_masses):
    scale = float(np.min(new_feed[:3] / old_feed[:3]))
    start = scale * old_amounts + new_feed - scale * old_feed
    invariants = np.vstack((reactions['balance_matrix'], molar_masses, reactions['charges']))
    if (not np.isfinite(scale) or scale <= 0 or np.any(~np.isfinite(start))
            or np.any(start <= 1e-12) or np.max(np.abs(invariants @ (start-new_feed))) > 1e-10):
        raise RuntimeError('Inadmissible conservative reactive initialization')
    return start


def state_input_jacobian(block, feed_jacobian, output_ids):
    """Map native invariants to [T, P, apparent inputs], reusing b6f5's A1 map."""
    invariant_jacobian = np.asarray(block.invariant_matrix) @ feed_jacobian
    rows = dict(zip(block.invariant_ids, invariant_jacobian, strict=True))
    units = dict(zip(block.invariant_ids, block.invariant_units, strict=True))
    units.update(temperature_k='kelvin', pressure_pa='pascal')
    mapping = np.zeros((len(block.input_identities), 2 + feed_jacobian.shape[1]))
    for i, (identity, unit) in enumerate(zip(block.input_identities, block.input_units, strict=True)):
        if unit != units[identity]:
            raise ValueError('Native derivative input units changed')
        if identity in ('temperature_k', 'pressure_pa'):
            mapping[i, 0 if identity == 'temperature_k' else 1] = 1
        else:
            mapping[i, 2:] = rows[identity]
    active = np.flatnonzero(np.any(mapping != 0, axis=1))
    if any(block.input_failures[i] is not None for i in active):
        raise RuntimeError(f'Native input sensitivities unavailable: {block.input_failures}')
    indices = [block.output_identities.index(name) for name in output_ids]
    derivative = np.asarray([[block.jacobian[i][j] for j in active] for i in indices], dtype=float)
    result = derivative @ mapping[active]
    if np.any(~np.isfinite(result)):
        raise RuntimeError('Nonfinite native state input derivatives')
    return result


@lru_cache(maxsize=2048)
def neutral_vapor_state(temperature_k, pressure_pa, composition, dataset_text=None, kij_scale=None):
    """Exact-state neutral vapor values and native T/P/composition derivatives.

    Component balances fix composition: no reactions or alternative vapor model.
    The immutable model identity is owned by the selected dataset mixture.
    """
    import epcsaft
    from epcsaft import equilibrium
    dataset = Path(dataset_text) if dataset_text is not None else DATASET
    model = mixture(str(dataset), ('carbon-dioxide', 'monoethanolamine', 'water'), kij_scale)
    root = model.state(T=temperature_k*epcsaft.unit_registry.kelvin,
                       P=pressure_pa*epcsaft.unit_registry.pascal,
                       x=composition, phase='vapor')
    if root.density_diagnostics is None or not root.density_diagnostics.stable:
        raise RuntimeError('Neutral vapor lacks a stable pressure root')
    phase_id = 'neutral-vapor'
    problem = dict(
        identity='absorber-fixed-composition-vapor',
        temperature=dict(role='fixed', unit='kelvin', value=temperature_k),
        pressure=dict(role='fixed', unit='pascal', value=pressure_pa),
        phases=[dict(identity=phase_id, fluid_role='vapor', amount_role='finite',
                     support=dict(kind='all_components', component_ids=[]),
                     model=dict(kind='eos', reference_id='installed-provider-eos',
                                admissible_packing_fraction_interval=[1e-6, .74]),
                     start=dict(amounts_mol=composition,
                                molar_volume_m3_per_mol=1./root.molar_density.to('mol/m^3').magnitude))],
        reaction_system=dict(species_ids=model.component_ids, charges=[0,0,0],
                             molar_masses_kg_per_mol=_molar_masses(dataset, model.component_ids),
                             balance_matrix=np.eye(3).tolist(), conserved_totals=composition,
                             reaction_matrix=[], feed_amounts_mol=composition,
                             equilibrium_constants=[], source_standard_state=None,
                             strict_interior_amount_floor_mol=1e-16),
        reaction_phase_ids=[], state_input_derivatives=True,
        outputs=[dict(identity='fugacity:carbon-dioxide', selector='phase.fugacity',
                      unit='pascal', basis='true-species-EOS', phase_identity=phase_id,
                      coefficients=[1.,0.,0.], solvent_mass_coefficients_kg_per_mol=[],
                      support='positive', censor_limit=None, aggregate_identity=None, covariance_identity=None)],
        continuation=None, feed=None, phase_reactions=None, intensive_boundaries=[])
    result = equilibrium.solve(model, equilibrium.general_reactive_equilibrium_problem_from_mapping(problem))
    if (result.status != 'evaluated' or result.physical_status != 'passed'
            or result.numerical_status != 'passed' or result.state_input_derivatives is None):
        raise RuntimeError(f'Native neutral-vapor derivative calculation failed: {result.failure}')
    density = result.phases[0].molar_density_mol_m3
    if not np.isclose(density, root.molar_density.to('mol/m^3').magnitude, rtol=1e-8, atol=0):
        raise RuntimeError('Neutral vapor derivative state changed pressure root')
    return float(result.values[0]), result.state_input_derivatives


def validate_reactive_bundle(dataset_text: str) -> dict:
    import epcsaft
    from epcsaft.records import ReactionCorrelationRecord

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
    parameters = epcsaft.Parameters.from_json(dataset / "parameters.json")
    for reaction in reactions["reactions"]:
        fitted = [record for record in parameters.records
                  if isinstance(record, ReactionCorrelationRecord)
                  and record.reaction_id == reaction["reaction_id"]]
        if fitted:
            kind = fitted[0].correlation_kind
            expected = ReactionCorrelationRecord.coefficient_units[kind]
            if (any(record.correlation_kind != kind for record in fitted)
                    or {record.coefficient_name for record in fitted} != set(expected)):
                raise ValueError(f"Incomplete typed reaction correlation: {reaction['reaction_id']}")
            reaction["coefficients"] = {
                record.coefficient_name: float(record.value.to(record.unit).magnitude)
                for record in fitted
            }
            reaction["standard_state_offset"] = 0.0
            reaction["ln_k_form"] = {
                "ln-k-a-plus-b-over-t": "a + b_k / T",
                "ln-k-a-plus-b-over-t-plus-c-ln-t-plus-d-t": "a + b_k / T + c * ln(T) + d_per_k * T",
                "negative-log10-temperature-polynomial": "-ln(10) * (a_k / T + b + c_per_k * T)",
            }[kind]
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


def compile_reaction_constants(dataset_text: str, temperature_k: float) -> tuple[list, ...]:
    reactions = validate_reactive_bundle(dataset_text)["reactions"]
    return _compile_reaction_constants(reactions, temperature_k)


def _compile_reaction_constants(reactions: dict, temperature_k: float) -> tuple[list, ...]:
    temperature = float(temperature_k)
    compiled = []
    for reaction in reactions["reactions"]:
        lower, upper = reaction["temperature_domain_k"]
        if not lower <= temperature <= upper:
            raise ValueError(
                f"{reaction['reaction_id']} temperature {temperature} K is outside "
                f"[{lower}, {upper}] K"
            )
        coefficients = reaction["coefficients"]
        form = reaction["ln_k_form"]
        metadata = None
        if form.startswith("a + b_k / T"):
            value = (
                coefficients["a"]
                + coefficients["b_k"] / temperature
                + coefficients.get("c", 0.0) * math.log(temperature)
                + coefficients.get("d_per_k", 0.0) * temperature
                + reaction.get("standard_state_offset", 0.0)
            )
            names = ("a", "b_k") if set(coefficients) == {"a", "b_k"} else ("a", "b_k", "c", "d_per_k")
            values = [float(coefficients.get(name, 0.0)) for name in names]
            values[0] += reaction.get("standard_state_offset", 0.0)
            metadata = {
                "reaction_id": reaction["reaction_id"],
                "kind": (
                    "ln-k-a-plus-b-over-t" if len(names) == 2
                    else "ln-k-a-plus-b-over-t-plus-c-ln-t-plus-d-t"
                ),
                "coefficient_identities": [
                    f"reaction:{reaction['reaction_id']}:correlation:{name}" for name in names
                ],
                "coefficient_values": values,
            }
        elif form == "-ln(10) * (a_k / T + b + c_per_k * T)":
            value = -math.log(10.0) * (
                coefficients["a_k"] / temperature
                + coefficients["b"]
                + coefficients["c_per_k"] * temperature
            )
            metadata = {
                "reaction_id": reaction["reaction_id"],
                "kind": "negative-log10-temperature-polynomial",
                "coefficient_identities": [
                    f"reaction:{reaction['reaction_id']}:correlation:{name}"
                    for name in ("a_k", "b", "c_per_k")
                ],
                "coefficient_values": [coefficients[name] for name in ("a_k", "b", "c_per_k")],
            }
        else:
            raise ValueError(f"Unsupported reaction correlation: {form}")
        entry = [
            float(value),
            "mea-reactive-epcsaft-parameter-bundle",
            reactions["source_standard_state"]["id"],
            "products_positive",
            "source-standard-state-to-eos-neutral-reference",
            True,
        ]
        if metadata is not None:
            entry.append(metadata)
        compiled.append(entry)
    return tuple(compiled)


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


def homogeneous_reactive_request(
    dataset_text: str,
    temperature_k: float,
    pressure_pa: float,
    apparent_amounts,
    *,
    reactions: dict | None = None,
    molar_masses=None,
) -> dict:
    dataset = Path(dataset_text)
    if reactions is None:
        reactions = validate_reactive_bundle(dataset_text)["reactions"]
    apparent = np.array(apparent_amounts, dtype=float, copy=True)
    if apparent.shape != (3,) or np.any(~np.isfinite(apparent)) or np.any(apparent <= 0.0):
        raise ValueError("Reactive ePC-SAFT requires positive finite CO2/MEA/H2O amounts")
    apparent /= float(apparent.sum())
    feed = [*apparent.tolist(), *([0.0] * 6)]
    balances = reactions["balance_matrix"]
    totals = [math.fsum(a * b for a, b in zip(row, feed, strict=True)) for row in balances]
    species_ids = reactions["species_ids"]
    return {
        "identity": "mea-absorber-homogeneous-nine-species",
        "temperature": {"role": "fixed", "unit": "kelvin", "value": float(temperature_k)},
        "pressure": {"role": "fixed", "unit": "pascal", "value": float(pressure_pa)},
        "phases": [
            {
                "identity": "mea-nine-species-liquid",
                "fluid_role": "liquid",
                "amount_role": "finite",
                "support": {"kind": "all_components", "component_ids": []},
                "model": {
                    "kind": "eos",
                    "reference_id": "installed-provider-eos",
                    "admissible_packing_fraction_interval": [1.0e-6, 0.74],
                },
                "start": None,
            }
        ],
        "reaction_system": {
            "species_ids": species_ids,
            "charges": reactions["charges"],
            "molar_masses_kg_per_mol": (
                _molar_masses(dataset, species_ids) if molar_masses is None else list(molar_masses)
            ),
            "balance_matrix": balances,
            "conserved_totals": totals,
            "reaction_matrix": [reaction["stoichiometry"] for reaction in reactions["reactions"]],
            "feed_amounts_mol": feed,
            "equilibrium_constants": _compile_reaction_constants(reactions, temperature_k),
            "strict_interior_amount_floor_mol": 1.0e-12,
            "source_standard_state": reactions["source_standard_state"],
        },
        "reaction_phase_ids": ["mea-nine-species-liquid"],
        "outputs": [
            {
                "identity": "system-pressure",
                "selector": "system.pressure",
                "unit": "pascal",
                "basis": "total-state-pressure",
                "phase_identity": None,
                "coefficients": [],
                "solvent_mass_coefficients_kg_per_mol": [],
                "support": "positive",
                "censor_limit": None,
                "aggregate_identity": None,
                "covariance_identity": None,
            },
            *[
                {
                    "identity": f"amount:{species}",
                    "selector": "phase.amount",
                    "unit": "mole",
                    "basis": "true-species-amount-on-normalized-feed-basis",
                    "phase_identity": "mea-nine-species-liquid",
                    "coefficients": [float(i == j) for j in range(len(species_ids))],
                    "solvent_mass_coefficients_kg_per_mol": [],
                    "support": "positive",
                    "censor_limit": None,
                    "aggregate_identity": None,
                    "covariance_identity": None,
                }
                for i, species in enumerate(species_ids)
            ],
        ],
        "continuation": None,
        "feed": None,
        "phase_reactions": None,
        "intensive_boundaries": [],
    }


def _solve_homogeneous_reactive_result(
    dataset_text: str,
    temperature_k: float,
    pressure_pa: float,
    apparent_amounts,
    *,
    model=None,
    seed_fraction: float = 1.0e-3,
    reactions: dict | None = None,
    molar_masses=None,
    state_input_derivatives: bool = False,
    thermochemistry=None,
    loading_anchor=None,
    max_log_loading_step=.1,
    max_loading_steps=32,
    _phase_start=None,
    water_per_mea_anchor=None,
    _diagnostics=None,
):
    import epcsaft
    from epcsaft import equilibrium

    request = homogeneous_reactive_request(
        dataset_text, temperature_k, pressure_pa, apparent_amounts,
        reactions=reactions, molar_masses=molar_masses,
    )
    liquid_path = loading_anchor is not None or _phase_start is not None
    # A1 mode admits only the declared start in GREPE, including intermediate
    # steps. It must not silently try generated starts after a path failure.
    state_input_derivatives = state_input_derivatives or liquid_path
    if state_input_derivatives:
        if not hasattr(equilibrium, "EquilibriumStateInputDerivatives"):
            raise RuntimeError(
                "Exact reactive film requires an immutable Engine A1 wheel exposing "
                "EquilibriumStateInputDerivatives; numerical derivative substitution is disabled"
            )
        request["state_input_derivatives"] = True
        for index, species in enumerate(request["reaction_system"]["species_ids"]):
            for selector, unit, identity in (
                ("phase.chemical_potential_over_rt", "dimensionless", "mu"),
                ("phase.fugacity", "pascal", "fugacity"),
            ):
                request["outputs"].append(dict(
                    request["outputs"][index + 1],
                    identity=f"{identity}:{species}", selector=selector, unit=unit,
                    basis="true-species-EOS", support="real" if identity == "mu" else "positive",
                ))
        request["outputs"].append(dict(
            request["outputs"][1], identity="liquid-molar-density",
            selector="phase.molar_density", unit="mole / meter**3",
            basis="true-species-EOS", coefficients=[],
        ))
    if model is None:
        model = epcsaft.Mixture(epcsaft.Parameters.from_json(Path(dataset_text) / "parameters.json"))
    if loading_anchor is not None:
        if (not np.isfinite(loading_anchor) or loading_anchor <= 0.
                or not np.isfinite(max_log_loading_step) or max_log_loading_step <= 0.
                or isinstance(max_loading_steps, bool) or not isinstance(max_loading_steps, int)
                or max_loading_steps < 1):
            raise ValueError("Loading anchor, log step and integer step budget must be finite positive")
        apparent = np.asarray(apparent_amounts, dtype=float)
        distance = math.log(apparent[0]) - math.log(apparent[1]) - math.log(loading_anchor)
        water_anchor = apparent[2]/apparent[1] if water_per_mea_anchor is None else float(water_per_mea_anchor)
        if not math.isfinite(water_anchor) or water_anchor <= 0:
            raise ValueError('Water/MEA anchor must be finite positive')
        water_distance = math.log(apparent[2]/apparent[1]/water_anchor)
        steps = math.ceil(max(abs(distance), abs(water_distance)) / max_log_loading_step)
        if steps > max_loading_steps:
            raise RuntimeError(f"Liquid loading path needs {steps} steps; declared budget is {max_loading_steps}")
        if steps:
            anchor = apparent.copy()
            anchor[0] = loading_anchor * apparent[1]
            anchor[2] = water_anchor * apparent[1]
            common = dict(model=model, seed_fraction=seed_fraction, reactions=reactions,
                          molar_masses=molar_masses, state_input_derivatives=True,
                          thermochemistry=thermochemistry, _diagnostics=_diagnostics)
            previous, previous_request = _solve_homogeneous_reactive_result(
                dataset_text, temperature_k, pressure_pa, anchor, **common)
            _require_liquid_pressure_root(model, previous, temperature_k, pressure_pa)
            for index in range(1, steps + 1):
                amounts = apparent.copy()
                if index < steps:
                    amounts[0] = anchor[0] * math.exp(distance * index / steps)
                    amounts[2] = anchor[2] * math.exp(water_distance * index / steps)
                next_request = (request if index == steps else homogeneous_reactive_request(
                    dataset_text, temperature_k, pressure_pa, amounts,
                    reactions=reactions, molar_masses=molar_masses))
                old_feed = np.asarray(previous_request["reaction_system"]["feed_amounts_mol"])
                new_feed = np.asarray(next_request["reaction_system"]["feed_amounts_mol"])
                old_amounts = np.asarray([row.value for row in previous.rows
                                          if row.identity.startswith("amount:")])
                scale = float(np.min(new_feed[:3] / old_feed[:3]))
                start = scale * old_amounts + new_feed - scale * old_feed
                reaction = next_request["reaction_system"]
                invariants = np.vstack((reaction["balance_matrix"],
                                        reaction["molar_masses_kg_per_mol"], reaction["charges"]))
                balance_error = np.max(np.abs(invariants @ (start - new_feed)))
                if (not np.isfinite(scale) or scale <= 0. or np.any(~np.isfinite(start))
                        or np.any(start <= reaction["strict_interior_amount_floor_mol"])
                        or balance_error > dict(previous.resolved_policy)["start.balance_inf_norm_max"]):
                    raise RuntimeError(f"Liquid loading path step {index}/{steps} has an inadmissible conservative start")
                _phase_start = dict(amounts_mol=start.tolist(),
                                    molar_volume_m3_per_mol=1. / previous.phases[0].molar_density_mol_m3)
                if index < steps:
                    previous, previous_request = _solve_homogeneous_reactive_result(
                        dataset_text, temperature_k, pressure_pa, amounts,
                        _phase_start=_phase_start, **common)
    # This deterministic start moves only along declared reaction extents. It
    # preserves every stoichiometric invariant and never changes the feed.
    if not np.isfinite(seed_fraction) or not 0.0 < seed_fraction < 1.0 / 9.0:
        raise ValueError("seed_fraction must be finite and between zero and 1/9")
    from .epcsaft_v02 import molar_density_value

    if _phase_start is None:
        reaction = request["reaction_system"]
        feed = np.asarray(reaction["feed_amounts_mol"], dtype=float)
        stoichiometry = np.asarray(reaction["reaction_matrix"], dtype=float)
        extent_direction = np.asarray([1.0, 4.0, 1.0, -1.0, -1.0])
        start = feed + stoichiometry.T @ extent_direction * (seed_fraction * min(feed[:3]))
        if np.any(start <= reaction["strict_interior_amount_floor_mol"]):
            raise ValueError("Reaction-extent start is outside the declared strict interior")
        initial_state = model.state(
            T=float(temperature_k) * epcsaft.unit_registry.kelvin,
            P=float(pressure_pa) * epcsaft.unit_registry.pascal,
            x=tuple(start / start.sum()),
            phase="liquid",
        )
        _phase_start = dict(amounts_mol=start.tolist(),
                            molar_volume_m3_per_mol=1.0 / molar_density_value(initial_state))
    request["phases"][0]["start"] = _phase_start
    problem = equilibrium.general_reactive_equilibrium_problem_from_mapping(request)
    if thermochemistry is not None:
        thermochemistry.validate_component_order(tuple(model.component_ids))
        problem = replace(problem, thermochemistry=thermochemistry)
    started = time.perf_counter()
    try:
        result = equilibrium.solve(model, problem)
    finally:
        if _diagnostics is not None:
            _diagnostics['native_solves'] += 1
            _diagnostics['native_seconds'] += time.perf_counter() - started
    if (
        result.status != "evaluated"
        or result.numerical_status != "passed"
        or result.physical_status != "passed"
    ):
        raise RuntimeError(
            f"Reactive ePC-SAFT equilibrium failed: {result.solver_status}; {result.failure}; "
            f"numerical={result.numerical_status}, physical={result.physical_status}; "
            f"evidence={dict(result.evidence)}"
        )
    if liquid_path:
        _require_liquid_pressure_root(model, result, temperature_k, pressure_pa)
    if loading_anchor is not None:
        result = replace(result, evidence=result.evidence + (
            ("absorber.loading_anchor", loading_anchor),
            ("absorber.max_log_loading_step", max_log_loading_step),
            ("absorber.loading_steps", steps),
            ("absorber.max_loading_steps", max_loading_steps),
        ))
    return result, request


def _require_liquid_pressure_root(model, result, temperature_k, pressure_pa):
    """Match the native liquid pressure root, without a density floor/global claim."""
    import epcsaft
    from .epcsaft_v02 import molar_density_value

    phase = result.phases[0]
    root = model.state(T=float(temperature_k) * epcsaft.unit_registry.kelvin,
                       P=float(pressure_pa) * epcsaft.unit_registry.pascal,
                       x=phase.mole_fractions, phase="liquid")
    diagnostics = root.density_diagnostics
    density = molar_density_value(root)
    # Same root-identity criterion as Engine density.hpp's root deduplication.
    if (diagnostics is None or not diagnostics.stable or diagnostics.branch != "liquid"
            or not np.isfinite(density) or not np.isfinite(phase.molar_density_mol_m3)
            or diagnostics.requested_phase != "liquid"
            or diagnostics.certificate_kind != "global_stable_root"
            or root.certified_branch is None
            or root.certified_branch.get("schema") != "epcsaft-certified-density-branch-v1"
            or abs(phase.molar_density_mol_m3 - density) > 1e-8 * max(1., density)):
        raise RuntimeError("Reactive equilibrium does not match the certified native liquid pressure root: "
                           f"GREPE density={phase.molar_density_mol_m3}, liquid root={density}")


def solve_homogeneous_reactive_state(
    dataset_text: str, temperature_k: float, pressure_pa: float, apparent_amounts,
    *, model=None, seed_fraction: float = 1.0e-3, reactions: dict | None = None,
    molar_masses=None, state_input_derivatives: bool = False, thermochemistry=None,
    loading_anchor=None, max_log_loading_step=.1, max_loading_steps=32,
    water_per_mea_anchor=None, _phase_start=None, _diagnostics=None,
) -> dict:
    import epcsaft

    result, request = _solve_homogeneous_reactive_result(
        dataset_text, temperature_k, pressure_pa, apparent_amounts,
        model=model, seed_fraction=seed_fraction, reactions=reactions,
        molar_masses=molar_masses, state_input_derivatives=state_input_derivatives,
        thermochemistry=thermochemistry,
        loading_anchor=loading_anchor, max_log_loading_step=max_log_loading_step,
        max_loading_steps=max_loading_steps,
        water_per_mea_anchor=water_per_mea_anchor, _phase_start=_phase_start,
        _diagnostics=_diagnostics,
    )
    reaction = request["reaction_system"]
    phase = result.phases[0]
    rows = {row.identity: row for row in result.rows}
    amount_rows = [rows[f"amount:{species}"] for species in reaction["species_ids"]]
    if any(row.value is None or row.status != "evaluated" for row in amount_rows):
        raise RuntimeError("Reactive ePC-SAFT did not certify all species amounts")
    amounts = np.asarray([row.value for row in amount_rows], dtype=float)
    resolved = {
        "composition": np.asarray(phase.mole_fractions, dtype=float),
        "amounts_mol": amounts,
        "feed_amounts_mol": np.asarray(
            request["reaction_system"]["feed_amounts_mol"], dtype=float
        ),
        "density_mol_m3": float(phase.molar_density_mol_m3),
        "chemical_potentials_over_rt": np.asarray(
            phase.chemical_potential_over_rt, dtype=float
        ),
        "parameter_fingerprint": result.descriptor.parameter_fingerprint,
        "evidence": dict(result.evidence),
    }
    if thermochemistry is not None:
        enthalpy = result.total_enthalpy
        if not isinstance(enthalpy, epcsaft.EquilibriumEnthalpy):
            raise RuntimeError(f"Engine total equilibrium enthalpy unavailable: {enthalpy}")
        if (enthalpy.reference_fingerprint != thermochemistry.scientific_fingerprint
                or enthalpy.parameter_fingerprint != result.descriptor.parameter_fingerprint):
            raise RuntimeError("Engine caloric reference or parameter identity changed")
        resolved["total_enthalpy_j"] = float(enthalpy.value.to("joule").magnitude)
        resolved["reference_fingerprint"] = enthalpy.reference_fingerprint
        if not np.isfinite(resolved["total_enthalpy_j"]):
            raise RuntimeError("Engine total equilibrium enthalpy is non-finite")
    if state_input_derivatives:
        block = result.state_input_derivatives
        if isinstance(block, epcsaft.NonEvaluableTrial) or block is None:
            raise RuntimeError(f"Engine equilibrium derivatives unavailable: {block}")
        resolved["state_input_derivatives"] = block
        for quantity, key in (("mu", "chemical_potentials_over_rt"), ("fugacity", "fugacities_pa")):
            output_rows = [rows[f"{quantity}:{species}"] for species in reaction["species_ids"]]
            if any(row.value is None or row.status != "evaluated" for row in output_rows):
                raise RuntimeError(f"Reactive ePC-SAFT did not certify all species {quantity} outputs")
            resolved[key] = np.asarray([row.value for row in output_rows], dtype=float)
        density_row = rows["liquid-molar-density"]
        if density_row.value is None or density_row.status != "evaluated":
            raise RuntimeError("Reactive ePC-SAFT did not certify liquid molar density")
        resolved["density_mol_m3"] = float(density_row.value)
    return resolved



DATASET = Path(__file__).resolve().parents[1] / "data/epcsaft_datasets/MEA_reactive_epcsaft_bundle"
MODEL = "epcsaft_reactive_nine"


@lru_cache(maxsize=1)
def reactive_liquid():
    """Cache immutable model inputs only; every solve starts a fresh loading path."""
    return ReactiveLiquid(
        DATASET,
        thermochemistry=load_reference_thermochemistry(DATASET / "anchored-reference-thermochemistry.json"),
        loading_anchor=.25, max_log_loading_step=.1, max_loading_steps=32,
    )


def parameter_set(dataset_text: str, kij_scale=None):
    """Selected inputs or one declared multiplicative binary-interaction perturbation."""
    import epcsaft
    validate_reactive_bundle(dataset_text)
    parameters = epcsaft.Parameters.from_json(Path(dataset_text) / 'parameters.json')
    if kij_scale is None:
        return parameters
    identity, factor = kij_scale
    factor = float(factor)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('Binary-interaction scale must be finite and positive')
    document = parameters.to_mapping()
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
    return epcsaft.Parameters.from_mapping(document)


@lru_cache(maxsize=2)
def mixture(dataset_text: str, components=None, kij_scale=None):
    import epcsaft
    parameters = parameter_set(dataset_text, kij_scale)
    if components is not None:
        parameters = parameters.select(components)
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
