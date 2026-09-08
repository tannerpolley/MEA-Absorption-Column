from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
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
    """One validated input set and native model, with no stored solved states.

    Optional loading_anchor is CO2/MEA, not the bulk-relative film coordinate.
    Each query follows a fresh fixed-step path at its own T/P and MEA/water
    ratio. max_loading_steps bounds work, not the physical loading domain.
    """

    def __init__(self, dataset: str | Path, *, thermochemistry=None,
                 loading_anchor=None, max_log_loading_step=.1, max_loading_steps=32):
        import epcsaft

        self.dataset = str(dataset)
        validated = validate_reactive_bundle(self.dataset)
        self._reactions = validated["reactions"]
        self.molar_masses = tuple(_molar_masses(Path(dataset), self._reactions["species_ids"]))
        self.model = epcsaft.Mixture(epcsaft.Parameters.from_json(Path(dataset) / "parameters.json"))
        if thermochemistry is not None:
            thermochemistry.validate_component_order(tuple(self.model.component_ids))
            expected = validated["bundle"].get("reference_scientific_fingerprint")
            if expected is not None and thermochemistry.scientific_fingerprint != expected:
                raise ValueError("Reference thermochemistry does not match the adopted parameter bundle")
        self.thermochemistry = thermochemistry
        self.loading_anchor = loading_anchor
        self.max_log_loading_step = max_log_loading_step
        self.max_loading_steps = max_loading_steps

    def solve(self, temperature_k, pressure_pa, apparent_amounts, *, state_input_derivatives=False):
        return solve_homogeneous_reactive_state(
            self.dataset, temperature_k, pressure_pa, apparent_amounts,
            model=self.model, reactions=self._reactions, molar_masses=self.molar_masses,
            state_input_derivatives=state_input_derivatives,
            thermochemistry=self.thermochemistry,
            loading_anchor=self.loading_anchor, max_log_loading_step=self.max_log_loading_step,
            max_loading_steps=self.max_loading_steps,
        )

    def solve_actions(self, temperature_k, pressure_pa, apparent_amounts, actions, output_ids):
        """Native selected-output actions, retaining central/A1 values and failures."""
        result, _ = _solve_homogeneous_reactive_result(
            self.dataset, temperature_k, pressure_pa, apparent_amounts,
            model=self.model, reactions=self._reactions, molar_masses=self.molar_masses,
            state_input_derivatives=True, thermochemistry=self.thermochemistry,
            state_input_actions=actions, output_ids=output_ids,
            loading_anchor=self.loading_anchor, max_log_loading_step=self.max_log_loading_step,
            max_loading_steps=self.max_loading_steps,
        )
        return result


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
    state_input_actions=(),
    output_ids=None,
    loading_anchor=None,
    max_log_loading_step=.1,
    max_loading_steps=32,
    _phase_start=None,
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
        steps = (0 if apparent[0] == loading_anchor * apparent[1]
                 else math.ceil(abs(distance) / max_log_loading_step))
        if steps > max_loading_steps:
            raise RuntimeError(f"Liquid loading path needs {steps} steps; declared budget is {max_loading_steps}")
        if steps:
            anchor = apparent.copy()
            anchor[0] = loading_anchor * apparent[1]
            common = dict(model=model, seed_fraction=seed_fraction, reactions=reactions,
                          molar_masses=molar_masses, state_input_derivatives=True,
                          thermochemistry=thermochemistry)
            previous, previous_request = _solve_homogeneous_reactive_result(
                dataset_text, temperature_k, pressure_pa, anchor, **common)
            _require_liquid_pressure_root(model, previous, temperature_k, pressure_pa)
            for index in range(1, steps + 1):
                amounts = apparent.copy()
                if index < steps:
                    amounts[0] = anchor[0] * math.exp(distance * index / steps)
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
    if output_ids is not None:
        outputs = {output.identity: output for output in problem.outputs}
        problem = replace(problem, outputs=tuple(outputs[identity] for identity in output_ids))
    problem = replace(problem, state_input_actions=tuple(state_input_actions))
    result = equilibrium.solve(model, problem)
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
) -> dict:
    import epcsaft

    result, request = _solve_homogeneous_reactive_result(
        dataset_text, temperature_k, pressure_pa, apparent_amounts,
        model=model, seed_fraction=seed_fraction, reactions=reactions,
        molar_masses=molar_masses, state_input_derivatives=state_input_derivatives,
        thermochemistry=thermochemistry,
        loading_anchor=loading_anchor, max_log_loading_step=max_log_loading_step,
        max_loading_steps=max_loading_steps,
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


def equilibrium_loading_direction(state: dict) -> np.ndarray:
    """Contract Engine equilibrium derivatives with the normalized feed path.

    lambda changes only the apparent CO2 amount by exp(lambda). Since the
    homogeneous solve uses a unit-total feed, dn/dlambda=n_CO2*(e_CO2-n).
    Engine supplies the complete independent invariant matrix, including mass;
    fixed T, fixed P and the fixed-zero charge invariant have zero direction.
    """
    block = state["state_input_derivatives"]
    if tuple(block.component_ids) != tuple(_FORMULAS):
        raise ValueError("Engine equilibrium derivative species order changed")
    feed = state["feed_amounts_mol"]
    feed_direction = -feed[0] * feed
    feed_direction[0] += feed[0]
    invariant_direction = np.asarray(block.invariant_matrix) @ feed_direction
    by_identity = dict(zip(block.invariant_ids, invariant_direction, strict=True))
    by_identity.update(temperature_k=0.0, pressure_pa=0.0)
    direction = np.asarray([by_identity[identity] for identity in block.input_identities])
    active = np.flatnonzero(direction)
    for index in active:
        if block.input_failures[index] is not None:
            raise RuntimeError(f"Engine loading derivative unavailable: {block.input_failures[index]}")
    output_indices = {identity: index for index, identity in enumerate(block.output_identities)}
    jacobian = np.asarray([
        [block.jacobian[output_indices[f"mu:{species}"]][index] for index in active]
        for species in block.component_ids
    ], dtype=float)
    if not np.all(np.isfinite(jacobian)):
        raise RuntimeError("Engine chemical-potential loading derivative is non-finite")
    return jacobian @ direction[active]
