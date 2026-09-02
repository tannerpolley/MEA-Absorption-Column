from __future__ import annotations

import hashlib
import json
import math
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


@lru_cache(maxsize=4)
def validate_reactive_bundle(dataset_text: str) -> dict:
    dataset = Path(dataset_text)
    bundle = _json(dataset / "bundle.json")
    expected = {
        Path(item["path"]).name: item["sha256"]
        for item in bundle["files"]
        if item["path"] in {"parameters/parameters.json", "chemistry/reaction-system.json"}
    }
    for name, digest in expected.items():
        actual = hashlib.sha256((dataset / name).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(
                f"Reactive ePC-SAFT {name} SHA-256 mismatch: expected {digest}, got {actual}"
            )
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
    return {"bundle": bundle, "reactions": reactions}


def compile_reaction_constants(dataset_text: str, temperature_k: float) -> tuple[list, ...]:
    reactions = validate_reactive_bundle(dataset_text)["reactions"]
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
            if set(coefficients) == {"a", "b_k"}:
                metadata = {
                    "reaction_id": reaction["reaction_id"],
                    "kind": "ln-k-a-plus-b-over-t",
                    "coefficient_identities": [
                        f"reaction:{reaction['reaction_id']}:correlation:{name}"
                        for name in coefficients
                    ],
                    "coefficient_values": list(coefficients.values()),
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
                    for name in coefficients
                ],
                "coefficient_values": list(coefficients.values()),
            }
        else:
            raise ValueError(f"Unsupported reaction correlation: {form}")
        entry = [
            float(value),
            "mea-reactive-epcsaft-parameter-bundle",
            reactions["source_standard_state"]["id"],
            "products_positive",
            "source-standard-state-to-provider-neutral-reference",
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
) -> dict:
    dataset = Path(dataset_text)
    reactions = validate_reactive_bundle(dataset_text)["reactions"]
    apparent = np.asarray(apparent_amounts, dtype=float)
    if apparent.shape != (3,) or np.any(~np.isfinite(apparent)) or np.any(apparent <= 0.0):
        raise ValueError("Reactive ePC-SAFT requires positive finite CO2/MEA/H2O amounts")
    apparent /= float(apparent.sum())
    trace = 1.0e-10
    feed = [
        *(float(value) * (1.0 - 9.0 * trace) for value in apparent),
        trace,
        trace,
        trace,
        trace,
        4.0 * trace,
        trace,
    ]
    charges = np.asarray(reactions["charges"], dtype=float)
    if abs(float(charges @ feed)) > 1.0e-15:
        raise RuntimeError("Reactive ePC-SAFT seed is not electroneutral")
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
                    "kind": "provider",
                    "reference_id": "installed-provider-eos",
                    "admissible_packing_fraction_interval": [1.0e-6, 0.74],
                },
                "start": None,
            }
        ],
        "reaction_system": {
            "species_ids": species_ids,
            "charges": reactions["charges"],
            "molar_masses_kg_per_mol": _molar_masses(dataset, species_ids),
            "balance_matrix": balances,
            "conserved_totals": totals,
            "reaction_matrix": [reaction["stoichiometry"] for reaction in reactions["reactions"]],
            "feed_amounts_mol": feed,
            "equilibrium_constants": compile_reaction_constants(dataset_text, temperature_k),
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
            }
        ],
        "continuation": None,
        "feed": None,
        "phase_reactions": None,
        "intensive_boundaries": [],
    }


def solve_homogeneous_reactive_state(
    dataset_text: str,
    temperature_k: float,
    pressure_pa: float,
    apparent_amounts,
) -> dict:
    import epcsaft
    from epcsaft import equilibrium

    request = homogeneous_reactive_request(
        dataset_text, temperature_k, pressure_pa, apparent_amounts
    )
    problem = equilibrium.general_reactive_equilibrium_problem_from_mapping(request)
    model = epcsaft.Mixture(epcsaft.Parameters.from_json(Path(dataset_text) / "parameters.json"))
    result = equilibrium.solve(model, problem)
    if (
        result.status != "evaluated"
        or result.numerical_status != "passed"
        or result.physical_status != "passed"
    ):
        raise RuntimeError(
            f"Reactive ePC-SAFT equilibrium failed: {result.solver_status}; {result.failure}"
        )
    phase = result.phases[0]
    return {
        "composition": np.asarray(phase.mole_fractions, dtype=float),
        "density_mol_m3": float(phase.molar_density_mol_m3),
        "chemical_potentials_over_rt": np.asarray(
            phase.chemical_potential_over_rt, dtype=float
        ),
        "parameter_fingerprint": result.descriptor.parameter_fingerprint,
        "evidence": dict(result.evidence),
    }
