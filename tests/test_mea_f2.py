import json
from importlib import import_module

import numpy as np
import pytest

from mea_absorption_column.Thermodynamics.reactive_bundle import ReactiveLiquid, ideal_gas_record
from mea_absorption_column.Thermodynamics.solvent_inputs import ROOT
from mea_absorption_column.Thermodynamics.thermo_models import ensure_epcsaft_importable


@pytest.mark.parametrize("dataset, expected", [
    ("MEA_F2_ae17803e", 118.8994811228556),  # Amine-Thermodynamics 761e38ae8, template 15-problem
    ("MEA_A_196faf11", 119.2512320828877),  # Amine-Thermodynamics a812493cc, reported fit A, replay row 15
])
def test_mea_co2_partial_pressure(dataset, expected):
    """vle_obs_0134-pco2 against the Amine-Thermodynamics prediction for each parameter record."""
    engine = ensure_epcsaft_importable()
    equilibrium = import_module("epcsaft.equilibrium")
    dataset = ROOT / "src/mea_absorption_column/data/epcsaft_datasets" / dataset
    parameters = engine.Parameters.from_mapping(json.loads((dataset / "parameters.json").read_bytes()))
    species = parameters.component_ids
    thermo = ideal_gas_record(dataset / "ideal-gas-thermochemistry.json", species)
    model = engine.Mixture(parameters, thermochemistry=thermo)
    record = json.loads((dataset / "engine-reactions.json").read_bytes())
    reactions = [
        dict(row, correlation=equilibrium.ReactionLogPolynomial(**row["correlation"]))
        for row in record["reactions"]
    ]
    problem = equilibrium.problem_from_mapping({
        "T": 313., "P": {"free": True, "guess": 100000.},
        "feed": {"basis": "amounts", "values": {
            "carbon-dioxide": .3889, "monoethanolamine": .99998, "water": 7.873352564834195,
            "protonated-monoethanolamine": 1e-5, "carbamate-anion": 1e-5, "bicarbonate-anion": 8e-5,
            "carbonate-anion": 1e-5, "hydronium-cation": .00011, "hydroxide-anion": 1e-5,
        }},
        "phases": [
            {"name": "mea-nine-species-liquid", "kind": "liquid", "support": species,
             "amount": {"status": "free"},
             "composition_guess": [
                 .0002838106172893405, .02954633922322694, .8834356036761588,
                 .043377306201163805, .0392633745564251, .004073199652147183,
                 2.036594667588587e-5, 1.3836563348558454e-11, 1.1307631006024982e-10,
             ]},
            {"name": "mea-neutral-incipient-vapor", "kind": "vapor", "support": species[:3],
             "amount": {"status": "pinned", "value": 0.},
             "composition_guess": [.8519140660203904, 8.081946279357392e-5, .14800511451681603]},
        ],
        "reactions": reactions, "neutral_reference": record["neutral_reference"],
        "topology_declared": True, "row_tolerance": 1e-10,
    })
    result = equilibrium.solve_equilibrium(model, problem)
    assert result.success and result.requested_tolerance_met, result.message
    actual = result.pressure * result.phase_mole_fractions(1)[species.index("carbon-dioxide")]
    assert actual == pytest.approx(expected, rel=1e-8, abs=0.)


def test_fit_a_liquid_matches_engine_equilibrium():
    engine = ensure_epcsaft_importable()
    equilibrium = import_module("epcsaft.equilibrium")
    dataset = ROOT / "src/mea_absorption_column/data/epcsaft_datasets/MEA_A_196faf11"

    temperature, pressure, feed = 313.15, 109500., np.array([.25, 1., 7.909])
    parameters = engine.Parameters.from_mapping(json.loads((dataset / "parameters.json").read_bytes()))
    model = engine.Mixture(
        parameters, thermochemistry=ideal_gas_record(dataset / "ideal-gas-thermochemistry.json", parameters.component_ids)
    )
    record = json.loads((dataset / "engine-reactions.json").read_bytes())
    problem = equilibrium.problem_from_mapping({
        "T": temperature, "P": pressure,
        "phases": [{"name": "liquid", "kind": "liquid"}],
        "feed": {"basis": "amounts", "values": dict(zip(parameters.component_ids[:3], feed / feed.sum()))},
        "reactions": [dict(row, correlation=equilibrium.ReactionLogPolynomial(**row["correlation"])) for row in record["reactions"]],
        "neutral_reference": record["neutral_reference"],
    })
    expected = equilibrium.solve_equilibrium(model, problem)
    assert expected.success, expected.message
    root = model.state(temperature, rho=expected.molar_densities[0], x=expected.mole_fractions)
    actual = ReactiveLiquid(dataset).solve(temperature, pressure, feed)
    # Use the relative precision of the independent native liquid-root check.
    np.testing.assert_allclose(actual["composition"], expected.mole_fractions, rtol=1e-8, atol=0.)
    assert actual["density_mol_m3"] == pytest.approx(expected.molar_densities[0], rel=1e-8)
    expected_fugacity = expected.mole_fractions[0] * np.exp(root.log_fugacity_coefficient[0]) * pressure
    assert actual["fugacities_pa"][0] == pytest.approx(expected_fugacity, rel=1e-8)
