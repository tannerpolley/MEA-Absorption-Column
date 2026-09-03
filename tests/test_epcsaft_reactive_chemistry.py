import csv

import numpy as np
import pandas as pd
import pytest

from mea_absorption_column.misc.Convert_Data import convert_data
from mea_absorption_column.Thermodynamics.Chemical_Equilibrium import chemical_equilibrium_with_model
from mea_absorption_column.Thermodynamics.thermo_models import (
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
    ensure_epcsaft_importable,
)
from mea_absorption_column.Thermodynamics.reactive_bundle import (
    compile_reaction_constants,
    homogeneous_reactive_request,
    solve_homogeneous_reactive_state,
    validate_reactive_bundle,
)

def _requires_reactive_epcsaft_dataset():
    try:
        ensure_epcsaft_importable()
    except RuntimeError as exc:
        pytest.skip(f"external ePC-SAFT native extension unavailable: {exc}")
    assert MEA_THERMODYNAMICS_EPCSAFT_DATASET.exists()


def _case_3c_liquid_state():
    df = pd.read_csv("src/mea_absorption_column/data/C_cases_data.csv", index_col=0)
    inputs, _, _ = convert_data(df, run=df.index.get_loc("3C"), type="mole", return_metadata=True)
    Fl, _, Tl, _, _, _, _, P, _ = inputs
    return list(Fl), float(Tl), float(P)


@pytest.mark.parametrize(
    "model",
    (
        "epcsaft_reactive_six_concentration",
        "epcsaft_reactive_six_activity",
        "epcsaft_reactive_six_activity_converted",
    ),
)
def test_legacy_reactive_modes_fail_closed_until_constants_meet_v02_contract(model):
    _requires_reactive_epcsaft_dataset()
    Fl, Tl, P = _case_3c_liquid_state()

    with pytest.raises(RuntimeError, match="independently sourced.*standard-state conversion"):
        chemical_equilibrium_with_model(Fl, Tl, model=model, P=P, diagnostics={})


def test_bundle_reactive_nine_species_state_compiles_temperature_dependent_constants():
    _requires_reactive_epcsaft_dataset()
    constants = compile_reaction_constants(str(MEA_THERMODYNAMICS_EPCSAFT_DATASET), 313.15)
    diagnostics = {}
    concentrations, composition = chemical_equilibrium_with_model(
        [0.1529, 1.0, 7.911],
        313.15,
        model="epcsaft_reactive_nine",
        P=101325.0,
        diagnostics=diagnostics,
    )

    assert [entry[0] for entry in constants] == pytest.approx(
        [-31.17540213354555, -14.505037067112221, -23.53653838436852, -2.7007932503784, -20.30164053537688]
    )
    assert concentrations.shape == composition.shape == (9,)
    assert composition.sum() == pytest.approx(1.0, abs=1.0e-12)
    assert np.dot(composition, [0, 0, 0, 1, -1, -1, -2, 1, -1]) == pytest.approx(0.0, abs=1.0e-12)
    assert diagnostics["epcsaft_chemistry_reaction_affinity_inf_norm"] < 1.0e-8
    state = solve_homogeneous_reactive_state(
        str(MEA_THERMODYNAMICS_EPCSAFT_DATASET),
        313.15,
        101325.0,
        (0.1529, 1.0, 7.911),
    )
    reactions = validate_reactive_bundle(str(MEA_THERMODYNAMICS_EPCSAFT_DATASET))[
        "reactions"
    ]["reactions"]
    reaction_matrix = np.asarray(
        [reaction["stoichiometry"] for reaction in reactions], dtype=float
    )
    assert reaction_matrix @ state["chemical_potentials_over_rt"] == pytest.approx(
        0.0, abs=1.0e-10
    )


def test_tabulated_reactive_equilibrium_interpolates_certified_amounts(tmp_path, monkeypatch):
    path = tmp_path / "speciation.csv"
    species = (
        "carbon-dioxide", "monoethanolamine", "water", "protonated-monoethanolamine",
        "carbamate-anion", "bicarbonate-anion", "carbonate-anion",
        "hydronium-cation", "hydroxide-anion",
    )
    fieldnames = ["status", "temperature_k", "loading", *(f"x_{name}" for name in species)]
    compositions = (
        (300.0, 0.2, [1e-6, .08, .88, .02, .015, .003, .002, 1e-8, 1e-6]),
        (300.0, 0.4, [2e-6, .06, .87, .035, .025, .006, .004, 1e-8, 1e-6]),
        (340.0, 0.2, [3e-6, .09, .87, .018, .014, .004, .003, 1e-8, 1e-6]),
        (340.0, 0.4, [4e-6, .07, .86, .032, .024, .008, .005, 1e-8, 1e-6]),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for temperature, loading, values in compositions:
            values = [value / sum(values) for value in values]
            writer.writerow({
                "status": "evaluated", "temperature_k": temperature, "loading": loading,
                **{f"x_{name}": value for name, value in zip(species, values)},
            })
    monkeypatch.setenv("MEA_EPCSAFT_REACTIVE_TABLE", str(path))
    diagnostics = {}
    _, x_true = chemical_equilibrium_with_model(
        [0.3, 1.0, 8.0], 320.0,
        model="epcsaft_reactive_nine_tabulated", diagnostics=diagnostics,
    )
    assert x_true.shape == (9,)
    assert x_true.sum() == pytest.approx(1.0)
    assert diagnostics["epcsaft_chemistry_table_hits"] == 1


def test_reactive_request_preserves_caller_amounts_and_exact_state():
    amounts = np.array([0.1529, 1.0, 7.911])
    original = amounts.copy()
    request = homogeneous_reactive_request(
        str(MEA_THERMODYNAMICS_EPCSAFT_DATASET), 313.15123, 101325.456, amounts
    )
    np.testing.assert_array_equal(amounts, original)
    assert request["temperature"]["value"] == 313.15123
    assert request["pressure"]["value"] == 101325.456
