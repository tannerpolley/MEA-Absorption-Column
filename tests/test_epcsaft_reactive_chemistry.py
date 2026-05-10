import numpy as np
import pandas as pd
import pytest

from mea_absorption_column.misc.Convert_Data import convert_data
from mea_absorption_column.Thermodynamics.Chemical_Equilibrium import chemical_equilibrium_with_model
from mea_absorption_column.Thermodynamics.thermo_models import (
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
    ensure_epcsaft_importable,
)

SPECIES_9 = ("CO2", "MEA", "H2O", "MEAH+", "MEACOO-", "HCO3-", "CO3^2-", "H3O+", "OH-")


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


def test_epcsaft_reactive_six_concentration_matches_legacy_case_3c_state():
    _requires_reactive_epcsaft_dataset()
    Fl, Tl, P = _case_3c_liquid_state()

    _, legacy_x = chemical_equilibrium_with_model(Fl, Tl, model="legacy", P=P)
    _, epcsaft_x = chemical_equilibrium_with_model(
        Fl,
        Tl,
        model="epcsaft_reactive_six_concentration",
        P=P,
        diagnostics={},
    )

    np.testing.assert_allclose(np.sum(epcsaft_x), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(epcsaft_x, legacy_x, rtol=1.0e-3, atol=5.0e-5)


def test_epcsaft_reactive_six_activity_basis_changes_case_3c_speciation():
    _requires_reactive_epcsaft_dataset()
    Fl, Tl, P = _case_3c_liquid_state()

    _, legacy_x = chemical_equilibrium_with_model(Fl, Tl, model="legacy", P=P)
    _, activity_x = chemical_equilibrium_with_model(
        Fl,
        Tl,
        model="epcsaft_reactive_six_activity",
        P=P,
        diagnostics={},
    )

    np.testing.assert_allclose(np.sum(activity_x), 1.0, atol=1.0e-12)
    assert activity_x[0] > 10.0 * legacy_x[0]
    assert activity_x[4] < legacy_x[4]


def test_epcsaft_reactive_six_activity_converted_uses_concentration_basis_units():
    _requires_reactive_epcsaft_dataset()
    Fl, Tl, P = _case_3c_liquid_state()

    _, legacy_x = chemical_equilibrium_with_model(Fl, Tl, model="legacy", P=P)
    _, activity_x = chemical_equilibrium_with_model(
        Fl,
        Tl,
        model="epcsaft_reactive_six_activity",
        P=P,
        diagnostics={},
    )
    _, converted_x = chemical_equilibrium_with_model(
        Fl,
        Tl,
        model="epcsaft_reactive_six_activity_converted",
        P=P,
        diagnostics={},
    )

    np.testing.assert_allclose(np.sum(converted_x), 1.0, atol=1.0e-12)
    assert converted_x[0] < activity_x[0]
    assert abs(converted_x[4] - legacy_x[4]) < abs(activity_x[4] - legacy_x[4])


def test_epcsaft_reactive_nine_activity_rebased_solves_case_3c_state():
    _requires_reactive_epcsaft_dataset()
    Fl, Tl, P = _case_3c_liquid_state()
    diagnostics = {}

    Cl_true, x_true = chemical_equilibrium_with_model(
        Fl,
        Tl,
        model="epcsaft_reactive_nine_activity_rebased",
        P=P,
        diagnostics=diagnostics,
    )

    np.testing.assert_allclose(np.sum(x_true), 1.0, atol=1.0e-12)
    assert len(x_true) == len(SPECIES_9)
    assert len(Cl_true) == len(SPECIES_9)
    assert diagnostics["epcsaft_chemistry_max_mass_residual"] < 1.0e-6
    assert diagnostics["epcsaft_chemistry_max_reaction_residual"] < 1.0e-6
    assert diagnostics["epcsaft_chemistry_max_charge_residual"] < 1.0e-6
    assert x_true[SPECIES_9.index("H3O+")] > 0.0
    assert x_true[SPECIES_9.index("OH-")] > 0.0
    assert x_true[SPECIES_9.index("CO3^2-")] > 0.0
