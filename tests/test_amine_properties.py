from dataclasses import replace

import numpy as np

from mea_absorption_column.Properties.Amine_Properties import MEA_PROPERTIES, resolve_amine_properties
from mea_absorption_column.Properties.Thermophysical_Properties import enthalpy, henrys_law
from mea_absorption_column.misc.Get_Temperature_Enthalpy import get_liquid_enthalpy, get_liquid_temperature


def _linear_enthalpy(temperature_K, liquid_mole_fractions, phase="liquid"):
    temperature_K = float(np.asarray(temperature_K).reshape(-1)[0])
    values = np.array([temperature_K, 2 * temperature_K, 3 * temperature_K], dtype=float)
    return values, float(np.dot(values, liquid_mole_fractions))


def test_default_mea_package_preserves_existing_correlations():
    x = [0.03, 0.10, 0.87]

    assert MEA_PROPERTIES.henry_co2(333.15, x) == henrys_law(333.15, x)
    assert np.allclose(MEA_PROPERTIES.enthalpy(333.15, x)[0], enthalpy(333.15, x, phase="liquid")[0])


def test_custom_amine_enthalpy_is_used_for_initialization_and_temperature_inversion():
    properties = replace(
        MEA_PROPERTIES,
        amine_id="test-amine",
        amine_molar_mass_kg_per_mol=0.11916,
        enthalpy_correlation=_linear_enthalpy,
    )
    flows = [0.1, 0.3, 0.6]

    mixture_enthalpy = get_liquid_enthalpy(flows, 340.0, properties)
    recovered_temperature = get_liquid_temperature(flows, mixture_enthalpy, properties)

    assert mixture_enthalpy == 850.0
    assert abs(recovered_temperature - 340.0) < 1.0e-6


def test_amine_properties_rejects_nonphysical_molar_mass():
    try:
        replace(MEA_PROPERTIES, amine_molar_mass_kg_per_mol=0.0)
    except ValueError as exc:
        assert "finite and positive" in str(exc)
    else:
        raise AssertionError("zero amine molar mass was accepted")


def test_amine_properties_rejects_missing_correlation():
    try:
        replace(MEA_PROPERTIES, diffusivity_correlation=None)
    except TypeError as exc:
        assert "diffusivity_correlation must be callable" in str(exc)
    else:
        raise AssertionError("missing amine diffusivity correlation was accepted")


def test_incomplete_amine_reports_all_missing_column_inputs():
    properties = replace(
        MEA_PROPERTIES,
        amine_id="test-amine",
        chemical_equilibrium_correlation=None,
        enhancement_factor_correlation=None,
        missing_column_inputs=("thermodynamic closure",),
    )

    try:
        resolve_amine_properties(properties, require_column_ready=True)
    except RuntimeError as exc:
        assert str(exc) == (
            "test-amine column model is incomplete; missing required inputs: "
            "thermodynamic closure; a solvent-specific chemical-equilibrium callable; "
            "a solvent-specific enhancement-factor callable"
        )
    else:
        raise AssertionError("incomplete amine was accepted for a column run")
