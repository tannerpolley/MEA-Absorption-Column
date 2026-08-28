import math
from pathlib import Path

import numpy as np
import pytest

import mea_absorption_column.Thermodynamics.thermo_models as thermo_models
from mea_absorption_column.Thermodynamics.Fugacity import fugacity
from mea_absorption_column.Thermodynamics.epcsaft_v02 import (
    fugacity_coefficients,
    parameter_document,
    parameters,
    state as epcsaft_state,
)
from mea_absorption_column.Thermodynamics.thermo_models import (
    IONIC_LIQUID_SPECIES,
    IONIC_LIQUID_SPECIES_9,
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
    ensure_epcsaft_importable,
    epcsaft_dataset_mixture,
    epcsaft_liquid_transport_state,
    epcsaft_source_fingerprint,
    ionic_liquid_composition,
    neutral_liquid_composition,
)


def test_ideal_henry_fugacity_returns_positive_finite_values():
    fl_co2, fv_co2, fl_h2o, fv_h2o, co2, h2o = fugacity(
        x=[0.07, 0.28, 0.65],
        y=[0.08, 0.05, 0.80, 0.07],
        x_true=[0.02, 0.24, 0.62, 0.06, 0.05, 0.01],
        Cl_true=[900.0, 9000.0, 30000.0, 1800.0, 1500.0, 200.0],
        Tl=323.15,
        Tv=319.15,
        alpha=0.25,
        H_CO2_mix=2.5e6,
        P=109500.0,
        P_sat_H2O=12000.0,
        thermo_model="ideal_henry",
    )

    values = [fl_co2, fv_co2, fl_h2o, fv_h2o, co2[0], co2[1], h2o[0], h2o[1]]
    assert all(math.isfinite(value) for value in values)
    assert fl_co2 > 0.0
    assert fv_co2 > 0.0


def test_neutral_liquid_composition_renormalizes_co2_mea_h2o_only():
    x_neutral = neutral_liquid_composition([0.02, 0.24, 0.62, 0.06, 0.05, 0.01])

    assert np.allclose(x_neutral.sum(), 1.0)
    assert x_neutral.shape == (3,)
    assert x_neutral[0] > 0.0


def test_epcsaft_parameter_document_has_all_species_molar_masses_and_pair_records():
    document = parameter_document(str(MEA_THERMODYNAMICS_EPCSAFT_DATASET))
    components = {record["name"]: record for record in document["components"]}

    assert set(components) == set(IONIC_LIQUID_SPECIES_9)
    assert all(
        record["fixed"]["molar_mass"]["value"]["magnitude"] > 0.0
        for record in components.values()
    )
    assert all(
        record["fixed"]["molar_mass"]["value"]["unit"] == "kilogram / mole"
        for record in components.values()
    )
    mea_water = next(
        record
        for record in document["pairs"]
        if {record["component_id_a"], record["component_id_b"]}
        == {"monoethanolamine", "water"}
    )
    assert mea_water["coefficients"][0]["value"]["magnitude"] == pytest.approx(-0.052)
    assert (
        "status=explicit-selected-dataset"
        in mea_water["coefficients"][0]["provenance"]["locator"]
    )


def test_epcsaft_neutral_fugacity_returns_positive_finite_values():
    try:
        ensure_epcsaft_importable()
    except RuntimeError as exc:
        pytest.skip(f"external ePC-SAFT native extension unavailable: {exc}")
    fl_co2, fv_co2, fl_h2o, fv_h2o, co2, h2o = fugacity(
        x=[0.07, 0.28, 0.65],
        y=[0.08, 0.05, 0.80, 0.07],
        x_true=[0.02, 0.24, 0.62, 0.06, 0.05, 0.01],
        Cl_true=[900.0, 9000.0, 30000.0, 1800.0, 1500.0, 200.0],
        Tl=323.15,
        Tv=319.15,
        alpha=0.25,
        H_CO2_mix=2.5e6,
        P=109500.0,
        P_sat_H2O=12000.0,
        thermo_model="epcsaft_neutral",
    )

    values = [fl_co2, fv_co2, fl_h2o, fv_h2o, co2[0], h2o[0]]
    assert all(math.isfinite(value) for value in values)
    assert fl_co2 > 0.0
    assert fv_co2 > 0.0


def test_epcsaft_fugacity_blend_interpolates_between_henry_and_epcsaft():
    try:
        ensure_epcsaft_importable()
    except RuntimeError as exc:
        pytest.skip(f"external ePC-SAFT native extension unavailable: {exc}")
    kwargs = {
        "x": [0.07, 0.28, 0.65],
        "y": [0.08, 0.05, 0.80, 0.07],
        "x_true": [0.02, 0.24, 0.62, 0.06, 0.05, 0.01],
        "Cl_true": [900.0, 9000.0, 30000.0, 1800.0, 1500.0, 200.0],
        "Tl": 323.15,
        "Tv": 319.15,
        "alpha": 0.25,
        "H_CO2_mix": 2.5e6,
        "P": 109500.0,
        "P_sat_H2O": 12000.0,
        "thermo_model": "epcsaft_neutral",
    }

    henry_like = fugacity(**kwargs, epcsaft_fugacity_blend=0.0)[:4]
    epcsaft = fugacity(**kwargs, epcsaft_fugacity_blend=1.0)[:4]
    blended = fugacity(**kwargs, epcsaft_fugacity_blend=0.25)[:4]

    assert np.allclose(
        blended, 0.75 * np.asarray(henry_like) + 0.25 * np.asarray(epcsaft)
    )


def test_epcsaft_fugacity_blend_is_clipped_to_valid_range():
    try:
        ensure_epcsaft_importable()
    except RuntimeError as exc:
        pytest.skip(f"external ePC-SAFT native extension unavailable: {exc}")
    kwargs = {
        "x": [0.07, 0.28, 0.65],
        "y": [0.08, 0.05, 0.80, 0.07],
        "x_true": [0.02, 0.24, 0.62, 0.06, 0.05, 0.01],
        "Cl_true": [900.0, 9000.0, 30000.0, 1800.0, 1500.0, 200.0],
        "Tl": 323.15,
        "Tv": 319.15,
        "alpha": 0.25,
        "H_CO2_mix": 2.5e6,
        "P": 109500.0,
        "P_sat_H2O": 12000.0,
        "thermo_model": "epcsaft_neutral",
    }

    henry_like = fugacity(**kwargs, epcsaft_fugacity_blend=0.0)[:4]
    epcsaft = fugacity(**kwargs, epcsaft_fugacity_blend=1.0)[:4]

    assert np.allclose(fugacity(**kwargs, epcsaft_fugacity_blend=-1.0)[:4], henry_like)
    assert np.allclose(fugacity(**kwargs, epcsaft_fugacity_blend=2.0)[:4], epcsaft)


def test_epcsaft_adapter_reports_external_source_without_modifying_it():
    try:
        ensure_epcsaft_importable()
    except RuntimeError as exc:
        pytest.skip(f"external ePC-SAFT package unavailable: {exc}")
    fingerprint = epcsaft_source_fingerprint()

    assert fingerprint["package"] == "epcsaft"
    assert fingerprint["installed"] is True
    assert fingerprint["exists"] is True
    assert Path(fingerprint["module_path"]).exists()
    assert fingerprint["source_kind"] in {
        "release",
        "pinned_git",
        "direct_url",
        "local_file",
    }
    assert "source_root" not in fingerprint


def test_ionic_epcsaft_document_declares_fixed_born_and_permittivity_models():
    assert MEA_THERMODYNAMICS_EPCSAFT_DATASET.name == "MEA_CO2_H2O_ionic_fit"
    document = parameter_document(str(MEA_THERMODYNAMICS_EPCSAFT_DATASET))
    families = {record["kind"]: record for record in document["model_families"]}
    charged = [
        record
        for record in document["components"]
        if record["fixed"]["charge_number"]["value"]["magnitude"] != 0
    ]

    assert families["electrolyte"]["choice"] == "born"
    assert families["electrolyte"]["c_shell"] == pytest.approx(1.0)
    assert families["electrolyte"]["c_dielectric"] == pytest.approx(1.0)
    assert families["permittivity"]["choice"] == "ion-fraction-suppression"
    assert charged
    assert all(
        any(value["family"] == "born_diameter" for value in record["coefficients"])
        for record in charged
    )


def test_ionic_epcsaft_state_uses_ion_and_born_contribution_terms():
    try:
        ensure_epcsaft_importable()
    except RuntimeError as exc:
        pytest.skip(f"external ePC-SAFT native extension unavailable: {exc}")

    mixture = epcsaft_dataset_mixture(tuple(IONIC_LIQUID_SPECIES), 323.2)
    composition = ionic_liquid_composition([0.02, 0.24, 0.62, 0.06, 0.05, 0.01])
    state = epcsaft_state(
        mixture,
        temperature_k=323.15,
        pressure_pa=109500.0,
        composition=composition,
        phase="liquid",
    )

    assert abs(float(state.debye_huckel)) > 1.0e-8
    assert abs(float(state.born)) > 1.0e-8
    assert all(value > 0.0 for value in fugacity_coefficients(state))


def test_full_species_ionic_epcsaft_state_uses_all_nine_species():
    try:
        ensure_epcsaft_importable()
    except RuntimeError as exc:
        pytest.skip(f"external ePC-SAFT native extension unavailable: {exc}")

    mixture = epcsaft_dataset_mixture(tuple(IONIC_LIQUID_SPECIES_9), 323.2)
    composition = ionic_liquid_composition(
        [0.02, 0.23, 0.62, 0.06, 0.05, 0.01, 1.0e-6, 2.0e-6, 1.0e-6]
    )
    state = epcsaft_state(
        mixture,
        temperature_k=323.15,
        pressure_pa=109500.0,
        composition=composition,
        phase="liquid",
    )
    phi = fugacity_coefficients(state)

    assert len(phi) == len(IONIC_LIQUID_SPECIES_9)
    assert abs(float(state.debye_huckel)) > 1.0e-8
    assert abs(float(state.born)) > 1.0e-8


def test_installed_epcsaft_exposes_exact_fixed_pressure_transport_tangent():
    composition = np.array([1.0, 20.0, 70.0, 3.0, 2.0, 0.5, 0.25, 0.5, 0.5])
    composition /= composition.sum()

    state = epcsaft_liquid_transport_state(318.15, 109500.0, composition)

    assert state.fugacities_pa.shape == (9,)
    assert state.log_composition_basis.shape == (9, 7)
    assert state.chemical_potential_derivatives_over_rt.shape == (9, 7)
    assert state.coordinate_component_ids == (
        "carbon-dioxide",
        "monoethanolamine",
        "carbamate-anion",
        "bicarbonate-anion",
        "carbonate-anion",
        "hydronium-cation",
        "hydroxide-anion",
    )
    assert state.dependent_component_ids == ("water", "protonated-monoethanolamine")
    assert state.fixed_other_concentrations_log_fugacity_derivative(0) == pytest.approx(
        0.9467107695956956, rel=1.0e-12
    )
    assert state.artifact_fingerprint.startswith("sha256:")


def test_installed_epcsaft_returns_structured_no_matrix_derivative_failure():
    mixture = epcsaft_dataset_mixture(tuple(IONIC_LIQUID_SPECIES_9), 318.2)
    composition = np.array([0.0, 20.0, 70.0, 3.0, 2.0, 0.5, 0.25, 0.5, 0.5])
    composition /= composition.sum()
    state = epcsaft_state(
        mixture,
        temperature_k=318.15,
        pressure_pa=109500.0,
        composition=composition,
        phase="liquid",
    )

    block = state.fixed_pressure_composition_derivatives
    assert block["status"] == "non_evaluable"
    assert block["failure"].code == "provider_domain_rejection"
    assert "log_composition_basis" not in block
    assert "chemical_potential_derivatives_over_rt" not in block


def test_retained_predictive_parameters_apply_co2_water_temperature_relationship():
    dataset = (
        Path(__file__).parents[1]
        / "src/mea_absorption_column/data/epcsaft_datasets/MEA_CO2_H2O_retained_predictive"
    )

    def co2_water_kij(temperature_k):
        mapping = parameters(
            str(dataset), tuple(IONIC_LIQUID_SPECIES_9), temperature_k
        ).to_mapping()
        return next(
            coefficient["value"]["magnitude"]
            for pair in mapping["pairs"]
            for coefficient in pair["coefficients"]
            if coefficient["identity"] == "pair/carbon-dioxide/water/k_ij"
        )

    assert co2_water_kij(313.15) == pytest.approx(0.0)
    assert co2_water_kij(333.15) == pytest.approx(0.006032)


def test_uncached_fugacity_uses_canonical_pressure_state(monkeypatch):
    sentinel_state = object()
    monkeypatch.setattr(thermo_models, "epcsaft_mixture", lambda: object())
    monkeypatch.setattr(
        thermo_models, "_v02_state", lambda *args, **kwargs: sentinel_state
    )
    monkeypatch.setattr(
        thermo_models,
        "_pressure_state_with_optional_rho_guess",
        lambda *args, **kwargs: pytest.fail(
            "uncached fugacity used the approximate warm-density solve"
        ),
    )
    monkeypatch.setattr(
        thermo_models,
        "_v02_fugacity_coefficients",
        lambda state: (2.0, 1.0, 1.0) if state is sentinel_state else (),
    )

    assert thermo_models.epcsaft_phi_co2(
        323.15,
        109500.0,
        np.array([0.02, 0.24, 0.74]),
        phase="liq",
        cache=False,
    ) == pytest.approx(2.0)
