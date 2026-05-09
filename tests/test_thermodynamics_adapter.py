import math
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mea_absorption_column.Thermodynamics.Fugacity import fugacity
from mea_absorption_column.Thermodynamics.thermo_models import (
    EPCSAFT_SOURCE_ROOT,
    IONIC_LIQUID_SPECIES,
    IONIC_LIQUID_SPECIES_9,
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
    build_epcsaft_params,
    ensure_epcsaft_importable,
    epcsaft_dataset_mixture,
    epcsaft_source_fingerprint,
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


def test_epcsaft_parameter_dataset_has_expected_shape_and_documented_species():
    params = build_epcsaft_params()

    assert params["m"].shape == (3,)
    assert params["k_ij"].shape == (3, 3)
    assert params["species"] == ["CO2", "MEA", "H2O"]
    assert params["assoc_scheme"] == [None, "3b", "4c"]
    assert np.allclose(params["k_ij"], params["k_ij"].T)
    assert np.isclose(params["k_ij"][1, 2], -0.052)
    assert Path(params["metadata_path"]).exists()


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

    assert np.allclose(blended, 0.75 * np.asarray(henry_like) + 0.25 * np.asarray(epcsaft))


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
    fingerprint = epcsaft_source_fingerprint()

    assert fingerprint["source_root"] == str(EPCSAFT_SOURCE_ROOT)
    assert fingerprint["exists"] is True
    assert "modified_at_utc" in fingerprint


def test_ionic_epcsaft_dataset_enables_ssm_ds_and_dborn_parameters():
    options_path = MEA_THERMODYNAMICS_EPCSAFT_DATASET / "user_options.json"
    pure_path = MEA_THERMODYNAMICS_EPCSAFT_DATASET / "pure" / "any_solvent.csv"

    with options_path.open("r", encoding="utf-8") as handle:
        options = json.load(handle)
    born_options = options["elec_model"]["born_model"]

    assert born_options["d_Born_mode"] == 3
    assert born_options["solvation_shell_model"] is True
    assert born_options["dielectric_saturation"] is True
    assert born_options["mu_born_model"]["comp_dep_delta_d"] is True

    with pure_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    charged_rows = [row for row in rows if abs(float(row["z"])) > 0.0]

    assert charged_rows
    assert {row["component"] for row in charged_rows}.issuperset({"MEAH+", "MEACOO-", "HCO3-"})
    assert all(float(row["d_born"]) > 0.0 for row in charged_rows)


def test_ionic_epcsaft_state_uses_ion_and_born_contribution_terms():
    try:
        ensure_epcsaft_importable()
    except RuntimeError as exc:
        pytest.skip(f"external ePC-SAFT native extension unavailable: {exc}")

    mixture = epcsaft_dataset_mixture(tuple(IONIC_LIQUID_SPECIES), 323.2)
    composition = np.array([0.02, 0.24, 0.62, 0.06, 0.05, 0.01], dtype=float)
    composition /= composition.sum()
    state = mixture.state(T=323.15, x=composition, P=109500.0, phase="liq")

    ares = state.residual_helmholtz(return_contribution_terms=True)
    lnfug = state.fugacity_coefficient(natural_log=True, return_contribution_terms=True)

    assert abs(float(ares["terms"]["ion"])) > 1.0e-8
    assert abs(float(ares["terms"]["born"])) > 1.0e-8
    assert np.any(np.abs(np.asarray(lnfug["terms"]["ion"], dtype=float)) > 1.0e-8)
    assert np.any(np.abs(np.asarray(lnfug["terms"]["born"], dtype=float)) > 1.0e-8)


def test_full_species_ionic_epcsaft_state_uses_all_nine_species():
    try:
        ensure_epcsaft_importable()
    except RuntimeError as exc:
        pytest.skip(f"external ePC-SAFT native extension unavailable: {exc}")

    mixture = epcsaft_dataset_mixture(tuple(IONIC_LIQUID_SPECIES_9), 323.2)
    composition = np.array([0.02, 0.23, 0.62, 0.06, 0.05, 0.01, 1.0e-6, 2.0e-6, 1.0e-6], dtype=float)
    composition /= composition.sum()
    state = mixture.state(T=323.15, x=composition, P=109500.0, phase="liq")
    phi = state.fugacity_coefficient(natural_log=True)
    lnfug = state.fugacity_coefficient(natural_log=True, return_contribution_terms=True)

    assert len(phi) == len(IONIC_LIQUID_SPECIES_9)
    assert len(lnfug["terms"]["ion"]) == len(IONIC_LIQUID_SPECIES_9)
    assert len(lnfug["terms"]["born"]) == len(IONIC_LIQUID_SPECIES_9)
