import math
from pathlib import Path

import numpy as np
import pytest

from mea_absorption_column.Thermodynamics.Fugacity import fugacity
from mea_absorption_column.Thermodynamics.thermo_models import (
    EPCSAFT_SOURCE_ROOT,
    build_epcsaft_params,
    ensure_epcsaft_importable,
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
