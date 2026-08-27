import json
import shutil

import numpy as np
import pytest

from mea_absorption_column.Thermodynamics.epcsaft_v02 import (
    dataset_content_sha256,
    parameter_document_content_sha256,
)
from mea_absorption_column.Thermodynamics.thermo_models import (
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
    compute_fugacity,
    clear_epcsaft_phi_cache,
    ensure_epcsaft_importable,
    epcsaft_phi_co2,
    epcsaft_state_contribution_diagnostics,
)


IONIC_X = np.array([1.0e-8, 0.055, 0.888, 0.028, 0.027, 0.001], dtype=float)
IONIC_X = IONIC_X / IONIC_X.sum()


def _requires_epcsaft():
    try:
        ensure_epcsaft_importable()
    except RuntimeError as exc:
        pytest.skip(f"external ePC-SAFT native extension unavailable: {exc}")
    assert MEA_THERMODYNAMICS_EPCSAFT_DATASET.exists()


def test_neutral_path_has_no_electrolyte_contributions():
    _requires_epcsaft()

    diagnostics = epcsaft_state_contribution_diagnostics(
        323.15,
        109500.0,
        np.array([0.02, 0.24, 0.74], dtype=float),
        phase="liq",
        mixture_kind="neutral",
    )

    assert abs(diagnostics["ares_terms"]["ion"]) < 1.0e-12
    assert abs(diagnostics["ares_terms"]["born"]) < 1.0e-12
    assert diagnostics["lnfugcoef_co2_terms"] == {}
    assert diagnostics["parameter_fingerprint"].startswith("sha256:")


def test_fixed_electrolyte_model_runs_and_reports_terms():
    _requires_epcsaft()

    diagnostics = epcsaft_state_contribution_diagnostics(
        323.15,
        109500.0,
        IONIC_X,
        phase="liq",
        mixture_kind="ionic",
    )

    assert diagnostics["mixture_kind"] == "ionic"
    assert diagnostics["phi_co2"] > 0.0
    assert abs(diagnostics["ares_terms"]["ion"]) > 1.0e-8
    assert abs(diagnostics["ares_terms"]["born"]) > 1.0e-8
    assert diagnostics["lnfugcoef_co2_terms"] == {}
    assert diagnostics["parameter_fingerprint"].startswith("sha256:")


def test_parameter_content_identity_is_relocation_invariant(tmp_path):
    relocated = tmp_path / MEA_THERMODYNAMICS_EPCSAFT_DATASET.name
    shutil.copytree(MEA_THERMODYNAMICS_EPCSAFT_DATASET, relocated)

    assert dataset_content_sha256(str(relocated)) == dataset_content_sha256(
        str(MEA_THERMODYNAMICS_EPCSAFT_DATASET)
    )
    assert parameter_document_content_sha256(str(relocated)) == parameter_document_content_sha256(
        str(MEA_THERMODYNAMICS_EPCSAFT_DATASET)
    )


def test_runtime_model_family_override_fails_clearly():
    _requires_epcsaft()

    with pytest.raises(ValueError, match="removed in API 0.2"):
        epcsaft_state_contribution_diagnostics(
            323.15,
            109500.0,
            IONIC_X,
            phase="liq",
            mixture_kind="ionic",
            user_options={"elec_model": {"include_born_model": False}},
        )


def test_runtime_derivative_backend_override_fails_clearly(monkeypatch):
    _requires_epcsaft()
    monkeypatch.setenv(
        "MEA_EPCSAFT_USER_OPTIONS_JSON",
        json.dumps({"elec_model": {"rel_perm": {"differential_mode": "numerical"}}}),
    )
    with pytest.raises(RuntimeError, match="CppAD.*sole production derivative authority"):
        epcsaft_phi_co2(323.15, 109500.0, IONIC_X, phase="liq", mixture_kind="ionic")


def test_fixed_ionic_fugacity_is_cache_stable():
    _requires_epcsaft()
    clear_epcsaft_phi_cache()
    first = epcsaft_phi_co2(323.15, 109500.0, IONIC_X, phase="liq", mixture_kind="ionic")
    second = epcsaft_phi_co2(323.15, 109500.0, IONIC_X, phase="liq", mixture_kind="ionic")

    assert first > 0.0
    assert second == pytest.approx(first)


@pytest.mark.parametrize("alias", ["epcsaft_electrolyte", "epcsaft_full_ionic"])
def test_electrolyte_aliases_route_to_ionic_fugacity(alias):
    _requires_epcsaft()
    y = np.array([0.10, 0.08], dtype=float)
    cl_true = np.array([1.0e-4, 2400.0, 39000.0, 1200.0, 1180.0, 20.0])
    expected = compute_fugacity(
        "epcsaft_ionic",
        y,
        IONIC_X,
        cl_true,
        Tl=323.15,
        Tv=323.15,
        H_CO2_mix=1.0,
        P=109500.0,
        P_sat_H2O=12000.0,
    )

    actual = compute_fugacity(
        alias,
        y,
        IONIC_X,
        cl_true,
        Tl=323.15,
        Tv=323.15,
        H_CO2_mix=1.0,
        P=109500.0,
        P_sat_H2O=12000.0,
    )

    np.testing.assert_allclose(actual, expected)
