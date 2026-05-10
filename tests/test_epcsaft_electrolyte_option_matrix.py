import copy
import json

import numpy as np
import pytest

from mea_absorption_column.Thermodynamics.thermo_models import (
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
    compute_fugacity,
    clear_epcsaft_phi_cache,
    ensure_epcsaft_importable,
    epcsaft_dataset_user_options,
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


def _options(
    *,
    include_born=True,
    d_born_mode=0,
    ssm=False,
    ds=False,
    mu_mode="analytical",
    rel_perm_rule="linear",
    rel_perm_mode="analytical",
):
    return {
        "elec_model": {
            "rel_perm": {"rule": rel_perm_rule, "differential_mode": rel_perm_mode},
            "include_born_model": include_born,
            "born_model": {
                "d_Born_mode": d_born_mode,
                "solvation_shell_model": ssm,
                "dielectric_saturation": ds,
                "mu_born_model": {
                    "differential_mode": mu_mode,
                    "comp_dep_delta_d": bool(ssm or ds),
                },
            },
        }
    }


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
    assert abs(diagnostics["lnfugcoef_co2_terms"]["ion"]) < 1.0e-12
    assert abs(diagnostics["lnfugcoef_co2_terms"]["born"]) < 1.0e-12


@pytest.mark.parametrize(
    ("name", "options", "expect_born_nonzero"),
    [
        ("dataset_default", None, True),
        ("ion_only_born_disabled", _options(include_born=False), False),
        ("classic_born_sigma_radius", _options(), True),
        (
            "fitted_born_ssm_only",
            _options(
                d_born_mode=3,
                ssm=True,
                ds=False,
                mu_mode="numerical",
                rel_perm_rule="empirical",
                rel_perm_mode="numerical",
            ),
            True,
        ),
        (
            "fitted_born_ds_only",
            _options(
                d_born_mode=3,
                ssm=False,
                ds=True,
                mu_mode="numerical",
                rel_perm_rule="empirical",
                rel_perm_mode="numerical",
            ),
            True,
        ),
        (
            "fitted_born_ssm_ds",
            _options(
                d_born_mode=3,
                ssm=True,
                ds=True,
                mu_mode="numerical",
                rel_perm_rule="empirical",
                rel_perm_mode="numerical",
            ),
            True,
        ),
    ],
)
def test_electrolyte_user_option_matrix_runs_and_reports_terms(name, options, expect_born_nonzero):
    _requires_epcsaft()
    if options is None:
        options = epcsaft_dataset_user_options()

    diagnostics = epcsaft_state_contribution_diagnostics(
        323.15,
        109500.0,
        IONIC_X,
        phase="liq",
        mixture_kind="ionic",
        user_options=copy.deepcopy(options),
    )

    assert diagnostics["mixture_kind"] == "ionic"
    assert diagnostics["phi_co2"] > 0.0
    assert abs(diagnostics["ares_terms"]["ion"]) > 1.0e-8, name
    if expect_born_nonzero:
        assert abs(diagnostics["ares_terms"]["born"]) > 1.0e-8, name
        assert abs(diagnostics["lnfugcoef_co2_terms"]["born"]) > 1.0e-8, name
    else:
        assert abs(diagnostics["ares_terms"]["born"]) < 1.0e-12, name
        assert abs(diagnostics["lnfugcoef_co2_terms"]["born"]) < 1.0e-12, name


def test_fitted_born_without_ssm_or_ds_fails_clearly():
    _requires_epcsaft()

    with pytest.raises(ValueError, match="fitted_param.*requires SSM/DS Born path"):
        epcsaft_state_contribution_diagnostics(
            323.15,
            109500.0,
            IONIC_X,
            phase="liq",
            mixture_kind="ionic",
            user_options=_options(d_born_mode="fitted_param", ssm=False, ds=False),
        )


def test_runtime_user_options_env_changes_ionic_fugacity_path(monkeypatch):
    _requires_epcsaft()
    monkeypatch.setenv(
        "MEA_EPCSAFT_USER_OPTIONS_JSON",
        json.dumps({"elec_model": {"rel_perm": {"rule": "linear"}, "include_born_model": False}}),
    )
    clear_epcsaft_phi_cache()
    no_born = epcsaft_phi_co2(323.15, 109500.0, IONIC_X, phase="liq", mixture_kind="ionic")

    monkeypatch.setenv(
        "MEA_EPCSAFT_USER_OPTIONS_JSON",
        json.dumps(
            _options(
                d_born_mode=3,
                ssm=True,
                ds=True,
                mu_mode="numerical",
                rel_perm_rule="empirical",
                rel_perm_mode="numerical",
            )
        ),
    )
    clear_epcsaft_phi_cache()
    with_born = epcsaft_phi_co2(323.15, 109500.0, IONIC_X, phase="liq", mixture_kind="ionic")

    assert no_born > 0.0
    assert with_born > 0.0
    assert abs(no_born - with_born) > 1.0e-6


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
