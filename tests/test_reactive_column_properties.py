"""Fit-A caloric and fugacity checks at retained column states."""
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

from mea_absorption_column.Thermodynamics.reactive_bundle import FIT_A, ReactiveLiquid


CHECKS = SimpleNamespace(**runpy.run_path(str(Path(__file__).resolve().parents[1]
    /'analyses/nccc_validation/scripts/check_epcsaft_properties.py')))


def test_fit_a_phase_enthalpy_and_transferred_enthalpy():
    references = CHECKS.reference_phases()
    _, liquid, vapor = CHECKS.profile_states('3C')[0]
    comparison = CHECKS.caloric_comparison(ReactiveLiquid(FIT_A, loading_anchor=.25), references, liquid, vapor)
    for name, limit in CHECKS.ABSOLUTE_LIMITS.items():
        actual, expected = comparison[name]
        assert actual == pytest.approx(expected, abs=limit, rel=0.)
    for name, limit in CHECKS.RELATIVE_LIMITS.items():
        actual, expected = comparison[name]
        assert actual == pytest.approx(expected, rel=limit)
    transfer = CHECKS.interface_comparison(references, liquid, vapor)
    np.testing.assert_allclose(transfer['phase_energy'], transfer['expected_phase_energy'], atol=10., rtol=0.)
    np.testing.assert_allclose(transfer['liquid_components'], transfer['expected_liquid_components'], atol=10., rtol=0.)


def test_fit_a_signed_water_flux_and_column_jacobian():
    liquid = (318.15, 109500., 2.5, 10., 79.)
    vapor = (316.75, 109500., 2., 1.67, 20., 2.)
    comparison = CHECKS.interface_comparison(CHECKS.reference_phases(), liquid, vapor)
    assert comparison['water_flux'] < 0
    assert comparison['water_flux'] == pytest.approx(comparison['expected_water_flux'], abs=1e-6, rel=0.)
    for actual, expected in CHECKS.directional_comparison(liquid, vapor):
        np.testing.assert_allclose(actual, expected, rtol=CHECKS.RTOL, atol=CHECKS.ATOL)
