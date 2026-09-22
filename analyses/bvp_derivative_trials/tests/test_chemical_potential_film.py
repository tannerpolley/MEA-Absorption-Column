from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/run_chemical_potential_film.py"
SPEC = spec_from_file_location("chemical_potential_film", SCRIPT)
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_flux_expansion_closes_total_and_current():
    independent = np.asarray([0, 1, 4, 5, 6, 7, 8])
    dependent = np.asarray([2, 3])
    flux = MODULE._expand_flux(np.arange(7, dtype=float), independent, dependent)
    assert np.sum(flux) == pytest.approx(0.0)
    assert np.dot(MODULE.CHARGES, flux) == pytest.approx(0.0)


def test_rank_deficiency_is_reported_without_regularization():
    with pytest.raises(MODULE.ChemicalPotentialFilmError, match=r"rank=1, expected=2"):
        MODULE._rank_or_stop(np.ones((2, 2)), 2, "test block")


def test_cases_close_and_refine_with_expected_signs():
    coarse = MODULE.run(5)
    refined = MODULE.run(11)
    assert coarse["state_disposition"] == "basis_unresolved_unadmitted_provisional_numerical_state"
    for name, sign in (("zero_drive", 0), ("absorption", 1), ("desorption", -1)):
        result = refined[name]["chemical_potential"]
        assert result["calculation_status"] == "fixed_path_constrained_quadrature_completed"
        assert result["minimum_composition"] > 0.0
        assert result["maximum_normalization_residual"] <= 1.0e-12
        assert result["maximum_electroneutrality_residual"] <= 1.0e-12
        assert result["maximum_zero_total_flux_residual"] <= 1.0e-12
        assert result["maximum_zero_current_residual"] <= 1.0e-12
        assert result["maximum_species_conservation_residual"] <= 1.0e-12
        assert result["minimum_dissipation"] >= -1.0e-12
        flux = result["co2_flux_mol_m2_s"]
        if sign == 0:
            assert flux == 0.0
        elif sign > 0:
            assert flux > 0.0
        else:
            assert flux < 0.0
        if sign:
            coarse_flux = coarse[name]["chemical_potential"]["co2_flux_mol_m2_s"]
            assert abs(coarse_flux - flux) / abs(flux) <= 1.0e-10
